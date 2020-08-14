# pylint: disable=no-member
import math
import random 
import time
from os.path import join, exists
import torch
from torchvision import transforms
import numpy as np
import pickle
import gym 
import gym.envs.box2d
from torch.distributions import Normal, Categorical 
#from utils import sample_mdrnn_latent
from models import RSSModel 
from models import UpsdModel, UpsdBehavior
from control import Planner
from envs import get_env_params
import random 

class WeightedNormal:
    def __init__(self, mu, sigma, beta=1):
        self.normal = Normal(mu, sigma)
        self.beta = beta

    def sample(self, nsamps):
        s = self.normal.sample([nsamps])
        s = s*torch.exp(s/self.beta)
        #print("desired reward sampled", s)
        return s

import scipy.signal
def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

class Agent:
    def __init__(self, gamename,
        model = None, 
        desire_scalings=None, 
        take_rand_actions = False,
        desired_reward_stats=(1,1),
        desired_horizon = 250,
        desired_state = None, 
        delta_state = False, 
        Levine_Implementation = False, 
        #desired_reward_dist_beta = 1.0,
        discount_factor=1.0, model_version = 'checkpoint',
        return_plan_images=False,
        advantage_model=None, 
        td_lambda=1.0):
        """ Build vae, forward model, and environment. """

        self.gamename = gamename
        self.env_params = get_env_params(gamename)
        self.action_noise = self.env_params['action_noise']
        self.take_rand_actions = take_rand_actions
        self.discount_factor = discount_factor
        self.Levine_Implementation = Levine_Implementation
        self.advantage_model = advantage_model

        self.desired_state = desired_state
        self.delta_state = delta_state
        self.td_lambda = td_lambda
            
        self.desired_rew_mu, self.desired_rew_std, self.desired_horizon = desired_reward_stats[0], desired_reward_stats[1], desired_horizon
        if self.Levine_Implementation:
            print('the desired stats for mu and std are:', desired_reward_stats)
            self.desired_reward_dist = Normal(self.desired_rew_mu, 
                self.desired_rew_std)
            #WeightedNormal(desired_reward_stats[0], 
            #    desired_reward_stats[1], beta=desired_reward_dist_beta)

        # top, bottom, left, right
        self.obs_trim = self.env_params['trim_shape']

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.time_limit = self.env_params['time_limit']
        self.num_action_repeats = self.env_params['num_action_repeats']

        # transform used on environment generated observations. 
        if self.env_params['use_vae']:
            self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((self.env_params['IMAGE_RESIZE_DIM'], self.env_params['IMAGE_RESIZE_DIM'])),
                    transforms.ToTensor()
                ])

        if model:
            self.model = model 
            self.model.eval()

        # can be set to true inside Simulate. 
        self.return_events = False

    def make_env(self, seed, render_mode=False, full_episode=False):
        """ Called every time a new rollout occurs. Creates a new environment and 
        sets a new random seed for it."""
        self.render_mode = render_mode
        self.env = gym.make(self.env_params['env_name'])
        self.env.reset()
        #if not seed: 
        #seed = np.random.randint(0,1e9,1)[0]

        self.env.seed(int(seed))
        self.env.action_space.np_random.seed(int(seed))
        if render_mode: 
            self.env.render(mode='rgb_array')

    def _add_action_noise(self, action, noise):
        if noise is not None:
            action = action + noise * torch.randn_like(action)
        return action

    def constrain_actions(self, actions):
        """ Ensures actions sampled from the gaussians are within the game bounds."""
        for ind, (l, h) in enumerate(zip(self.env.action_space.low, self.env.action_space.high)):
            actions[:,ind] = torch.clamp(actions[:,ind], min=l, max=h)
        return actions

    def rollout(self, render=False, display_monitor=None,
            greedy=False):
        """ Executes a rollout and returns cumulative reward along with
        the time point the rollout stopped and 
        optionally, all observations/actions/rewards/terminal outputted 
        by the environment.

        :args:
            - rand_env_seed: int. Used to help guarentee this rollout is random.

        :returns: (cumulative, t, [rollout_dict])
            - cumulative: float. cumulative reward
            - time: int. timestep of termination
            - rollout_dict: OPTIONAL. Dictionary with keys: 'obs', 'rewards',
                'actions', 'terminal'. Each is a PyTorch Tensor with the first
                dimension corresponding to time. 
        """

        obs = self.env.reset()
        if self.env_params['give_raw_pixels']:
            obs = self.env.render(mode='rgb_array')
            #self.env.viewer.window.dispatch_events()

        # sample a desired reward
        if not self.take_rand_actions:
            if self.Levine_Implementation:
                #curr_desired_reward = torch.Tensor([np.random.uniform(self.desired_rew_mu, self.desired_rew_mu+self.desired_rew_std)])
                curr_desired_reward = self.desired_reward_dist.sample([1])
                curr_desired_state = torch.Tensor([self.desired_state])
            else: 
                curr_desired_reward = np.random.uniform(self.desired_rew_mu, self.desired_rew_mu+self.desired_rew_std)
                curr_desired_reward = torch.Tensor([min(curr_desired_reward, self.env_params['max_reward']  )])
                curr_desired_horizon = torch.Tensor([self.desired_horizon])
                curr_desired_state = torch.Tensor([self.desired_state])

        # useful if use an LSTM. 
        #hidden, state, action = self.model.init_hidden_state_action(1)

        if self.gamename == 'carracing':
            sim_rewards_queue = []
        cumulative = 0
        time = 0
        hit_done = False

        if self.return_events:
            rollout_dict = {k:[] for k in ['obs', 'obs2', 'rew', 'act', 'terminal']}
        
        while not hit_done:

            # NOTE: maybe make this unique to carracing here? 
            if self.obs_trim is not None:
                # trims the control panel at the base for the carracing environment. 
                obs = obs[self.obs_trim[0]:self.obs_trim[1], 
                            self.obs_trim[2]:self.obs_trim[3], :]

            if render: 
                if display_monitor:
                    display_monitor.set_data(obs)
                self.env.render()

            if self.env_params['use_vae']:
                obs = self.transform(obs).unsqueeze(0)#.to(self.device)
            else: 
                obs = torch.Tensor(obs).unsqueeze(0)#.to(self.device)

            if self.take_rand_actions:
                action = self.env.action_space.sample()
            else: 
                # use upside down model: 
                #print("desired state is:", curr_desired_state.shape )
                if self.Levine_Implementation:
                    desires = [curr_desired_reward.unsqueeze(1), curr_desired_state]
                else: 
                    desires = [curr_desired_reward.unsqueeze(1), curr_desired_horizon.unsqueeze(1), curr_desired_state]
                action = self.model(obs, desires )
                # need to constrain the action! 
                if self.env_params['continuous_actions']:
                    if not greedy: 
                        action = self._add_action_noise(action, self.action_noise)
                    action = self.constrain_actions(action)
                    action = action[0].detach().numpy()
                else: 
                    #sample action
                    # to do add temperature noise. 
                    #print('action is:', action)
                    #if self.Levine_Implementation:
                    if greedy: 
                        action = torch.argmax(action).squeeze().detach().numpy()
                    else: 
                        action = torch.softmax(action*self.action_noise, dim=-1)
                        action = Categorical(probs=action).sample([1])
                        action = action.squeeze().detach().numpy()
            
            # using action repeats
            action_rep_rewards = 0
            for _ in range(self.num_action_repeats):
                next_obs, reward, done, _ = self.env.step(action)
                action_rep_rewards += reward
                # ensures the done indicator is not missed during the action repeats.
                if done: hit_done = True

            # reward is all of the rewards collected during the action repeats. 
            reward = action_rep_rewards

            if self.env_params['give_raw_pixels']:
                next_obs = self.env.render(mode='rgb_array')
                #self.env.viewer.window.dispatch_events()

            if not hit_done and time>=self.time_limit:
                # add in any penalty for hitting the time limit and still not being done. 
                reward += self.env_params['over_max_time_limit_penalty']

            if time >= self.time_limit:
                hit_done=True
            
            time += 1
                
            # if the last n steps (real steps independent of action repeats)
            # have all given -0.1 reward then cut the rollout early. 
            if self.gamename == 'carracing':
                if len(sim_rewards_queue) < 50:
                    sim_rewards_queue.append(reward)
                else: 
                    sim_rewards_queue.pop(0)
                    sim_rewards_queue.append(reward)
                    #print('lenght of sim rewards',  len(sim_rewards_queue),round(sum(sim_rewards_queue),3))
                    if round(sum(sim_rewards_queue), 3) == -5.0:
                        hit_done=True
            # done checking for hit_done. 

            cumulative += reward
            if self.env_params['sparse']:
                # store in cumulative first. Dont need to worry about
                # storing in cumulative later as reward set to 0!
                reward = cumulative if hit_done else 0.0

            # update reward desires! 
            if not self.take_rand_actions:
                
                if self.Levine_Implementation: 
                    if self.advantage_model:
                        # sample a new desired reward
                        curr_desired_reward = self.desired_reward_dist.sample([1])
                    else:
                        pass 
                    # dont touch curr_desired reward. 
                    # or state or horizon. 
                else: 
                    curr_desired_reward = torch.Tensor( [min(curr_desired_reward-reward, self.env_params['max_reward'])])
                    curr_desired_horizon = torch.Tensor ( [max( curr_desired_horizon-1, 1)])
                # TODO: implement delta states here. in the buffer. and in the
                # training loop. 
                if self.delta_state:
                    curr_desired_state = torch.Tensor(obs-[curr_desired_state])
                else: 
                    pass
           
            # save out things.
            # doesnt save out the time so dont need to worry about it here. 
            if self.return_events:
                for key, var in zip(['obs', 'obs2', 'rew', 'act', 'terminal'], 
                                        [obs, next_obs, reward, action, hit_done ]):
                    if key=='obs':
                        var = var.squeeze().detach().numpy()
                    elif key=='obs2' and self.env_params['use_vae']:
                        var = self.transform(var).detach().numpy()
                    rollout_dict[key].append(var)

            # This is crucial. 
            obs = next_obs

        #print('time at end of rollout!!!', time)
        #print('done with this simulation')
        if render: 
            print('the last state for agent is:', rollout_dict['obs'][-1].round(3)  )

        if self.return_events:

            list_of_keys = list(rollout_dict.keys())
            for k in list_of_keys: # list of tensors arrays.
                # rewards to go here for levine
                if k =='rew':
                    if self.advantage_model:
                        rollout_dict['raw_rew'] = np.asarray(rollout_dict[k]) # need for TD lambda

                        # computing the TD(lambda) advantage values
                        # do this before the rewards are set as being discounted. 
                        rollout_dict['desire'] = discount_cumsum(np.asarray(rollout_dict[k]), self.discount_factor)
                        to_desire = rollout_dict['desire'][0] 
                    else: 
                        # discounted rewards to go. 
                        rollout_dict['desire'] = discount_cumsum(np.asarray(rollout_dict[k]), self.discount_factor)
                        to_desire = rollout_dict['desire'][0]
                else: 
                    rollout_dict[k] = np.asarray(rollout_dict[k])
            # repeat the cum reward up to length times. 
            # setting cum_rew to be the first reward to go. This is equivalent to the cum reward 
            # but accounts too for any discounting factor.

            rollout_dict['cum_rew'] = np.repeat(cumulative, time)
            rollout_dict['rollout_length'] = np.repeat(time, time)
            rollout_dict['horizon'] = time - np.arange(0, time) 
            rollout_dict['final_obs'] = np.repeat(np.expand_dims(rollout_dict['obs'][-1],0), time, axis=0)
            #print('rollout final obs is:', rollout_dict['final_obs'].shape )
            # so that the horizon is always 1 away
            #print(rollout_dict['final_obs'], rollout_dict['horizon'], rollout_dict['cum_rew'])
            #print('lenghts of things being added:', time, len(rollout_dict['cum_rew']), len(rollout_dict['horizon']), len(rollout_dict['desire']), len(rollout_dict['terminal']) )
            # discounted cumulative!
            return cumulative, to_desire, time, rollout_dict # passed back to simulate. 
        else: 
            return cumulative, time # ending time and cum reward
                
    def simulate(self, seed, return_events=False, num_episodes=16, 
        render_mode=False, antithetic=False, greedy=False):
        """ Runs lots of rollouts with different random seeds. Optionally,
        can keep track of all outputs from the environment in each rollout 
        (adding each to a list which contains dictionaries for the rollout). 
        And it can compute the FEEF at the end of each rollout. 
        
        :returns: (cum_reward_list, terminal_time_list, [data_dict_list], [feef_losses_list])
            - cum_reward_list: list. cumulative rewards of each rollout. 
            - terminal_time_list: list. timestep the rollout ended at. 
            - data_dict_list: list. OPTIONAL. Dictionaries from each rollout.
                Has keys: 'obs', 'rewards',
                'actions', 'terminal'. Each is a PyTorch Tensor with the first
                dimension corresponding to time.  
            - feef_losses_list: list. OPTIONAL. Free Energy of Expected Future value
            from the whole rollout. 
        """

        # have these here in case I wanted to use them. 
        # Render also currently doesnt do anything. 
        recording_mode = False

        self.return_events = return_events

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.make_env(seed=seed)

        cum_reward_list = []
        terminal_time_list = []
        to_desire_list = []
        if self.return_events:
            data_dict_list = []

        with torch.no_grad():
            for i in range(num_episodes):
                # for every second rollout. reset the rand seed if using antithetic. 
                if antithetic and i%2==1:
                    # uses the previous rand_seed
                    self.env.seed(int(rand_env_seed))
                    rand_env_seed = np.random.randint(0,1e9,1)[0]
                else: 
                    rand_env_seed = np.random.randint(0,1e9,1)[0]
                    self.env.seed(int(rand_env_seed))

                if self.return_events: 
                    rew, to_desire, time, data_dict = self.rollout(render=render_mode, greedy=greedy)
                    # data dict has the keys 'obs', 'rewards', 'actions', 'terminal'
                    data_dict_list.append(data_dict)
                    to_desire_list.append(to_desire)
                else: 
                    rew, time = self.rollout(render=render_mode, greedy=greedy)
                if render_mode: 
                    print('Cumulative Reward is:', rew, 'Termination time is:', time)
                    #print('Last Desired reward is:',curr_des "Last Desired Horizon is:", )
                cum_reward_list.append(rew)
                terminal_time_list.append(time)

        self.env.close()

        if render_mode:
            print('Mean reward over', num_episodes, 'episodes is:', np.mean(cum_reward_list))

        if self.return_events: 
            return cum_reward_list, to_desire_list, terminal_time_list, data_dict_list 
        else: 
            return cum_reward_list, terminal_time_list