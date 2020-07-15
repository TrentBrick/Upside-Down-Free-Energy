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
    def __init__(self, gamename, logdir,
        model = None, 
        desire_scalings=None, 
        take_rand_actions = False,
        desired_reward_stats=(1,1),
        Levine_Implementation = False, 
        desired_reward_dist_beta = 1.0,
        desired_horizon = 250, 
        discount_factor=1.0, model_version = 'checkpoint',
        return_plan_images=False):
        """ Build vae, forward model, and environment. """

        self.gamename = gamename
        self.env_params = get_env_params(gamename)
        self.action_noise = self.env_params['action_noise']
        self.take_rand_actions = take_rand_actions
        self.discount_factor = discount_factor
        self.Levine_Implementation = Levine_Implementation

        if self.Levine_Implementation:
            print('the desired stats are:', desired_reward_stats)
            self.desired_reward_dist = WeightedNormal(desired_reward_stats[0], 
                desired_reward_stats[1], beta=desired_reward_dist_beta)

        else:
            self.desired_rew_mu, self.desired_rew_std, self.desired_horizon = desired_reward_stats[0], desired_reward_stats[1], desired_horizon
            
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

    def make_env(self, seed=None, render_mode=False, full_episode=False):
        """ Called every time a new rollout occurs. Creates a new environment and 
        sets a new random seed for it."""
        self.render_mode = render_mode
        self.env = gym.make(self.env_params['env_name'])
        self.env.reset()
        if not seed: 
            seed = np.random.randint(0,1e9,1)[0]

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
                curr_desired_reward = self.desired_reward_dist.sample(1)
            else: 
                curr_desired_reward = np.random.uniform(self.desired_rew_mu, self.desired_rew_mu+self.desired_rew_std)
                curr_desired_reward = torch.Tensor([min(curr_desired_reward, self.env_params['max_reward']  )])
                curr_desired_horizon = torch.Tensor([self.desired_horizon])

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
                desires = torch.cat([curr_desired_reward.unsqueeze(1), torch.Tensor([time]).unsqueeze(1)], dim=1)
                action = self.model(obs, desires )
                # need to constrain the action! 
                if self.env_params['continuous_actions']:
                    action = self._add_action_noise(action, self.action_noise)
                    action = self.constrain_actions(action)
                    action = action[0].cpu().numpy()
                else: 
                    #sample action
                    # to do add temperature noise. 
                    #print('action is:', action)
                    #if self.Levine_Implementation:
                    if greedy: 
                        action = torch.argmax(action).squeeze().cpu().numpy()
                    else: 
                        action = torch.softmax(action*self.action_noise, dim=1)
                        action = Categorical(probs=action).sample([1])
                        action = action.squeeze().cpu().numpy()
            
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

            # has gone over and still not crashed: 
            if not hit_done and time>=self.time_limit:
                reward = self.env_params['over_max_time_limit']

            if time >= self.time_limit:
                hit_done=True
            else:
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
                    #curr_desired_reward -= reward
                    pass
                else: 
                    curr_desired_reward = torch.Tensor( [min(curr_desired_reward-reward, self.env_params['max_reward'])])
                    curr_desired_horizon = torch.Tensor ( [max( curr_desired_horizon-1, 1)])

            # save out things.
            # doesnt save out the time so dont need to worry about it here. 
            if self.return_events:
                for key, var in zip(['obs', 'obs2', 'rew', 'act', 'terminal'], 
                                        [obs, next_obs, reward, action, hit_done ]):
                    if key=='obs':
                        var = var.squeeze().cpu().numpy()
                    elif key=='obs2' and self.env_params['use_vae']:
                        var = self.transform(var).numpy()
                    rollout_dict[key].append(var)

            # This is crucial. 
            obs = next_obs

        #print('time at end of rollout!!!', time)
        #print('done with this simulation')
        if self.return_events:
            for k,v in rollout_dict.items(): # list of tensors arrays.
                # rewards to go here for levine
                if self.Levine_Implementation and k =='rew':
                    rollout_dict[k] = discount_cumsum(np.asarray(v), self.discount_factor)
                else: 
                    rollout_dict[k] = np.asarray(v)
            # repeat the cum reward up to length times. 
            rollout_dict['terminal_rew'] = np.repeat(cumulative, time)
            rollout_dict['time'] = np.arange(time, 0, -1)
            #print('lenghts of things being added:', time, len(rollout_dict['terminal_rew']), len(rollout_dict['time']), len(rollout_dict['rew']) )
            return cumulative, time, rollout_dict # passed back to simulate. 
        else: 
            return cumulative, time # ending time and cum reward
                
    def simulate(self, seed, return_events=False, num_episodes=16, 
        compute_feef=False,
        render_mode=False, antithetic=False, greedy=False):
        """ Runs lots of rollouts with different random seeds. Optionally,
        can keep track of all outputs from the environment in each rollout 
        (adding each to a list which contains dictionaries for the rollout). 
        And it can compute the FEEF at the end of each rollout. 
        
        :returns: (cum_reward_list, t_list, [data_dict_list], [feef_losses_list])
            - cum_reward_list: list. cumulative rewards of each rollout. 
            - t_list: list. timestep the rollout ended at. 
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
        t_list = []
        if self.return_events:
            data_dict_list = []
            if compute_feef:
                feef_losses_list = []

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
                    rew, time, data_dict = self.rollout(render=render_mode, greedy=greedy)
                    # data dict has the keys 'obs', 'rewards', 'actions', 'terminal'
                    data_dict_list.append(data_dict)
                    if compute_feef: 
                        feef_losses_list.append(  0.0 )#self.feef_loss(data_dict)  )
                else: 
                    rew, time = self.rollout(render=render_mode, greedy=greedy)
                if render_mode: 
                    print('Cumulative Reward is:', rew, 'Horizon is:', time)
                cum_reward_list.append(rew)
                t_list.append(time)

        self.env.close()

        if render_mode:
            print('Mean reward over', num_episodes, 'episodes is:', np.mean(cum_reward_list))

        if self.return_events: 
            if compute_feef:
                return cum_reward_list, t_list, data_dict_list, feef_losses_list
            else: 
                return cum_reward_list, t_list, data_dict_list 
        else: 
            return cum_reward_list, t_list