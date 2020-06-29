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
from torch.distributions.normal import Normal
#from utils import sample_mdrnn_latent
from models import RSSModel 
from control import Planner
from envs import get_env_params

class Agent:
    def __init__(self, gamename, logdir, decoder_reward_condition,
        planner_n_particles=100,
        cem_iters=10, 
        discount_factor=0.90):
        """ Build vae, forward model, and environment. """

        self.gamename = gamename
        self.env_params = get_env_params(gamename)

        # top, bottom, left, right
        self.obs_trim = self.env_params['trim_shape']

        self.device = 'cpu'#torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.time_limit = self.env_params['time_limit']

        # transform used on environment generated observations. 
        self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.env_params['IMAGE_RESIZE_DIM'], self.env_params['IMAGE_RESIZE_DIM'])),
                transforms.ToTensor()
            ])

        # Loading world model and vae
        # NOTE: the checkpoint, ie most up to date model. Is being loaded in. 
        
        self.num_action_repeats = self.env_params['num_action_repeats']
        # the real horizon 
        self.horizon = self.env_params['actual_horizon']
        self.planner_n_particles = planner_n_particles

        self.cem_iters = cem_iters 
        self.discount_factor = discount_factor
        self.k_top = int(planner_n_particles*0.10)

        self.rssm = RSSModel(
            self.env_params['ACTION_SIZE'],
            self.env_params['LATENT_RECURRENT_SIZE'],
            self.env_params['LATENT_SIZE'],
            self.env_params['EMBEDDING_SIZE'],
            self.env_params['NODE_SIZE'],
            decoder_reward_condition,
            False, #decoder make sigmas
            device=self.device,
        )
        load_file = join(logdir, 'rssm_checkpoint.tar')
        assert exists(load_file), "Could not find file: " + load_file + " to load in!"
        state = torch.load(load_file, map_location={'cuda:0': str(self.device)})
        print("Loading model_type {} at epoch {} "
            "with test error {}".format('rssm',
                state['epoch'], state['precision']))

        self.rssm.load_state_dict(state['state_dict'])
        self.rssm.eval()

        self.make_env()

        assert str(type(self.env.action_space)) == "<class 'gym.spaces.box.Box'>", "Need to constrain discrete actions in planner and sample from a non Normal distribution!!"

        self.planner = Planner(
            self.rssm,
            self.env_params['ACTION_SIZE'],
            (self.env.action_space.low, self.env.action_space.high),
            plan_horizon=self.horizon,
            optim_iters=self.cem_iters,
            candidates=self.planner_n_particles,
            top_candidates=self.k_top,
            init_cem_params=self.env_params['init_cem_params']
        )

        # TODO: make an RSSM ensemble. 
        '''self.mdrnn_ensemble = [self.rssm]
        self.ensemble_size = len(self.mdrnn_ensemble)
        assert self.planner_n_particles  % self.ensemble_size==0, "Need planner n particles and ensemble size to be perfectly divisible!"
        self.ensemble_batchsize = self.planner_n_particles//self.ensemble_size '''
        
    def make_env(self, seed=-1, render_mode=False, full_episode=False):
        """ Called every time a new rollout occurs. Creates a new environment and 
        sets a new random seed for it."""
        self.render_mode = render_mode
        self.env = gym.make(self.env_params['env_name'])
        self.env.seed(int(seed))
        self.env.render('rgb_array')

    def get_action_and_transition(self, obs, reward, hidden, action=None, latent_s=None):
        """ Get action and transition.

        Encode obs to latent using the VAE, then obtain estimation for next
        latent and next hidden state using the MDRNN and compute the controller
        corresponding action.

        :args:
            - obs: current observation (1 x 3 x 64 x 64) torch tensor
            - hidden: current hidden state (1 x 256) torch tensor

        :returns: (action, next_hidden)
            - action: 1D np array
            - next_hidden (1 x 256) torch tensor
        """

        # why is none of this batched?? Because can't run lots of environments at a single point in parallel I guess. 
        
        if self.use_rssm: 
            encoder_obs = self.rssm.encode_obs(obs)
            encoder_obs = encoder_obs.unsqueeze(0)

            if action is not None: 
                action = action.unsqueeze(0)

            dyn_dict = self.rssm.perform_rollout(action, hidden=hidden, state=latent_s, encoder_output=encoder_obs)
            next_hidden = dyn_dict['hiddens'].squeeze(0)
            latent_s = dyn_dict['posterior_states'].squeeze(0)
            action = self.rssm_planner(hidden, latent_s)
            return action.squeeze().cpu().numpy(), next_hidden, action, latent_s

        else:
            mu, logsigma = self.vae.encoder(obs, reward)
            latent_s =  mu + logsigma.exp() * torch.randn_like(mu) 
            assert latent_s.shape == (1, self.env_params['LATENT_SIZE']), 'latent z in controller is the wrong shape!!'
            action = self.planner(latent_s, hidden, reward)
            if self.deterministic:
                _, _, _, _, _, next_hidden = self.mdrnn(action, latent_s, reward, last_hidden=hidden)
            else: 
                _, _, _, _, _, next_hidden = self.mdrnn(action, latent_s, hidden, reward)
    
            return action.squeeze().cpu().numpy(), next_hidden

    def rollout(self, rand_env_seed, render=False):
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

        # base random seed set. Then sample new random numbers
        # for each individual rollout. 
        np.random.seed(rand_env_seed)
        torch.manual_seed(rand_env_seed)

        # not sure what the need for this is: 
        dispatch_events = False

        obs = self.env.reset()
        if dispatch_events:
            self.env.viewer.window.dispatch_events()
        hidden, state, action = self.rssm.init_hidden_state_action(1)
        reward = 0
        done = 0
        #action = np.array([0.,0.,0.])

        sim_rewards_queue = []
        cumulative = 0
        time = 0

        if self.return_events:
            rollout_dict = {k:[] for k in ['obs', 'rewards', 'actions', 'terminal']}
        
        while True:
            #print('iteration of the rollout', time)

            # NOTE: maybe make this unique to carracing here? 
            if self.obs_trim is not None:
                # trims the control panel at the base for the carracing environment. 
                obs = obs[self.obs_trim[0]:self.obs_trim[1], 
                            self.obs_trim[2]:self.obs_trim[3], 
                            :]

            obs = self.transform(obs).unsqueeze(0).to(self.device)
            reward = torch.Tensor([reward]).to(self.device).unsqueeze(0)

            encoded_obs = self.rssm.encode_obs(obs)
            encoded_obs = encoded_obs.unsqueeze(0)
            action = action.unsqueeze(0)

            rollout = self.rssm.perform_rollout(
                    action, hidden=hidden, state=state, encoder_output=encoded_obs
                )
            hidden = rollout["hiddens"].squeeze(0)
            state = rollout["posterior_states"].squeeze(0)

            action = self.planner(hidden, state)
            #action = self._add_action_noise(action, action_noise)

            # needed to accomodate the action repeats. 
            hit_done = False 

            # using action repeats
            for _ in range(self.num_action_repeats):
                next_obs, reward, done, _ = self.env.step(action[0].cpu())
                if dispatch_events:
                    self.env.viewer.window.dispatch_events()
                cumulative += reward

                # if the last n steps (real steps independent of action repeats)
                # have all given -0.1 reward then cut the rollout early. 
                # TODO: move into get_env_params? 
                if self.gamename == 'carracing':
                    if len(sim_rewards_queue) < 50:
                        sim_rewards_queue.append(reward)
                    else: 
                        sim_rewards_queue.pop(0)
                        sim_rewards_queue.append(reward)
                        #print('lenght of sim rewards',  len(sim_rewards_queue),round(sum(sim_rewards_queue),3))
                        if round(sum(sim_rewards_queue), 3) == -5.0:
                            done=True

                if done: 
                    hit_done = True 

            if self.return_events:
                # NOTE: adding obs not next_obs
                for key, var in zip(['obs', 'rewards', 'actions', 'terminal'], 
                                        [obs, reward, action, hit_done ]):
                    if key == 'actions' or key=='terminal':
                        var = torch.Tensor([var])
                    rollout_dict[key].append(var.squeeze())

            if hit_done or time > self.time_limit:
            #print('done with this simulation')
                if self.return_events:
                    for k,v in rollout_dict.items(): # list of tensors arrays.
                        rollout_dict[k] = torch.stack(v)
                    return cumulative, time, rollout_dict # passed back to simulate. 
                else: 
                    return cumulative, time # ending time and cum reward
                
            obs = next_obs 
            time += 1

            if render: 
                self.env.render()

    def simulate(self, return_events=False, num_episodes=16, 
        seed=27, compute_feef=False,
        render_mode=False):
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
        penalize_turning = False

        self.return_events = return_events

        #random(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        cum_reward_list = []
        t_list = []
        if self.return_events:
            data_dict_list = []
            if compute_feef:
                feef_losses_list = []

        with torch.no_grad():
            for i in range(num_episodes):

                rand_env_seed = np.random.randint(0,1e9,1)[0]

                self.make_env(seed=rand_env_seed)
                
                if self.return_events: 
                    rew, time, data_dict = self.rollout(rand_env_seed, render=render_mode)
                    # data dict has the keys 'obs', 'rewards', 'actions', 'terminal'
                    data_dict_list.append(data_dict)
                    if compute_feef: 
                        feef_losses_list.append(  0.0 )#self.feef_loss(data_dict)  )
                else: 
                    rew, time = self.rollout(rand_env_seed, render=render_mode)
                cum_reward_list.append(rew)
                t_list.append(time)

                self.env.close()

        if self.return_events: 
            if compute_feef:
                return cum_reward_list, t_list, data_dict_list, feef_losses_list
            else: 
                return cum_reward_list, t_list, data_dict_list 
        else: 
            return cum_reward_list, t_list


    

