# pylint: disable=no-member
"""
Simulated carracing environment.
"""
import argparse
from os.path import join, exists
import torch
from torch.distributions.categorical import Categorical
from torchvision import transforms
import gym
from gym import spaces
import matplotlib.pyplot as plt
import numpy as np
from pyglet.window import key
import json 
import copy 
from torchvision.utils import save_image
import time
import pickle 
#from .. import control 
from control import Agent
from utils import reshape_to_img
from envs import get_env_params
from models import RSSModel

class SimulatedCarracing(gym.Env): # pylint: disable=too-many-instance-attributes
    """
    Simulated Car Racing.

    Gym environment using learnt RSSM to simulate the
    CarRacing-v0 environment.
    """
    def __init__(self, directory, num_action_repeats=5):
        #self.num_action_repeats = num_action_repeats

        rssm_file = join(directory, 'joint', 'rssm_best.tar')
        assert exists(rssm_file), "No RSSM model in the directory..."

        env_params = get_env_params(args.gamename)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.IMAGE_RESIZE_DIM = env_params['IMAGE_RESIZE_DIM']

        decoder_reward_condition = False
        decoder_make_sigmas = False 

        # init models
        self.rssm = RSSModel(
            env_params['ACTION_SIZE'],
            env_params['LATENT_RECURRENT_SIZE'],
            env_params['LATENT_SIZE'],
            env_params['EMBEDDING_SIZE'],
            env_params['NODE_SIZE'],
            decoder_reward_condition,
            decoder_make_sigmas,
            device=device,
        )
        state = torch.load(rssm_file, map_location={'cuda:0': str(device)})
        self.rssm.load_state_dict(state['state_dict'])

        # spaces
        self.action_space = spaces.Box(np.array([-1, 0, 0]), np.array([1, 1, 1]))
        self.observation_space = spaces.Box(low=0, high=255, shape=(64, 64, 3),
                                            dtype=np.uint8)

        self.hidden, self.state = self.rssm.init_hidden_state(1)
        #self.hidden, self.state =self.hidden.unsqueeze(0), self.state.unsqueeze(0) 

        # obs
        self._obs = None
        self._visual_obs = None

        # rendering
        self.monitor = None
        self.figure = None

        self.reset()

    def reset(self):
        """ Resetting """
        import matplotlib.pyplot as plt
        
        '''l_and_h_starter = pickle.load(open('latent_and_hidden_starters.pkl' ,'rb'))
        self._lstate = l_and_h_starter[0] #torch.randn(1, LATENT_SIZE)
        self._hstate = l_and_h_starter[1] #2 * [torch.zeros(1, LATENT_RECURRENT_SIZE)]
        '''

        # also reset monitor
        if not self.monitor:
            self.figure = plt.figure()
            self.monitor = plt.imshow(
                np.zeros((self.IMAGE_RESIZE_DIM, self.IMAGE_RESIZE_DIM, 3),
                         dtype=np.uint8))

    def step(self, action, r):
        """ One step forward in the dream like environment!!! """
        with torch.no_grad():

            action = torch.Tensor(action).unsqueeze(0).unsqueeze(0)
            single_dream_step_dict = self.rssm.perform_rollout(action, hidden=self.hidden, state=self.state )

            #predict next reward
            pred_rewards = self.rssm.decode_sequence_reward(single_dream_step_dict['hiddens'], single_dream_step_dict['prior_states'])
            mus, logsigmas = self.rssm.decode_sequence_obs(single_dream_step_dict['hiddens'], single_dream_step_dict['prior_states'], pred_rewards )
            self._obs = reshape_to_img(mus, self.IMAGE_RESIZE_DIM) 
            np_obs = self._obs.cpu().numpy()
            
            np_obs = np.clip(np_obs, 0, 1) * 255
            np_obs = np.transpose(np_obs, (0, 2, 3, 1))
            np_obs = np_obs.squeeze()
            np_obs = np_obs.astype(np.uint8)
            self._visual_obs = np_obs

            self.hidden = single_dream_step_dict['hiddens'].squeeze(0)
            self.state = single_dream_step_dict['prior_states'].squeeze(0)

            return np_obs, pred_rewards.item(), False #d.item() > 0

    def render(self): # pylint: disable=arguments-differ
        """ Rendering """
        import matplotlib.pyplot as plt
        if not self.monitor:
            self.figure = plt.figure()
            self.monitor = plt.imshow(
                np.zeros((self.IMAGE_RESIZE_DIM, self.IMAGE_RESIZE_DIM, 3),
                         dtype=np.uint8))
        self.monitor.set_data(self._visual_obs)
        plt.pause(.01)

        '''#print('updated cross entropies')
        # choose the best next action out of all of them. 
        best_actions_ind = torch.argmax(all_particles_cum_rewards)
        worst_actions_ind = torch.argmin(all_particles_cum_rewards)

        best_action = all_particles_sequential_actions[best_actions_ind,0, :]

        best_rewards_seq = sequential_rewards[best_actions_ind]
        worst_rewards_seq = sequential_rewards[worst_actions_ind]

        print('best sequential actions', all_particles_sequential_actions[best_actions_ind] )
        print('worst sequential actions', all_particles_sequential_actions[worst_actions_ind] )

        print('best rewards seq', best_rewards_seq)
        print('worst rewards seq', worst_rewards_seq)

        sequence_best_pred_obs = all_particles_all_predicted_obs[best_actions_ind]
        sequence_worst_pred_obs = all_particles_all_predicted_obs[worst_actions_ind]

        # saving this out. 
        sequence_best_pred_obs = sequence_best_pred_obs.view(sequence_best_pred_obs.shape[0], 3, 64, 64)
        sequence_worst_pred_obs = sequence_worst_pred_obs.view(sequence_worst_pred_obs.shape[0], 3, 64, 64)
        save_image(sequence_best_pred_obs,
                        'exp_dir/simulation/samples/best_sample_' + str(time_step) + '.png')
        save_image(sequence_worst_pred_obs,
                        'exp_dir/simulation/samples/worst_sample_' + str(time_step) + '.png')

        print('predicted reward of best particle', torch.max(all_particles_cum_rewards))
        print('predicted reward of worst particle', torch.min(all_particles_cum_rewards))
        #print('best action is:', best_action)
        return best_action.unsqueeze(0)'''


if __name__ == '__main__':
    # argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, help='Directory from which MDRNN and VAE are '
                        'retrieved.')
    parser.add_argument('--gamename', type=str, default='carracing', help='The game being played.')
    parser.add_argument('--num_action_repeats', type=int, default=5, help='Number of times action is repeated.')
    parser.add_argument('--real_obs', action='store_true',
                    help="Use the real observations not dream ones")
    parser.add_argument('--test_agent', action='store_true',
                    help="Put the trained controller to the test!")
    parser.add_argument('--render', type=int, default=1,
                    help="Turn off if want to run lots of experiments")
    args = parser.parse_args()
    args.render = bool(args.render)

    if args.gamename == 'carracing':
        game_env = 'CarRacing-v0'
    if args.gamename == 'pendulum':
        game_env = 'Pendulum-v0'

    def on_key_press(event):
        """ Defines key pressed behavior """
        if event.key == 'up':
            action[1] = 1
        if event.key == 'down':
            action[2] = .8
        if event.key == 'left':
            action[0] = -1
        if event.key == 'right':
            action[0] = 1

    def on_key_release(event):
        """ Defines key pressed behavior """
        if event.key == 'up':
            action[1] = 0
        if event.key == 'down':
            action[2] = 0
        if event.key == 'left' and action[0] == -1:
            action[0] = 0
        if event.key == 'right' and action[0] == 1:
            action[0] = 0

    # this is for the real observation based driving! 
    def key_press(k, mod):
        global restart
        if k == 0xff0d: restart = True
        if k == key.LEFT:  action[0] = -1.0
        if k == key.RIGHT: action[0] = +1.0
        if k == key.UP:    action[1] = +1.0
        if k == key.DOWN:  action[2] = +0.8   # set 1.0 for wheels to block to zero rotation

    def key_release(k, mod):
        if k == key.LEFT  and action[0] == -1.0: action[0] = 0
        if k == key.RIGHT and action[0] == +1.0: action[0] = 0
        if k == key.UP:    action[1] = 0
        if k == key.DOWN:  action[2] = 0

    decoder_reward_condition = False

    if args.real_obs: 
        if args.test_agent: 
            # loads in the planner and rssm
            agent = Agent(args.gamename, 'exp_dir/joint/', decoder_reward_condition, 
                planner_n_particles=700, cem_iters=7, model_version='best', return_plan_images=True)
            env = agent.env
        else: 
            env = gym.make(game_env)
            _ = env.reset()

        if args.render:
            figure = plt.figure()
            monitor = plt.imshow(
                np.zeros((96, 96, 3),
                    dtype=np.uint8))
            env.viewer.window.on_key_press = key_press
            env.viewer.window.on_key_release = key_release

    else: 
        env = SimulatedCarracing(args.logdir, num_action_repeats=args.num_action_repeats)
        env.figure.canvas.mpl_connect('key_press_event', on_key_press)
        env.figure.canvas.mpl_connect('key_release_event', on_key_release)
        
    obs = env.reset()
    action = np.array([0., 0., 0.])
    reward = torch.Tensor([0]).unsqueeze(0)

    IMAGE_RESIZE_DIM = 64

    cum_reward = 0
    t = 0

    transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(IMAGE_RESIZE_DIM, IMAGE_RESIZE_DIM),
                transforms.ToTensor()
            ])

    sim_rewards = []

    if args.real_obs:
        # need to generate actions!
        if args.test_agent: 
            with torch.no_grad():
                if args.render: 
                    agent.rollout(10, render=True, display_monitor=monitor, discount_factor=0.9 )
                else: 
                    nsims = 20
                    print('Running ', nsims, 'simulations!')
                    cum_rews, terminals = agent.simulate(num_episodes=nsims)
                    cum_rews = np.array(cum_rews)
                    print('cum rewards:', cum_rews)
                    print('mean reward:', cum_rews.mean(), 'max:', cum_rews.max(), 'min:', cum_rews.min() )
        else:
            # allow person to play. 
            while True: 
                for _ in range(args.num_action_repeats):
                    print(action)
                    # this is the real environment. 
                    obs, reward, done, _ = env.step(action)
                    cum_reward += reward
                    t+=1

                    if done:
                        print(cum_reward)
                        break

                monitor.set_data(obs)
                env.render()

    else:
        # dream like state using environment made by the simulated agent! 
        while True:
            #for _ in range(args.num_action_repeats):
            reward = torch.Tensor([reward]).unsqueeze(0)
            print('action is:', action)
            # simulated environment. 
            _, reward, done = env.step(action, reward)
            
            cum_reward += reward
            t+=1

            if done:
                print(cum_reward)
                break

            env.render()
            #time.sleep(0.2)
        
            print('time step of simulation', t)
            print('=============================')
        
        
