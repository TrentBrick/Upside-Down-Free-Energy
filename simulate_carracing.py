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
from models.vae import VAE
from models.mdrnn import MDRNNCell
from models.controller import Controller
from utils.misc import NUM_IMG_CHANNELS, NUM_GAUSSIANS_IN_MDRNN, LATENT_SIZE, LATENT_RECURRENT_SIZE, IMAGE_RESIZE_DIM, ACTION_SIZE
import matplotlib.pyplot as plt
import numpy as np
from pyglet.window import key
import json 
import copy 
from torchvision.utils import save_image
import time
from utils.misc import sample_mdrnn_latent
import pickle 
class SimulatedCarracing(gym.Env): # pylint: disable=too-many-instance-attributes
    """
    Simulated Car Racing.

    Gym environment using learnt VAE and MDRNN to simulate the
    CarRacing-v0 environment.

    :args directory: directory from which the vae and mdrnn are
    loaded.
    """
    def __init__(self, directory, real_obs=False, test_agent=False, use_planner=False, num_action_repeats=5):
        self.real_obs = real_obs 
        self.test_agent = test_agent
        self.use_planner = use_planner
        self.num_action_repeats = num_action_repeats
        self.condition = True 

        if test_agent and not use_planner: 
            # load in controller trained using CMA-ES.: 
            self.controller = Controller(LATENT_SIZE, LATENT_RECURRENT_SIZE, ACTION_SIZE, 'carracing', condition=self.condition).to('cpu')
            
            with open('es_log/carracing.cma.12.64.best.json', 'r') as f:
                ctrl_params = json.load(f)
            print("Loading in the best controller model, its average eval score was:", ctrl_params[1])
            self.controller = load_parameters(ctrl_params[0], self.controller)

        vae_file = join(directory, 'vae', 'best.tar')
        rnn_file = join(directory, 'mdrnn', 'best.tar')
        #vae_file = join(directory, 'joint', 'vae_best.tar')
        #rnn_file = join(directory, 'joint', 'mdrnn_best.tar')
        assert exists(vae_file), "No VAE model in the directory..."
        assert exists(rnn_file), "No MDRNN model in the directory..."

        # spaces
        self.action_space = spaces.Box(np.array([-1, 0, 0]), np.array([1, 1, 1]))
        self.observation_space = spaces.Box(low=0, high=255, shape=(IMAGE_RESIZE_DIM, IMAGE_RESIZE_DIM, 3),
                                            dtype=np.uint8)

        # load VAE
        self.vae = VAE(NUM_IMG_CHANNELS, LATENT_SIZE, conditional=self.condition)
        vae_state = torch.load(vae_file, map_location=lambda storage, location: storage)
        print("Loading VAE at epoch {}, "
            "with test error {}...".format(
                vae_state['epoch'], vae_state['precision']))
        self.vae.load_state_dict(vae_state['state_dict'])

        # load MDRNN
        self.rnn = MDRNNCell(LATENT_SIZE, ACTION_SIZE, LATENT_RECURRENT_SIZE, NUM_GAUSSIANS_IN_MDRNN)
        rnn_state = torch.load(rnn_file, map_location=lambda storage, location: storage)
        print("Loading MDRNN at epoch {}, "
            "with test error {}...".format(
                rnn_state['epoch'], rnn_state['precision']))
        rnn_state_dict = {k.strip('_l0'): v for k, v in rnn_state['state_dict'].items()}
        self.rnn.load_state_dict(rnn_state_dict)

        # obs
        self._obs = None
        self._visual_obs = None

        # rendering
        self.monitor = None
        self.figure = None

    def agent_action(self, obs, hidden, reward ):

        mu, logsigma = self.vae.encoder(obs)
        latent_z =  mu + logsigma.exp() * torch.randn_like(mu) 

        assert latent_z.shape == (1, LATENT_SIZE), 'latent z in controller is the wrong shape!!'

        action = self.controller(latent_z, hidden[0], reward)
        _, _, _, _, _, next_hidden = self.rnn(action, latent_z, hidden)
        return action.squeeze().cpu().numpy(), next_hidden

    def reset(self):
        """ Resetting """
        import matplotlib.pyplot as plt
        l_and_h_starter = pickle.load(open('latent_and_hidden_starters.pkl' ,'rb'))

        self._lstate = l_and_h_starter[0] #torch.randn(1, LATENT_SIZE)
        self._hstate = l_and_h_starter[1] #2 * [torch.zeros(1, LATENT_RECURRENT_SIZE)]

        # also reset monitor
        if not self.monitor:
            self.figure = plt.figure()
            self.monitor = plt.imshow(
                np.zeros((IMAGE_RESIZE_DIM, IMAGE_RESIZE_DIM, 3),
                         dtype=np.uint8))

    def step(self, action, r):
        """ One step forward in the dream like environment!!! """
        with torch.no_grad():
            action = torch.Tensor(action).unsqueeze(0)
            print('========= shapes', r.shape, action.shape, self._lstate.shape, self._hstate[0].shape)

            mus, sigmas, logpi, r, d, n_h = self.rnn(action, self._lstate, self._hstate, r) #next_z, next_hidden)

            next_z =  sample_mdrnn_latent(mus, sigmas, logpi, self._lstate)

            #mu, sigma, pi, r, d, n_h = self.rnn(action, self._lstate, self._hstate)
            #pi = pi.squeeze()
            #mixt = Categorical(torch.exp(pi)).sample().item()

            self._lstate = next_z
            self._hstate = n_h

            #self._lstate = mu[:, mixt, :] # + sigma[:, mixt, :] * torch.randn_like(mu[:, mixt, :])
            # TODO: THINK THAT THIS ALSO HAS THE ACTUAL OUTPUTS RATHER THAN JUST THE HIDDEN!!
            #self._hstate = n_h

            self._obs, dec_sigma = self.vae.decoder(self._lstate, r.unsqueeze(1))
            self._obs = self._obs.view(self._obs.shape[0], 3, IMAGE_RESIZE_DIM, IMAGE_RESIZE_DIM)
            np_obs = self._obs.cpu().numpy()
            
            np_obs = np.clip(np_obs, 0, 1) * 255
            np_obs = np.transpose(np_obs, (0, 2, 3, 1))
            np_obs = np_obs.squeeze()
            np_obs = np_obs.astype(np.uint8)
            self._visual_obs = np_obs

            return np_obs, r.item(), False #d.item() > 0

    def render(self): # pylint: disable=arguments-differ
        """ Rendering """
        import matplotlib.pyplot as plt
        if not self.monitor:
            self.figure = plt.figure()
            self.monitor = plt.imshow(
                np.zeros((IMAGE_RESIZE_DIM, IMAGE_RESIZE_DIM, 3),
                         dtype=np.uint8))
        self.monitor.set_data(self._visual_obs)
        plt.pause(.01)

    def planner(self, latent_s, full_hidden, reward, time_step, starting_obs):
        # predicts into the future up to horizon. 
        # returns the immediate action that will lead to the largest
        # cumulative reward

        # reset at the start of each planning run
        self.cem_mus = torch.Tensor([0,0.5,0]) 
        self.cem_sigmas = torch.Tensor([0.7,0.3,0.1])

        cem_iters = 10

        desired_horizon = 15
        actual_horizon = desired_horizon//self.num_action_repeats
        self.horizon = actual_horizon
        ensemble_batchsize = 200
        self.ensemble_batchsize = ensemble_batchsize
        self.planner_n_particles = ensemble_batchsize
        self.mdrnn_ensemble = [self.rnn]
        self.k_top = int(0.1*ensemble_batchsize)
        self.discount_factor =0.9

        starting_obs = starting_obs.view(starting_obs.shape[0], -1)
        starting_obs = starting_obs.repeat(ensemble_batchsize, *len(starting_obs.shape[1:])*[1])
        all_particles_all_predicted_obs = []
        all_particles_all_predicted_obs.append(starting_obs)

        for cem_iter in range(cem_iters):

            all_particles_cum_rewards = torch.zeros((self.planner_n_particles))
            all_particles_sequential_actions = torch.zeros((self.planner_n_particles, actual_horizon, 3))
            
            sequential_rewards = []
        
            for mdrnn_ind, mdrnn_boot in enumerate(self.mdrnn_ensemble):

                # initialize particles for a single ensemble model. 
                
                ens_latent_s, ens_reward = [var.clone().repeat(ensemble_batchsize, *len(var.shape[1:])*[1]) for var in [latent_s, reward]]
                hidden_0 = full_hidden[0].clone().repeat(ensemble_batchsize, *len(full_hidden[0].shape[1:])*[1])
                hidden_1 = full_hidden[1].clone().repeat(ensemble_batchsize, *len(full_hidden[1].shape[1:])*[1])
                ens_full_hidden = [hidden_0,hidden_1]
                # only repeat the first dimension. 

                # indices for logging the actions and rewards during horizon planning across the ensemble. 
                start_ind = ensemble_batchsize*mdrnn_ind
                end_ind = start_ind+ensemble_batchsize

                for t in range(0, actual_horizon):

                    # need to produce a batch of first actions here. 
                    ens_action = self.sample_cross_entropy_method() 
                    md_mus, md_sigmas, md_logpi, ens_reward, d, ens_full_hidden = mdrnn_boot(ens_action, ens_latent_s, ens_full_hidden, ens_reward)
                    
                    # get the next latent state
                    ens_latent_s =  sample_mdrnn_latent(md_mus, md_sigmas, md_logpi, ens_latent_s)
                    
                    # store these cumulative rewards and action
                    all_particles_cum_rewards[start_ind:end_ind] += (self.discount_factor**t)*ens_reward
                    all_particles_sequential_actions[start_ind:end_ind, t, :] = ens_action

                    # unsqueeze for the next iteration. 
                    ens_reward = ens_reward.unsqueeze(1)

                    if cem_iter==(cem_iters-1):
                        dec_mu, dec_logsigma = self.vae.decoder( ens_latent_s, ens_reward )
                        all_particles_all_predicted_obs.append( dec_mu )
                        sequential_rewards.append(ens_reward)

            # update these after going through each ensemble. 
            self.update_cross_entropy_method(all_particles_sequential_actions, all_particles_cum_rewards)
            if cem_iter==(cem_iters-1):
                print('Last CEM iter: new cem params:::', self.cem_mus, self.cem_sigmas)

        sequential_rewards = torch.stack(sequential_rewards).squeeze().permute(1,0)
        all_particles_all_predicted_obs = torch.stack(all_particles_all_predicted_obs).permute(1,0,2)

        print('AFTER ALL ITERS: new cem params', self.cem_mus, self.cem_sigmas)

        #print('updated cross entropies')
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
        return best_action.unsqueeze(0)

    def sample_cross_entropy_method(self):
        actions = torch.distributions.Normal(self.cem_mus, self.cem_sigmas).sample([self.ensemble_batchsize])
        # constrain these actions:
        actions = self.constrain_actions(actions)
        return actions

    def update_cross_entropy_method(self, all_actions, rewards):
        # for carracing we have 3 independent gaussians
        smoothing = 0.8
        vals, inds = torch.topk(rewards, self.k_top, sorted=False )
        elite_actions = all_actions[inds]

        num_elite_actions = self.k_top*self.horizon 

        new_mu = elite_actions.sum(dim=(0,1))/num_elite_actions
        new_sigma = torch.sqrt(torch.sum( (elite_actions - self.cem_mus)**2, dim=(0,1))/num_elite_actions)
        self.cem_mus = smoothing*new_mu + (1-smoothing)*(self.cem_mus) 
        self.cem_sigmas = smoothing*new_sigma+(1-smoothing)*(self.cem_sigmas )

    def constrain_actions(self, out):
        out[:,0] = torch.clamp(out[:,0], min=-1.0, max=1.0)
        out[:,1] = torch.clamp(out[:,1], min=0.0, max=1.0)
        out[:,2] = torch.clamp(out[:,2], min=0.0, max=1.0)
        return out

    def random_shooting(self, batch_size):
        out = torch.distributions.Uniform(-1,1).sample((batch_size, 3))
        out = self.constrain_actions(out)
        return out

if __name__ == '__main__':
    # argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, help='Directory from which MDRNN and VAE are '
                        'retrieved.')
    parser.add_argument('--num_action_repeats', type=int, default=5, help='Number of times action is repeated.')
    parser.add_argument('--real_obs', action='store_true',
                    help="Use the real observations not dream ones")
    parser.add_argument('--test_agent', action='store_true',
                    help="Put the trained controller to the test!")
    parser.add_argument('--use_planner', action='store_true',
                    help="Use a planner rather than the controller")
    args = parser.parse_args()

    if args.use_planner: 
        args.test_agent = True 

    if args.real_obs: 
        env = gym.make("CarRacing-v0")
        figure = plt.figure()
        monitor = plt.imshow(
            np.zeros((96, 96, 3),
                dtype=np.uint8))
    else: 
        #starter_real_env = gym.make("CarRacing-v0")
        #starter_real_obs = starter_real_env.reset()
        env = SimulatedCarracing(args.logdir)
        
    obs = env.reset()
    action = np.array([0., 0., 0.])
    reward = torch.Tensor([0]).unsqueeze(0)

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

    if args.real_obs: 
        env.viewer.window.on_key_press = key_press
        env.viewer.window.on_key_release = key_release
        #figure.canvas.mpl_connect('key_press_event', on_key_press)
        #figure.canvas.mpl_connect('key_release_event', on_key_release)
    else: 
        env.figure.canvas.mpl_connect('key_press_event', on_key_press)
        env.figure.canvas.mpl_connect('key_release_event', on_key_release)

    if args.real_obs:

        #if not args.use_planner: 
        agent_class = SimulatedCarracing( args.logdir, args.real_obs, args.test_agent, args.use_planner, args.num_action_repeats)

    cum_reward = 0
    t = 0
    hidden = [torch.zeros(1, LATENT_RECURRENT_SIZE)
                    for _ in range(2)]

    transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((IMAGE_RESIZE_DIM, IMAGE_RESIZE_DIM)),
                transforms.ToTensor()
            ])

    sim_rewards = []

    while True:
        if args.real_obs:
            
            # need to generate actions!
            if args.test_agent: 

                with torch.no_grad():

                    obs_t = obs[:84, :, :]
                    obs_t = transform(obs_t).unsqueeze(0).to('cpu')

                    if args.use_planner: 
                        reward = torch.Tensor([reward]).unsqueeze(0)
                        mu, logsigma = agent_class.vae.encoder(obs_t, reward)
                        latent_z =  mu + logsigma.exp() * torch.randn_like(mu) 
                        action = agent_class.planner(latent_z, hidden, reward, t, obs_t)
                        _, _, _, _, _, hidden = agent_class.rnn(action, latent_z, hidden, reward)
                        
                        print('t is:', t)
                        if t == 60:
                            print('saving out the latent state and hidden')
                            pickle.dump((latent_z, hidden),open('latent_and_hidden_starters.pkl' ,'wb'))

                        action = action.squeeze().cpu().numpy()
                    else:  
                        action, hidden = agent_class.agent_action(obs_t, hidden, reward)

            for _ in range(args.num_action_repeats):
                print(action)
                # this is the real environment. 
                obs, reward, done, _ = env.step(action)
                cum_reward += reward
                t+=1

                print(sim_rewards)
                if len(sim_rewards) <30:
                    sim_rewards.append(reward)
                else: 
                    sim_rewards.pop(0)
                    sim_rewards.append(reward)
                    print('lenght of sim rewards',  len(sim_rewards),round(sum(sim_rewards),3))
                    if round(sum(sim_rewards), 3) == -3.0:
                        print('we have equality!!!')
                        break 

                if done:
                    print(cum_reward)
                    break

            monitor.set_data(obs)

        else:
            # dream like state using environment made by the simulated agent! 
            start_dream_state = 20

            '''# priming the latent and hidden states!!! 
            
            action = np.array([0., 0., 0.])
            starter_real_obs = starter_real_obs[:84, :, :]
            env._visual_obs = starter_real_obs
            starter_real_obs, reward, done, _ = starter_real_env.step(action)
            
            reward = torch.Tensor([reward]).unsqueeze(0)
            obs_t = transform(starter_real_obs).unsqueeze(0).to('cpu')
            mu, logsigma = env.vae.encoder(obs_t, reward)
            latent_z =  mu + logsigma.exp() * torch.randn_like(mu)
            env._lstate = latent_z
            action = torch.Tensor(action).unsqueeze(0)
            _, _, _, _, _, hidden = env.rnn(action, latent_z, hidden, reward)
            env._hstate = hidden
            t+=1'''
            #else: 
            for _ in range(args.num_action_repeats):
                
                reward = torch.Tensor([reward]).unsqueeze(0)
                print('action is:', action)
                # simulated environment. 
                _, reward, done, = env.step(action, reward)
                
                cum_reward += reward
                t+=1

                if done:
                    print(cum_reward)
                    break

        env.render()
        time.sleep(0.5)
        
        print('time step of simulation', t)
        print('=============================')
        
        
