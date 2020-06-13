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
from utils.misc import LATENT_SIZE, LATENT_RECURRENT_SIZE, IMAGE_RESIZE_DIM, ACTION_SIZE
import matplotlib.pyplot as plt
import numpy as np
from pyglet.window import key
import json 
from controller_model import load_parameters
import copy 

class SimulatedCarracing(gym.Env): # pylint: disable=too-many-instance-attributes
    """
    Simulated Car Racing.

    Gym environment using learnt VAE and MDRNN to simulate the
    CarRacing-v0 environment.

    :args directory: directory from which the vae and mdrnn are
    loaded.
    """
    def __init__(self, directory, real_obs=False, test_agent=False, use_planner=False):
        self.real_obs = real_obs 
        self.test_agent = test_agent
        self.use_planner = use_planner
        self.condition = True 

        if test_agent: 

            # load in controller: 
            self.controller = Controller(LATENT_SIZE, LATENT_RECURRENT_SIZE, ACTION_SIZE, 'carracing', condition=self.condition).to('cpu')
            
            with open('es_log/carracing.cma.12.64.best.json', 'r') as f:
                ctrl_params = json.load(f)
            print("Loading in the best controller model, its average eval score was:", ctrl_params[1])
            self.controller = load_parameters(ctrl_params[0], self.controller)

        vae_file = join(directory, 'vae', 'best.tar')
        rnn_file = join(directory, 'mdrnn', 'best.tar')
        assert exists(vae_file), "No VAE model in the directory..."
        assert exists(rnn_file), "No MDRNN model in the directory..."

        # spaces
        self.action_space = spaces.Box(np.array([-1, 0, 0]), np.array([1, 1, 1]))
        self.observation_space = spaces.Box(low=0, high=255, shape=(IMAGE_RESIZE_DIM, IMAGE_RESIZE_DIM, 3),
                                            dtype=np.uint8)

        # load VAE
        self.vae = VAE(3, LATENT_SIZE)
        vae_state = torch.load(vae_file, map_location=lambda storage, location: storage)
        print("Loading VAE at epoch {}, "
            "with test error {}...".format(
                vae_state['epoch'], vae_state['precision']))
        self.vae.load_state_dict(vae_state['state_dict'])
        self._decoder = self.vae.decoder

        # load MDRNN
        self._rnn = MDRNNCell(32, 3, LATENT_RECURRENT_SIZE, 5)
        rnn_state = torch.load(rnn_file, map_location=lambda storage, location: storage)
        print("Loading MDRNN at epoch {}, "
            "with test error {}...".format(
                rnn_state['epoch'], rnn_state['precision']))
        rnn_state_dict = {k.strip('_l0'): v for k, v in rnn_state['state_dict'].items()}
        self._rnn.load_state_dict(rnn_state_dict)

        # init state
        self._lstate = torch.randn(1, LATENT_SIZE)
        self._hstate = 2 * [torch.zeros(1, LATENT_RECURRENT_SIZE)]

        # obs
        self._obs = None
        self._visual_obs = None

        # rendering
        self.monitor = None
        self.figure = None

    def agent_action(self, obs, hidden, reward ):

        # why is none of this batched?? Because can't run lots of environments at a single point in parallel I guess. 
        mu, logsigma = self.vae.encoder(obs)
        latent_z =  mu + logsigma.exp() * torch.randn_like(mu) 

        assert latent_z.shape == (1, LATENT_SIZE), 'latent z in controller is the wrong shape!!'

        action = self.controller(latent_z, hidden[0], reward)
        _, _, _, _, _, next_hidden = self._rnn(action, latent_z, hidden)
        return action.squeeze().cpu().numpy(), next_hidden

    def reset(self):
        """ Resetting """
        import matplotlib.pyplot as plt
        self._lstate = torch.randn(1, LATENT_SIZE)
        self._hstate = 2 * [torch.zeros(1, LATENT_RECURRENT_SIZE)]

        # also reset monitor
        if not self.monitor:
            self.figure = plt.figure()
            self.monitor = plt.imshow(
                np.zeros((IMAGE_RESIZE_DIM, IMAGE_RESIZE_DIM, 3),
                         dtype=np.uint8))

    def step(self, action):
        """ One step forward """
        with torch.no_grad():
            action = torch.Tensor(action).unsqueeze(0)
            mus, sigmas, logpi, r, d, n_h = self._rnn(action, self._lstate, self._hstate) #next_z, next_hidden)

            g_probs = Categorical(probs=torch.exp(logpi).permute(0,2,1))
            which_g = g_probs.sample()
            mus_g, sigs_g = torch.gather(mus.squeeze(), 0, which_g), torch.gather(sigmas.squeeze(), 0, which_g)
            #print(mus_g.shape)
            next_z = mus_g + sigs_g * torch.randn_like(mus_g)

            #mu, sigma, pi, r, d, n_h = self._rnn(action, self._lstate, self._hstate)
            #pi = pi.squeeze()
            #mixt = Categorical(torch.exp(pi)).sample().item()

            self._lstate = next_z
            self._hstate = n_h

            #self._lstate = mu[:, mixt, :] # + sigma[:, mixt, :] * torch.randn_like(mu[:, mixt, :])
            # TODO: THINK THAT THIS ALSO HAS THE ACTUAL OUTPUTS RATHER THAN JUST THE HIDDEN!!
            #self._hstate = n_h

            self._obs = self._decoder(self._lstate)
            np_obs = self._obs.numpy()
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


""" Testing the planner by giving it the real world dynamics """


def random_shooting(batch_size):
    out = torch.distributions.Uniform(-1,1).sample((batch_size, 3))
    out = torch.tanh(out)
    out[0,1] = (out[0,1]+1)/2.0 # this converts tanh to sigmoid
    out[0,2] = torch.clamp(out[0,2], min=0.0, max=1.0)

    return out

def planner(env):
    # predicts into the future up to horizon. 
    # returns the immediate action that will lead to the largest
    # cumulative reward

    env.render(close=True)

    planner_n_particles = 10
    horizon = 30

    cum_rewards = []
    all_particles_first_action = []
    
    # initialize particles for a single ensemble model. 
    #ens_action, ens_latent_s, ens_full_hidden, ens_reward = [var.clone().repeat(self.ensemble_interval, *len(var.shape[1:])*[1]) for var in [action, latent_s, full_hidden, reward]]:
    # only repeat the first dimension. 
    #print('planner tensors, should now have larger batch size of', self.ensemble_interval, action.shape,latent_s.shape )
    for p in range(planner_n_particles):

        env_clone = copy.deepcopy(env)
        print(env_clone)
        ens_reward = 0

        for t in range(horizon):

            # sample actions for each particle!!! 
            # TODO: implement CEM here: 
            ens_action = random_shooting(1)
            #print(ens_action)

            obs, reward, done, _ = env_clone.step(ens_action.squeeze())

            # save these cumulative rewards
            ens_reward += reward 
            
            if t==0:
                # store next action to be taken
                ens_first_action = ens_action

        cum_rewards.append(ens_reward)
        all_particles_first_action.append(ens_first_action)

    all_particles_first_action = torch.stack(all_particles_first_action)
    # choose the best next action out of all of them. 
    best_actions_ind = np.argmax(cum_reward)

    env.render()

    return all_particles_first_action[best_actions_ind]


if __name__ == '__main__':
    # argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, help='Directory from which MDRNN and VAE are '
                        'retrieved.')
    parser.add_argument('--real_obs', action='store_true',
                    help="Use the real observations not dream ones")
    parser.add_argument('--test_agent', action='store_true',
                    help="Put the trained controller to the test!")
    parser.add_argument('--use_planner', action='store_true',
                    help="Use a planner rather than the controller")
    args = parser.parse_args()
    if args.real_obs: 
        env = gym.make("CarRacing-v0")
        figure = plt.figure()
        monitor = plt.imshow(
            np.zeros((96, 96, 3),
                dtype=np.uint8))
    else: 
        env = SimulatedCarracing(args.logdir)

    obs = env.reset()
    action = np.array([0., 0., 0.])

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

        if not args.use_planner: 

            hidden = [
                torch.zeros(1, LATENT_RECURRENT_SIZE).to('cpu')
                for _ in range(2)]
            reward = 0

            transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((IMAGE_RESIZE_DIM, IMAGE_RESIZE_DIM)),
                    transforms.ToTensor()
                ])

            agent_class = SimulatedCarracing( args.logdir, args.real_obs, args.test_agent, args.use_planner)

    cum_reward = 0
    while True:
        if args.real_obs:
            
            # need to generate actions!
            if args.test_agent: 

                if args.use_planner: 

                    print(env)

                    action = planner(env)

                else: 
                    with torch.no_grad():
                        obs_t = transform(obs).unsqueeze(0).to('cpu')
                        action, hidden = agent_class.agent_action(obs_t, hidden, reward)

            print(action)
            obs, reward, done, _ = env.step(action)
            monitor.set_data(obs)

        else:
            print(action)
            _, reward, done = env.step(action)
        env.render()
        cum_reward += reward
        if done:
            print(cum_reward)
            break
