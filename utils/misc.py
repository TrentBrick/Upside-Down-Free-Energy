""" Various auxiliary utilities """
import math
import time 
from os.path import join, exists
import torch
from torchvision import transforms
import numpy as np
from models import MDRNNCell, VAE, Controller
import gym
import gym.envs.box2d

# A bit dirty: manually change size of car racing env
# BUG: this makes the images very grainy!!!
#gym.envs.box2d.car_racing.STATE_W, gym.envs.box2d.car_racing.STATE_H = 64, 64

# Hardcoded for now
ACTION_SIZE, LATENT_SIZE, LATENT_RECURRENT_SIZE, IMAGE_RESIZE_DIM, SIZE =\
    3, 32, 256, 64, 96

# Same. used for Rollout Generator below. 
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_RESIZE_DIM, IMAGE_RESIZE_DIM)),
    transforms.ToTensor()
])

def sample_continuous_policy(action_space, seq_len, dt):
    """ Sample a continuous policy.

    Atm, action_space is supposed to be a box environment. The policy is
    sampled as a brownian motion a_{t+1} = a_t + sqrt(dt) N(0, 1).

    :args action_space: gym action space
    :args seq_len: number of actions returned
    :args dt: temporal discretization

    :returns: sequence of seq_len actions
    """
    actions = [action_space.sample()]
    for _ in range(seq_len):
        daction_dt = np.random.randn(*actions[-1].shape)
        actions.append(
            np.clip(actions[-1] + math.sqrt(dt) * daction_dt,
                    action_space.low, action_space.high))
    return actions

def save_checkpoint(state, is_best, filename, best_filename):
    """ Save state in filename. Also save in best_filename if is_best. """
    start_time = time.time()
    torch.save(state, filename)
    if is_best:
        torch.save(state, best_filename)
    print('seconds taken to save checkpoint.',(time.time()-start_time) )

def flatten_parameters(params):
    """ Flattening parameters.

    :args params: generator of parameters (as returned by module.parameters())

    :returns: flattened parameters (i.e. one tensor of dimension 1 with all
        parameters concatenated)
    """
    return torch.cat([p.detach().view(-1) for p in params], dim=0).cpu().numpy()

def unflatten_parameters(params, example, device):
    """ Unflatten parameters.

    :args params: parameters as a single 1D np array
    :args example: generator of parameters (as returned by module.parameters()),
        used to reshape params
    :args device: where to store unflattened parameters

    :returns: unflattened parameters
    """
    params = torch.Tensor(params).to(device) # why werent these put on the device earlier? 
    idx = 0
    unflattened = []
    for e_p in example:
        # makes a list of parameters in the same format and shape as the network. 
        unflattened += [params[idx:idx + e_p.numel()].view(e_p.size())]
        idx += e_p.numel()
    return unflattened

def load_parameters(params, controller):
    """ Load flattened parameters into controller.

    :args params: parameters as a single 1D np array
    :args controller: module in which params is loaded
    """
    proto = next(controller.parameters())
    params = unflatten_parameters(
        params, controller.parameters(), proto.device) # dont see the need to pass the device here only to put them into it later. 

    for p, p_0 in zip(controller.parameters(), params):
        p.data.copy_(p_0)

class RolloutGenerator(object):
    """ Utility to generate rollouts.

    Encapsulate everything that is needed to generate rollouts in the TRUE ENV
    using a controller with previously trained VAE and MDRNN.

    :attr vae: VAE model loaded from mdir/vae
    :attr mdrnn: MDRNN model loaded from mdir/mdrnn
    :attr controller: Controller, either loaded from mdir/ctrl or randomly
        initialized
    :attr env: instance of the CarRacing-v0 gym environment
    :attr device: device used to run VAE, MDRNN and Controller
    :attr time_limit: rollouts have a maximum of time_limit timesteps
    """
    def __init__(self, device, time_limit, mdir=None, return_events=False, give_models=None):
        """ Build vae, rnn, controller and environment. """

        self.env = gym.make('CarRacing-v0')
        self.device = device
        self.return_events = return_events
        self.time_limit = time_limit

        if give_models:
            self.vae = give_models['vae']

            if 'controller' in give_models.key():
                self.controller = give_models['controller']
            else: 
                self.controller = Controller(LATENT_SIZE, LATENT_RECURRENT_SIZE, ACTION_SIZE).to(device)
                # load controller if it was previously saved
                ctrl_file = join(mdir, 'ctrl', 'best.tar')
                if exists(ctrl_file):
                    ctrl_state = torch.load(ctrl_file, map_location={'cuda:0': str(device)})
                    print("Loading Controller with reward {}".format(
                        ctrl_state['reward']))
                    self.controller.load_state_dict(ctrl_state['state_dict'])

            # need to load in the cell based version!
            self.mdrnn = MDRNNCell(LATENT_SIZE, ACTION_SIZE, LATENT_RECURRENT_SIZE, 5).to(device)
            self.mdrnn.load_state_dict( 
                {k.strip('_l0'): v for k, v in give_models['mdrnn'].state_dict.items()})

        else:
            # Loading world model and vae
            vae_file, rnn_file, ctrl_file = \
                [join(mdir, m, 'best.tar') for m in ['vae', 'mdrnn', 'ctrl']]

            assert exists(vae_file) and exists(rnn_file),\
                "Either vae or mdrnn is untrained or the file is in the wrong place."

            vae_state, rnn_state = [
                torch.load(fname, map_location={'cuda:0': str(device)})
                for fname in (vae_file, rnn_file)]

            for m, s in (('VAE', vae_state), ('MDRNN', rnn_state)):
                print("Loading {} at epoch {} "
                    "with test loss {}".format(
                        m, s['epoch'], s['precision']))

            self.vae = VAE(3, LATENT_SIZE).to(device)
            self.vae.load_state_dict(vae_state['state_dict'])

            self.mdrnn = MDRNNCell(LATENT_SIZE, ACTION_SIZE, LATENT_RECURRENT_SIZE, 5).to(device)
            self.mdrnn.load_state_dict(
                {k.strip('_l0'): v for k, v in rnn_state['state_dict'].items()})

            self.controller = Controller(LATENT_SIZE, LATENT_RECURRENT_SIZE, ACTION_SIZE).to(device)

            # load controller if it was previously saved
            if exists(ctrl_file):
                ctrl_state = torch.load(ctrl_file, map_location={'cuda:0': str(device)})
                print("Loading Controller with reward {}".format(
                    ctrl_state['reward']))
                self.controller.load_state_dict(ctrl_state['state_dict'])

    def get_action_and_transition(self, obs, hidden):
        """ Get action and transition.

        Encode obs to latent using the VAE, then obtain estimation for next
        latent and next hidden state using the MDRNN and compute the controller
        corresponding action.

        :args obs: current observation (1 x 3 x 64 x 64) torch tensor
        :args hidden: current hidden state (1 x 256) torch tensor

        :returns: (action, next_hidden)
            - action: 1D np array
            - next_hidden (1 x 256) torch tensor
        """

        # why is none of this batched?? Because can't run lots of environments at a single point in parallel I guess. 
        mu, logsigma = self.vae.encoder(obs)
        latent_z =  mu + logsigma.exp() * torch.randn_like(mu) 

        assert latent_z.shape == (1, LATENT_SIZE), 'latent z in controller is the wrong shape!!'

        action = self.controller(latent_z, hidden[0])
        _, _, _, _, _, next_hidden = self.mdrnn(action, latent_z, hidden)
        return action.squeeze().cpu().numpy(), next_hidden

    def rollout(self, params, rand_env_seed, render=False):
        """ Execute a rollout and returns minus cumulative reward.

        Load :params: into the controller and execute a single rollout. This
        is the main API of this class.

        :args params: parameters as a single 1D np array

        :returns: minus cumulative reward
        # Why is this the minus cumulative reward?!?!!?
        """
        # copy params into the controller
        if params is not None:
            load_parameters(params, self.controller)

        self.env.seed(int(rand_env_seed)) # ensuring that each rollout has a differnet random seed. 
        obs = self.env.reset()

        # This first render is required !
        self.env.render()

        hidden = [
            torch.zeros(1, LATENT_RECURRENT_SIZE).to(self.device)
            for _ in range(2)]

        cumulative = 0
        i = 0
        if self.return_events: 
            rollout_dict = {k:[] for k in ['obs', 'rew', 'act', 'term']}
        while True:
            print('iteration of the rollout', i)
            obs = transform(obs).unsqueeze(0).to(self.device)
            action, hidden = self.get_action_and_transition(obs, hidden)
            obs, reward, done, _ = self.env.step(action)

            if self.return_events: 
                for key, var in zip(['obs', 'rew', 'act', 'term'], [obs,reward, action, done]):
                    rollout_dict[key].append(var)

            if render:
                self.env.render()

            cumulative += reward
            if done or i > self.time_limit:
                print('done with this simulation')
                if self.return_events:
                    for k,v in rollout_dict.items():
                        rollout_dict[k] = np.array(v)

                    return - cumulative, rollout_dict
                else: 
                    return - cumulative
            i += 1

if __name__ == "__main__":
    env = gym.make("CarRacing-v0")
    env.reset()
    seq_len=1000
    dt = 1. / 50
    actions = [env.action_space.sample()]
    print(actions)
    print(*actions)
    print(*actions[-1])
    for _ in range(seq_len):
        # getting rid of the list and then array structure. 
        # sampling 3 random actions from the last action in the list. 
        daction_dt = np.random.randn(*actions[-1].shape)
        actions.append(
            np.clip(actions[-1] + math.sqrt(dt) * daction_dt,
                    env.action_space.low, env.action_space.high))