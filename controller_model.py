
import math
import random 
import time 
from os.path import join, exists
import torch
from torchvision import transforms
import numpy as np
from models import MDRNNCell, VAE, Controller
import gym
import gym.envs.box2d
from utils.misc import ACTION_SIZE, LATENT_RECURRENT_SIZE, LATENT_SIZE, IMAGE_RESIZE_DIM

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

    return controller


class Models:

    def __init__(self, time_limit, 
        mdir=None, return_events=False, give_models=None):
        """ Build vae, rnn, controller and environment. """

        self.env = gym.make('CarRacing-v0')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.return_events = return_events
        self.time_limit = time_limit

        self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((IMAGE_RESIZE_DIM, IMAGE_RESIZE_DIM)),
                transforms.ToTensor()
            ])

        if give_models:
            self.vae = give_models['vae']

            if 'controller' in give_models.key():
                self.controller = give_models['controller']
            else: 
                self.controller = Controller(LATENT_SIZE, LATENT_RECURRENT_SIZE, ACTION_SIZE).to(self.device)
                # load controller if it was previously saved
                ctrl_file = join(mdir, 'ctrl', 'best.tar')
                if exists(ctrl_file):
                    ctrl_state = torch.load(ctrl_file, map_location={'cuda:0': str(self.device)})
                    print("Loading Controller with reward {}".format(
                        ctrl_state['reward']))
                    self.controller.load_state_dict(ctrl_state['state_dict'])

            # need to load in the cell based version!
            self.mdrnn = MDRNNCell(LATENT_SIZE, ACTION_SIZE, LATENT_RECURRENT_SIZE, 5).to(self.device)
            self.mdrnn.load_state_dict( 
                {k.strip('_l0'): v for k, v in give_models['mdrnn'].state_dict.items()})

        else:
            # Loading world model and vae
            vae_file, rnn_file, ctrl_file = \
                [join(mdir, m, 'best.tar') for m in ['vae', 'mdrnn', 'ctrl']]

            assert exists(vae_file) and exists(rnn_file),\
                "Either vae or mdrnn is untrained or the file is in the wrong place."

            vae_state, rnn_state = [
                torch.load(fname, map_location={'cuda:0': str(self.device)})
                for fname in (vae_file, rnn_file)]

            #print('about to load in the states')
            for m, s in (('VAE', vae_state), ('MDRNN', rnn_state)):
                print("Loading {} at epoch {} "
                    "with test loss {}".format(
                        m, s['epoch'], s['precision']))
            #print('loading in vae from: ', vae_file, device)
            self.vae = VAE(3, LATENT_SIZE).to(self.device)
            self.vae.load_state_dict(vae_state['state_dict'])

            #print('loading in mdrnn')
            self.mdrnn = MDRNNCell(LATENT_SIZE, ACTION_SIZE, LATENT_RECURRENT_SIZE, 5).to(self.device)
            self.mdrnn.load_state_dict(
                {k.strip('_l0'): v for k, v in rnn_state['state_dict'].items()})

            #print('loadin in controller.')
            self.controller = Controller(LATENT_SIZE, LATENT_RECURRENT_SIZE, ACTION_SIZE).to(self.device)

            # load controller if it was previously saved
            if exists(ctrl_file):
                ctrl_state = torch.load(ctrl_file, map_location={'cuda:0': str(self.device)})
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

    def rollout(self, rand_env_seed, params=None, render=False, time_limit=None):
        """ Execute a rollout and returns minus cumulative reward.

        Load :params: into the controller and execute a single rollout. This
        is the main API of this class.

        :args params: parameters as a single 1D np array

        :returns: minus cumulative reward
        # Why is this the minus cumulative reward?!?!!?
        """

        # copy params into the controller
        if params is not None:
            self.controller = load_parameters(params, self.controller)

        if not time_limit: 
            time_limit = self.time_limit 

        #random(rand_env_seed)
        np.random.seed(rand_env_seed)
        torch.manual_seed(rand_env_seed)
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
            #print('iteration of the rollout', i)
            obs = self.transform(obs).unsqueeze(0).to(self.device)
            action, hidden = self.get_action_and_transition(obs, hidden)
            obs, reward, done, _ = self.env.step(action)

            if self.return_events: 
                for key, var in zip(['obs', 'rew', 'act', 'term'], [obs,reward, action, done]):
                    rollout_dict[key].append(var)

            if render:
                self.env.render()

            cumulative += reward
            if done or i > time_limit:
                #print('done with this simulation')
                if self.return_events:
                    for k,v in rollout_dict.items():
                        rollout_dict[k] = np.array(v)
                    return cumulative, rollout_dict
                else: 
                    return cumulative, i # ending time and cum reward
            i += 1

    def simulate(self, params, train_mode=True, 
        render_mode=False, num_episode=16, 
        seed=27, max_len=1000): # run lots of rollouts 


        #print('seed recieved for this set of simulations', seed)
        # update params into the controller
        self.controller = load_parameters(params, self.controller)

        #random(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        reward_list = []
        t_list = []

        if max_len ==-1:
            max_len = 1000 # making it very long

        with torch.no_grad():
            for i in range(num_episode):
                rand_env_seed = np.random.randint(0,1e9,1)[0]
                rew, t = self.rollout(rand_env_seed, render=render_mode, 
                            params=None, time_limit=max_len)
                reward_list.append(rew)
                t_list.append(t)

        return reward_list, t_list


    

