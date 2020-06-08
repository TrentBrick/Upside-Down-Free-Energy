import math
import random 
import time
from os.path import join, exists
import torch
from torchvision import transforms
import numpy as np
from models import MDRNNCell, VAE, Controller
import pickle
import gym 
from feef import feef_loss 
from ha_env import make_env
#import gym
#import gym.envs.box2d
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from utils.misc import ACTION_SIZE, LATENT_RECURRENT_SIZE, LATENT_SIZE, IMAGE_RESIZE_DIM
from trainvae import loss_function as original_vae_loss_function

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

testing_old_controller = False

class Models:

    def __init__(self, env_name, time_limit, use_old_gym=True, 
        mdir=None, return_events=False, give_models=None, conditional=True):
        """ Build vae, rnn, controller and environment. """

        #self.env = gym.make('CarRacing-v0')
        self.env_name = env_name
        self.use_old_gym = use_old_gym

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.return_events = return_events
        self.time_limit = time_limit

        self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((IMAGE_RESIZE_DIM, IMAGE_RESIZE_DIM)),
                transforms.ToTensor()
            ])

        self.fixed_ob = pickle.load(open('notebooks/image_array.pkl', 'rb'))

        if give_models:
            self.vae = give_models['vae']

            if 'controller' in give_models.key():
                self.controller = give_models['controller']
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
            if testing_old_controller: 
                self.controller = Controller(LATENT_SIZE, LATENT_RECURRENT_SIZE, ACTION_SIZE, conditional=False).to(self.device)
            else: 
                self.controller = Controller(LATENT_SIZE, LATENT_RECURRENT_SIZE, ACTION_SIZE, conditional=conditional).to(self.device)

            # load controller if it was previously saved
            if exists(ctrl_file):
                ctrl_state = torch.load(ctrl_file, map_location={'cuda:0': str(self.device)})
                print("Loading Controller with reward {}".format(
                    ctrl_state['reward']))
                self.controller.load_state_dict(ctrl_state['state_dict'])

    def make_env(self, seed=-1, render_mode=False, full_episode=False):
        self.render_mode = render_mode
        if self.use_old_gym:
            self.env = make_env(self.env_name, seed=seed, render_mode=render_mode, full_episode=full_episode)
        else: 
            self.env = gym.make("CarRacing-v0")
            self.env.seed(int(seed))
    def get_action_and_transition(self, obs, hidden, reward):
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
        mu, logsigma = self.vae.encoder(obs, reward)
        latent_s =  mu + logsigma.exp() * torch.randn_like(mu) 

        assert latent_s.shape == (1, LATENT_SIZE), 'latent z in controller is the wrong shape!!'

        if testing_old_controller: 
            action = self.controller(latent_s, hidden[0])
        else: 
            action = self.controller(latent_s, hidden[0], reward)
        _, _, _, _, _, next_hidden = self.mdrnn(action, latent_s, hidden, reward)
        
        return action.squeeze().cpu().numpy(), next_hidden

    def rollout(self, rand_env_seed, params=None, render=False, time_limit=None, trim_controls=True):
        """ Execute a rollout and returns minus cumulative reward.

        Load :params: into the controller and execute a single rollout. This
        is the main API of this class.

        :args params: parameters as a single 1D np array

        :returns: minus cumulative reward
        # Why is this the minus cumulative reward?!?!!?
        """

        if self.use_old_gym:
            self.env.render('rgb_array')
        self.trim_controls = trim_controls

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
        if self.use_old_gym:
            self.env.viewer.window.dispatch_events()

        if not self.use_old_gym:
            # This first render is required !
            self.env.render()

        hidden = [
            torch.zeros(1, LATENT_RECURRENT_SIZE).to(self.device)
            for _ in range(2)]
        reward = 0
        done = 0
        action = np.array([0.,0.,0.])

        cumulative = 0
        i = 0
        if self.return_events: 
            rollout_dict = {k:[] for k in ['obs', 'rewards', 'actions', 'terminal']}
        while True:
            #print('iteration of the rollout', i)

            if self.trim_controls:
                # need to trim the observation first
                obs = obs[:84, :, :]

            obs = self.transform(obs).unsqueeze(0).to(self.device)
            reward = torch.Tensor([reward]).to(self.device).unsqueeze(0)
            
            action, hidden = self.get_action_and_transition(obs, hidden, reward)
            
            if self.return_events: 
                for key, var in zip(['obs', 'rewards', 'actions', 'terminal'], 
                                        [obs,reward, action, done ]):
                    rollout_dict[key].append(var)

            #obs, reward, done = self.fixed_ob, np.random.random(1)[0], False
            obs, reward, done, _ = self.env.step(action)
            if self.use_old_gym:
                self.env.viewer.window.dispatch_events()

            if render:
                pass
                #self.env.render()

            cumulative += reward
            if done or i > time_limit:
                #print('done with this simulation')
                if self.return_events:
                    for k,v in rollout_dict.items():
                        rollout_dict[k] = torch.Tensor(v, requires_grad=False)
                    return cumulative, rollout_dict, i
                else: 
                    return cumulative, i # ending time and cum reward
            i += 1

    def elbo_calculation(self, recon_obs, obs, mu, logsigma):

        # vae loss function but also computes the normalizing constant. 
        # and things are all positive not negative because we arent doing minimization here. 
        # NOTE: I dont have free bits in here and dont think that I should. this enables a looser bound but doesnt affect the probability calculation. 
        
        log_p_obs_given_s = Normal(torch.flatten(recon_obs), 1.0)).log_prob(torch.flatten(obs))
        
        KLD = -0.5 * torch.sum(1 + 2 * logsigma - mu.pow(2) - (2 * logsigma).exp())
        
        return log_p_obs_given_s - KLD

    def p_policy(self, rollout_dict, num_s_samps, num_next_s_samps):

        # rollout_dict values are of the size and shape seq_len, values dimensions. 

        '''#Should have access already
        #transform then encode the observations and sample latent zs
        #obs = transform(obs).unsqueeze(0).to(device)
        # batch, channels, image dim, image dim
        #mu, logsigma = vae.encoder(obs)'''

        print('this might give OOM and need to break into batch')
        # TODO: break the sequence length into smaller batches. 
        
        total_samples = num_s_samps * num_next_s_samps
        expected_kld = 0

        mus, logsigmas = self.vae.encoder(rollout_dict['obs'], rollout_dict['rewards'])

        for _ in range(num_s_samps): # vectorize this somehow 
            latent_s =  mus + logsigmas.exp() * torch.randn_like(mus)

            # predict the next state from this one. 
            # I already have access to the action from the runs generated. 
            # TODO: make both of these run with a batch. DONT NEED TO GENERATE OR PASS AROUND HIDDEN AS A RESULT. 

            md_mus, md_sigmas, md_logpi, next_r, d, next_hidden = self.mdrnn(rollout_dict['actions'], 
                                                                        latent_s, rollout_dict['rewards'])

            # reward loss 
            log_reward_surprise = self.reward_prior.log_prob(next_r)

            g_probs = Categorical(probs=torch.exp(logpi).permute(0,2,1))
            which_g = g_probs.sample()
            mus_g, sigs_g = torch.gather(mus.squeeze(), 0, which_g), torch.gather(sigmas.squeeze(), 0, which_g)
            #print(mus_g.shape)
            for _ in range(num_next_s_samps): # this performs the same function as the ELBO sampling. 
                next_s = mus_g + sigs_g * torch.randn_like(mus_g)
                recon_obs = self.vae.decoder(next_s, next_r)
                # ELBO: 
                log_p_obs = elbo_calculation(recon_obs, rollout_dict['obs'], mus_g, torch.log(sigs_g), kl_tolerance=False)
                per_time_loss = torch.log(BCE) + log_reward_surprise
                print(per_time_loss.shape)
                # can sum across time with these logs. 
                expected_loss += torch.sum(per_time_loss, dim=0)
                # multiply all of these probabilities together within a single batch. 

        # average across the all of the sample rollouts. 
        return expected_loss / total_samples, mus, logsigmas # use these for the tilde computation

    def p_tilde(self, rollout_dict, elbo_samps,  mus, logsigmas):
        # for the actual observations need to compute the prob of seeing it and its reward
        # the rollout will also contain the reconstruction loss so: 

        # compute the ELBO: 
        log_p_obs = 0
        for _ in range(elbo_samps):
            latent_s =  mus + logsigmas.exp() * torch.randn_like(mus)
            recon_obs = self.vae.decoder(latent_s, rollout_dict['rewards'])
            # NOTE: should I have the free bits in here or not??? 
            # negative to make it approximate ln(p(x)). At least be proportional to this. Ignoring normalizing constants. 
            log_p_obs += elbo_calculation( recon_obs, rollout_dict['obs'], mus, logsigmas, kl_tolerance=False)

        # note that elbos are already in log. 
        log_p_obs /= elbo_samps # average here. 

        # all of the rewards
        log_reward_surprise = self.reward_prior.log_prob(rollout_dict['rewards'])

        per_time_loss = log_p_obs+reward_surprise

        # can sum across time with these logs. 
        expected_loss += torch.sum(per_time_loss, dim=0)

        return expected_loss

    def feef_loss(self, data_dict_rollout):

        # choose the action that minimizes the following reward.
        # provided with information from a single rollout 
        # this includes the observation, the actions taken, the VAE mu and sigma, and the next hidden state predictions. 
        # for p_opi
        # should see what the difference in variance is between using a single value and taking an expectation over many. 
        # as calculating a single value would be so much faster. 

        num_s_samps = 3
        num_next_s_samps = 3 
        elbo_samps_tilde = 5
        data_dict_rollout = {k:v.to(self.device) for k, v in data_dict_rollout.items()}

        self.reward_prior = Normal(1.5,0.5)

        with torch.no_grad():
            log_policy_loss, mus, logsigmas = self.p_policy(data_dict_rollout, num_s_samps, num_next_s_samps)
            log_tilde_loss =  torch.log( self.p_tilde(data_dict_rollout, elbo_samps_tilde, mus, logsigmas ) )

        return log_policy_loss - log_tilde_loss

    def simulate(self, params, train_mode=True, 
        render_mode=False, num_episode=16, 
        seed=27, max_len=1000, compute_feef=False): # run lots of rollouts 

        # update params into the controller
        self.controller = load_parameters(params, self.controller)

        recording_mode = False
        penalize_turning = False

        if train_mode and max_len < 0:
            max_episode_length = max_len

        #random(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        reward_list = []
        t_list = []
        if self.return_events:
            if compute_feef:
                feef_losses = []
            else: 
                data_dict_list = []

        if max_len ==-1:
            max_len = 1000 # making it very long

        with torch.no_grad():
            for i in range(num_episode):

                rand_env_seed = np.random.randint(0,1e9,1)[0]
                if self.return_events: 
                    rew, data_dict, t = self.rollout(rand_env_seed, render=render_mode, 
                                params=None, time_limit=max_len)
                    # data dict has the keys 'obs', 'rewards', 'actions', 'terminal'
                
                    if compute_feef: 
                        feef_losses.append(  self.feef_loss(data_dict)  )
                    else: 
                        data_dict_list.append(data_dict)
                else: 
                    rew, t = self.rollout(rand_env_seed, render=render_mode, 
                                params=None, time_limit=max_len)
                reward_list.append(rew)
                t_list.append(t)

        if self.return_events: 
            if compute_feef:
                return reward_list, t_list, feef_losses # no need to return the data.
            else: 
                return reward_list, t_list, data_dict_list
        else: 
            return reward_list, t_list


    

