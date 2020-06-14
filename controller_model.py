import math
import random 
import time
from os.path import join, exists
import torch
from torchvision import transforms
import numpy as np
from models import MDRNN, MDRNNCell, VAE, Controller
import pickle
import gym 
import gym.envs.box2d
from ha_env import make_env
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

    def __init__(self, env_name, time_limit, use_old_gym=False, 
        mdir=None, return_events=False, give_models=None, conditional=True, 
        joint_file_dir=False, planner_n_particles=100, cem_params=None, horizon=30):
        """ Build vae, rnn, controller and environment. """

        #self.env = gym.make('CarRacing-v0')
        self.env_name = env_name
        self.use_old_gym = use_old_gym

        self.device = 'cpu' #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.return_events = return_events
        self.time_limit = time_limit
        
        if cem_params:
            
            self.cem_mus = cem_params[0]
            self.cem_sigmas = cem_params[1]

        '''self.ensemble_batchsize = []
        for i in range(ensemble_size):
            self.ensemble_batchsize += [i]*indices_split
        self.ensemble_batchsize = np.asarray(self.ensemble_batchsize)'''

        self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((IMAGE_RESIZE_DIM, IMAGE_RESIZE_DIM)),
                transforms.ToTensor()
            ])

        #self.fixed_ob = pickle.load(open('notebooks/image_array.pkl', 'rb'))

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
            if joint_file_dir:
                vae_file, rnn_file, ctrl_file = \
                    [join(mdir, m+'_best.tar') for m in ['vae', 'mdrnn', 'ctrl']]
            else: 
                vae_file, rnn_file, ctrl_file = \
                    [join(mdir, m, 'best.tar') for m in ['vae', 'mdrnn', 'ctrl']]

            assert exists(vae_file) and exists(rnn_file),\
                "Either vae or mdrnn is untrained or the file is in the wrong place. "+vae_file+' '+rnn_file

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

            self.mdrnn_full = MDRNN(LATENT_SIZE, ACTION_SIZE, LATENT_RECURRENT_SIZE, 5, conditional=conditional ).to(self.device)
            self.mdrnn_full.load_state_dict(rnn_state['state_dict'])

            #print('loadin in controller.')
            '''if testing_old_controller: 
                self.controller = Controller(LATENT_SIZE, LATENT_RECURRENT_SIZE, ACTION_SIZE, conditional=False).to(self.device)
            else: 
                self.controller = Controller(LATENT_SIZE, LATENT_RECURRENT_SIZE, ACTION_SIZE, conditional=conditional).to(self.device)

            # load controller if it was previously saved
            if exists(ctrl_file):
                ctrl_state = torch.load(ctrl_file, map_location={'cuda:0': str(self.device)})
                print("Loading Controller with reward {}".format(
                    ctrl_state['reward']))
                self.controller.load_state_dict(ctrl_state['state_dict'])'''

        self.horizon = horizon
        self.planner_n_particles = planner_n_particles
        # TODO: make an MDRNN ensemble. 
        self.mdrnn_ensemble = [self.mdrnn]
        self.ensemble_size = len(self.mdrnn_ensemble)
        assert self.planner_n_particles  % self.ensemble_size==0, "Need planner n particles and ensemble size to be perfectly divisible!"
        self.k_top = int(planner_n_particles*0.10)
        self.ensemble_batchsize = self.planner_n_particles//self.ensemble_size 

    def make_env(self, seed=-1, render_mode=False, full_episode=False):
        self.render_mode = render_mode
        if self.use_old_gym:
            self.env = make_env(self.env_name, seed=seed, render_mode=render_mode, full_episode=full_episode)
        else: 
            self.env = gym.make("CarRacing-v0")
            self.env.seed(int(seed))
            self.env.render('rgb_array')

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

        '''if testing_old_controller: 
            action = self.controller(latent_s, hidden[0])
        else: 
            action = self.controller(latent_s, hidden[0], reward)'''

        action = self.planner(latent_s, hidden, reward)

        _, _, _, _, _, next_hidden = self.mdrnn(action, latent_s, hidden, reward)
        
        return action.squeeze().cpu().numpy(), next_hidden

    def planner(self, latent_s, full_hidden, reward):
        # predicts into the future up to horizon. 
        # returns the immediate action that will lead to the largest
        # cumulative reward

        cum_reward = torch.zeros((self.planner_n_particles))
        all_particles_first_action = []
        
        for mdrnn_ind, mdrnn_boot in enumerate(self.mdrnn_ensemble):

            # initialize particles for a single ensemble model. 
            ens_latent_s, ens_reward = [var.clone().repeat(self.ensemble_batchsize, *len(var.shape[1:])*[1]) for var in [latent_s, reward]]
            hidden_0 = full_hidden[0].clone().repeat(self.ensemble_batchsize, *len(full_hidden[0].shape[1:])*[1])
            hidden_1 = full_hidden[1].clone().repeat(self.ensemble_batchsize, *len(full_hidden[1].shape[1:])*[1])
            ens_full_hidden = [hidden_0,hidden_1]
            # only repeat the first dimension. 

            # need to produce a batch of first actions here. 
            ens_action = self.sample_cross_entropy_method() 

            for t in range(self.horizon):
                
                md_mus, md_sigmas, md_logpi, ens_reward, d, ens_full_hidden = mdrnn_boot(ens_action, ens_latent_s, ens_full_hidden, ens_reward)
                
                # get the next latent state
                g_probs = Categorical(probs=torch.exp(md_logpi.squeeze()).permute(0,2,1))
                which_g = g_probs.sample()
                mus_g, sigs_g = torch.gather(md_mus.squeeze(), 1, which_g.unsqueeze(1)).squeeze(), torch.gather(md_sigmas.squeeze(), 1, which_g.unsqueeze(1)).squeeze()
                ens_latent_s = mus_g + (sigs_g * torch.randn_like(mus_g))

                # sample actions for each particle!!! 
                # TODO: implement CEM here: 
                #ens_action = self.random_shooting(self.ensemble_batchsize)
                ens_action = self.sample_cross_entropy_method() 

                # store these cumulative rewards
                cum_reward[self.ensemble_batchsize*mdrnn_ind:self.ensemble_batchsize*mdrnn_ind+self.ensemble_batchsize] += ens_reward

                ens_reward = ens_reward.unsqueeze(1)

                if t==0:
                    # store next action to be taken
                    all_particles_first_action.append(ens_action)

        all_particles_first_action = torch.stack(all_particles_first_action).squeeze()

        # update CEM parameters: 
        # TODO: should CEM learn from all of the actions adn their rewards to go? 
        # or only from the first actions? 
        #print('best actions ind', all_particles_first_action.shape, cum_reward)
        self.update_cross_entropy_method(all_particles_first_action, cum_reward)
        #print('updated cross entropies')
        # choose the best next action out of all of them. 
        best_actions_ind = torch.argmax(cum_reward)
        best_action = all_particles_first_action[best_actions_ind]
        #print('best action is:', best_action)
        return best_action.unsqueeze(0)

    def sample_cross_entropy_method(self):
        actions = Normal(self.cem_mus, self.cem_sigmas).sample([self.ensemble_batchsize])
        # constrain these actions:
        if self.env_name=='carracing':
            actions = self.constrain_actions(actions)
        else:
            raise NotImplementedError("The environment you are trying to use does not have random shooting actions implemented.")
        return actions

    def update_cross_entropy_method(self, first_actions, rewards):
        # for carracing we have 3 independent gaussians
        smoothing = 0.5
        vals, inds = torch.topk(rewards, self.k_top )
        elite_actions = first_actions[inds]
        self.cem_mus = smoothing*self.cem_mus + (1-smoothing)*(elite_actions.sum(dim=0)/self.k_top) 
        self.cem_sigmas = smoothing*self.cem_sigmas+(1-smoothing)*(torch.sum( (elite_actions - self.cem_mus)**2, dim=0)/self.k_top )
        self.cem_sigmas = torch.clamp(self.cem_sigmas, min=0.2)
        #print('updated cems',self.cem_mus, self.cem_sigmas )

    def constrain_actions(self, out):
        #print('before tanh', out)
        out = torch.tanh(out)
        out[:,1] = (out[:,1]+1)/2.0 # this converts tanh to sigmoid
        out[:,2] = torch.clamp(out[:,2], min=0.0, max=1.0)
        #print('after all processing', out)
        return out

    def random_shooting(self, batch_size):
        if self.env_name=='carracing':
            out = torch.distributions.Uniform(-1,1).sample((batch_size, 3))
            out = self.constrain_actions(out)
        else:
            raise NotImplementedError("The environment you are trying to use does not have random shooting actions implemented.")

        return out

    def rollout(self, rand_env_seed, params=None, render=False, time_limit=None, trim_controls=True):
        """ Execute a rollout and returns minus cumulative reward.

        Load :params: into the controller and execute a single rollout. This
        is the main API of this class.

        :args params: parameters as a single 1D np array

        :returns: minus cumulative reward
        # Why is this the minus cumulative reward?!?!!?
        """

        #if self.use_old_gym:
        # setting in make env. 
        #self.env.render('rgb_array')
        self.trim_controls = trim_controls

        # copy params into the controller
        if params is not None:
            self.controller = load_parameters(params, self.controller)

        if not time_limit: 
            time_limit = self.time_limit 

        #random(rand_env_seed)
        np.random.seed(rand_env_seed)
        torch.manual_seed(rand_env_seed)
        #now setting in make env.
        #self.env.seed(int(rand_env_seed)) # ensuring that each rollout has a differnet random seed. 
        obs = self.env.reset()
        if self.use_old_gym:
            self.env.viewer.window.dispatch_events()

        # NOTE: TODO: call first render in the make env. note this is before I call reset though...
        #if not self.use_old_gym:
        # This first render is required !
        #    self.env.render()

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
            
            # using planner!
            action, hidden = self.get_action_and_transition(obs, hidden, reward)
            
            if self.return_events: 
                for key, var in zip(['obs', 'rewards', 'actions', 'terminal'], 
                                        [obs, reward, action, done ]):
                    if key == 'actions' or key=='terminal':
                        var = torch.Tensor([var])
                    rollout_dict[key].append(var.squeeze())
            #obs, reward, done = self.fixed_ob, np.random.random(1)[0], False
            obs, reward, done, _ = self.env.step(action)
            #print('reward recieved:', reward)
            if self.use_old_gym:
                self.env.viewer.window.dispatch_events()

            if render:
                pass
                #self.env.render()

            cumulative += reward
            if done or i > time_limit:
                #print('done with this simulation')
                if self.return_events:
                    for k,v in rollout_dict.items(): # list of tensors arrays.
                        #print(k, v[0].shape, len(v))
                        rollout_dict[k] = torch.stack(v)
                    return cumulative, i, rollout_dict # passed back to simulate. 
                else: 
                    return cumulative, i # ending time and cum reward
            i += 1

    def importance_sampling(self, num_samps, real_obs, encoder_mu, encoder_logsigma, cond_reward):
        """
        Returns a full batch. 
        """

        real_obs = real_obs.view(real_obs.size(0), -1) # flatten all but batch. 
        log_p_v = torch.zeros(encoder_mu.shape[0]) # batch size

        for _ in range(num_samps):

            z = encoder_mu + (encoder_logsigma.exp() * torch.randn_like(encoder_mu))

            decoder_mu, decoder_logsigma = self.vae.decoder(z, cond_reward)

            log_P_OBS_GIVEN_S = Normal(decoder_mu, decoder_logsigma.exp()).log_prob(real_obs)
            log_P_OBS_GIVEN_S = log_P_OBS_GIVEN_S.sum(dim=-1) #multiply the probabilities within the batch. 

            log_P_S = Normal(0.0, 1.0).log_prob(z).sum(dim=-1)
            log_Q_S_GIVEN_X = Normal(encoder_mu, encoder_logsigma.exp()).log_prob(z).sum(dim=-1)

            log_p_v += log_P_OBS_GIVEN_S + log_P_S - log_Q_S_GIVEN_X

        return log_p_v/num_samps

    def p_policy(self, rollout_dict, num_s_samps, num_next_encoder_samps, 
            num_importance_samps, encoder_mus, encoder_logsigmas):

        # rollout_dict values are of the size and shape seq_len, values dimensions. 

        # TODO: break the sequence length into smaller batches. 
        
        total_samples = num_s_samps * num_next_encoder_samps * num_importance_samps
        expected_loss = 0

        for i in range(num_s_samps): # vectorize this somehow 
            latent_s =  encoder_mus + encoder_logsigmas.exp() * torch.randn_like(encoder_mus)

            # predict the next state from this one. 
            # I already have access to the action from the runs generated. 
            # TODO: make both of these run with a batch. DONT NEED TO GENERATE OR PASS AROUND HIDDEN AS A RESULT. 

            #print(rollout_dict['actions'].shape, latent_s.shape, rollout_dict['rewards'].shape)
            pres_actions, pres_rewards = rollout_dict['actions'][:-1], rollout_dict['rewards'][:-1]

            # need to unsqueeze everything to add a batch dimension of 1. 
            md_mus, md_sigmas, md_logpi, next_r, d = self.mdrnn_full(pres_actions.unsqueeze(0), 
                                                                        latent_s.unsqueeze(0), pres_rewards.unsqueeze(0))

            next_r = next_r.squeeze()

            # reward loss 
            log_p_r = self.reward_prior.log_prob(next_r)

            g_probs = Categorical(probs=torch.exp(md_logpi.squeeze()).permute(0,2,1))
            for j in range(num_next_encoder_samps):
                which_g = g_probs.sample()
                mus_g, sigs_g = torch.gather(md_mus.squeeze(), 1, which_g.unsqueeze(1)).squeeze(), torch.gather(md_sigmas.squeeze(), 1, which_g.unsqueeze(1)).squeeze()
                #print('samples from mdrnn', mus_g.shape, sigs_g.shape)

                # importance sampling which has its own number of iterations: 
                next_obs = rollout_dict['obs'][1:]
                log_p_v = self.importance_sampling(num_importance_samps, next_obs, mus_g, torch.log(sigs_g), next_r.unsqueeze(1))
                
                # can sum across time with these logs. (as the batch is the different time points)
                expected_loss += torch.sum(log_p_v+log_p_r)

        # average across the all of the sample rollouts. 
        return expected_loss / total_samples

    def p_tilde(self, rollout_dict, num_importance_samps, encoder_mus, encoder_logsigmas):
        # for the actual observations need to compute the prob of seeing it and its reward
        # the rollout will also contain the reconstruction loss so: 

        pres_rewards = rollout_dict['rewards'][:-1]

        # all of the rewards
        log_p_r = self.reward_prior.log_prob(pres_rewards.squeeze())

        # compute the probability of the visual observations: 
        curr_obs = rollout_dict['obs'][:-1]
        log_p_v = self.importance_sampling(num_importance_samps, curr_obs, encoder_mus, encoder_logsigmas, pres_rewards)
        #print('p tilde', log_p_r.shape, log_p_v.shape)
        # can sum across time with these logs. (as the batch is the different time points)
        expected_loss = torch.sum(log_p_v+log_p_r)

        return expected_loss

    def feef_loss(self, data_dict_rollout, reward_prior_mu = 4.0, reward_prior_sigma=0.1):

        # choose the action that minimizes the following reward.
        # provided with information from a single rollout 
        # this includes the observation, the actions taken, the VAE mu and sigma, and the next hidden state predictions. 
        # for p_opi
        # should see what the difference in variance is between using a single value and taking an expectation over many. 
        # as calculating a single value would be so much faster. 
        print('computing feef loss')
        num_s_samps = 1
        num_next_encoder_samps = 1 
        num_importance_samps = 1 # I think 250 is ideal but probably too slow
        data_dict_rollout = {k:v.to(self.device) for k, v in data_dict_rollout.items()}
        data_dict_rollout['rewards'] = data_dict_rollout['rewards'].unsqueeze(1)

        self.reward_prior = Normal(reward_prior_mu,reward_prior_sigma) # treating this as basically a half normal. it should be higher than the max reward available for any run. 

        with torch.no_grad():

            # this computation is shared across both
            encoder_mus, encoder_logsigmas = self.vae.encoder(data_dict_rollout['obs'], data_dict_rollout['rewards'])

            # remove last one that is in the future. 
            encoder_mus, encoder_logsigmas = encoder_mus[:-1], encoder_logsigmas[:-1]
            #print('encoder mu going into feef calcs: ', encoder_mus.shape)

            log_policy_loss = self.p_policy(data_dict_rollout, num_s_samps, num_next_encoder_samps, 
                                            num_importance_samps, encoder_mus, encoder_logsigmas)
            log_tilde_loss =  self.p_tilde(data_dict_rollout, num_importance_samps, 
                                            encoder_mus, encoder_logsigmas )

        return log_policy_loss - log_tilde_loss

    def simulate(self, train_mode=True, 
        render_mode=False, num_episode=16, 
        seed=27, max_len=1000, compute_feef=False): # run lots of rollouts 

        # update params into the controller
        #self.controller = load_parameters(params, self.controller)

        recording_mode = False
        penalize_turning = False

        #if train_mode and max_len < 0:
        #    max_episode_length = max_len

        #random(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        reward_list = []
        t_list = []
        if self.return_events:
            data_dict_list = []
            if compute_feef:
                feef_losses_list = []
                
        if max_len ==-1:
            max_len = 1000 # making it very long

        with torch.no_grad():
            for i in range(num_episode):

                rand_env_seed = np.random.randint(0,1e9,1)[0]

                self.make_env(seed=rand_env_seed)
                
                if self.return_events: 
                    rew, t, data_dict = self.rollout(rand_env_seed, render=render_mode, 
                                params=None, time_limit=max_len)
                    # data dict has the keys 'obs', 'rewards', 'actions', 'terminal'
                
                    data_dict_list.append(data_dict)
                    if compute_feef: 
                        feef_losses_list.append(  self.feef_loss(data_dict)  )
                    
                else: 
                    rew, t = self.rollout(rand_env_seed, render=render_mode, 
                                params=None, time_limit=max_len)
                reward_list.append(rew)
                t_list.append(t)

                self.env.close()

        if self.return_events: 
            if compute_feef:
                return reward_list, t_list, data_dict_list, feef_losses_list  # no need to return the data.
            else: 
                return reward_list, t_list, data_dict_list 
        else: 
            return reward_list, t_list


    

