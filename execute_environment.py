# pylint: disable=no-member
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
from utils.misc import NUM_IMG_CHANNELS, NUM_GAUSSIANS_IN_MDRNN, ACTION_SIZE, LATENT_RECURRENT_SIZE, LATENT_SIZE, IMAGE_RESIZE_DIM, NODE_SIZE, EMBEDDING_SIZE
from trainvae import loss_function as original_vae_loss_function
from utils.misc import sample_mdrnn_latent
from planet_models.rssm import RSSModel 
from planet_models.planner import Planner

class EnvSimulator:

    def __init__(self, gamename, logdir, vae_conditional, mdrnn_conditional,
        deterministic, use_lstm,
        time_limit=1000, planner_n_particles=100, horizon=30, 
        num_action_repeats=5, init_cem_params=None, cem_iters=10, 
        discount_factor=0.90):
        """ Build vae, forward model, and environment. """

        self.gamename = gamename
        self.deterministic = deterministic
        self.use_lstm = use_lstm

        self.use_rssm = True 

        self.device = 'cpu'#torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.time_limit = time_limit

        # transform used on environment generated observations. 
        self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((IMAGE_RESIZE_DIM, IMAGE_RESIZE_DIM)),
                transforms.ToTensor()
            ])

        # Loading world model and vae
        # NOTE: the checkpoint, ie most up to date model. Is being loaded in. 
        
        self.num_action_repeats = num_action_repeats
        # the real horizon 
        self.horizon = horizon
        self.planner_n_particles = planner_n_particles

        if init_cem_params:
            self.init_cem_mus = init_cem_params[0]
            self.init_cem_sigmas = init_cem_params[1]
        self.cem_iters = cem_iters 
        self.discount_factor = discount_factor
        self.k_top = int(planner_n_particles*0.10)

        if self.use_rssm:

            self.rssm = RSSModel(
                ACTION_SIZE,
                LATENT_RECURRENT_SIZE,
                LATENT_SIZE,
                EMBEDDING_SIZE,
                NODE_SIZE,
                device=self.device,
            )
            load_file = join(logdir, 'rssm_checkpoint.tar')

            assert exists(load_file), "Could not find file: " + load_file + " to load in!"
            state = torch.load(load_file, map_location={'cuda:0': str(self.device)})
            print("Loading model_type {} at epoch {} "
                "with test error {}".format('rssm',
                    state['epoch'], state['precision']))

            self.rssm.load_state_dict(state['state_dict'])

            self.rssm_planner = Planner(
                self.rssm,
                ACTION_SIZE,
                plan_horizon=horizon,
                optim_iters=cem_iters,
                candidates=planner_n_particles,
                top_candidates=self.k_top,
            )

            self.rssm.eval()

        else:
            vae_file, rnn_file = \
                [join(logdir, m+'_checkpoint.tar') for m in ['vae', 'mdrnn']]
        
            assert exists(vae_file) and exists(rnn_file),\
                "Either vae or mdrnn is untrained or the file is in the wrong place. "+vae_file+' '+rnn_file

            vae_state, rnn_state = [
                torch.load(fname, map_location={'cuda:0': str(self.device)})
                for fname in (vae_file, rnn_file)]

            for m, s in (('VAE', vae_state), ('MDRNN', rnn_state)):
                print("Loading {} at epoch {} "
                    "with test loss {}".format(
                        m, s['epoch'], s['precision']))
            print('loading in vae from:', vae_file, self.device)
            self.vae = VAE(NUM_IMG_CHANNELS, LATENT_SIZE, conditional=vae_conditional).to(self.device)
            self.vae.load_state_dict(vae_state['state_dict'])

            print('loading in mdrnn from:', rnn_file, self.device)
            if self.deterministic:
                self.mdrnn = MDRNN(LATENT_SIZE, ACTION_SIZE, LATENT_RECURRENT_SIZE, NUM_GAUSSIANS_IN_MDRNN, conditional=mdrnn_conditional, use_lstm=self.use_lstm).to(self.device)
                self.mdrnn.load_state_dict(rnn_state['state_dict'])
            else: 
                self.mdrnn = MDRNNCell(LATENT_SIZE, ACTION_SIZE, LATENT_RECURRENT_SIZE, NUM_GAUSSIANS_IN_MDRNN, conditional=mdrnn_conditional).to(self.device)
                self.mdrnn.load_state_dict(
                    {k.strip('_l0'): v for k, v in rnn_state['state_dict'].items()})

                self.mdrnn_full = MDRNN(LATENT_SIZE, ACTION_SIZE, LATENT_RECURRENT_SIZE, NUM_GAUSSIANS_IN_MDRNN, conditional=mdrnn_conditional, use_lstm=self.use_lstm ).to(self.device)
                self.mdrnn_full.load_state_dict(rnn_state['state_dict'])

            self.vae.eval()
            self.mdrnn.eval()
            self.mdrnn_full.eval()

        # TODO: make an MDRNN ensemble. 
        if self.use_rssm: 
            self.mdrnn_ensemble = [self.rssm]
        else: 
            self.mdrnn_ensemble = [self.mdrnn]

        self.ensemble_size = len(self.mdrnn_ensemble)
        assert self.planner_n_particles  % self.ensemble_size==0, "Need planner n particles and ensemble size to be perfectly divisible!"
        self.ensemble_batchsize = self.planner_n_particles//self.ensemble_size 
        
    def make_env(self, seed=-1, render_mode=False, full_episode=False):
        """ Called every time a new rollout occurs. Creates a new environment and 
        sets a new random seed for it."""
        self.render_mode = render_mode
        self.env = gym.make("CarRacing-v0")
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
            assert latent_s.shape == (1, LATENT_SIZE), 'latent z in controller is the wrong shape!!'
            action = self.planner(latent_s, hidden, reward)
            if self.deterministic:
                _, _, _, _, _, next_hidden = self.mdrnn(action, latent_s, reward, last_hidden=hidden)
            else: 
                _, _, _, _, _, next_hidden = self.mdrnn(action, latent_s, hidden, reward)
    
            return action.squeeze().cpu().numpy(), next_hidden

    def planner(self, latent_s, full_hidden, reward, discount_factor=0.90):
        """ Predicts into the future up to horizon. 
        Returns the immediate action that will lead to the largest 
        cumulative reward.
        
        :args: 
            BATCH_SIZE AND SEQUENCE LENGTH HERE ARE BOTH 1. 
            - latent_s: Torch tensor. (1, 1, LATENT_SIZE) starting latent state to plan from. 
            - full_hidden: tuple/list of 2 Torch tensors: (1, 1, LATENT_SIZE)
            if using RNN based forward model. 
            Gives memory associated with latent.
            - reward: Torch tensor (1, 1) last reward. 
            - discount_factor: float. used to discount future rewards exponentially.

        :returns: 
            - best_action: Torch tensor (1, ACTION_DIM). 
        
        """

        # starting CEM from scratch each time 
        self.cem_mus = self.init_cem_mus.clone()
        self.cem_sigmas = self.init_cem_sigmas.clone()

        # used to refine the CEM parameters. 
        for cem_iter in range(self.cem_iters):

            all_particles_cum_rewards = torch.zeros((self.planner_n_particles))
            all_particles_sequential_actions = torch.zeros((self.planner_n_particles, self.horizon, self.cem_mus.shape[0]))
            
            for mdrnn_ind, mdrnn_boot in enumerate(self.mdrnn_ensemble):

                # initialize particles for a single ensemble model. 
                # only repeat the first dimension. 
                ens_latent_s, ens_reward = [var.clone().repeat(self.ensemble_batchsize, *len(var.shape[1:])*[1]) for var in [latent_s, reward]]
                hidden_0 = full_hidden[0].clone().repeat(self.ensemble_batchsize, *len(full_hidden[0].shape[1:])*[1])
                hidden_1 = full_hidden[1].clone().repeat(self.ensemble_batchsize, *len(full_hidden[1].shape[1:])*[1])
                ens_full_hidden = [hidden_0,hidden_1]
                
                # indices for logging the actions and rewards during horizon planning across the ensemble. 
                start_ind = self.ensemble_batchsize*mdrnn_ind
                end_ind = start_ind+self.ensemble_batchsize
                # NOTE: I could do all of these CEM iters and updates within a 
                # single ensemble. Would be much slower and likely not much better though...

                # need to produce a batch of first actions here. 
                #ens_action = self.sample_cross_entropy_method() 
                #all_particles_sequential_actions[start_ind:end_ind, 0, :] = ens_action

                for t in range(0, self.horizon):

                    # sample an action and reward.
                    ens_action = self.sample_cross_entropy_method() 
                    if self.deterministic: 
                        md_mus, md_sigmas, md_logpi, ens_reward, d, ens_full_hidden = mdrnn_boot(ens_action, ens_latent_s, ens_reward, last_hidden=ens_full_hidden)
                        ens_latent_s = ens_latent_s + md_mus
                    else: 
                        md_mus, md_sigmas, md_logpi, ens_reward, d, ens_full_hidden = mdrnn_boot(ens_action, ens_latent_s, ens_full_hidden, ens_reward)
                        # predict the next latent state
                        ens_latent_s = sample_mdrnn_latent(md_mus, md_sigmas, md_logpi, ens_latent_s)

                    # store these cumulative rewards and action
                    all_particles_sequential_actions[start_ind:end_ind, t, :] = ens_action
                    all_particles_cum_rewards[start_ind:end_ind] += (self.discount_factor**t)*ens_reward

                    # unsqueeze for the next iteration. 
                    ens_reward = ens_reward.unsqueeze(1)
        
            self.update_cross_entropy_method(all_particles_sequential_actions, all_particles_cum_rewards)
        
        # choose the best next action out of all of them. 
        best_actions_ind = torch.argmax(all_particles_cum_rewards)
        best_action = all_particles_sequential_actions[best_actions_ind, 0, :]
        #print('best action is:', best_action)
        return best_action.unsqueeze(0)

    def sample_cross_entropy_method(self):
        """ Sample from the most up to date CEM parameters.
        
        :returns: 
            - actions: Torch tensor (num_particles, ACTION_DIM)
        """
        actions = Normal(self.cem_mus, self.cem_sigmas).sample([self.ensemble_batchsize])
        # constrain these actions:
        actions = self.constrain_actions(actions)
        
        return actions

    def update_cross_entropy_method(self, all_actions, rewards):
        """ Updates CEM. 
        
        :args: 
            - all_actions: Torch tensor (num_particles, horizon, action ). All 
                            actions taken by all particles in the CEM iteration. 
            - rewards: Torch tensor (num_particles). cumulative rewards from each particle. 
        
        Returns updated versions of cem_mu and cem_sigma.
        """
        # for carracing we have 3 independent gaussians
    
        # determines how quickly the CEM values update from those in the prev iteration.
        smoothing = 0.8
        vals, inds = torch.topk(rewards, self.k_top, sorted=False )
        elite_actions = all_actions[inds]

        num_elite_actions = self.k_top*self.horizon 

        new_mu = elite_actions.sum(dim=(0,1))/num_elite_actions
        new_sigma = torch.sqrt(torch.sum( (elite_actions - new_mu)**2, dim=(0,1))/num_elite_actions)
        self.cem_mus = smoothing*new_mu + (1-smoothing)*(self.cem_mus) 
        self.cem_sigmas = smoothing*new_sigma+(1-smoothing)*(self.cem_sigmas )

    def constrain_actions(self, out):
        """ Ensures actions sampled from the gaussians are within the game bounds."""
        if self.gamename=='carracing':
            out[:,0] = torch.clamp(out[:,0], min=-1.0, max=1.0)
            out[:,1] = torch.clamp(out[:,1], min=0.0, max=1.0)
            out[:,2] = torch.clamp(out[:,2], min=0.0, max=1.0)
        else:
            raise NotImplementedError("The environment you are trying to use does not have constrain actions implemented.")
        return out

    def random_shooting(self, batch_size):
        """ Generate random actions. Alternative to CEM."""
        if self.gamename=='carracing':
            out = torch.distributions.Uniform(-1,1).sample((batch_size, 3))
            out = self.constrain_actions(out)
        else:
            raise NotImplementedError("The environment you are trying to use does not have random shooting actions implemented.")

        return out

    def rollout(self, rand_env_seed, render=False):
        """ Executes a rollout and returns cumulative reward along with
        the time point the rollout stopped and 
        optionally, all observations/actions/rewards/terminal outputted 
        by the environment.

        :args:
            - rand_env_seed: int. Used to help guarentee this rollout is random.

        :returns: (cumulative, t, [rollout_dict])
            - cumulative: float. cumulative reward
            - t: int. timestep of termination
            - rollout_dict: OPTIONAL. Dictionary with keys: 'obs', 'rewards',
                'actions', 'terminal'. Each is a PyTorch Tensor with the first
                dimension corresponding to time. 
        """

        # base random seed set. Then sample new random numbers
        # for each individual rollout. 
        np.random.seed(rand_env_seed)
        torch.manual_seed(rand_env_seed)

        obs = self.env.reset()

        if self.use_rssm: 
            hidden, latent_s, rssm_action = self.rssm.init_hidden_state_action(1)

        else: 
            hidden = [
                torch.zeros(1, LATENT_RECURRENT_SIZE).to(self.device)
                for _ in range(2)]
        reward = 0
        done = 0
        action = np.array([0.,0.,0.])

        sim_rewards = []
        cumulative = 0
        t = 0
        if self.return_events: 
            rollout_dict = {k:[] for k in ['obs', 'rewards', 'actions', 'terminal']}
        while True:
            #print('iteration of the rollout', t)

            if self.gamename=='carracing':
                # trims the control panel at the base
                obs = obs[:84, :, :]

            obs = self.transform(obs).unsqueeze(0).to(self.device)
            reward = torch.Tensor([reward]).to(self.device).unsqueeze(0)
            
            # using planner!
            if self.use_rssm: 
                action, hidden, rssm_action, latent_s = self.get_action_and_transition(obs, reward, hidden, rssm_action, latent_s)
            else: 
                action, hidden = self.get_action_and_transition(obs, reward, hidden)
            
            # have this here rather than further down in order to capture the very first 
            # observation and actions.
            if self.return_events: 
                for key, var in zip(['obs', 'rewards', 'actions', 'terminal'], 
                                        [obs, reward, action, done ]):
                    if key == 'actions' or key=='terminal':
                        var = torch.Tensor([var])
                    rollout_dict[key].append(var.squeeze())

            # using action repeats
            for _ in range(self.num_action_repeats):
                obs, reward, done, _ = self.env.step(action)
                cumulative += reward

                # if the last 40 steps (real steps independent of action repeats)
                # have all given -0.1 reward then cut the rollout early. 
                if self.gamename == 'carracing':
                    if len(sim_rewards) <40:
                        sim_rewards.append(reward)
                    else: 
                        sim_rewards.pop(0)
                        sim_rewards.append(reward)
                        #print('lenght of sim rewards',  len(sim_rewards),round(sum(sim_rewards),3))
                        if round(sum(sim_rewards), 3) == -4.0:
                            done=True

                if done or t > self.time_limit:
                #print('done with this simulation')
                    if self.return_events:
                        for k,v in rollout_dict.items(): # list of tensors arrays.
                            #print(k, v[0].shape, len(v))
                            rollout_dict[k] = torch.stack(v)
                        return cumulative, t, rollout_dict # passed back to simulate. 
                    else: 
                        return cumulative, t # ending time and cum reward
                t += 1

    def importance_sampling(self, num_samps, real_obs, latent_s, encoder_mu, 
        encoder_logsigma, cond_reward, delta_prediction=False, pres_latent_s=None):
        """
        Returns a full batch. 
        """

        real_obs = real_obs.view(real_obs.size(0), -1) # flatten all but batch. 
        log_p_v = torch.zeros(encoder_mu.shape[0]) # batch size

        for _ in range(num_samps):

            decoder_mu, decoder_logsigma = self.vae.decoder(latent_s, cond_reward)

            log_P_OBS_GIVEN_S = Normal(decoder_mu, decoder_logsigma.exp()).log_prob(real_obs)
            log_P_OBS_GIVEN_S = log_P_OBS_GIVEN_S.sum(dim=-1) #multiply the probabilities within the batch. 

            log_P_S = Normal(0.0, 1.0).log_prob(latent_s).sum(dim=-1)
            log_Q_S_GIVEN_X = Normal(encoder_mu, encoder_logsigma.exp()).log_prob(latent_s).sum(dim=-1)

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
            latent_s = latent_s.unsqueeze(0)
            # predict the next state from this one. 
            # I already have access to the action from the runs generated. 
            # TODO: make both of these run with a batch. DONT NEED TO GENERATE OR PASS AROUND HIDDEN AS A RESULT. 

            #print(rollout_dict['actions'].shape, latent_s.shape, rollout_dict['rewards'].shape)
            pres_actions, pres_rewards = rollout_dict['actions'][:-1], rollout_dict['rewards'][:-1]

            # need to unsqueeze everything to add a batch dimension of 1. 
            
            md_mus, md_sigmas, md_logpi, next_r, d = self.mdrnn_full(pres_actions.unsqueeze(0), 
                                                                        latent_s, pres_rewards.unsqueeze(0))

            next_r = next_r.squeeze()

            # reward loss 
            log_p_r = self.reward_prior.log_prob(next_r)

            next_obs = rollout_dict['obs'][1:]

            for j in range(num_next_encoder_samps):

                next_encoder_sample, mus_g, sigs_g = sample_mdrnn_latent(md_mus, md_sigmas, 
                                        md_logpi, latent_s, 
                                        return_chosen_mus_n_sigs=True)

                # importance sampling which has its own number of iterations: 
                log_p_v = self.importance_sampling(num_importance_samps, next_obs, 
                    next_encoder_sample, mus_g, torch.log(sigs_g), 
                    next_r.unsqueeze(1))
                
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
                    rew, t, data_dict = self.rollout(rand_env_seed, render=render_mode)
                    # data dict has the keys 'obs', 'rewards', 'actions', 'terminal'
                
                    data_dict_list.append(data_dict)
                    if compute_feef: 
                        feef_losses_list.append(  0.0 )#self.feef_loss(data_dict)  )
                    
                else: 
                    rew, t = self.rollout(rand_env_seed, render=render_mode)
                cum_reward_list.append(rew)
                t_list.append(t)

                self.env.close()

        if self.return_events: 
            if compute_feef:
                return cum_reward_list, t_list, data_dict_list, feef_losses_list  # no need to return the data.
            else: 
                return cum_reward_list, t_list, data_dict_list 
        else: 
            return cum_reward_list, t_list


    

