# pylint: disable=no-member
""" 
Useful utilities for Joint Training.  
"""
import numpy as np
import sys 
from os.path import join, exists
from os import mkdir, unlink, listdir, getpid
import torch
import torch.utils.data
from torchvision import transforms
import gym
from bisect import bisect
import time
from torchvision.utils import save_image
from control import Agent
from ray.util.multiprocessing import Pool
#from multiprocessing import Pool 
import copy

generic_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_RESIZE_DIM, IMAGE_RESIZE_DIM)),
    transforms.ToTensor()
])

def write_logger(logger_filename, train_loss_dict, test_loss_dict):
    # Header at the top of logger file written once at the start of new training run.
    if not exists(logger_filename): 
        header_string = ""
        for loss_dict, train_or_test in zip([train_loss_dict, test_loss_dict], ['train', 'test']):
            for k in loss_dict.keys():
                header_string+=train_or_test+'_'+k+' '
        header_string+= '\n'
        with open(logger_filename, "w") as file:
            file.write(header_string) 

    # write out all of the logger losses.
    with open(logger_filename, "a") as file:
        log_string = ""
        for loss_dict in [train_loss_dict, test_loss_dict]:
            for k, v in loss_dict.items():
                log_string += str(v)+' '
        log_string+= '\n'
        file.write(log_string)


def save_checkpoint(state, is_best, filename, best_filename):
    """ Save state in filename. Also save in best_filename if is_best. """
    start_time = time.time()
    torch.save(state, filename)
    if is_best:
        torch.save(state, best_filename)
    print('seconds taken to save checkpoint.',(time.time()-start_time) )


def sample_mdrnn_latent(mus, sigmas, logpi, latent_s, no_delta=False, return_chosen_mus_n_sigs=False):
    if NUM_GAUSSIANS_IN_MDRNN > 1:
        assert len(mus.shape) == len(latent_s.shape)+1, "Need shape of latent to be one more than sufficient stats! Shape of mus and then latents."+str(mus.shape)+' '+str(latent_s.shape)
        if len(logpi.shape) == 3: 
            g_probs = Categorical(probs=torch.exp(logpi.squeeze()).permute(0,2,1))
            which_g = g_probs.sample()
            mus, sigmas = torch.gather(mus.squeeze(), 1, which_g.unsqueeze(1)).squeeze(), torch.gather(sigmas.squeeze(), 1, which_g.unsqueeze(1)).squeeze()
        elif len(logpi.shape) == 4:
            g_probs = torch.distributions.Categorical(probs=torch.exp(logpi.squeeze()).permute(0,1,3,2))
            which_g = g_probs.sample()
            print('how are the gaussian probabilities distributed??', logpi[0,0,:,0].exp(), logpi[0,0,:,1].exp())
            print('the gaussian mus are:', mus[0,0,:,0], mus[0,0,:,1])
            print('g_probs are:', which_g.shape)
            # this is selecting where there are 4 dimensions rather than just 3. 
            mus, sigmas = torch.gather(mus.squeeze(), 2, which_g.unsqueeze(2)).squeeze(), torch.gather(sigmas.squeeze(), 2, which_g.unsqueeze(2)).squeeze()
        else:
            print('size of mus and sigmas is neither 3D nor 4D.')
            raise ValueError
    else: 
        mus, sigmas = mus.squeeze(), sigmas.squeeze()
        latent_s = latent_s.squeeze()

    # predict the next latent state. 
    pred_latent = mus + (sigmas * torch.randn_like(mus))
    #print('size of predicted deltas and real', pred_latent_deltas.shape, latent_s.shape)
    if no_delta:
        latent_s = pred_latent
    else:
        latent_s = latent_s+pred_latent

    if return_chosen_mus_n_sigs: 
        return latent_s, mus, sigmas
    else: 
        return latent_s 


def generate_rssm_samples(rssm, for_vae_n_mdrnn_sampling, deterministic,
                            samples_dir, SEQ_LEN, example_length, 
                            memory_adapt_period, e, device,
                            make_vae_samples=False,
                            make_mdrnn_samples=True, 
                            transform_obs = False):

    # need to restrict the data to a random segment. Important in cases where 
    # sequence length is too long
    start_sample_ind = np.random.randint(0, SEQ_LEN-example_length,1)[0]
    end_sample_ind = start_sample_ind+example_length

    # ensuring this is the same length as everything else. 
    #for_vae_n_mdrnn_sampling[0] = for_vae_n_mdrnn_sampling[0][1:, :, :, :]

    last_test_observations, last_test_decoded_obs, \
    last_test_hiddens, last_test_prior_states, \
    last_test_pres_rewards, last_test_next_rewards, \
    last_test_latent_obs, \
    last_test_actions = [var[start_sample_ind:end_sample_ind] for var in for_vae_n_mdrnn_sampling]

    last_test_encoded_obs = rssm.encode_sequence_obs(last_test_observations.unsqueeze(1))

    print('last test obs before reshaping:', last_test_observations.shape)

    last_test_observations = last_test_observations.view(last_test_observations.shape[0], 3, IMAGE_RESIZE_DIM, IMAGE_RESIZE_DIM).cpu()
    last_test_decoded_obs = last_test_decoded_obs.view(last_test_decoded_obs.shape[0], 3, IMAGE_RESIZE_DIM, IMAGE_RESIZE_DIM).cpu()

    if make_vae_samples:
        with torch.no_grad():
            # get test samples
            to_save = torch.cat([last_test_observations, last_test_decoded_obs], dim=0)
            print('Generating VAE samples of the shape:', to_save.shape)
            save_image(to_save,
                    join(samples_dir, 'vae_sample_' + str(e) + '.png'))

        print('====== Done Generating VAE Samples')

    if make_mdrnn_samples: 
        with torch.no_grad():
            # print examples of the prior

            horizon_one_step = rssm.decode_obs(last_test_hiddens, last_test_prior_states)
            horizon_one_step_obs = horizon_one_step.view(horizon_one_step.shape[0],3, IMAGE_RESIZE_DIM, IMAGE_RESIZE_DIM)
            
            # print multi horizon examples. 

            # set memory and context: 
            last_test_actions = last_test_actions.unsqueeze(1)
            adapt_dict = rssm.perform_rollout(last_test_actions[:memory_adapt_period], 
                encoder_output=last_test_encoded_obs[:memory_adapt_period] ) 
            #print('into decoder:', adapt_dict['hiddens'].shape, adapt_dict['posterior_states'].shape)
            adapt_obs = rssm.decode_sequence_obs(adapt_dict['hiddens'], adapt_dict['posterior_states'])
            adapt_obs = adapt_obs.view(adapt_obs.shape[0], 3, IMAGE_RESIZE_DIM, IMAGE_RESIZE_DIM)

            #print('adapt dict keys', adapt_dict.keys())
            #print('into horizon predictions', last_test_actions[memory_adapt_period:].shape, 
            #    adapt_dict['hiddens'][-1].shape , 
            #    adapt_dict['posterior_states'][-1].shape)

            horizon_multi_step_dict = rssm.perform_rollout(last_test_actions[memory_adapt_period:], hidden=adapt_dict['hiddens'][-1] , 
                state=adapt_dict['posterior_states'][-1] )
            
            horizon_multi_step_obs = rssm.decode_sequence_obs(horizon_multi_step_dict['hiddens'], horizon_multi_step_dict['prior_states'])
            horizon_multi_step_obs = horizon_multi_step_obs.view(horizon_multi_step_obs.shape[0],3, IMAGE_RESIZE_DIM, IMAGE_RESIZE_DIM)

            to_save = torch.cat([last_test_observations, last_test_decoded_obs, 
                horizon_one_step_obs.cpu(), adapt_obs.cpu(), horizon_multi_step_obs.cpu()], dim=0)

            print('Generating MDRNN samples of the shape:', to_save.shape)
            save_image(to_save,
                    join(samples_dir, 'horizon_pred_sample_' + str(e) + '.png'))


class GeneratedDataset(torch.utils.data.Dataset):
    """ This dataset is inspired by those from dataset/loaders.py but it 
    doesn't need to apply any transformations to the data or load in any 
    files.

    :args:
        - transform: any tranforms desired. Currently these are done by each rollout
        and sent back to avoid performing redundant transforms.
        - data: a dictionary containing a list of Pytorch Tensors. 
                Each element of the list corresponds to a separate full rollout.
                Each full rollout has its first dimension corresponding to time. 
        - seq_len: desired length of rollout sequences. Anything shorter must have 
        already been dropped. (currently done in 'combine_worker_rollouts()')

    :returns:
        - a subset of length 'seq_len' from one of the rollouts with all of its relevant features.
    """
    def __init__(self, transform, data, seq_len): 
        self._transform = transform
        self.data = data
        self._cum_size = [0]
        self._buffer_index = 0
        self._seq_len = seq_len

        # set the cum size tracker by iterating through the data:
        for d in self.data['terminal']:
            self._cum_size += [self._cum_size[-1] +
                                   (len(d)-self._seq_len)]

    def __getitem__(self, i): # kind of like the modulo operator but within rollouts of batch size. 
        # binary search through cum_size
        rollout_index = bisect(self._cum_size, i) - 1 # because it finds the index to the right of the element. 
        # within a specific rollout. will linger on one rollout for a while iff random sampling not used. 
        seq_index = i - self._cum_size[rollout_index] # references the previous file length. so normalizes to within this file's length. 
        obs_data = self.data['obs'][rollout_index][seq_index:seq_index + self._seq_len + 1]
        if self._transform:
            obs_data = self._transform(obs_data.astype(np.float32))
        action = self.data['actions'][rollout_index][seq_index:seq_index + self._seq_len + 1]
        reward, terminal = [self.data[key][rollout_index][seq_index:
                                      seq_index + self._seq_len + 1]
                            for key in ('rewards', 'terminal')]
        return obs_data, action, reward.unsqueeze(1), terminal
        
    def __len__(self):
        return self._cum_size[-1]

def combine_worker_rollouts(inp, seq_len, dim=1):
    """Combine data across the workers and their rollouts (making each rollout a separate element in a batch)
    
    :args:
        - inp: list. containing outputs of each worker. Assumes that each worker output inside this list is itself a list (or tuple)
        - seq_len: int. given to ensure each rollout is equal to or longer. 
        - dim: int. for a given worker, the element in its returned list that corresponds to the rollouts to be combined.

    :returns:
        - dictionary of rollouts organized by element type (eg. actions, pixel based observations). 
            Each dictionary key corresponds to a list of full rollouts. 
            Each full rollout has its first dimension corresponding to time. 
    """
    
    print('RUNNING COMBINE WORKER')
    first_iter = True
    # iterate through the worker outputs. 
    for worker_rollouts in inp: 
        # iterate through each of the rollouts within the list of rollouts inside this worker.
        for rollout_data_dict in worker_rollouts[dim]: # this is pulling from a list!
            if len(rollout_data_dict[list(rollout_data_dict.keys())[0]])-seq_len <= 0:
                # this rollout is too small so it is being ignored. 
                # getting one of the keys from the dictionary in a dict agnostic way.
                # NOTE: assumes that all keys in the dictionary correspond to lists of the same length
                print('!!!!!! Combine_worker_rollouts is ignoring rollout of length', len(rollout_data_dict[list(rollout_data_dict.keys())[0]]), 'for being smaller than the sequence length')
                continue

            if first_iter:
                # init the combo dictionary with lists for each key. 
                combo_dict = {k:[v] for k, v in rollout_data_dict.items()} 
                first_iter = False
            else: 
                # append items to the list for each key
                for k, v in combo_dict.items():
                    v.append(rollout_data_dict[k])

    return combo_dict

def generate_rollouts_using_planner(num_workers, seq_len, worker_package): 

    """ Uses ray.util.multiprocessing Pool to create workers that each 
    run environment rollouts using planning with CEM (Cross Entropy Method). 
    Each worker (that runs code in execute_environment.py) passes back the 
    rollouts which have already been transformed (done for the planner to work). 
    These are split between train and test. The FEEF (Free energy of the expected future)
    and cumulative rewards from the rollout are also passed back. 

    Arguments and returns should be self explanatory. 
    All inputs are ints, strings or bools except 'init_cem_params' 
    which is a tuple/list containing two tensors. One for mus and other
    for sigmas. 
    """

    # 10% of the rollouts to use for test data. 
    ten_perc = np.floor(num_workers*0.1)
    if ten_perc == 0.0:
        ten_perc=1
    ninety_perc = int(num_workers-ten_perc)

    # generate a random number to give as a random seed to each pool worker. 
    rand_ints = np.random.randint(0, 1e9, num_workers)

    worker_data = []
    #NOTE: currently not using joint_file_directory here (this is used for each worker to know to pull files from joint or from a subsection)
    for i in range(num_workers):

        package_w_rand = copy.copy(worker_package)
        package_w_rand.append(rand_ints[i])

        worker_data.append( tuple(package_w_rand)  )

    # deploy all of the workers and wait for results. 
    with Pool(processes=num_workers) as pool:
        res = pool.map(worker, worker_data) 

    # res is a list with tuples for each worker containing: reward_list, data_dict_list, t_list
    print('===== Done with pool of workers')

    # get all of the feef losses aggregated across the different workers: 
    feef_losses, reward_list = [], []
    for worker_rollouts in res:
        for ind, li in zip([3, 0], [feef_losses, reward_list]): # output order from simulate.
            li += worker_rollouts[ind]

    print("Number of rollouts being given to test:", len(res[ninety_perc:]))

    return combine_worker_rollouts(res[:ninety_perc], seq_len, dim=2), \
                combine_worker_rollouts(res[ninety_perc:], seq_len, dim=2), \
                feef_losses, reward_list

def worker(inp): # run lots of rollouts 

    """ Parses its input before creating an environment simulator instance
    and using it to generate the desired number of rollouts. These rollouts
    save and return the observations, rewards, and FEEF calculations."""

    gamename, vae_conditional, mdrnn_conditional, \
        deterministic, use_lstm, \
        time_limit, logdir, num_episodes, \
        horizon, num_action_repeats, \
        planner_n_particles, init_cem_params, \
        cem_iters, discount_factor, \
        compute_feef, seed = inp
    
    env_simulator = EnvSimulator(gamename, logdir, vae_conditional, mdrnn_conditional,
            deterministic, use_lstm,
            time_limit=time_limit,
            
            planner_n_particles = planner_n_particles, 
            horizon=horizon, num_action_repeats=num_action_repeats,
            init_cem_params=init_cem_params, cem_iters=cem_iters, 
            discount_factor=discount_factor)

    return env_simulator.simulate(return_events=True,
            compute_feef=compute_feef,
            num_episodes=num_episodes, seed=seed)
