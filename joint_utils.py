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
from utils.misc import ACTION_SIZE, LATENT_RECURRENT_SIZE, LATENT_SIZE
import time
from execute_environment import EnvSimulator
from ray.util.multiprocessing import Pool
import copy
#from multiprocessing import Pool 

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

        (time_limit, horizon, num_action_repeats, 
            planner_n_particles, init_cem_params, cem_iters, discount_factor,
            rand_ints[i], num_rolls_per_worker, 
            time_limit, logdir, compute_feef )

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
        time_limit, logdir, num_episodes, \
        horizon, num_action_repeats, \
        planner_n_particles, init_cem_params, \
        cem_iters, discount_factor, \
        compute_feef, seed = inp
    
    model = EnvSimulator(gamename, logdir, vae_conditional, mdrnn_conditional,
            time_limit=time_limit,
            vae_conditional=vae_conditional, mdrnn_conditional=mdrnn_conditional,
            return_events=True,
            planner_n_particles = planner_n_particles, 
            horizon=horizon, num_action_repeats=num_action_repeats,
            init_cem_params=init_cem_params, cem_iters=cem_iters, 
            discount_factor=discount_factor, use_old_gym=False)

    return model.simulate(train_mode=True, render_mode=False, 
            num_episode=num_episodes, seed=seed, 
            compute_feef=compute_feef)
