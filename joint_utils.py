
import numpy as np
import sys 
from os.path import join, exists
from os import mkdir, unlink, listdir, getpid
from bisect import bisect
import torch
import torch.utils.data
from torchvision import transforms
import gym 
from utils.misc import ACTION_SIZE, LATENT_RECURRENT_SIZE, LATENT_SIZE
import time
import cma
from controller_model import Models, load_parameters, flatten_parameters
from ray.util.multiprocessing import Pool
#from multiprocessing import Pool 

class GeneratedDataset(torch.utils.data.Dataset):
    """ """
    def __init__(self, transform, data, seq_len): 
        self._transform = transform
        self.data = data
        self._cum_size = [0]
        self._buffer_index = 0
        self._seq_len = seq_len

        # set the cum size tracker by iterating through the data:
        for d in self.data['terminal']:
            #print(d, 'whats being added to cum size')
            self._cum_size += [self._cum_size[-1] +
                                   (len(d)-self._seq_len)]

    def __getitem__(self, i): # kind of like the modulo operator but within rollouts of batch size. 
        # binary search through cum_size
        rollout_index = bisect(self._cum_size, i) - 1 # because it finds the index to the right of the element. 
        # within a specific rollout. will linger on one rollout for a while
        seq_index = i - self._cum_size[rollout_index] # references the previous file length. so normalizes to within this file's length. 
        
        obs_data = self.data['obs'][rollout_index][seq_index:seq_index + self._seq_len + 1]
        if self._transform:
            obs_data = self._transform(obs_data.astype(np.float32))
        action = self.data['actions'][rollout_index][seq_index+1:seq_index + self._seq_len + 1]
        #action = action.astype(np.float32)
        reward, terminal = [self.data[key][rollout_index][seq_index:
                                      seq_index + self._seq_len + 1]#.astype(np.float32)
                            for key in ('rewards', 'terminal')]
        # this transform is already being done and saved. 
        #reward = np.expand_dims(reward, 1)
        return obs_data, action, reward.unsqueeze(1), terminal
        
    def __len__(self):
        return self._cum_size[-1]

def combine_worker_rollouts(inp, seq_len, dim=1):
    """Combine data across the workers and their rollouts (making each rollout a separate batch)
    I currently only care about the data_dict_list. This is a list of each rollout: 
    each one has the dictionary keys: 'obs', 'rewards', 'actions', 'terminal'"""
    print('RUNNING COMBINE WORKER')

    first_iter = True
    for worker_rollouts in inp: 
        for rollout_data_dict in worker_rollouts[dim]: # this is pulling from a list!
            # this rollout is too small so it is being ignored. 
            # getting one of the keys from the dictionary
            if len(rollout_data_dict[list(rollout_data_dict.keys())[0]])-seq_len <= 0:
                print('!!!!!! Combine_worker_rollouts is ignoring rollout of length', len(rollout_data_dict[list(rollout_data_dict.keys())[0]]), 'for being smaller than the sequence length')
                continue

            if first_iter:
                combo_dict = {k:[v] for k, v in rollout_data_dict.items()} 
                first_iter = False
            else: 
                for k, v in combo_dict.items():
                    v.append(rollout_data_dict[k])

    return combo_dict

def generate_rollouts_using_planner(cem_params, horizon, num_action_repeats, planner_n_particles, seq_len, 
    time_limit, logdir, num_rolls_per_worker=2, num_workers=16, transform=None, joint_file_dir=True): # this makes 32 pieces of data.

    # 10% of the rollouts to use for test data. 
    ten_perc = np.floor(num_workers*0.1)
    if ten_perc == 0.0:
        ten_perc=1
    ninety_perc = int(num_workers- ten_perc)

    # generate a random number to give as a random seed to each pool worker. 
    rand_ints = np.random.randint(0, 1e9, num_workers)

    worker_data = []
    #NOTE: currently not using joint_file_directory here (this is used for each worker to know to pull files from joint or from a subsection)
    for i in range(num_workers):
        worker_data.append( (cem_params, horizon, num_action_repeats, planner_n_particles, rand_ints[i], num_rolls_per_worker, time_limit, logdir, True ) ) # compute FEEF.

    #res = ray.get( [worker.remote() ] )
    with Pool(processes=num_workers) as pool:
        all_worker_outputs = pool.map(worker, worker_data) 

    res = []
    for ind, worker_output in enumerate(all_worker_outputs): 
        res.append(worker_output[0])
        if ind==0:
            cem_mus = worker_output[1]  
            cem_sigmas = worker_output[2]
        else:
            cem_mus += worker_output[1] 
            cem_sigmas += worker_output[2]
    # averaging across the workers for the new CEM parameters. 
    cem_mus /= num_workers
    cem_sigmas /= num_workers

    # cem_smoothing: 
    alpha_smoothing=0.2
    cem_mus = cem_mus*alpha_smoothing + (1-alpha_smoothing)*cem_params[0]
    cem_sigmas = cem_sigmas*alpha_smoothing + (1-alpha_smoothing)*cem_params[1]

    # res is a list with tuples for each worker containing: reward_list, data_dict_list, t_list
    print('===== Done with pool of workers')

    # get all of the feef losses aggregated across the different workers: 
    feef_losses, reward_list = [], []
    for worker_rollouts in res:
        for ind, li in zip([3, 0], [feef_losses, reward_list]): # output order from simulate.
            li += worker_rollouts[ind]

    print("Number of rollouts being given to test:", len(res[ninety_perc:]))

    return (cem_mus, cem_sigmas), GeneratedDataset(transform, combine_worker_rollouts(res[:ninety_perc], seq_len, dim=2), seq_len),  \
                GeneratedDataset(transform, combine_worker_rollouts(res[ninety_perc:], seq_len, dim=2), seq_len), \
                feef_losses, reward_list

#@ray.remote
def worker(inp): # run lots of rollouts 
    cem_params, horizon, num_action_repeats, planner_n_particles, seed, num_episodes, max_len, logdir, compute_feef = inp
    gamename = 'carracing'
    model = Models(gamename, 1000, mdir = logdir, conditional=True, 
            return_events=True, use_old_gym=False, joint_file_dir=True,
            planner_n_particles = planner_n_particles, cem_params=cem_params, 
            horizon=horizon, num_action_repeats=num_action_repeats)

    return model.simulate(train_mode=True, render_mode=False, 
            num_episode=num_episodes, seed=seed, max_len=max_len, 
            compute_feef=compute_feef), model.cem_mus, model.cem_sigmas
