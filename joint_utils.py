
import numpy as np
from os.path import join, exists
from os import mkdir, unlink, listdir, getpid
from multiprocessing import Pool 
from bisect import bisect
import torch
import torch.utils.data
from torchvision import transforms
import gym 
from utils.misc import RolloutGenerator, ACTION_SIZE, LATENT_RECURRENT_SIZE, LATENT_SIZE

class GeneratedDataset(torch.utils.data.Dataset): # pylint: disable=too-few-public-methods
    def __init__(self, transform, data, seq_len): 
        self._transform = transform
        self.data = data
        self._cum_size = None
        self._buffer_index = 0
        self._seq_len = seq_len

        # set the cum size tracker by iterating through the data:
        for d in self.data['term']:
            self._cum_size += [self._cum_size[-1] +
                                   (d.shape[0]-self._seq_len)]

    def __getitem__(self, i): # kind of like the modulo operator but within rollouts of batch size. 
        # binary search through cum_size
        rollout_index = bisect(self._cum_size, i) - 1 # because it finds the index to the right of the element. 
        # within a specific rollout. will linger on one rollout for a while
        seq_index = i - self._cum_size[rollout_index] # references the previous file length. so normalizes to within this file's length. 
        
        obs_data = self.data['obs'][rollout_index][seq_index:seq_index + self._seq_len + 1]
        obs_data = self._transform(obs_data.astype(np.float32))
        action = self.data['act'][rollout_index][seq_index+1:seq_index + self._seq_len + 1]
        action = action.astype(np.float32)
        reward, terminal = [self.data[key][rollout_index][seq_index:
                                      seq_index + self._seq_len + 1].astype(np.float32)
                            for key in ('rew', 'term')]
        reward = np.expand_dims(reward, 1)
        return obs_data, action, reward, terminal
        
    def __len__(self):
        return self._cum_size[-1]

def generate_rollouts(ctrl_params, transform, seq_len, 
    time_limit, logdir, num_rolls=2, num_workers=16): # this makes 32 pieces of data.

    # 10% of the rollouts to use for test data. 
    ninety_perc = ten_perc - np.floor(num_workers*0.1)

    # generate a random number to give as a random seed to each pool worker. 
    rand_ints = np.random.randint(0, 1e9, num_workers)

    worker_data = []
    for i in range(num_workers):
        worker_data.append( (ctrl_params, rand_ints[i], num_rolls, time_limit, logdir) )

    with Pool(processes=num_workers) as pool:
        res = pool.map(worker, worker_data) 
    # res is a list with tuples for each worker containing: reward_list, data_dict_list, t_list
    print('done with pool')

    # combine data across the workers and their rollouts (making each rollout a separate batch)
    # I currently only care about the data_dict_list. This is a list of each rollout: 
    # each one has the dictionary keys: 'obs', 'rewards', 'actions', 'terminal'
    
    def combine_worker_rollouts(input):
    
        first_iter = True
        for worker_rollouts in input: 
            for rollout_data_dict in worker_rollouts[1]:
                if first_iter:
                    combo_dict = rollout_data_dict
                    first_iter = False
                else: 
                    combo_dict = {k:v.append(rollout_data_dict[k]) for k, v in combo_dict.items()}
        
        #combo_dict = {k:torch.stack(v) for k, v in combo_dict.items()}
        return combo_dict

    return GeneratedDataset(transform, combine_worker_rollouts(res[:ninety_perc]) , seq_len),  \
                GeneratedDataset(transform, combine_worker_rollouts(res[ninety_perc:]), seq_len)

def worker(inp): # run lots of rollouts 
    ctrl_params, seed, num_episodes, max_len, logdir = inp
    print('worker has started')
    from controller_model import Models, load_parameters, flatten_parameters
    gamename = 'carracing'
    model = Models(gamename, 1000, mdir = logdir, conditional=True, 
            return_events=True, use_old_gym=False)

    model.make_env(seed)

    return model.simulate(ctrl_params, train_mode=True, render_mode=False, 
        num_episode=num_episodes, seed=seed, max_len=max_len)

if __name__ == '__main__':

    import json 

    with open('es_log/carracing.cma.12.64.best.json', 'r') as f:
        ctrl_params = json.load(f)

    transform = transforms.Lambda(
        lambda x: np.transpose(x, (0, 3, 1, 2)) / 255)

    print(len(ctrl_params[0]))

    generate_rollouts(ctrl_params[0], transform, 30, 1000, 
        'exp_dir', num_rolls = 4, num_workers = 1 )




