
import numpy as np
from os.path import join, exists
from os import mkdir, unlink, listdir, getpid
from multiprocessing import Queue, Process, cpu_count
from bisect import bisect
import torch
import torch.utils.data
from utils.misc import RolloutGenerator, ACTION_SIZE, LATENT_RECURRENT_SIZE, LATENT_SIZE
from traincontroller import slave_routine

class GeneratedDataset(torch.utils.data.Dataset): # pylint: disable=too-few-public-methods
    def __init__(self, transform, data, seq_len): 
        self._transform = transform
        self.data = data
        self._cum_size = None
        self._buffer_index = 0
        self.seq_len

        # set the cum size tracker by iterating through the data:
        for d in self.data['term']:
            self._cum_size += [self._cum_size[-1] +
                                   (d.shape[0]-self.seq_len)]

    def __getitem__(self, i): # kind of like the modulo operator but within rollouts of batch size. 
        # binary search through cum_size
        rollout_index = bisect(self._cum_size, i) - 1 # because it finds the index to the right of the element. 
        # within a specific rollout. will linger on one rollout for a while
        seq_index = i - self._cum_size[rollout_index] # references the previous file length. so normalizes to within this file's length. 
        
        obs_data = data['obs'][rollout_index][seq_index:seq_index + self._seq_len + 1]
        obs_data = self._transform(obs_data.astype(np.float32))
        action = data['act'][rollout_index][seq_index+1:seq_index + self._seq_len + 1]
        action = action.astype(np.float32)
        reward, terminal = [data[key][rollout_index][seq_index+1:
                                      seq_index + self._seq_len + 1].astype(np.float32)
                            for key in ('rew', 'term')]
        return obs_data, action, reward, terminal
        
    def __len__(self):
        return self._cum_size[-1]

def generate_rollouts(model_variables_dict, transform, seq_len, 
    time_limit, logdir, num_rolls=16, num_workers=16):

    # create tmp dir if non existent and clean it if existent
    tmp_dir = join(logdir, 'tmp')
    if not exists(tmp_dir):
        mkdir(tmp_dir)
    else:
        for fname in listdir(tmp_dir):
            unlink(join(tmp_dir, fname))

    assert num_workers <= cpu_count()

    # 10% of the rollouts to use for test data. 
    ten_perc = np.floor(num_rolls*0.1)

    # start processes for generating rollouts of the current policy. 
    p_queue = Queue()
    r_queue = Queue()
    e_queue = Queue()

    # generate a random number to give as a random seed to each process. 
    rand_ints = np.random.randint(0, 1e9,num_workers)

    for p_index in range(num_workers):
        Process(target=slave_routine, args=(p_queue, r_queue, e_queue, p_index, 
            rand_ints[p_index], time_limit, logdir, tmp_dir, model_variables_dict, return_events=True)).start()

    r_list = [0] * pop_size  # cum rewards list
    data_dict = {k:[] for k in ['obs', 'rew', 'act', 'term']}
    for _ in range(num_rolls):
        while r_queue.empty():
            sleep(.1)
        r_s_id, cum_rew, rollout_data_dict = r_queue.get() # data is currently all in numpy arrays in dictionary
        r_list[r_s_id] += cum_rew / n_samples

        for k, v in rollout_data_dict.items():
            data_dict[k].append(v) # list of full rollouts inside each. 

    e_queue.put('EOP')

    return GeneratedDataset(transform, data[:ten_perc], seq_len), GeneratedDataset(transform, data[ten_perc:], seq_len)

        

            