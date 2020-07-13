
import numpy as np 
import torch 
def combined_shape(length, shape=None):
    # taken from openAI spinning up. 
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
    # Taken from OpenAI spinning up. 
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.terminal_rew = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def add_rollouts(self, rollouts_dict):
        for np_buf, key in zip([self.obs_buf, self.obs2_buf, 
                            self.act_buf, 
                            self.rew_buf, self.done_buf, self.terminal_rew],
                            ['obs', 'obs2','act', 'rew', 'terminal', 'terminal_rew'] ):
            # TODO: combine these from the rollouts much earlier on. 
            temp_flat = None
            for rollout in rollouts_dict[key]:
                if temp_flat is None: 
                    temp_flat=rollout
                else: 
                    temp_flat = np.concatenate([temp_flat, rollout], axis=0)
            temp_flat = np.asarray(temp_flat)
            if key =='rew':
                print('number of iters being added', key, temp_flat.shape)
            #print('shape of cum rewards overall', len(cum_rewards_per_rollout))
            temp_flat = temp_flat.reshape(combined_shape(-1, np_buf.shape[1:]))
            iters_adding = len(temp_flat)
           
            if (self.ptr+iters_adding)>self.max_size:
                amount_pre_loop = self.max_size-self.ptr
                amount_post_loop = iters_adding-amount_pre_loop
                np_buf[self.ptr:] = temp_flat[:amount_pre_loop]
                np_buf[:amount_post_loop] = temp_flat[amount_pre_loop:]
            else: 
                np_buf[self.ptr:self.ptr+iters_adding] = temp_flat

        self.ptr = (self.ptr+iters_adding) % self.max_size
        self.size = min(self.size+iters_adding, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     terminal=self.done_buf[idxs],
                     terminal_rew=self.terminal_rew[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}

class TrainBufferDataset:
    def __init__(self, init_train_data, max_buffer_size, 
        key_to_check_lengths='terminal'):

        self.key_to_check_lengths = key_to_check_lengths
        self.max_buffer_size = max_buffer_size
        # init the buffer.
        self.buffer = init_train_data
        self.buffer_index = len(init_train_data[self.key_to_check_lengths])

    def add(self, train_data):
        curr_buffer_size = len(self.buffer[self.key_to_check_lengths])
        length_data_to_add = len(train_data[self.key_to_check_lengths])
        # dict agnostic length checker::: len(self.buffer[list(self.buffer.keys())[0]])
        if curr_buffer_size < self.max_buffer_size:
            print('growing buffer')
            for k in self.buffer.keys():
                self.buffer[k] += train_data[k]
            print('new buffer size', len(self.buffer[self.key_to_check_lengths]))
            self.buffer_index += length_data_to_add
            #if now exceeded buffer size: 
            if self.buffer_index>self.max_buffer_size:
                self.max_buffer_size=self.buffer_index
                self.buffer_index = 0
        else: 
            # buffer is now full. Rewrite to the correct index.
            if self.buffer_index > self.max_buffer_size-length_data_to_add:
                print('looping!')
                # going to go over so needs to loop around. 
                amount_pre_loop = self.max_buffer_size-self.buffer_index
                amount_post_loop = length_data_to_add-amount_pre_loop

                for k in self.buffer.keys():
                    self.buffer[k][self.buffer_index:] = train_data[k][:amount_pre_loop]

                for k in self.buffer.keys():
                    self.buffer[k][:amount_post_loop] = train_data[k][amount_pre_loop:]
                self.buffer_index = amount_post_loop
            else: 
                print('clean add')
                for k in self.buffer.keys():
                    self.buffer[k][self.buffer_index:self.buffer_index+length_data_to_add] = train_data[k]
                # update the index. 
                self.buffer_index += length_data_to_add
                self.buffer_index = self.buffer_index % self.max_buffer_size

    def __len__(self):
        return len(self.buffer[self.key_to_check_lengths])


# Datasets: 
class ForwardPlanner_GeneratedDataset(torch.utils.data.Dataset):
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