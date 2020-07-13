import numpy as np 
from utils import ReplayBuffer

train_buffer = ReplayBuffer(obs_dim=3, act_dim=1, size=100)

for e in range(30):
    amount_of_training_data = np.random.randint(10,12,1)[0]
    obs = [np.random.random((amount_of_training_data,3))]
    obs2 = obs 
    act = [np.random.random((amount_of_training_data,1))]
    rew = [np.random.random((amount_of_training_data))]
    terminal = [np.random.random((amount_of_training_data))]
    train_data = dict(terminal=terminal, rew=rew, 
        obs=obs, obs2=obs2, act=act )

    train_buffer.add_rollouts(train_data)
    
    print('epoch', e, 'size of buffer', train_buffer.size , 'buffer index', train_buffer.ptr )
    print('==========')
