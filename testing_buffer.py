
import numpy as np 

max_buffer_size = 100

for e in range(30):
    amount_of_training_data = np.random.randint(10,20,1)[0]
    terminal = np.random.random((amount_of_training_data,3))
    reward = np.random.random((amount_of_training_data,3))
    train_data = dict(terminal=terminal, reward=reward)

    print('amount of training data', amount_of_training_data)

    if e==0:
        # init buffers
        buffer_train_data = train_data
        buffer_index = len(train_data['terminal'])
    else: 
        curr_buffer_size = len(buffer_train_data['terminal'])
        length_data_added = len(train_data['terminal'])
        # dict agnostic length checker::: len(buffer_train_data[list(buffer_train_data.keys())[0]])
        if curr_buffer_size < max_buffer_size:
            # grow the buffer
            print('growing buffer')
            for k, v in buffer_train_data.items():
                #buffer_train_data[k] = np.concatenate([v, train_data[k]], axis=0)
                buffer_train_data[k] = torch.cat([v, train_data[k]], dim=0)
            print('new buffer size', len(buffer_train_data['terminal']))
            buffer_index += length_data_added
            #if now exceeded buffer size: 
            if buffer_index>max_buffer_size:
                max_buffer_size=buffer_index
                buffer_index = 0
        else: 
            # buffer is max size. Rewrite the correct index.
            if buffer_index > max_buffer_size-length_data_added:
                print('looping!')
                # going to go over so needs to loop around. 
                amount_pre_loop = max_buffer_size-buffer_index
                amount_post_loop = length_data_added-amount_pre_loop

                for k in buffer_train_data.keys():
                    buffer_train_data[k][buffer_index:] = train_data[k][:amount_pre_loop]

                for k in buffer_train_data.keys():
                    buffer_train_data[k][:amount_post_loop] = train_data[k][amount_pre_loop:]
                buffer_index = amount_post_loop
            else: 
                print('clean add')
                for k in buffer_train_data.keys():
                    buffer_train_data[k][buffer_index:buffer_index+length_data_added] = train_data[k]
                # update the index. 
                buffer_index += length_data_added
                buffer_index = buffer_index % max_buffer_size

    print('epoch', e, 'size of buffer', len(buffer_train_data['terminal']), 'buffer index', buffer_index)
    print('==========')
