
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