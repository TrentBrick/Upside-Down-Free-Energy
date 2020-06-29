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


"""def sample_mdrnn_latent(mus, sigmas, logpi, latent_s, no_delta=False, return_chosen_mus_n_sigs=False):
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
        return latent_s """


def generate_rssm_samples(rssm, for_vae_n_mdrnn_sampling,
                            samples_dir, SEQ_LEN, IMAGE_RESIZE_DIM, example_length, 
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
            adapt_obs = adapt_obs.view(adapt_obs.shape[0], 3, adapt_obs.shape[-1], adapt_obs.shape[-1])

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
