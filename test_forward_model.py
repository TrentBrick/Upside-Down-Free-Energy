
import numpy as np
import torch
from utils.misc import sample_mdrnn_latent
import pickle 
from models.vae import VAE
from models.mdrnn import MDRNNCell
from utils.misc import NUM_IMG_CHANNELS, NUM_GAUSSIANS_IN_MDRNN, LATENT_SIZE, LATENT_RECURRENT_SIZE, IMAGE_RESIZE_DIM, ACTION_SIZE
from os.path import join, exists
from torchvision.utils import save_image

directory = 'exp_dir'

#vae_file = join(directory, 'vae', 'best.tar')
#mdrnn_file = join(directory, 'mdrnn', 'best.tar')
vae_file = join(directory, 'joint', 'vae_best.tar')
mdrnn_file = join(directory, 'joint', 'mdrnn_best.tar')
assert exists(vae_file), "No VAE model in the directory..."
assert exists(mdrnn_file), "No MDRNN model in the directory..."

condition=True 

# load VAE
vae = VAE(NUM_IMG_CHANNELS, LATENT_SIZE, conditional=condition)
vae_state = torch.load(vae_file, map_location=lambda storage, location: storage)
print("Loading VAE at epoch {}, "
    "with test error {}...".format(
        vae_state['epoch'], vae_state['precision']))
vae.load_state_dict(vae_state['state_dict'])

# load MDRNN
mdrnn = MDRNNCell(LATENT_SIZE, ACTION_SIZE, LATENT_RECURRENT_SIZE, NUM_GAUSSIANS_IN_MDRNN)
mdrnn_state = torch.load(mdrnn_file, map_location=lambda storage, location: storage)
print("Loading MDRNN at epoch {}, "
    "with test error {}...".format(
        mdrnn_state['epoch'], mdrnn_state['precision']))
mdrnn_state_dict = {k.strip('_l0'): v for k, v in mdrnn_state['state_dict'].items()}
mdrnn.load_state_dict(mdrnn_state_dict)

turn_right = torch.Tensor([0.8, 0.8, 0.0]).unsqueeze(0)
turn_left = torch.Tensor([-0.8, 0.8, 0.0]).unsqueeze(0)
go_straight = torch.Tensor([0.0, 0.8, 0.0]).unsqueeze(0)
sit_still = torch.Tensor([0.0, 0.0, 0.0]).unsqueeze(0)

l_and_h_starter = pickle.load(open('latent_and_hidden_starters.pkl' ,'rb'))
lstart, hstart = l_and_h_starter[0], l_and_h_starter[1]

with torch.no_grad():
    # continuously turn right: 
    for name, rep_action in zip(['right', 'left', 'straight', 'still'],
                            [turn_right, turn_left, go_straight, sit_still]):

        all_particles_all_predicted_obs = []
        sequential_rewards = []

        # reset starting latent. 
        ens_reward = torch.Tensor([0.0]).unsqueeze(0)
        ens_latent_s, hidden = lstart, hstart

        # save the starting state
        dec_mu, dec_logsigma = vae.decoder(ens_latent_s, ens_reward)
        all_particles_all_predicted_obs.append( dec_mu )
        sequential_rewards.append(ens_reward)
        
        for t in range(20):
            md_mus, md_sigmas, md_logpi, ens_reward, d, hidden = mdrnn(rep_action, 
                                                                    ens_latent_s, 
                                                                    hidden, ens_reward)

            # get the next latent state
            ens_latent_s =  sample_mdrnn_latent(md_mus, md_sigmas, md_logpi, ens_latent_s)
            
            sequential_rewards.append(ens_reward)
            ens_reward = ens_reward.unsqueeze(0)
            dec_mu, dec_logsigma = vae.decoder(ens_latent_s, ens_reward)
            all_particles_all_predicted_obs.append( dec_mu )
            
        all_particles_all_predicted_obs = torch.stack(all_particles_all_predicted_obs).view(len(all_particles_all_predicted_obs), 3, 64, 64)
        save_image(all_particles_all_predicted_obs,
                    'exp_dir/forward_model_test/direction_' + name + '.png')

        print('reward sequence while action:', name, 'is:', sequential_rewards)
        print('========================')

                    