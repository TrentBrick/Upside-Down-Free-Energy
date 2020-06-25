# pylint: disable=no-member
""" Various auxiliary utilities """
import math
import time 
from os.path import join, exists
import torch
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
from models import MDRNNCell, VAE, Controller
import gym
import gym.envs.box2d
from torch.distributions.categorical import Categorical
import torch.nn.functional as f

# A bit dirty: manually change size of car racing env
# BUG: this makes the images very grainy!!!
#gym.envs.box2d.car_racing.STATE_W, gym.envs.box2d.car_racing.STATE_H = 64, 64

# Hardcoded for now
NUM_IMG_CHANNELS, ACTION_SIZE, LATENT_SIZE, LATENT_RECURRENT_SIZE, IMAGE_RESIZE_DIM, SIZE =\
    3, 3, 32, 512, 64, 96
NUM_GAUSSIANS_IN_MDRNN = 1
NODE_SIZE, EMBEDDING_SIZE = 256, 256

# Same. used for Rollout Generator below. 
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_RESIZE_DIM, IMAGE_RESIZE_DIM)),
    transforms.ToTensor()
])

def sample_continuous_policy(action_space, seq_len, dt):
    """ Sample a continuous policy.

    Atm, action_space is supposed to be a box environment. The policy is
    sampled as a brownian motion a_{t+1} = a_t + sqrt(dt) N(0, 1).

    :args action_space: gym action space
    :args seq_len: number of actions returned
    :args dt: temporal discretization

    :returns: sequence of seq_len actions
    """
    actions = [action_space.sample()]
    # bias towards more forward driving at the start in order to produce diverse observations. 
    actions[0][1] = 0.9
    # and not having the brakes on!
    actions[0][2] = 0.0
    print('first action being used', actions)
    for _ in range(seq_len):
        daction_dt = np.random.randn(*actions[-1].shape)
        next_action = np.clip(actions[-1] + math.sqrt(dt) * daction_dt,
                    action_space.low, action_space.high)
        next_action[2] = np.clip(next_action[2], 0.0, 0.2)
        actions.append( next_action)
    return actions

def save_checkpoint(state, is_best, filename, best_filename):
    """ Save state in filename. Also save in best_filename if is_best. """
    start_time = time.time()
    torch.save(state, filename)
    if is_best:
        torch.save(state, best_filename)
    print('seconds taken to save checkpoint.',(time.time()-start_time) )

def flatten_parameters(params):
    """ Flattening parameters.

    :args params: generator of parameters (as returned by module.parameters())

    :returns: flattened parameters (i.e. one tensor of dimension 1 with all
        parameters concatenated)
    """
    return torch.cat([p.detach().view(-1) for p in params], dim=0).cpu().numpy()

def unflatten_parameters(params, example, device):
    """ Unflatten parameters.

    :args params: parameters as a single 1D np array
    :args example: generator of parameters (as returned by module.parameters()),
        used to reshape params
    :args device: where to store unflattened parameters

    :returns: unflattened parameters
    """
    params = torch.Tensor(params).to(device) # why werent these put on the device earlier? 
    idx = 0
    unflattened = []
    for e_p in example:
        # makes a list of parameters in the same format and shape as the network. 
        unflattened += [params[idx:idx + e_p.numel()].view(e_p.size())]
        idx += e_p.numel()
    return unflattened

def load_parameters(params, controller):
    """ Load flattened parameters into controller.

    :args params: parameters as a single 1D np array
    :args controller: module in which params is loaded
    """
    proto = next(controller.parameters())
    params = unflatten_parameters(
        params, controller.parameters(), proto.device) # dont see the need to pass the device here only to put them into it later. 

    for p, p_0 in zip(controller.parameters(), params):
        p.data.copy_(p_0)
    return controller

def sample_mdrnn_latent(mus, sigmas, logpi, latent_s, no_delta=False, return_chosen_mus_n_sigs=False):
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
        return latent_s 


def generate_rssm_samples(rssm, for_vae_n_mdrnn_sampling, deterministic,
                            samples_dir, SEQ_LEN, example_length, 
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

    last_test_observations = last_test_observations.view(last_test_observations.shape[0], 3, IMAGE_RESIZE_DIM, IMAGE_RESIZE_DIM).cpu()
    last_test_decoded_obs = last_test_decoded_obs.view(last_test_decoded_obs.shape[0], 3, IMAGE_RESIZE_DIM, IMAGE_RESIZE_DIM).cpu()

    if transform_obs:
        transform_for_mdrnn_samples = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((IMAGE_RESIZE_DIM, IMAGE_RESIZE_DIM)),
                transforms.ToTensor(),
                ])

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
            print('into decoder:', adapt_dict['hiddens'].shape, adapt_dict['posterior_states'].shape)
            adapt_obs = rssm.decode_sequence_obs(adapt_dict['hiddens'], adapt_dict['posterior_states'])
            adapt_obs = adapt_obs.view(adapt_obs.shape[0], 3, IMAGE_RESIZE_DIM, IMAGE_RESIZE_DIM)

            print('into horizon predictions', ast_test_actions[memory_adapt_period:].shape, hidden=adapt_dict['hiddens'][-1].shape , 
                state=adapt_dict['posterior_states'][-1].shape)

            horizon_multi_step_dict = rssm.perform_rollout(last_test_actions[memory_adapt_period:], hidden=adapt_dict['hiddens'][-1] , 
                state=adapt_dict['posterior_states'][-1] )
            
            horizon_multi_step_obs = rssm.decode_sequence_obs(horizon_multi_step_dict['hiddens'], horizon_multi_step_dict['prior_states'])
            horizon_multi_step_obs = horizon_multi_step_obs.view(horizon_multi_step_obs.shape[0],3, IMAGE_RESIZE_DIM, IMAGE_RESIZE_DIM)

            to_save = torch.cat([last_test_observations, last_test_decoded_obs, 
                horizon_one_step_obs.cpu(), adapt_obs.cpu(), horizon_multi_step_obs.cpu()], dim=0)

            print('Generating MDRNN samples of the shape:', to_save.shape)
            save_image(to_save,
                    join(samples_dir, 'horizon_preds_sample_' + str(e) + '.png'))


def generate_samples(vae, mdrnn, for_vae_n_mdrnn_sampling, deterministic,
                            samples_dir, SEQ_LEN, example_length, 
                            memory_adapt_period, e, device,
                            make_vae_samples=False,
                            make_mdrnn_samples=True, 
                            transform_obs = False):

    # need to restrict the data to a random segment. Important in cases where 
    # sequence length is too long
    start_sample_ind = np.random.randint(0, SEQ_LEN-example_length,1)[0]
    end_sample_ind = start_sample_ind+example_length

    # ensuring this is the same length as everything else. 
    for_vae_n_mdrnn_sampling[0] = for_vae_n_mdrnn_sampling[0][1:, :, :, :]

    last_test_observations, \
    last_test_pres_rewards, last_test_next_rewards, \
    last_test_latent_pres_obs, last_test_latent_next_obs, \
    last_test_pres_action = [var[start_sample_ind:end_sample_ind] for var in for_vae_n_mdrnn_sampling]

    if transform_obs:
        transform_for_mdrnn_samples = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((IMAGE_RESIZE_DIM, IMAGE_RESIZE_DIM)),
                transforms.ToTensor(),
                ])

    if make_vae_samples:
        with torch.no_grad():
            # get test samples
            decoder_mu, decoder_logsigma = vae.decoder(last_test_latent_next_obs, last_test_next_rewards)
            recon_batch = decoder_mu + (decoder_logsigma.exp() * torch.randn_like(decoder_mu))
            recon_batch = recon_batch.view(recon_batch.shape[0], 3, IMAGE_RESIZE_DIM, IMAGE_RESIZE_DIM)
            decoder_mu = decoder_mu.view(decoder_mu.shape[0], 3, IMAGE_RESIZE_DIM, IMAGE_RESIZE_DIM)
            to_save = torch.cat([last_test_observations.cpu(), recon_batch.cpu(), decoder_mu.cpu()], dim=0)
            print('Generating VAE samples of the shape:', to_save.shape)
            save_image(to_save,
                    join(samples_dir, 'vae_sample_' + str(e) + '.png'))

        print('====== Done Generating VAE Samples')

    if make_mdrnn_samples: 
        with torch.no_grad():

            # update MDRNN with the new samples: 
            # TODO: would this update have happened by default?? 
            
            #mdrnn_cell.load_state_dict( 
            #    {k.strip('_l0'): v for k, v in mdrnn.state_dict().items()})

            # vae of one in the future
            decoder_mu, decoder_logsigma = vae.decoder(last_test_latent_next_obs, last_test_next_rewards)
            real_next_vae_decoded_observation = decoder_mu.view(decoder_mu.shape[0], 3, IMAGE_RESIZE_DIM, IMAGE_RESIZE_DIM)
            
            # unsqueeze all to make a batch.
            # predict MDRNN one into the future
            #print('shape of actions going into mdrnn', last_test_pres_action.shape)
            mus, sigmas, logpi, rs, ds = mdrnn(last_test_pres_action.unsqueeze(0), last_test_latent_pres_obs.unsqueeze(0), last_test_pres_rewards.unsqueeze(0))
            # squeeze just the first dimension!
            mus, sigmas, logpi, rs, ds = [var.squeeze(0) for var in [mus, sigmas, logpi, rs, ds]]
            #print('MDRNN Debugger shouldnt have batch dimensions. :', mus.shape, sigmas.shape, logpi.shape)
            
            #print('Eval of MDRNN: Real rewards and latent squeezed or not? :', last_test_next_rewards.squeeze(), last_test_next_rewards.shape, last_test_latent_pres_obs.shape)
            #print('Predicted Rewards:', rs, rs.shape)

            if deterministic: 
                pred_latent_obs = last_test_latent_pres_obs + mus.squeeze(-2)
            else: 
                pred_latent_obs = sample_mdrnn_latent(mus, sigmas, logpi, last_test_latent_pres_obs)
                
            #print('shape of pred latent obs', pred_latent_obs.shape )

            # squeeze it back again:
            decoder_mu, decoder_logsigma = vae.decoder(pred_latent_obs, rs.unsqueeze(1))
            #print('shape of decoder mu', decoder_mu.shape)
            pred_next_vae_decoded_observation = decoder_mu.view(decoder_mu.shape[0], 3, IMAGE_RESIZE_DIM, IMAGE_RESIZE_DIM)
            
            # need to transform the last_test_observations here: 
            # THIS IS ONLY NEEDED HERE, NOT IN JOINT TRAINING AS IN JOINT ALL THE TRANSFORMS HAVE
            # ALREADY OCCURED. 
            if transform_obs:
                last_test_observations = last_test_observations.permute(0,2,3,1) * 255
                last_test_observations = last_test_observations.cpu().numpy().astype(np.uint8)
                trans_imgs = []
                for i in range(last_test_observations.shape[0]):
                    trans_imgs.append( transform_for_mdrnn_samples(last_test_observations[i,:,:,:]) )
                last_test_observations = torch.stack(trans_imgs)
            else: 
                last_test_observations = last_test_observations.cpu()

            #print('trying to save all out', last_test_observations.shape, 
            #    real_next_vae_decoded_observation.shape, 
            #    pred_next_vae_decoded_observation.shape )

            to_save = torch.cat([last_test_observations, 
                real_next_vae_decoded_observation.cpu(), 
                pred_next_vae_decoded_observation.cpu()], dim=0)

            print('Generating MDRNN samples of the shape:', to_save.shape)
            save_image(to_save,
                    join(samples_dir, 'next_pred_sample_' + str(e) + '.png'))

            ##################

            # Generating multistep predictions from the first latent. 
            horizon_pred_obs = []
            mse_losses = []
            # first latent state. straight from the VAE. 
            
            horizon_next_hidden = [
                torch.zeros(1, 1, LATENT_RECURRENT_SIZE).to(device)
                for _ in range(2)]

            for t in range(example_length):
                next_action = last_test_pres_action[t].unsqueeze(0).unsqueeze(0)
                horizon_next_latent_state = last_test_latent_pres_obs[t].unsqueeze(0).unsqueeze(0)
                horizon_next_reward = last_test_pres_rewards[t].unsqueeze(0).unsqueeze(0)

                #print('going into horizon pred', next_action.shape, horizon_next_latent_state.shape, horizon_next_hidden[0].shape, horizon_next_reward.shape)
                md_mus, md_sigmas, md_logpi, horizon_next_reward, d, horizon_next_hidden = mdrnn(next_action, 
                            horizon_next_latent_state, horizon_next_reward, horizon_next_hidden)
                horizon_next_reward = horizon_next_reward.unsqueeze(0)
                #print('going into sample mdrnn latent', md_mus.shape, horizon_next_latent_state.shape)

                if deterministic: 
                    horizon_next_latent_state = horizon_next_latent_state + md_mus.squeeze(-2)
                    horizon_next_latent_state = horizon_next_latent_state.squeeze(0)
                else: 
                    horizon_next_latent_state = sample_mdrnn_latent(md_mus, md_sigmas, md_logpi, horizon_next_latent_state)
                    horizon_next_latent_state = horizon_next_latent_state.unsqueeze(0)
                
                # mse between this and the real one. 
                mse_losses.append(  f.mse_loss(last_test_latent_next_obs[t], horizon_next_latent_state.squeeze()) )
                
                print('going into vae', horizon_next_latent_state.shape, horizon_next_reward.shape )
                decoder_mu, decoder_logsigma = vae.decoder(horizon_next_latent_state, horizon_next_reward.squeeze(0))
                #print('shape of decoder mu', decoder_mu.shape)
                pred_next_vae_decoded_observation = decoder_mu.view(decoder_mu.shape[0], 3, IMAGE_RESIZE_DIM, IMAGE_RESIZE_DIM)
                horizon_pred_obs.append(pred_next_vae_decoded_observation)

            print('===== MSE losses between horizon prediction and real', mse_losses)

            horizon_pred_obs_given_next = torch.stack(horizon_pred_obs).squeeze()

            ##################

            # Generating multistep predictions from the first latent. 
            horizon_pred_obs = []
            mse_losses = []
            # first latent state. straight from the VAE. 
            horizon_next_hidden = [
                torch.zeros(1, 1, LATENT_RECURRENT_SIZE).to(device)
                for _ in range(2)]

            for t in range(example_length):

                if t< memory_adapt_period:
                    # giving the real observation still 
                    horizon_next_latent_state = last_test_latent_pres_obs[t].unsqueeze(0).unsqueeze(0)
                    horizon_next_reward = last_test_pres_rewards[t].unsqueeze(0).unsqueeze(0)

                next_action = last_test_pres_action[t].unsqueeze(0).unsqueeze(0)
                #print('going into horizon pred', next_action.shape, horizon_next_latent_state.shape, horizon_next_hidden[0].shape, horizon_next_reward.shape)
                md_mus, md_sigmas, md_logpi, horizon_next_reward, d, horizon_next_hidden = mdrnn(next_action, 
                            horizon_next_latent_state, horizon_next_reward, horizon_next_hidden)
                horizon_next_reward = horizon_next_reward.unsqueeze(0)
                #print('going into sample mdrnn latent', md_mus.shape, horizon_next_latent_state.shape)

                if deterministic: 
                    horizon_next_latent_state = horizon_next_latent_state + md_mus.squeeze(-2)
                    horizon_next_latent_state = horizon_next_latent_state.squeeze(0)
                else: 
                    horizon_next_latent_state = sample_mdrnn_latent(md_mus, md_sigmas, md_logpi, horizon_next_latent_state)
                    horizon_next_latent_state = horizon_next_latent_state.unsqueeze(0)

                # mse between this and the real one. 
                mse_losses.append(  f.mse_loss(last_test_latent_next_obs[t], horizon_next_latent_state.squeeze()) )
                #print('going into vae', horizon_next_latent_state.shape, horizon_next_reward.shape )
                decoder_mu, decoder_logsigma = vae.decoder(horizon_next_latent_state, horizon_next_reward.squeeze(0))
                horizon_next_latent_state = horizon_next_latent_state.unsqueeze(0)
                #print('shape of decoder mu', decoder_mu.shape)
                pred_next_vae_decoded_observation = decoder_mu.view(decoder_mu.shape[0], 3, IMAGE_RESIZE_DIM, IMAGE_RESIZE_DIM)
                horizon_pred_obs.append(pred_next_vae_decoded_observation)

            print('===== MSE losses between horizon prediction and real', mse_losses)

            horizon_pred_obs_full_based = torch.stack(horizon_pred_obs).squeeze()

            #######################

            to_save = torch.cat([last_test_observations, real_next_vae_decoded_observation.cpu(), horizon_pred_obs_given_next.cpu(),
                horizon_pred_obs_full_based.cpu() ], dim=0)

            print('Generating MDRNN samples of the shape:', to_save.shape)
            save_image(to_save,
                    join(samples_dir, 'horizon_pred_sample_' + str(e) + '.png'))

if __name__ == "__main__":
    env = gym.make("CarRacing-v0")
    env.reset()
    seq_len=1000
    dt = 1. / 50
    actions = [env.action_space.sample()]
    print(actions)
    print(*actions)
    print(*actions[-1])
    for _ in range(seq_len):
        # getting rid of the list and then array structure. 
        # sampling 3 random actions from the last action in the list. 
        daction_dt = np.random.randn(*actions[-1].shape)
        actions.append(
            np.clip(actions[-1] + math.sqrt(dt) * daction_dt,
                    env.action_space.low, env.action_space.high))