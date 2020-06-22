""" 
Joint training of the VAE and MDRNN (forward model) using 
CEM based probabilistic Planner. 
"""
import argparse
from functools import partial
from os.path import join, exists
from os import mkdir, unlink
import torch
import torch.nn.functional as f
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
import json
from tqdm import tqdm
from joint_utils import generate_rollouts_using_planner, GeneratedDataset
from utils.misc import save_checkpoint, load_parameters, flatten_parameters
from utils.misc import ACTION_SIZE, LATENT_SIZE, LATENT_RECURRENT_SIZE, IMAGE_RESIZE_DIM, SIZE, NUM_GAUSSIANS_IN_MDRNN, NUM_IMG_CHANNELS
from utils.learning import EarlyStopping, ReduceLROnPlateau
import sys
from data.loaders import RolloutSequenceDataset
from models.vae import VAE
from models.mdrnn import MDRNN, gmm_loss
from multiprocessing import cpu_count
from trainvae import loss_function as trainvae_loss_function
from trainmdrnn import get_loss as trainmdrnn_loss_function
from collections import OrderedDict
import time 
from utils.misc import sample_mdrnn_latent

def main(args):

    assert args.num_workers <= cpu_count(), "Providing too many workers! Need one less than total amount." 

    condition_on_rewards =True
    make_vae_samples, make_mdrnn_samples = True, True
    # used for saving which models are the best based upon their train performance. 
    vae_n_mdrnn_cur_best=None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_feef = True # NOTE: currently not actually computing it inside of simulate!!!!

    include_reward = condition_on_rewards # has the MDRNN return rewards. this is very important for the conditioning on rewards 
    include_terminal = False
    
    # Constants
    BATCH_SIZE = 32
    SEQ_LEN = 100 # number of sequences in a row used during training
    epochs = 500
    time_limit =1000 # max time limit for the rollouts generated
    num_vae_mdrnn_training_rollouts_per_worker = 3

    # Planning values
    planner_n_particles = 700
    desired_horizon = 30
    num_action_repeats = 5 # number of times the same action is performed repeatedly. 
    # this makes the environment accelerate by this many frames. 
    actual_horizon = (desired_horizon//num_action_repeats)+1
    discount_factor = 0.90
    init_cem_params = ( torch.Tensor([0,0.7,0]), torch.Tensor([0.5,0.7,0.3]) )
    cem_iters = 7

    # for plotting example horizons:
    example_length = 16
    assert example_length<= SEQ_LEN, "Example length must be smaller."
    memory_adapt_period = example_length - actual_horizon

    # memory buffer:
    # making a memory buffer for previous rollouts too. 
    # buffer contains a dictionary full of tensors. 
    use_training_buffer=True
    num_new_rollouts = args.num_workers*num_vae_mdrnn_training_rollouts_per_worker
    num_prev_epochs_to_store = 4
    # NOTE: this is a lower bound. could go over this depending on how stochastic the buffer adding is!
    max_buffer_size = num_new_rollouts*num_prev_epochs_to_store

    kl_tolerance=0.5
    kl_tolerance_scaled = torch.Tensor([kl_tolerance*LATENT_SIZE]).to(device)

    # Data Loading. Cant use previous transform directly as it is a seq len sized batch of observations!!
    # DONT NEED AS I AM ALREADY SAVING THE DATA TRANSFORMS GENERATED BY THE ROLLOUTS. 
    #initial_data_transform = transforms.Lambda(
    #    lambda x: np.transpose(x, (0, 3, 1, 2)) / 255) #why is this necessary?

    # Init save filenames 
    joint_dir = join(args.logdir, 'joint')
    filenames_dict = {m+'_'+bc:join(joint_dir, m+'_'+bc+'.tar') for bc in ['best', 'checkpoint'] \
                                                    for m in ['vae', 'mdrnn']}
    # make directories if they dont exist
    samples_dir = join(joint_dir, 'samples')
    for dirr in [joint_dir, samples_dir]:
        if not exists(dirr):
            mkdir(dirr)

    logger_filename = join(joint_dir, 'logger.txt')

    # init models
    vae = VAE(NUM_IMG_CHANNELS, LATENT_SIZE, condition_on_rewards).to(device)
    mdrnn = MDRNN(LATENT_SIZE, ACTION_SIZE, LATENT_RECURRENT_SIZE, NUM_GAUSSIANS_IN_MDRNN, condition_on_rewards).to(device)
    
    # TODO: consider learning these parameters with different optimizers and learning rates for each network. 
    optimizer = torch.optim.Adam(list(vae.parameters())+list(mdrnn.parameters()), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    earlystopping = EarlyStopping('min', patience=100) # NOTE: this needs to be esp high as the epochs are heterogenous buffers!! not all data. 

    # Loading in trained models: 
    if not args.no_reload:
        for model_var, name in zip([vae, mdrnn],['vae', 'mdrnn']):
            load_file = filenames_dict[name+'_best']
            assert exists(load_file), "Could not find file: " + load_file + " to load in!"
            state = torch.load(load_file, map_location={'cuda:0': str(device)})
            print("Loading model_type {} at epoch {} "
                "with test error {}".format(name,
                    state['epoch'], state['precision']))

            model_var.load_state_dict(state['state_dict'])

            # load in the training loop states only if all jointly trained together before
            # but if it is from given pretrained models then this should not be done. 
            if not args.giving_pretrained and name =='mdrnn': 
                # this info is currently saved with the vae and mdrnn on their own pulling from mdrnn as its currently the last.
                print(' Loading in training state info (eg optimizer state) from last model in iter list:', name)
                vae_n_mdrnn_cur_best = state['precision']
                optimizer.load_state_dict(state["optimizer"])
                #scheduler.load_state_dict(state['scheduler'])
                #earlystopping.load_state_dict(state['earlystopping'])
                    
    # save init models
    if args.no_reload or args.giving_pretrained: 
        print("Overwriting checkpoint because pretrained models or no reload was called and removing the old logger file!")
        print("NB! This will overwrite the best and checkpoint models!\nSleeping for 5 seconds to allow you to change your mind...")
        time.sleep(5.0)
        for model_var, model_name in zip([vae, mdrnn],['vae', 'mdrnn']):
            save_checkpoint({
                "state_dict": model_var.state_dict(),
                "optimizer": optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'earlystopping': earlystopping.state_dict(),
                "precision": None,
                "epoch": -1}, True, filenames_dict[model_name+'_checkpoint'],
                            filenames_dict[model_name+'_best'])
                        # saves file to is best AND checkpoint

        # unlinking the old logger too
        if exists(logger_filename):
            unlink(logger_filename)

    # making learning rate lower because models are already pretrained!
    if args.giving_pretrained: 
        optimizer = torch.optim.Adam(list(vae.parameters())+list(mdrnn.parameters()), lr=1e-4)
        vae_n_mdrnn_cur_best = None

    vae_output_names = ['encoder_mu', 'encoder_logsigma', 'latent_s', 'decoder_mu', 'decoder_logsigma']

    def run_vae(obs, rewards):
        """ Use VAE to map observations to latent space.
        Assumes that the observations have already been transformed. 
        This is done in generate_rollouts() where the transformations are saved before
        being passed back already. (See trainmdrnn.py for an example of 
        the tranforms that need to be done to a *batch* of new observations 
        from the environment)

        :args: 
            - obs: 5D torch tensor (BATCH_SIZE, SEQ_LEN, CHANNEL_DIM, SIZE, SIZE)
            - rewards: 3D torch tensor (BATCH_SIZE, SEQ_LEN, 1)
        
        :returns: vae_res_dict: 
                keys are: 
                - encoder_mu: 3D torch tensor (BATCH_SIZE, SEQ_LEN, LATENT_SIZE)
                - encoder_logsigma: 3D torch tensor (BATCH_SIZE, SEQ_LEN, LATENT_SIZE)
                - latent_s: 3D torch tensor (BATCH_SIZE, SEQ_LEN, LATENT_SIZE)
                - decoder_mu: 3D torch tensor (BATCH_SIZE, SEQ_LEN, CHANNEL_DIM*SIZE*SIZE)
                - decoder_logsigma: 3D torch tensor (BATCH_SIZE, SEQ_LEN, CHANNEL_DIM*SIZE*SIZE)
        """

        # need to loop over the batch because within it is a "batch" in the form of sequence of observations
        vae_res_dict = {n:[] for n in vae_output_names}
        for x, r in zip(obs, rewards):

            vae_outputs = vae(x, r)
            # TODO: make this more pythonic and efficient. shouldnt have to loop over the VAE outputs. 
            for ind, n in enumerate(vae_output_names):
                vae_res_dict[n].append(vae_outputs[ind])

        # stacking everything.
        for k in vae_res_dict.keys():
            vae_res_dict[k] = torch.stack(vae_res_dict[k])
        
        return vae_res_dict

    # Reconstruction + KL divergence losses summed over all elements and batch
    def vae_loss_function(real_obs, vae_res_dict):
        """ VAE loss function. Calls the function from trainvae.py.
        Collapses the batch and time dimensions because the VAE is currently
        time independent. Would like to test a sequential VAE at some point!

        :args: 
            - real_obs: 5D torch tensor (BATCH_SIZE, SEQ_LEN, CHANNEL_DIM, SIZE, SIZE)
            - vae_res_dict:
                keys are: 
                - encoder_mu: 4D torch tensor (BATCH_SIZE, SEQ_LEN, LATENT_SIZE)
                - encoder_logsigma: 4D torch tensor (BATCH_SIZE, SEQ_LEN, LATENT_SIZE)
                - latent_s: 4D torch tensor (BATCH_SIZE, SEQ_LEN, LATENT_SIZE)
                - decoder_mu: 4D torch tensor (BATCH_SIZE, SEQ_LEN, CHANNEL_DIM*SIZE*SIZE)
                - decoder_logsigma: 4D torch tensor (BATCH_SIZE, SEQ_LEN, CHANNEL_DIM*SIZE*SIZE)

        :returns: Dictionary of total vaeloss, recon loss and kld loss
            - loss: float. combination of reconstruction and KLD loss. Made negative 
                    to account for the minimization 
            - recon: float. reconstruction loss. Computed here using a real probability distribution
                    rather than just the MSE loss with the decoder mu as is typically done. 
            - KLD: float. uses free bits to encourage a more expressive latent space. 
        """

        # flatten the batch and seq length tensors together.
        flat_tensors = [ vae_res_dict[k].flatten(end_dim=1) for k in ['encoder_mu', 'encoder_logsigma', 'decoder_mu', 'decoder_logsigma']]
        vae_loss, recon, kld = trainvae_loss_function(real_obs.flatten(end_dim=1), *flat_tensors, kl_tolerance_scaled)
        
        return dict(loss=vae_loss, recon=recon, kld=kld)

    def data_pass(epoch, train): # pylint: disable=too-many-locals
        """One pass through full epoch pass through the data either testing or training (with torch.no_grad()).
        NB. One epoch here is all of the data collected from the workers using
        the planning algorithm. 
        This is num_workers * num_vae_mdrnn_training_rollouts_per_worker.

        :args:
            - epoch: int
            - train: bool

        :returns:
            - cumloss_dict - All of the losses collected from this epoch. 
                            Averaged across the batches and sequence lengths 
                            to be on a per element basis. 
            if test also returns information used to generate the VAE and MDRNN samples
            these are useful for evaluating performance: 
                - first in the batch of:
                    - obs[0,:,:,:,:]
                    - pres_reward[0,:,:] 
                    - next_reward[0,:,:]
                    - latent_obs[0,:,:]
                    - latent_next_obs[0,:,:]
                    - pres_action[0,:,:]
        """
        
        if train:
            loader = train_loader
            for model_var in [vae, mdrnn]:
                model_var.train()
            
        else:
            loader = test_loader
            for model_var in [vae, mdrnn]:
                model_var.eval()

        pbar = tqdm(total=len(loader.dataset), desc="Epoch {}".format(epoch))

        # store all of the losses for this data pass. 
        cumloss_dict = {n:0 for n in ['loss', 'loss_vae', 'loss_mdrnn','kld', 'recon', 'gmm', 'bce', 'mse']}

        def forward_and_loss(return_for_vae_n_mdrnn_sampling=False):
            """ 
            Run the VAE and MDRNNs and return their losses. 
            """
            # transform obs
            vae_res_dict = run_vae(obs, reward)
            vae_loss_dict = vae_loss_function(obs, vae_res_dict)

            #split into previous and next observations:
            latent_next_obs = vae_res_dict['latent_s'][:,1:,:].clone() #possible BUG: need to ensure these tensors are different to each other. Tensors arent being modified though so should be ok? Test it anyways.
            latent_obs = vae_res_dict['latent_s'][:,:-1,:]

            next_reward = reward[:, 1:,:].clone()
            pres_reward = reward[:, :-1,:]

            pres_action = action[:, :-1]

            mdrnn_loss_dict = trainmdrnn_loss_function(mdrnn, latent_obs, latent_next_obs, pres_action, 
                                pres_reward, next_reward,
                                terminal, include_reward, include_terminal )

            if return_for_vae_n_mdrnn_sampling: 
                for_vae_n_mdrnn_sampling = [obs[0,:,:,:,:], pres_reward[0,:,:], next_reward[0,:,:], latent_obs[0,:,:], latent_next_obs[0,:,:], pres_action[0,:,:]]
                return vae_loss_dict, mdrnn_loss_dict, for_vae_n_mdrnn_sampling
            else: 
                return vae_loss_dict, mdrnn_loss_dict

        # iterate through an epoch of data. 
        for i, data in enumerate(loader):
            obs, action, reward, terminal = [arr.to(device) for arr in data]

            if train:
                vae_loss_dict, mdrnn_loss_dict = forward_and_loss()
                # coefficient balancing!
                mdrnn_loss_dict['loss'] = 10000*mdrnn_loss_dict['loss']
                total_loss = vae_loss_dict['loss'] + mdrnn_loss_dict['loss']

                # taking grad step after every batch. 
                optimizer.zero_grad()
                total_loss.backward()
                # TODO: consider adding gradient clipping like Ha.  
                torch.nn.utils.clip_grad_norm_(list(vae.parameters())+list(mdrnn.parameters()), 100.0)
                optimizer.step()
            else:
                with torch.no_grad():
                    vae_loss_dict, mdrnn_loss_dict, for_vae_n_mdrnn_sampling = forward_and_loss(return_for_vae_n_mdrnn_sampling = True)
                    # coefficient balancing!
                    mdrnn_loss_dict['loss'] = 10000*mdrnn_loss_dict['loss']
                    #total_loss = vae_loss_dict['loss'] + mdrnn_loss_dict['loss']

            # add results from each batch to cumulative losses
            for k in cumloss_dict.keys():
                for loss_dict in [vae_loss_dict, mdrnn_loss_dict]:
                    if k in loss_dict.keys():
                        cumloss_dict[k] += loss_dict[k].item() if hasattr(loss_dict[k], 'item') else \
                                                loss_dict[k]
            # store separately vae and mdrnn losses: 
            cumloss_dict['loss_vae'] += vae_loss_dict['loss'].item()
            cumloss_dict['loss_mdrnn'] += mdrnn_loss_dict['loss'].item()

            # Display training progress bar with current losses
            postfix_str = ""
            for k,v in cumloss_dict.items():
                v = v / (i + 1)
                postfix_str+= k+'='+str(round(v,4))+', '
            pbar.set_postfix_str(postfix_str)
            pbar.update(BATCH_SIZE)
        pbar.close()

        # puts losses on a per element level.
        cumloss_dict = {k: (v*BATCH_SIZE) / len(loader.dataset) for k, v in cumloss_dict.items()}
        # sort the order so they are added to the logger in the same order!
        cumloss_dict = OrderedDict(sorted(cumloss_dict.items()))
        if train: 
            return cumloss_dict 
        else: 
            return cumloss_dict, for_vae_n_mdrnn_sampling
            # return the last observation and reward to generate the VAE examples. 

    train = partial(data_pass, train=True)
    test = partial(data_pass, train=False)

    ################## Main Training Loop ############################

    for e in range(epochs):
        print('====== New Epoch: ',e)
        ## run the current policy with the current VAE and MDRNN

        # NOTE: each worker loads in the checkpoint model not the best model! Want to use up to date. 
        print('====== Generating Rollouts to train the MDRNN and VAE') 
        train_data, test_data, feef_losses, reward_losses = generate_rollouts_using_planner( 
                actual_horizon, num_action_repeats, planner_n_particles, 
                SEQ_LEN, time_limit, joint_dir, init_cem_params, cem_iters, discount_factor,
                 num_rolls_per_worker=num_vae_mdrnn_training_rollouts_per_worker, 
                num_workers=args.num_workers, compute_feef=use_feef )

        if use_training_buffer:
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
                        buffer_train_data[k] += train_data[k]
                        #buffer_train_data[k] = torch.cat([v, train_data[k]], dim=0)
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
        else: 
            buffer_train_data = train_data

        # NOTE: currently not applying any transformations as saving those that happen when the 
        # rollouts are actually generated. 
        train_dataset = GeneratedDataset(None, buffer_train_data, SEQ_LEN)
        test_dataset = GeneratedDataset(None, test_data, SEQ_LEN)
        # TODO: set number of workers higher. Here it doesn;t matter much as already have tensors ready. 
        # (dont need any loading or tranformations) 
        # and before these workers were clashing with the ray workers for generating rollouts. 
        train_loader = DataLoader(train_dataset,
            batch_size=BATCH_SIZE, num_workers=0, shuffle=True, drop_last=False)
        test_loader = DataLoader(test_dataset, shuffle=True,
            batch_size=BATCH_SIZE, num_workers=0, drop_last=False)
        print('====== Starting Training of VAE and MDRNN')
        # train VAE and MDRNN. uses partial(data_pass)
        train_loss_dict = train(e)
        print('====== Done Training VAE and MDRNN')
        # returns the last ones in order to produce samples!
        test_loss_dict, for_vae_n_mdrnn_sampling = test(e)
        print('====== Done Testing VAE and MDRNN')
        scheduler.step(test_loss_dict['loss'])

        # append the planning results to the TEST loss dictionary. 
        for name, var in zip(['reward_planner', 'feef_planner'], [reward_losses, feef_losses]):
            test_loss_dict['avg_'+name] = np.mean(var)
            test_loss_dict['std_'+name] = np.std(var)
            test_loss_dict['max_'+name] = np.max(var)
            test_loss_dict['min_'+name] = np.min(var)

        print('========== Test Loss dictionary:', test_loss_dict)

        # checkpointing the model. Necessary to ensure the workers load in the most up to date checkpoint.
        # save_checkpoint function always saves a checkpoint and may also update the best. 
        is_best = not vae_n_mdrnn_cur_best or test_loss_dict['loss'] < vae_n_mdrnn_cur_best
        if is_best:
            vae_n_mdrnn_cur_best = test_loss_dict['loss']
            print('========== New Best for the Test Loss! Updating *MODEL_best.tar*')
        for model_var, model_name in zip([vae, mdrnn],['vae', 'mdrnn']):
            save_checkpoint({
                "state_dict": model_var.state_dict(),
                "optimizer": optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'earlystopping': earlystopping.state_dict(),
                "precision": test_loss_dict['loss_'+model_name],
                "epoch": e}, is_best, filenames_dict[model_name+'_checkpoint'],
                            filenames_dict[model_name+'_best'])
        print('====== Done Saving VAE and MDRNN')

        if make_vae_samples or make_mdrnn_samples:

            # need to restrict the data to a random segment. Important in cases where 
            # sequence length is too long
            start_sample_ind = np.random.randint(0, SEQ_LEN-example_length,1)[0]
            end_sample_ind = start_sample_ind+example_length

            # ensuring that the last_test_observations are the same length as everything else. 
            # and one into the future. 
            for_vae_n_mdrnn_sampling[0] = for_vae_n_mdrnn_sampling[0][1:, :, :, :]
            
            last_test_observations, \
            last_test_pres_rewards, last_test_next_rewards, \
            last_test_latent_pres_obs, last_test_latent_next_obs, \
            last_test_pres_action = [var[start_sample_ind:end_sample_ind] for var in for_vae_n_mdrnn_sampling]

        # generating and saving VAE samples
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

        # Generate MDRNN examples. 

        if make_mdrnn_samples:
            with torch.no_grad():
                # some of this compute is repeated from the VAE sample but it all runs quickly and doesn't happen v often. 
                # vae of one in the future
                decoder_mu, decoder_logsigma = vae.decoder(last_test_latent_next_obs, last_test_next_rewards)
                real_next_vae_decoded_observation = decoder_mu.view(decoder_mu.shape[0], 3, IMAGE_RESIZE_DIM, IMAGE_RESIZE_DIM)
                # unsqueeze all to make a batch.
                # predict MDRNN one into the future
                print('shape of actions going into mdrnn', last_test_pres_action.shape)
                mus, sigmas, logpi, rs, ds = mdrnn(last_test_pres_action.unsqueeze(0), last_test_latent_pres_obs.unsqueeze(0), last_test_pres_rewards.unsqueeze(0))
                # squeeze just the first dimension!
                mus, sigmas, logpi, rs, ds = [var.squeeze(0) for var in [mus, sigmas, logpi, rs, ds]]
                print('MDRNN Debugger shouldnt have batch dimensions. :', mus.shape, sigmas.shape, logpi.shape)
                print('Eval of MDRNN: Real rewards and latent squeezed or not? :', last_test_next_rewards.squeeze(), last_test_next_rewards.shape, last_test_latent_pres_obs.shape)
                print('Predicted Rewards:', rs, rs.shape)
                pred_latent_obs = sample_mdrnn_latent(mus, sigmas, logpi, last_test_latent_pres_obs)
                print('shape of pred latent obs', pred_latent_obs.shape )
                # squeeze it back again:
                decoder_mu, decoder_logsigma = vae.decoder(pred_latent_obs, rs.unsqueeze(1))
                print('shape of decoder mu', decoder_mu.shape)
                pred_next_vae_decoded_observation = decoder_mu.view(decoder_mu.shape[0], 3, IMAGE_RESIZE_DIM, IMAGE_RESIZE_DIM)
                
                print('trying to save all out', last_test_observations.shape, 
                    real_next_vae_decoded_observation.shape, 
                    pred_next_vae_decoded_observation.shape )
                to_save = torch.cat([last_test_observations.cpu(), 
                    real_next_vae_decoded_observation.cpu(), 
                    pred_next_vae_decoded_observation.cpu()], dim=0)
                print('Generating MDRNN samples of the shape:', to_save.shape)
                save_image(to_save,
                        join(samples_dir, 'mdrnn_next_pred_sample_' + str(e) + '.png'))
                
                ################## Predicting next observation using the full RNN
                # Generating multistep predictions from the first latent. 
                horizon_pred_obs = []
                mse_losses = [] 
                horizon_next_hidden = [
                    torch.zeros(1, 1, LATENT_RECURRENT_SIZE).to(device)
                    for _ in range(2)]
                # TODO: convert to all in one go for speed up at some point
                # this could be done all in one go but I used code from single iterations during debugging. 
                for t in range(example_length):
                    next_action = last_test_pres_action[t].unsqueeze(0).unsqueeze(0)
                    horizon_next_latent_state = last_test_latent_pres_obs[t].unsqueeze(0).unsqueeze(0)
                    horizon_next_reward = last_test_pres_rewards[t].unsqueeze(0).unsqueeze(0)

                    print('going into horizon pred', next_action.shape, horizon_next_latent_state.shape, horizon_next_hidden[0].shape, horizon_next_reward.shape)
                    md_mus, md_sigmas, md_logpi, horizon_next_reward, d, horizon_next_hidden = mdrnn(next_action, 
                                horizon_next_latent_state, horizon_next_reward, horizon_next_hidden)
                    horizon_next_reward = horizon_next_reward.unsqueeze(0)
                    print('going into sample mdrnn latent', md_mus.shape, horizon_next_latent_state.shape)
                    horizon_next_latent_state = sample_mdrnn_latent(md_mus, md_sigmas, md_logpi, horizon_next_latent_state)
                    
                    # mse between this and the real one. 
                    mse_losses.append(  f.mse_loss(last_test_latent_next_obs[t], horizon_next_latent_state) )
                    
                    print('going into vae', horizon_next_latent_state.shape, horizon_next_reward.shape )
                    decoder_mu, decoder_logsigma = vae.decoder(horizon_next_latent_state.squeeze(0), horizon_next_reward.squeeze(0))
                    print('shape of decoder mu', decoder_mu.shape)
                    pred_next_vae_decoded_observation = decoder_mu.view(decoder_mu.shape[0], 3, IMAGE_RESIZE_DIM, IMAGE_RESIZE_DIM)
                    horizon_pred_obs.append(pred_next_vae_decoded_observation)
                print('===== MSE losses between horizon prediction and real', mse_losses)
                horizon_pred_obs_given_next = torch.stack(horizon_pred_obs).squeeze()
                
                ##################
                # Generating multistep predictions from the first latent. 
                horizon_pred_obs = []
                mse_losses = []
                # first latent state. straight from the VAE. 
                #horizon_next_latent_state = last_test_latent_pres_obs[0].unsqueeze(0).unsqueeze(0)
                #horizon_next_reward = last_test_pres_rewards[0].unsqueeze(0).unsqueeze(0)
                horizon_next_hidden = [
                    torch.zeros(1, 1, LATENT_RECURRENT_SIZE).to(device)
                    for _ in range(2)]
    
                for t in range(example_length):

                    if t< memory_adapt_period:
                        # giving the real observation still 
                        horizon_next_latent_state = last_test_latent_pres_obs[t].unsqueeze(0).unsqueeze(0)
                        horizon_next_reward = last_test_pres_rewards[t].unsqueeze(0).unsqueeze(0)

                    next_action = last_test_pres_action[t].unsqueeze(0).unsqueeze(0)
                    print('going into horizon pred', next_action.shape, horizon_next_latent_state.shape, horizon_next_hidden[0].shape, horizon_next_reward.shape)
                    md_mus, md_sigmas, md_logpi, horizon_next_reward, d, horizon_next_hidden = mdrnn(next_action, 
                                horizon_next_latent_state, horizon_next_reward, horizon_next_hidden)
                    horizon_next_reward = horizon_next_reward.unsqueeze(0)
                    print('going into sample mdrnn latent', md_mus.shape, horizon_next_latent_state.shape)
                    horizon_next_latent_state = sample_mdrnn_latent(md_mus, md_sigmas, md_logpi, horizon_next_latent_state)
                    # mse between this and the real one. 
                    mse_losses.append(  f.mse_loss(last_test_latent_next_obs[t], horizon_next_latent_state) )
                    print('going into vae', horizon_next_latent_state.shape, horizon_next_reward.shape )
                    decoder_mu, decoder_logsigma = vae.decoder(horizon_next_latent_state.squeeze(0), horizon_next_reward.squeeze(0))
                    print('shape of decoder mu', decoder_mu.shape)
                    pred_next_vae_decoded_observation = decoder_mu.view(decoder_mu.shape[0], 3, IMAGE_RESIZE_DIM, IMAGE_RESIZE_DIM)
                    horizon_pred_obs.append(pred_next_vae_decoded_observation)

                    
                print('===== MSE losses between horizon prediction and real', mse_losses)
                horizon_pred_obs_full_based = torch.stack(horizon_pred_obs).squeeze()

                to_save = torch.cat([last_test_observations.cpu(), real_next_vae_decoded_observation.cpu(), 
                    horizon_pred_obs_given_next.cpu(),
                    horizon_pred_obs_full_based.cpu()], dim=0)

                print('Generating MDRNN samples of the shape:', to_save.shape)
                save_image(to_save,
                        join(samples_dir, 'mdrnn_horizon_pred_sample_' + str(e) + '.png'))

        # Header at the top of logger file written once at the start of new training run.
        if not exists(logger_filename) or e==0: 
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
        print('====== Done Writing out to the Logger')
        # accounts for all of the different module losses. 
        earlystopping.step(test_loss_dict['loss'])

        if earlystopping.stop:
            print("End of Training because of early stopping at epoch {}".format(e))
            break

if __name__ =='__main__':
    parser = argparse.ArgumentParser("Joint training")
    parser.add_argument('--logdir', type=str, default='exp_dir',
                        help="Where things are logged and models are loaded from.")
    parser.add_argument('--gamename', type=str, default='carracing',
                        help="What Gym environment to train in.")
    parser.add_argument('--no_reload', action='store_true',
                        help="Won't load in models for VAE and MDRNN from the joint file. \
                        NB. This will create new models with random inits and will overwrite \
                        the best and checkpoints!")
    parser.add_argument('--giving_pretrained', action='store_true',
                        help="If pretrained models are being provided, avoids loading in an optimizer \
                        or previous lowest loss score.")
    parser.add_argument('--num_workers', type=int, help='Maximum number of workers.',
                        default=16)
    parser.add_argument('--display', action='store_true', help="Use progress bars if "
                        "specified.")
    args = parser.parse_args()
    main(args)