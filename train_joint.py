""" Joint training of the VAE and MDRNN (forward model) using 
    CEM based probabilistic Planner """
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
from joint_utils import generate_rollouts_using_planner
from utils.misc import save_checkpoint, load_parameters, flatten_parameters
from utils.misc import RolloutGenerator, ACTION_SIZE, LATENT_SIZE, LATENT_RECURRENT_SIZE, IMAGE_RESIZE_DIM, SIZE
from utils.learning import EarlyStopping, ReduceLROnPlateau
import sys
from data.loaders import RolloutSequenceDataset
from models.vae import VAE
from models.mdrnn import MDRNN, gmm_loss
from multiprocessing import cpu_count
from trainvae import loss_function as trainvae_loss_function
from trainmdrnn import get_loss as trainmdrnn_loss_function
from collections import OrderedDict

def main(args):

    assert args.num_workers <= cpu_count(), "Providing too many workers!" 

    conditional =True
    make_vae_samples = True 
    # used for saving which models are the best based upon their train performance. 
    vae_n_mdrnn_cur_best=None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_feef = False

    include_reward = conditional # this is very important for the conditional 
    include_terminal = False

    # Constants
    BATCH_SIZE = 64
    SEQ_LEN = 64 # number of sequences in a row used during training
    epochs = 50
    time_limit =1000 # max time limit for the rollouts generated
    num_vae_mdrnn_training_rollouts_per_worker = 3

    # Planning values
    planner_n_particles = 500
    desired_horizon = 30
    num_action_repeats = 5 # number of times the same action is performed repeatedly. 
    # this makes the environment accelerate by this many frames. 
    actual_horizon = desired_horizon//num_action_repeats

    kl_tolerance=0.5
    kl_tolerance_scaled = torch.Tensor([kl_tolerance*LATENT_SIZE]).to(device)

    model_types = ['ctrl', 'vae', 'mdrnn']
    # Init save filenames 
    joint_dir = join(args.logdir, 'joint')
    filenames_dict = {m+'_'+bc:join(joint_dir, m+'_'+bc+'.tar') for bc in ['best', 'checkpoint'] \
                                                    for m in model_types}
    # make directories if they dont exist
    samples_dir = join(joint_dir, 'samples')
    for dirr in [joint_dir, samples_dir]:
        if not exists(dirr):
            mkdir(dirr)

    logger_filename = join(joint_dir, 'logger.txt')

    # init models
    vae = VAE(3, LATENT_SIZE, conditional).to(device)
    mdrnn = MDRNN(LATENT_SIZE, ACTION_SIZE, LATENT_RECURRENT_SIZE, 5,conditional).to(device)
    
    # TODO: consider learning these parameters with different optimizers and learning rates for each network. 
    optimizer = torch.optim.Adam(list(vae.parameters())+list(mdrnn.parameters()), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    earlystopping = EarlyStopping('min', patience=30) # NOTE: this needs to be esp high as the epochs are heterogenous buffers!! not all data. 

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
            if name =='mdrnn': 
                # this info is currently saved with the vae and mdrnn on their own pulling from mdrnn as its currently the last.
                print(' Loading in training state info (eg optimizer state) from last model in iter list:', name)
                vae_n_mdrnn_cur_best = state['precision']
                optimizer.load_state_dict(state["optimizer"])
                #scheduler.load_state_dict(state['scheduler'])
                #earlystopping.load_state_dict(state['earlystopping'])
                    
    else: 
        print("Starting new models from scratch and removing the old logger file!")
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
        unlink(logger_filename)

    # NOTE: just for now because the losses to be stored are very different. 
    vae_n_mdrnn_cur_best = None
    #unlink(logger_filename)
    optimizer = torch.optim.Adam(list(vae.parameters())+list(mdrnn.parameters()), lr=1e-4)

    # dont need as saving the observations after their transforms in the rollout itself. 
    #transform = transforms.Lambda( lambda x: np.transpose(x, (0, 3, 1, 2)) / 255)

    vae_output_names = ['encoder_mu', 'encoder_logsigma', 'latent_s', 'decoder_mu', 'decoder_logsigma']

    def run_vae(obs, rewards):
        # TODO: update this documentation. 
        """ Transform observations to latent space.

        :args obs: 5D torch tensor (BATCH_SIZE, SEQ_LEN, ACTION_SIZE, SIZE, SIZE)
        
        :returns: (latent_obs, latent_next_obs)
            - latent_obs: 4D torch tensor (BATCH_SIZE, SEQ_LEN, LATENT_SIZE)
        """

        # TODO: make this more pythonic and efficient. shouldnt have to loop over the VAE outputs. 
        vae_res_dict = {n:[] for n in vae_output_names}
        for x, r in zip(obs, rewards):

            # the rollout generator returns observations that have already been resized and VAE transformed
            #x = f.upsample(x.view(-1, 3, 84, SIZE), size=IMAGE_RESIZE_DIM, 
            #               mode='bilinear', align_corners=True)

            vae_outputs = vae(x, r)
            for ind, n in enumerate(vae_output_names):
                vae_res_dict[n].append(vae_outputs[ind])

        # stacking everything.
        for k in vae_res_dict.keys():
            vae_res_dict[k] = torch.stack(vae_res_dict[k])
        
        return vae_res_dict

    # Reconstruction + KL divergence losses summed over all elements and batch
    def vae_loss_function(real_obs, vae_res_dict):
        """ VAE loss function 
        Images (recon_x and x) are: (BATCH_SIZE, SEQ_LEN, NUM_CHANNELS, IMG_RESIZE, IMG_RESIZE)
        mu and logsigma: (BATCH_SIZE, SEQ_LEN, LATENT_SIZE)
        treating time independently here. 
        """

        # flatten the batch and seq length tensors.
        flat_tensors = [ vae_res_dict[k].flatten(end_dim=1) for k in ['encoder_mu', 'encoder_logsigma', 'decoder_mu', 'decoder_logsigma']]
        vae_loss, recon, kld = trainvae_loss_function(real_obs.flatten(end_dim=1), *flat_tensors, kl_tolerance_scaled)
        
        return dict(loss=vae_loss, recon=recon, kld=kld)

    def data_pass(epoch, train): # pylint: disable=too-many-locals
        """ One pass through the data """
        
        if train:
            loader = train_loader
            for model_var in [vae, mdrnn]:
                model_var.train()
            
        else:
            loader = test_loader
            for model_var in [vae, mdrnn]:
                model_var.eval()

        pbar = tqdm(total=len(loader.dataset), desc="Epoch {}".format(epoch))

        cumloss_dict = {n:0 for n in ['loss', 'loss_vae', 'loss_mdrnn','kld', 'recon', 'gmm', 'bce', 'mse']}

        def forward_and_loss():
            # transform obs
            vae_res_dict = run_vae(obs, reward)
            vae_loss_dict = vae_loss_function(obs, vae_res_dict)

            #split into previous and next observations:
            latent_next_obs = vae_res_dict['latent_s'][:,1:,:].clone() #possible BUG: need to ensure these tensors are different to each other. Tensors arent being modified though so should be ok? Test it anyways.
            latent_obs = vae_res_dict['latent_s'][:,:-1,:]

            next_reward = reward[:, 1:,:].clone()
            pres_reward = reward[:, :-1,:]

            mdrnn_loss_dict = trainmdrnn_loss_function(mdrnn, latent_obs, latent_next_obs, action, 
                                pres_reward, next_reward,
                                terminal, include_reward, include_terminal )

            return vae_loss_dict, mdrnn_loss_dict

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
                    vae_loss_dict, mdrnn_loss_dict = forward_and_loss()
                    # coefficient balancing!
                    mdrnn_loss_dict['loss'] = 10000*mdrnn_loss_dict['loss']
                    #total_loss = vae_loss_dict['loss'] + mdrnn_loss_dict['loss']

            # add to cumulative losses
            for k in cumloss_dict.keys():
                for loss_dict in [vae_loss_dict, mdrnn_loss_dict]:
                    if k in loss_dict.keys():
                        cumloss_dict[k] += loss_dict[k].item() if hasattr(loss_dict[k], 'item') else \
                                                loss_dict[k]
            # separate vae and mdrnn losses: 
            cumloss_dict['loss_vae'] += vae_loss_dict['loss'].item()
            cumloss_dict['loss_mdrnn'] += mdrnn_loss_dict['loss'].item()

            # TODO: make this much more modular. 
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
            return cumloss_dict, obs[0,:,:,:,:], reward[0,:,:]
            # return the last observation and reward to generate the VAE examples. 

    train = partial(data_pass, train=True)
    test = partial(data_pass, train=False)

    # TODO: store the CEM parameters across full command line based runs
    cem_params = ( torch.Tensor([0,0.8,0]) , torch.Tensor([0.3,0.2,0.2]) )

    ################## Main Training Loop ############################

    for e in range(epochs):
        print('====== New Epoch: ',e)
        ## run the current policy with the current VAE and MDRNN
        # TODO: implement memory buffer as data doesnt need to be on policy. 
        # TODO: sample a set of parameters from the controller rather than just using the same one.
        print('====== Generating Rollouts to train the MDRNN and VAE') 
        # TODO: dont feed in similar time sequences, have some gap between them or slicing of them.
        # TODO: get rollouts from the agent CMA-ES evaluation and use them here for training. 
        
        # Uses planning
        # TODO: Have CEM collect the best results across distributed CPUs. 
        # each worker will load in the checkpoint model not the best model! Want to use up to date. 
        cem_params, train_dataset, test_dataset, feef_losses, reward_losses = generate_rollouts_using_planner( 
                cem_params, actual_horizon, num_action_repeats, planner_n_particles, 
                SEQ_LEN, time_limit, joint_dir, num_rolls_per_worker=num_vae_mdrnn_training_rollouts_per_worker, 
                num_workers=args.num_workers, joint_file_dir=True, transform=None )

        print('cem parameters after generating rollouts!!', cem_params)

        # TODO: ensure these workers are freed up after the vae/mdrnn training is Done. 
        train_loader = DataLoader(train_dataset,
            batch_size=BATCH_SIZE, num_workers=0, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_dataset, shuffle=True,
            batch_size=BATCH_SIZE, num_workers=0, drop_last=True)
        print('====== Starting Training of VAE and MDRNN')
        # train VAE and MDRNN. uses partial(data_pass)
        train_loss_dict = train(e)
        print('====== Done Training VAE and MDRNN')
        # returns the last ones in order to produce samples!
        test_loss_dict, last_test_observations, last_test_rewards = test(e)
        print('====== Done Testing VAE and MDRNN')
        scheduler.step(test_loss_dict['loss'])

        # append the planning results 
        test_loss_dict['avg_reward_planner'] = np.mean(reward_losses)
        test_loss_dict['std_reward_planner'] = np.std(reward_losses)
        test_loss_dict['max_reward_planner'] = np.max(reward_losses)
        test_loss_dict['min_reward_planner'] = np.min(reward_losses)
        test_loss_dict['avg_feef_planner'] = np.mean(feef_losses)
        test_loss_dict['std_feef_planner'] = np.std(feef_losses)
        test_loss_dict['max_feef_planner'] = np.max(feef_losses)
        test_loss_dict['min_feef_planner'] = np.min(feef_losses)

        print('========== test loss dictionary:', test_loss_dict)

        # checkpointing the model. Need to checkpoint these separately!: 
        # needs to be here so that the policy learning workers below can load in the new parameters.
        is_best = not vae_n_mdrnn_cur_best or test_loss_dict['loss'] < vae_n_mdrnn_cur_best
        if is_best:
            vae_n_mdrnn_cur_best = test_loss_dict['loss']
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

        # generating and saving VAE samples
        if make_vae_samples:
            with torch.no_grad():
                # get test samples
                encoder_mu, encoder_logsigma, latent_s, decoder_mu, decoder_logsigma = vae(last_test_observations, last_test_rewards)
                recon_batch = decoder_mu + (decoder_logsigma.exp() * torch.randn_like(decoder_mu))
                recon_batch = recon_batch.view(recon_batch.shape[0], 3, IMAGE_RESIZE_DIM, IMAGE_RESIZE_DIM)
                decoder_mu = decoder_mu.view(decoder_mu.shape[0], 3, IMAGE_RESIZE_DIM, IMAGE_RESIZE_DIM)
                to_save = torch.cat([last_test_observations.cpu(), recon_batch.cpu(), decoder_mu.cpu()], dim=0)
                print('Generating VAE samples of the shape:', to_save.shape)
                save_image(to_save,
                        join(samples_dir, 'sample_' + str(e) + '.png'))
        print('====== Done Generating VAE Samples')

        # TODO: generate MDRNN examples. 

        # header at the top of logger file written once. at the start of each epoch
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
                        help="Do not reload if specified.")
    
    parser.add_argument('--num_workers', type=int, help='Maximum number of workers.',
                        default=16)
    parser.add_argument('--display', action='store_true', help="Use progress bars if "
                        "specified.")
    args = parser.parse_args()
    main(args)