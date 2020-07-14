# pylint: disable=no-member
""" 
Joint training of the VAE and RNN (forward model) using 
CEM based probabilistic Planner. 
"""
import argparse
from functools import partial
from os.path import join, exists
from os import mkdir, unlink
import torch
import torch.nn.functional as f
from torch.distributions.kl import kl_divergence
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import json
from tqdm import tqdm
from utils import save_checkpoint, generate_rssm_samples, generate_rollouts_using_planner, GeneratedDataset, TrainBufferDataset, write_logger, combine_single_worker
from envs import get_env_params
import sys
from models.rssm import RSSModel 
from torch.distributions.normal import Normal
from multiprocessing import cpu_count
from collections import OrderedDict
from control import Agent 
import time 
from utils import set_seq_and_batch_vals

def main(args):

    assert args.num_workers <= cpu_count(), "Providing too many workers! Need one less than total amount." 

    make_vae_samples, make_mdrnn_samples = False, False 

    # used for saving which models are the best based upon their train performance. 
    model_cur_best=None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    compute_feef = True # NOTE: currently not actually computing it inside of agent.py simulate!!!!

    # TODO: enable predicting the terminal state. 
    #include_terminal = False # option to include predicting terminal state. 
    #include_overshoot = False #NOTE: no longer support overshoot training. have old code for this though.  

    # what to condition rewards upon. (encoder, decoder, rnn)
    decoder_reward_condition = False
    decoder_make_sigmas = False
    antithetic = True 

    # get environment parameters: 
    env_params = get_env_params(args.gamename)
    
    # Constants
    batch_size_to_seq_len_multiple = 4096
    SEQ_LEN = 12 # number of sequences in a row used during training
    # needs one more than this for the forward predictions. 
    BATCH_SIZE = batch_size_to_seq_len_multiple//SEQ_LEN
    epochs = 500
    random_action_epochs = 100

    training_rollouts_per_worker = 1

    # Planning values
    planner_n_particles = 700
    discount_factor = 0.95
    cem_iters = 7

    # for plotting example horizons:
    example_length = 12
    assert example_length<= SEQ_LEN, "Example length must be smaller."
    memory_adapt_period = example_length - env_params['actual_horizon']
    assert memory_adapt_period >0, "need horizon or example length to be longer!"

    # memory buffer:
    # making a memory buffer for previous rollouts too. 
    # buffer contains a dictionary full of lists of tensors which correspond to full rollouts. 
    use_training_buffer = False  
    # TODO: need to purge buffer elements that are shorter than the new seq length!
    # or keep the seq len to satisfy the buffer rather than just the new training data. 
    # TODO: keep better track of the sequence length anyways... and compute it more efficiently. 
    print('using training buffer?', use_training_buffer)
    num_new_rollouts = args.num_workers*training_rollouts_per_worker
    num_prev_epochs_to_store = 4
    # NOTE: this is a lower bound. could go over this depending on how stochastic the buffer adding is!
    max_buffer_size = num_new_rollouts*num_prev_epochs_to_store

    if use_training_buffer:
        iters_through_buffer_each_epoch = 1 #10 // num_prev_epochs_to_store
    else: 
        iters_through_buffer_each_epoch = 1

    kl_tolerance=0.5
    free_nats = torch.Tensor([kl_tolerance*env_params['LATENT_SIZE']]).to(device)

    # Init save filenames 
    game_dir = join(args.logdir, args.gamename)
    filenames_dict = { 'rssm_'+bc:join(game_dir, 'rssm_'+bc+'.tar') for bc in ['best', 'checkpoint'] }
    # make directories if they dont exist
    samples_dir = join(game_dir, 'samples')
    for dirr in [game_dir, samples_dir]:
        if not exists(dirr):
            mkdir(dirr)
    logger_filename = join(game_dir, 'logger.txt')

    # init models
    rssm = RSSModel(
        env_params['ACTION_SIZE'],
        env_params['LATENT_RECURRENT_SIZE'],
        env_params['LATENT_SIZE'],
        env_params['EMBEDDING_SIZE'],
        env_params['NODE_SIZE'],
        env_params['use_vae'],
        decoder_reward_condition,
        decoder_make_sigmas,
        device=device,
    )
    optimizer = torch.optim.Adam(rssm.parameters(), lr=1e-3)

    # Loading in trained models: 
    if not args.no_reload:
        for model_var, name in zip([rssm],['rssm']):
            load_file = filenames_dict[name+'_best']
            assert exists(load_file), "Could not find file: " + load_file + " to load in!"
            state = torch.load(load_file, map_location={'cuda:0': str(device)})
            print("Loading model_type {} at epoch {} "
                "with test error {}".format(name,
                    state['epoch'], state['precision']))

            model_var.load_state_dict(state['state_dict'])
            model_cur_best = state['precision']
            optimizer.load_state_dict(state["optimizer"])
                    
    # save init models
    if args.no_reload or args.giving_pretrained: 
        print("Overwriting checkpoint because pretrained models or no reload was called and removing the old logger file!")
        print("NB! This will overwrite the best and checkpoint models!\nSleeping for 5 seconds to allow you to change your mind...")
        time.sleep(5.0)
        for model_var, model_name in zip([rssm],['rssm']):
            save_checkpoint({
                "state_dict": model_var.get_save_dict(),
                "optimizer": optimizer.state_dict(),
                "precision": None,
                "epoch": -1}, True, filenames_dict[model_name+'_checkpoint'],
                            filenames_dict[model_name+'_best'])
                        # saves file to is best AND checkpoint

        # unlinking the old logger too
        if exists(logger_filename):
            unlink(logger_filename)

    # consider making the learning rate lower because models are already pretrained!
    if args.giving_pretrained:
        optimizer = torch.optim.Adam(rssm.parameters(), lr=1e-3)
        model_cur_best = None

    def train_batch(data, return_for_rssm_sampling=False):

        obs, acts, rews, terminals = [arr.to(device) for arr in data]

        # need to flip the seq and batch dimensions 
        # also set the terminals to non terms. 
        # I am inputting these with batch first. Need to flip this around. 

        print('data shapes', obs.shape, acts.shape, rews.shape, terminals.shape)

        if env_params['use_vae']:
            obs = obs.permute(1, 0, 2, 3, 4).contiguous()
        else: 
            obs = obs.permute(1, 0, 2).contiguous()
            print('obs shape in train batch', obs.shape)
        if args.gamename=='pendulum':
            acts = acts.unsqueeze(-1)
        acts = acts.permute(1,0,2).contiguous()
        rews = rews.permute(1,0,2).contiguous()
        terminals = terminals.unsqueeze(-1).permute(1,0,2).contiguous()
        non_terms = (terminals==0.).float().contiguous()

        # shift everything: 
        obs = obs[1:]
        acts = acts[:-1]
        rews = rews[:-1]
        non_terms = non_terms[:-1]

        """ (seq_len, batch_size, *dims) """
        #obs, acts, rews, non_terms = buffer.sample_and_shift(batch_size, seq_len)

        """ (seq_len, batch_size, embedding_size) """
        encoded_obs = rssm.encode_sequence_obs(obs)

        """ (seq_len, batch_size, dim) """
        rollout = rssm.perform_rollout(acts, encoder_output=encoded_obs, non_terms=non_terms)

        if env_params['use_vae']:
            """ (seq_len, batch_size, *dims) """
            decoded_mus, decoded_logsigmas = rssm.decode_sequence_obs(
                rollout["hiddens"], rollout["posterior_states"], rews
            )
            
            if decoder_make_sigmas:
                obs_loss = _observation_loss(decoded_mus, obs, decoded_logsigmas)
            else: 
                obs_loss = _observation_loss(decoded_mus, obs)

            #print('shape of decoded reward', decoded_reward.shape )
            # NOTE: do I also need this KLD loss here? this is on its own sufficient? 
            posterior = Normal(rollout["posterior_means"], rollout["posterior_stds"])
            prior = Normal(rollout["prior_means"], rollout["prior_stds"])
            kl_loss = _kl_loss(posterior, prior)

        else: 
            # NOTE: not sure i need this because of the KLD loss. 
            obs_loss = f.mse_loss(rollout["posterior_states"], rollout["prior_states"],
                reduction='none').sum(dim=2).mean(dim=(0,1))
            kl_loss = torch.Tensor([0])

        """ (seq_len, batch_size) """
        decoded_reward = rssm.decode_sequence_reward(
            rollout["hiddens"], rollout["posterior_states"]
        )
        
        print('going into reward loss', decoded_reward.shape, rews.squeeze(-1).shape)
        reward_loss = _reward_loss(decoded_reward, rews.squeeze(-1))

        if return_for_rssm_sampling: 
            if env_params['use_vae']:
                # return last batch for sample generation! 
                for_rssm_sampling = [obs[:,0,:,:,:], decoded_mus[:,0,:,:,:], 
                    rollout["hiddens"][:,0,:], rollout["prior_states"][:,0,:], 
                    acts[:,0,:], rews[:,0,:]]
                # dont need to flip batch dimensions as I get rid of the batch!
            else: 
                for_rssm_sampling = None
                print('prior and posterior states', rollout["prior_states"][:10,0,:],  rollout["posterior_states"][:10,0,:])
                print('reward prediction vs reality. :', decoded_reward[:10,0], rews.squeeze(-1)[:10,0])
                print('actions taken', acts[:10, 0])
            return obs_loss, reward_loss, kl_loss, for_rssm_sampling
        else: 
            return obs_loss, reward_loss, kl_loss

    def _observation_loss(decoded_mus, obs, decoded_logsigmas=None):
        #print('obs loss', decoded_mus.shape, obs.shape)
        if decoded_logsigmas is not None: 
            # TODO: add a min std value here? 
            # NOTE: sampling and using mse rather than maximizing log prob of the real observation. 
            pred_obs = decoded_mus + (decoded_logsigmas.exp() * torch.randn_like(decoded_mus))
            print('using logsigmas in loss. shape is:', pred_obs.shape, obs.shape )
        else: 
            pred_obs = decoded_mus
        return (
            f.mse_loss(obs, pred_obs, reduction="none")
            .sum(dim=(2, 3, 4))
            .mean(dim=(0, 1))
        )

    def _reward_loss(decoded_reward, reward):
        #print(decoded_reward.shape, reward.shape)
        return f.mse_loss(reward, decoded_reward, reduction="none").mean(dim=(0, 1))

    def _kl_loss(posterior, prior):
        if free_nats is not None:
            return torch.max(
                kl_divergence(posterior, prior).sum(dim=2), free_nats
            ).mean(dim=(0, 1))
        else:
            return kl_divergence(posterior, prior).sum(dim=2).mean(dim=(0, 1))

    def data_pass(epoch, train): # pylint: disable=too-many-locals
        """One pass through full epoch pass through the data either testing or training (with torch.no_grad()).
        NB. One epoch here is all of the data collected from the workers using
        the planning algorithm. 
        This is num_workers * training_rollouts_per_worker.

        :args:
            - epoch: int
            - train: bool

        :returns:
            - cumloss_dict - All of the losses collected from this epoch. 
                            Averaged across the batches and sequence lengths 
                            to be on a per element basis. 
            if test also returns information used to generate the RSSM samples
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
            for model_var in [rssm]:
                model_var.train()
            
        else:
            loader = test_loader
            for model_var in [rssm]:
                model_var.eval()

        pbar = tqdm(total=len(loader.dataset), desc="Epoch {}".format(epoch))

        # store all of the losses for this data pass. 
        #cumloss_dict = {n:0 for n in ['loss', 'loss_vae', 'loss_mdrnn','kld', 'recon', 'gmm', 'bce', 'mse']}
        cumloss_dict = {n:0 for n in ['loss', 'obs_loss', 'reward_loss', 'kl_loss']}

        # iterate through an epoch of data. 
        num_rollouts_shown = 0
        for i, data in enumerate(loader):
            # -1 here is the terminals, easiest to load in. 
            cur_batch_size = data[-1].shape[0]
            num_rollouts_shown+= cur_batch_size

            if train:
                obs_loss, reward_loss, kl_loss = train_batch(data)
                # taking grad step after every batch. 
                optimizer.zero_grad()
                if env_params['use_vae']:
                    (obs_loss + reward_loss + kl_loss).backward()
                else:
                    (obs_loss + reward_loss).backward()
                # TODO: consider adding gradient clipping like Ha.  
                torch.nn.utils.clip_grad_norm_(rssm.parameters(), 1000.0)
                optimizer.step()
            else:
                with torch.no_grad():
                    obs_loss, reward_loss, kl_loss, for_rssm_sampling = train_batch(data,return_for_rssm_sampling=True)

            rssm_loss_dict = dict(obs_loss=obs_loss, reward_loss=reward_loss, kl_loss=kl_loss)

            # add results from each batch to cumulative losses
            for k in cumloss_dict.keys():
                for loss_dict in [rssm_loss_dict]:
                    if k in loss_dict.keys():
                        cumloss_dict[k] += loss_dict[k].item()*cur_batch_size if hasattr(loss_dict[k], 'item') else \
                                                loss_dict[k]
            
            # store separately vae and mdrnn losses: 
            for k,v in rssm_loss_dict.items():
                cumloss_dict['loss'] += v.item()*cur_batch_size

            # Display training progress bar with current losses
            postfix_str = ""
            for k,v in cumloss_dict.items():
                v = (v /num_rollouts_shown)/SEQ_LEN
                postfix_str+= k+'='+str(round(v,4))+', '
            pbar.set_postfix_str(postfix_str)
            pbar.update(cur_batch_size)
        pbar.close()

        # puts losses on a per element level. independent of batch sizes and seq lengths.
        cumloss_dict = {k: (v/num_rollouts_shown)/SEQ_LEN for k, v in cumloss_dict.items()}
        # sort the order so they are added to the logger in the same order!
        cumloss_dict = OrderedDict(sorted(cumloss_dict.items()))
        if train: 
            return cumloss_dict 
        else: 
            return cumloss_dict, for_rssm_sampling
            # return the last observation and reward to generate the VAE examples. 

    train = partial(data_pass, train=True)
    test = partial(data_pass, train=False)

    ################## Main Training Loop ############################

    # all of these are static across epochs. 
    worker_package = [ args.gamename, decoder_reward_condition,
                game_dir, training_rollouts_per_worker,
                planner_n_particles, 
                cem_iters, discount_factor,
                compute_feef, antithetic ]

    for e in range(epochs):
        print('====== New Epoch:', e)
        ## run the current policy with the current VAE and MDRNN

        # NOTE: each worker loads in the checkpoint model not the best model! Want to use up to date. 
        print('====== Generating Rollouts to train the RSSM') 
        
        # TODO: antithetic sampling. use same seed twice. 
        if args.num_workers <= 1:
            # dont use multiprocessing. 
            agent = Agent(args.gamename, game_dir, decoder_reward_condition, 
                planner_n_particles=planner_n_particles, cem_iters=cem_iters, discount_factor=discount_factor)
            seed = np.random.randint(0, 1e9, 1)[0]
            output = agent.simulate(return_events=True,
                                    compute_feef=compute_feef,
                                    num_episodes=training_rollouts_per_worker, seed=seed)
            # reward_losses, terminals, sim_data, feef_losses 
            # TODO: clean this up.
            SEQ_LEN, BATCH_SIZE = set_seq_and_batch_vals([output], batch_size_to_seq_len_multiple,dim=2)
            train_data = combine_single_worker(output[2][:-1], SEQ_LEN )
            test_data = {k:[v]for k, v in output[2][-1].items()}
            reward_losses, feef_losses = output[0], output[3]

        else: 

            if e<random_action_epochs:
                SEQ_LEN, BATCH_SIZE, train_data, test_data, feef_losses, \
                reward_losses = generate_rollouts_using_planner( 
                        args.num_workers, batch_size_to_seq_len_multiple, 
                        worker_package, take_rand_actions=True)

            else: 
            #if e==0: # can be used to overfit to a single rollout for debugging. 
                SEQ_LEN, BATCH_SIZE, train_data, test_data, feef_losses, reward_losses = generate_rollouts_using_planner( 
                        args.num_workers, batch_size_to_seq_len_multiple, worker_package)

        if use_training_buffer:
            if e==0:
                buffer_train_data = TrainBufferDataset(train_data, max_buffer_size, 
                                        key_to_check_lengths='terminal')
            else: 
                buffer_train_data.add(train_data)
            train_data = buffer_train_data.buffer
            print('epoch', e, 'size of buffer', len(buffer_train_data), 'buffer index', buffer_train_data.buffer_index)

        # NOTE: currently not applying any transformations as saving those that happen when the 
        # rollouts are actually generated. 
        train_dataset = GeneratedDataset(None, train_data, SEQ_LEN)
        test_dataset = GeneratedDataset(None, test_data, SEQ_LEN)
        # TODO: set number of workers higher. Here it doesn;t matter much as already have tensors ready. 
        # (dont need any loading or tranformations) 
        # and before these workers were clashing with the ray workers for generating rollouts. 
        
        for it in range(iters_through_buffer_each_epoch):
            train_loader = DataLoader(train_dataset,
                batch_size=BATCH_SIZE, num_workers=0, shuffle=True, drop_last=False)
            print('====== Starting Training Models')
            # train VAE and MDRNN. uses partial(data_pass)
            train_loss_dict = train(e)
            print('====== Done Training Models')
        
        test_loader = DataLoader(test_dataset, shuffle=True,
                batch_size=BATCH_SIZE, num_workers=0, drop_last=False)
        # returns the last ones in order to produce samples!
        test_loss_dict, for_rssm_sampling = test(e)
        print('====== Done Testing Models')
        
        #scheduler.step(test_loss_dict['loss'])
        # append the planning results to the TEST loss dictionary. 
        for name, var in zip(['reward_planner', 'feef_planner'], [reward_losses, feef_losses]):
            test_loss_dict['avg_'+name] = np.mean(var)
            test_loss_dict['std_'+name] = np.std(var)
            test_loss_dict['max_'+name] = np.max(var)
            test_loss_dict['min_'+name] = np.min(var)

        print('====== Test Loss dictionary:', test_loss_dict)

        # checkpointing the rssm. Necessary to ensure the workers load in the most up to date checkpoint.
        # save_checkpoint function always saves a checkpoint and may also update the best. 
        is_best = not model_cur_best or test_loss_dict['loss'] < model_cur_best
        if is_best:
            model_cur_best = test_loss_dict['loss']
            print('====== New Best for the Test Loss! Updating *MODEL_best.tar*')
        for model_var, model_name in zip([rssm],['rssm']):
            save_checkpoint({
                "state_dict": model_var.get_save_dict(),
                "optimizer": optimizer.state_dict(),
                "precision": test_loss_dict['loss'],
                "epoch": e}, is_best, filenames_dict[model_name+'_checkpoint'],
                            filenames_dict[model_name+'_best'])
        print('====== Done Saving VAE and MDRNN')

        if make_vae_samples or make_mdrnn_samples:
            generate_rssm_samples( rssm, for_rssm_sampling, 
                            samples_dir, SEQ_LEN, env_params['IMAGE_RESIZE_DIM'],
                            example_length,
                            memory_adapt_period, e, device, 
                            make_vae_samples=make_vae_samples,
                            make_mdrnn_samples=make_mdrnn_samples, 
                            transform_obs=False  )
            print('====== Done Generating Samples')
        
        write_logger(logger_filename, train_loss_dict, test_loss_dict)
        print('====== Done Writing out to the Logger')

if __name__ =='__main__':
    parser = argparse.ArgumentParser("Training Script")
    parser.add_argument('--gamename', type=str,
                        help="What Gym environment to train in.")
    parser.add_argument('--logdir', type=str, default='exp_dir',
                        help="Where things are logged and models are loaded from.")
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