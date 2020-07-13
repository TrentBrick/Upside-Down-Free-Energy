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
from utils import save_checkpoint, generate_model_samples, \
    generate_rollouts, write_logger, ReplayBuffer, \
    RingBuffer, combine_single_worker
from envs import get_env_params
import sys
#from models.rssm import RSSModel 
from models import UpsdModel, UpsdBehavior
from torch.distributions.normal import Normal
from multiprocessing import cpu_count
from collections import OrderedDict
from control import Agent 
import time 
import random 
from utils import set_seq_and_batch_vals

def main(args):

    assert args.num_workers <= cpu_count(), "Providing too many workers! Need one less than total amount." 

    make_vae_samples, make_mdrnn_samples = False, False 

    Levine_Implementation = False 
    if args.seed:
        print('Setting the random seed!!!')
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # used for saving which models are the best based upon their train performance. 
    model_cur_best=None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    compute_feef = True # NOTE: currently not actually computing it inside of agent.py simulate!!!!

    # get environment parameters: 
    env_params = get_env_params(args.gamename)

    # Constants
    if Levine_Implementation:
        batch_size_to_seq_len_multiple = 256 #4096
    else: 
        batch_size_to_seq_len_multiple = 768
    SEQ_LEN = 1 # number of sequences in a row used during training
    # needs one more than this for the forward predictions. 
    BATCH_SIZE = batch_size_to_seq_len_multiple//SEQ_LEN
    epochs = 2000
    random_action_epochs = 1
    evaluate_every = 10

    desired_reward_dist_beta = 1000
    weight_loss = True

    training_rollouts_total = 20
    training_rollouts_per_worker = training_rollouts_total//args.num_workers
    if args.num_workers==1: 
        assert training_rollouts_per_worker>1, "need more workers to also make test data!"

    num_new_rollouts = args.num_workers*training_rollouts_per_worker
    #num_prev_epochs_to_store = random_action_epochs
    # NOTE: this is a lower bound. could go over this depending on how stochastic the buffer adding is!
    
    if Levine_Implementation:
        num_iters_in_buffer = 100000
        max_buffer_size = num_iters_in_buffer #num_new_rollouts*num_prev_epochs_to_store
        training_num_grad_steps = 1000 #max_buffer_size//BATCH_SIZE
        discount_factor = 0.99
        train_buffer = RingBuffer(obs_dim=env_params['STORED_STATE_SIZE'], act_dim=env_params['STORED_ACTION_SIZE'], size=max_buffer_size)
        test_buffer = RingBuffer(obs_dim=env_params['STORED_STATE_SIZE'], 
            act_dim=env_params['STORED_ACTION_SIZE'], size=batch_size_to_seq_len_multiple*10)
        desire_scalings = None
    else: 
        max_buffer_size = 500 
        training_num_grad_steps = 100
        desire_scalings = (0.02, 0.01) # reward then horizon
        discount_factor = 1.0
        last_few = 75
        train_buffer = ReplayBuffer(max_buffer_size, args.seed)
        test_buffer = ReplayBuffer(batch_size_to_seq_len_multiple*10, args.seed)

    # Planner and Forward Model Parameters!
    planner_n_particles = 700
    cem_iters = 7
    decoder_reward_condition = False
    decoder_make_sigmas = False
    antithetic = True 

    # for plotting example horizons. Useful with VAE:
    #example_length = 12
    #assert example_length<= SEQ_LEN, "Example length must be smaller."
    #memory_adapt_period = example_length - env_params['actual_horizon']
    #assert memory_adapt_period >0, "need horizon or example length to be longer!"
        
    #iters_through_buffer_each_epoch = 1
    kl_tolerance=0.5
    free_nats = torch.Tensor([kl_tolerance*env_params['LATENT_SIZE']]).to(device)

    # Init save filenames 
    game_dir = join(args.logdir, args.gamename)
    filenames_dict = { 'model_'+bc:join(game_dir, 'model_'+bc+'.tar') for bc in ['best', 'checkpoint'] }
    # make directories if they dont exist
    samples_dir = join(game_dir, 'samples')
    for dirr in [game_dir, samples_dir]:
        if not exists(dirr):
            mkdir(dirr)
    logger_filename = join(game_dir, 'logger.txt')

    # init models
    '''model = RSSModel(
        env_params['ACTION_SIZE'],
        env_params['LATENT_RECURRENT_SIZE'],
        env_params['LATENT_SIZE'],
        env_params['EMBEDDING_SIZE'],
        env_params['NODE_SIZE'],
        env_params['use_vae'],
        decoder_reward_condition,
        decoder_make_sigmas,
        device=device,
    )'''
    if Levine_Implementation:
        model = UpsdModel(env_params['STORED_STATE_SIZE'], 
            env_params['desires_size'], 
            env_params['ACTION_SIZE'], 
            env_params['NODE_SIZE'], desire_scalings=desire_scalings)
        lr = 0.0003
    else: 
        model = UpsdBehavior(env_params['STORED_STATE_SIZE'], 
            env_params['desires_size'], 
            env_params['ACTION_SIZE'], 
            env_params['NODE_SIZE'], desire_scalings=desire_scalings)
        lr = 0.0003
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Loading in trained models: 
    if not args.no_reload:
        for model_var, name in zip([model],['model']):
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
        time.sleep(0.1)
        for model_var, model_name in zip([model],['model']):
            save_checkpoint({
                "state_dict": model_var.state_dict(),
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
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model_cur_best = None

    def train_batch(data, return_for_model_sampling=False):

        if Levine_Implementation: 
            obs, obs2, act, rew, terminal, terminal_rew, time = data['obs'].to(device), data['obs2'].to(device), data['act'].to(device), data['rew'].to(device), data['terminal'].to(device), data['terminal_rew'].to(device), data['time'].to(device)
        else: 
            # this is actually delta time. 
            obs, act, rew, time = data['obs'].to(device), data['act'].to(device), data['rew'].to(device), data['time'].to(device)
        # need to flip the seq and batch dimensions 
        # also set the terminals to non terms. 
        # I am inputting these with batch first. Need to flip this around. 
        
        #print('data shapes', obs.shape, obs[0:5], obs2.shape, act.shape, rew.shape, terminal.shape)
        
        # feed obs and rew into the forward model. 
        if not Levine_Implementation: 
            desires = torch.cat([rew.unsqueeze(1), time.unsqueeze(1)], dim=1)
        pred_action = model.forward(obs, desires)
        if not env_params['continuous_actions']:
            #pred_action = torch.sigmoid(pred_action)
            act = act.squeeze().long()
        pred_loss = _pred_loss(pred_action, act, continous=env_params['continuous_actions'])
        if Levine_Implementation and weight_loss:
            #print('weights used ', torch.exp(rew/desired_reward_dist_beta))
            pred_loss = pred_loss*torch.exp(terminal_rew/desired_reward_dist_beta)
        pred_loss = pred_loss.mean(dim=0)

        #print('loss from this train batch', pred_loss)

        if return_for_model_sampling: 
            return pred_loss, (rew[0:5], pred_action[0:5], pred_action[0:5].argmax(-1), act[0:5])
        return pred_loss 

    def _pred_loss(pred_action, real_action, continous=True):
        if continous:
            # add a sigmoid activation layer.: 
            return f.mse_loss(pred_action, real_action ,reduction='none').sum(dim=1)
        else: 
            return f.cross_entropy(pred_action, real_action, reduction='none')

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
            if test also returns information used to generate the Model samples
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
            buffer = train_buffer
            model.train()
            # TODO: I need to adjust this!!!
            num_grad_steps = training_num_grad_steps
            
        else:
            buffer = test_buffer
            model.eval()
            num_grad_steps = 5

        pbar = tqdm(total=BATCH_SIZE*num_grad_steps, desc="Epoch {}".format(epoch))

        # store all of the losses for this data pass. 
        cumloss_dict = {n:0 for n in ['loss', 'pred_loss']}

        # iterate through an epoch of data. 
        num_iters_shown = 0
        for i in range(num_grad_steps):
            data = buffer.sample_batch(BATCH_SIZE)
        
            # -1 here is the terminals, easiest to load in. 
            #BATCH_SIZE = len(data) #data[-1].shape[0]
            num_iters_shown+= BATCH_SIZE

            if train:
                pred_loss = train_batch(data)
                # taking grad step after every batch. 
                optimizer.zero_grad()
                #if env_params['use_vae']:
                #    (obs_loss + reward_loss + kl_loss).backward()
                #else:
                pred_loss.backward()
                # TODO: consider adding gradient clipping like Ha.  
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1000.0)
                optimizer.step()
            else:
                with torch.no_grad():
                    pred_loss, for_upsd_sampling = train_batch(data,return_for_model_sampling=True)
                    if i == num_grad_steps-1:
                        print('reward, pred, and real action', for_upsd_sampling)
                        print('action probs', for_upsd_sampling[1].softmax(-1))
                        print("pred action", for_upsd_sampling[-2])
                        print('real action', for_upsd_sampling[-1])
                        # getting the diversity of actions taken. 
                        acts_observed, counts = np.unique(for_upsd_sampling[-2].cpu().numpy(), return_counts=True)
                        print("diversity of the pred actions", acts_observed, counts/len(for_upsd_sampling[-1]) )
                        acts_observed, counts = np.unique(for_upsd_sampling[-1].cpu().numpy(), return_counts=True)
                        print("diversity of the real actions", acts_observed, counts/len(for_upsd_sampling[-1]) )

            upsd_loss_dict = dict(pred_loss=pred_loss)

            # add results from each batch to cumulative losses
            for k in cumloss_dict.keys():
                for loss_dict in [upsd_loss_dict]:
                    if k in loss_dict.keys():
                        cumloss_dict[k] += loss_dict[k].item()*BATCH_SIZE if hasattr(loss_dict[k], 'item') else \
                                                loss_dict[k]
            
            for k,v in upsd_loss_dict.items():
                # the total overall loss combining all separate losses: 
                cumloss_dict['loss'] += v.item()*BATCH_SIZE

            # Display training progress bar with current losses
            postfix_str = ""
            for k,v in cumloss_dict.items():
                v = (v /num_iters_shown) #/SEQ_LEN
                postfix_str+= k+'='+str(round(v,4))+', '
            pbar.set_postfix_str(postfix_str)
            pbar.update(BATCH_SIZE)
        pbar.close()

        # puts losses on a per element level. independent of batch sizes and seq lengths.
        
        # /SEQ_LEN 
        cumloss_dict = {k: (v/num_iters_shown) for k, v in cumloss_dict.items()}
        # sort the order so they are added to the logger in the same order!
        cumloss_dict = OrderedDict(sorted(cumloss_dict.items()))
        if train: 
            return cumloss_dict 
        else: 
            return cumloss_dict, for_upsd_sampling
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
        print('====== Generating Rollouts to train the Model') 
        
        # TODO: antithetic sampling. use same seed twice. 
        #if args.num_workers <= 1:
            # dont use multiprocessing. 
        if e<random_action_epochs:

            agent = Agent(args.gamename, game_dir, decoder_reward_condition, 
                take_rand_actions=True,
                planner_n_particles=planner_n_particles, cem_iters=cem_iters, discount_factor=discount_factor)

        else: 
            agent = Agent(args.gamename, game_dir, decoder_reward_condition, 
                model = model, 
                Levine_Implementation= Levine_Implementation,
                desired_reward_stats = reward_from_epoch_stats, 
                desired_horizon = desired_horizon,
                desired_reward_dist_beta=desired_reward_dist_beta,
                planner_n_particles=planner_n_particles, cem_iters=cem_iters, discount_factor=discount_factor)
        
        seed = np.random.randint(0, 1e9, 1)[0]
        
        output = agent.simulate(return_events=True,
                                compute_feef=compute_feef,
                                num_episodes=training_rollouts_per_worker, seed=seed)
        #SEQ_LEN, BATCH_SIZE = set_seq_and_batch_vals([output], batch_size_to_seq_len_multiple,dim=2)
        train_data = combine_single_worker(output[2][:-1], SEQ_LEN )
        test_data = {k:[v]for k, v in output[2][-1].items()}
        reward_losses, feef_losses = output[0], output[3]

        '''else: 
            print('need to give the agents the current reward that is desired. ')
            
            if e<random_action_epochs:
                SEQ_LEN, BATCH_SIZE, train_data, test_data, feef_losses, \
                reward_losses = generate_rollouts( 
                        args.num_workers, batch_size_to_seq_len_multiple, 
                        worker_package, take_rand_actions=True)

            else: 
            #if e==0: # can be used to overfit to a single rollout for debugging. 
                SEQ_LEN, BATCH_SIZE, train_data, test_data, feef_losses, reward_losses = generate_rollouts( 
                        args.num_workers, batch_size_to_seq_len_multiple, worker_package)'''

        # modify the training data how I want to now while its in a list of rollouts. 
        # dictionary of items with lists inside of each rollout. 

        # add data to the buffer. 
        if Levine_Implementation: 
            train_buffer.add_rollouts(train_data)
            test_buffer.add_rollouts(test_data)
        else: 
            for r in range(len(train_data['terminal'])):
                train_buffer.add_sample(  train_data['obs'][r], train_data['act'][r], 
                    train_data['rew'][r], len(train_data['terminal'][r]) )

            for r in range(len(test_data['terminal'])):
                test_buffer.add_sample(  test_data['obs'][r], test_data['act'][r], 
                    test_data['rew'][r], len(test_data['terminal'][r]) )

        #for it in range(iters_through_buffer_each_epoch):
        #train_loader = DataLoader(train_dataset,
        #    batch_size=BATCH_SIZE, num_workers=0, shuffle=True, drop_last=False)
        print('====== Starting Training Models')
        # train VAE and MDRNN. uses partial(data_pass)
        train_loss_dict = train(e)
        print('====== Done Training Models')
        #test_loader = DataLoader(test_dataset, shuffle=True,
        #        batch_size=BATCH_SIZE, num_workers=0, drop_last=False)
        # returns the last ones in order to produce samples!
        test_loss_dict, for_upsd_sampling = test(e)
        print('====== Done Testing Models')
        
        #scheduler.step(test_loss_dict['loss'])
        # append the planning results to the TEST loss dictionary. 
        for name, var in zip(['reward', 'feef_'], [reward_losses, feef_losses]):
            test_loss_dict['avg_'+name] = np.mean(var)
            test_loss_dict['std_'+name] = np.std(var)
            test_loss_dict['max_'+name] = np.max(var)
            test_loss_dict['min_'+name] = np.min(var)

        if Levine_Implementation:
            desired_horizon = 99999 
            reward_from_epoch_stats = (test_loss_dict['avg_reward'], test_loss_dict['std_reward'])
        else: 
            last_few_mean_returns, last_few_std_returns, desired_horizon  = train_buffer.get_desires(last_few=last_few)
            reward_from_epoch_stats = (last_few_mean_returns, last_few_std_returns)
    
        print('====== Test Loss dictionary:', test_loss_dict)

        # checkpointing the model. Necessary to ensure the workers load in the most up to date checkpoint.
        # save_checkpoint function always saves a checkpoint and may also update the best. 
        is_best = not model_cur_best or test_loss_dict['loss'] < model_cur_best
        if is_best:
            model_cur_best = test_loss_dict['loss']
            print('====== New Best for the Test Loss! Updating *MODEL_best.tar*')
        for model_var, model_name in zip([model],['model']):
            save_checkpoint({
                "state_dict": model_var.state_dict(), #get_save_dict for RSSM
                "optimizer": optimizer.state_dict(),
                "precision": test_loss_dict['loss'],
                "epoch": e}, is_best, filenames_dict[model_name+'_checkpoint'],
                            filenames_dict[model_name+'_best'])
        print('====== Done Saving VAE and MDRNN')

        if make_vae_samples or make_mdrnn_samples:
            generate_model_samples( model, for_upsd_sampling, 
                            samples_dir, SEQ_LEN, env_params['IMAGE_RESIZE_DIM'],
                            example_length,
                            memory_adapt_period, e, device, 
                            make_vae_samples=make_vae_samples,
                            make_mdrnn_samples=make_mdrnn_samples, 
                            transform_obs=False  )
            print('====== Done Generating Samples')
        
        write_logger(logger_filename, train_loss_dict, test_loss_dict)
        print('====== Done Writing out to the Logger')

        if e%evaluate_every==0:
            print('======= Evaluating the agent')
            seed = np.random.randint(0, 1e9, 1)[0]
            cum_rewards, finish_times = agent.simulate(num_episodes=5, greedy=True)
            print('Evaluation, mean reward:', np.mean(cum_rewards), 'mean horizon length:', np.mean(finish_times))
            print('===========================')
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
    parser.add_argument('--seed', type=int, default=27,
                        help="Starter seed for reproducible results")
    args = parser.parse_args()
    main(args)