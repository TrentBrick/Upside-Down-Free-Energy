# pylint: disable=no-member
""" 
Training of the UpsideDown RL model.
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
from models import UpsdModel, UpsdBehavior, LightningTemplate
from torch.distributions.normal import Normal
from multiprocessing import cpu_count
from collections import OrderedDict
from control import Agent 
import time 
import random 
from utils import set_seq_and_batch_vals
# TODO: test if this seed everything actually works! 
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger

def main(args):

    assert args.num_workers <= cpu_count(), "Providing too many workers!" 

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
    epochs = 1500
    random_action_epochs = 1
    evaluate_every = 10
    training_rollouts_total = 20
    training_rollouts_per_worker = training_rollouts_total//args.num_workers
    num_new_rollouts = args.num_workers*training_rollouts_per_worker
    antithetic = False  
    if Levine_Implementation:
        config= dict(
            lr=0.01,
            batch_size = 256,
            max_buffer_size = 100000,
            discount_factor = 0.99,
            desired_reward_dist_beta = 1000,
            weight_loss = True,
            desire_scalings =None
        )
        train_buffer = RingBuffer(obs_dim=env_params['STORED_STATE_SIZE'], act_dim=env_params['STORED_ACTION_SIZE'], size=config['max_buffer_size'])
        test_buffer = RingBuffer(obs_dim=env_params['STORED_STATE_SIZE'], 
            act_dim=env_params['STORED_ACTION_SIZE'], size=config['batch_size']*10)
        
    else: 
        config= dict(
        lr=0.0003,
        batch_size = 768,
        max_buffer_size = 500,
        desire_scalings = (0.02, 0.01), # reward then horizon
        discount_factor = 1.0,
        last_few = 75,
        )
        # TODO: do I need to provide seeds to the buffer like this? 
        train_buffer = ReplayBuffer(config['max_buffer_size'], args.seed, config['batch_size'], args.num_grad_steps)
        test_buffer = ReplayBuffer(config['batch_size']*10, args.seed, config['batch_size'], 5)

    # for plotting example horizons. Useful with VAE:
    '''if env_params['use_vae']:
        make_vae_samples = True 
        example_length = 12
        assert example_length<= SEQ_LEN, "Example length must be smaller."
        memory_adapt_period = example_length - env_params['actual_horizon']
        assert memory_adapt_period >0, "need horizon or example length to be longer!"
        kl_tolerance=0.5
        free_nats = torch.Tensor([kl_tolerance*env_params['LATENT_SIZE']], device=device )
    '''

    # Init save filenames 
    base_game_dir = join(args.logdir, args.gamename)
    if not exists(base_game_dir):
        mkdir(base_game_dir)
    game_dir = join(base_game_dir, 'seed_'+str(args.seed)+'_gradsteps_'+str(args.num_grad_steps))
    filenames_dict = { 'model_'+bc:join(game_dir, 'model_'+bc+'.tar') for bc in ['best', 'checkpoint'] }
    # make directories if they dont exist
    samples_dir = join(game_dir, 'samples')
    for dirr in [game_dir, samples_dir]:
        if not exists(dirr):
            mkdir(dirr)
    logger_filename = join(game_dir, 'logger.txt')
    logger = TensorBoardLogger(logger_filename, name="my_model")

    # Load in the Model, Loggers, etc:
    seed_everything(args.seed)

    model = LightningTemplate(config, env_params)

    if not args.no_reload:
        # load in: 
        model = LightningTemplate.load_from_checkpoint(filenames_dict['best'])

    trainer = Trainer(deterministic=True, logger=logger,
         default_root_dir=game_dir, max_epochs=1)

    cum_iters_generated = 0 
    for e in range(epochs):
        print('====== New Epoch:', e)
        ## run the current policy with the current MODEL

        # NOTE: each worker loads in the checkpoint model not the best model! Want to use up to date. 
        print('====== Generating Rollouts to train the Model') 
        
        # TODO: antithetic sampling. use same seed twice. 
        #if args.num_workers <= 1:
            # dont use multiprocessing. 
        if e<random_action_epochs:

            agent = Agent(args.gamename, game_dir, 
                take_rand_actions=True,
                discount_factor=config['discount_factor'])

        else: 
            agent = Agent(args.gamename, game_dir, 
                model = model, 
                Levine_Implementation= Levine_Implementation,
                desired_reward_stats = reward_from_epoch_stats, 
                desired_horizon = desired_horizon,
                desired_reward_dist_beta=config['desired_reward_dist_beta'],
                discount_factor=config['discount_factor'])
        
        seed = np.random.randint(0, 1e9, 1)[0]
        
        output = agent.simulate(seed, return_events=True,
                                compute_feef=compute_feef,
                                num_episodes=training_rollouts_per_worker)
        #SEQ_LEN, config['batch_size'] = set_seq_and_batch_vals([output], config['batch_size'],dim=2)
        if Levine_Implementation: 
            train_data = combine_single_worker(output[2][:-1], 1 )
            test_data = {k:[v]for k, v in output[2][-1].items()}
        else: 
            train_data =output[2][:-1]
            test_data = [output[2][-1]]
        reward_losses, termination_times, feef_losses = output[0], output[1], output[3]

        # modify the training data how I want to now while its in a list of rollouts. 
        # dictionary of items with lists inside of each rollout. 

        # add data to the buffer. 
        iters_generated = 0
        if Levine_Implementation: 
            train_buffer.add_rollouts(train_data)
            test_buffer.add_rollouts(test_data)
        else: 
            for r in range(len(train_data)):
                train_buffer.add_sample(  train_data[r]['obs'], train_data[r]['act'], 
                    train_data[r]['rew'], termination_times[r] )
                iters_generated+= termination_times[r]

            for r in range(len(test_data)):
                test_buffer.add_sample(  test_data[r]['obs'], test_data[r]['act'], 
                    test_data[r]['rew'], len(test_data[r]['terminal']) )
        cum_iters_generated+= iters_generated
        
        # train the model on all of the new data that has been gathered. 

        # number of batch samples. 
        
        results = trainer.fit(model, train_dataloader=DataLoader(train_buffer, batch_size=1), 
            val_dataloaders=DataLoader(test_buffer, batch_size=1))

        # get out the results needed to compute the new desires
        # get out everything that I want to be logging. 
        # ensure the model is being checkpointed. 

        if Levine_Implementation:
            desired_horizon = 99999 
            reward_from_epoch_stats = (np.mean(reward_losses), np.std(reward_losses))
        else: 
            last_few_mean_returns, last_few_std_returns, desired_horizon  = train_buffer.get_desires(last_few=config['last_few'])
            reward_from_epoch_stats = (last_few_mean_returns, last_few_std_returns)
    
            #for name, var in zip(['mu', 'std', 'horizon'], [last_few_mean_returns, last_few_std_returns, desired_horizon]):
            #    test_loss_dict[name] = var

            #test_loss_dict['cum_num_iters']=cum_iters_generated
        
if __name__ =='__main__':
    parser = argparse.ArgumentParser("Training Script")
    parser.add_argument('--gamename', type=str,
                        help="What Gym environment to train in.")
    parser.add_argument('--logdir', type=str, default='exp_dir',
                        help="Where things are logged and models are loaded from.")
    parser.add_argument('--no_reload', action='store_true',
                        help="Won't load in models for MODEL from the joint file. \
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
    parser.add_argument('--num_grad_steps', type=int, default=100,
                        help="Grad steps per data collection")
    args = parser.parse_args()
    main(args)