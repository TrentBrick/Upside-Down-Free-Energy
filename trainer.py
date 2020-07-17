# pylint: disable=no-member
""" 
Training of the UpsideDown RL model.
"""
import argparse
from functools import partial
from os.path import join, exists
from os import mkdir, unlink
import torch
import torch.nn.functional as F 
from torch.distributions.kl import kl_divergence
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import json
from tqdm import tqdm
from envs import get_env_params
import sys
from lightning_trainer import LightningTemplate
from multiprocessing import cpu_count
from collections import OrderedDict
from utils import ReplayBuffer, \
    RingBuffer, SortedBuffer
import time 
import random 
from utils import set_seq_and_batch_vals
# TODO: test if this seed everything actually works! 
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateLogger
#from pytorch_lightning.utilities.cloud_io import load as pl_load
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining

class TuneReportCallback(Callback):
    def on_epoch_end(self, trainer, pl_module):
        tune.report(
            loss=trainer.callback_metrics["train_loss"],
            mean_reward=pl_module.mean_reward_rollouts,
            mean_reward_20_epochs = sum(pl_module.mean_reward_over_20_epochs[-20:])/20,
            epoch=trainer.current_epoch)

def main(args):

    assert args.num_workers <= cpu_count(), "Providing too many workers!" 

    Levine_Implementation = False 
    if args.seed:
        print('Setting the random seed!!!')
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # get environment parameters: 
    env_params = get_env_params(args.gamename)

    # Constants
    epochs = 2000
    training_rollouts_total = 20
    #training_rollouts_total//args.num_workers
    constants = dict(
        random_action_epochs = 1,
        eval_every = 10,
        eval_episodes=10,
        training_rollouts_total = training_rollouts_total,
        training_rollouts_per_worker = 20, #tune.choice( [10, 20, 30, 40]),
        num_rand_action_rollouts = 10,
        antithetic = False,
        Levine_Implementation=Levine_Implementation,
        num_val_batches = 5
    )
    if Levine_Implementation:
        config= dict(
            lr=0.01,
            batch_size = 256,
            max_buffer_size = 100000,
            discount_factor = 0.99,
            desired_reward_dist_beta = 1000,
            weight_loss = True,
            desire_scalings =None, 
            num_grad_steps = 1000
        )
        train_buffer = RingBuffer(obs_dim=env_params['STORED_STATE_SIZE'], act_dim=env_params['STORED_ACTION_SIZE'], size=config['max_buffer_size'])
        test_buffer = RingBuffer(obs_dim=env_params['STORED_STATE_SIZE'], 
            act_dim=env_params['STORED_ACTION_SIZE'], size=config['batch_size']*10)
        
        # want to store 500 episodes
    else: 
        config= dict(
        lr= 0.0003, #tune.choice(np.logspace(-4, -2, num = 101)),
        batch_size = 768, #tune.choice([512, 768, 1024, 1536, 2048]),
        max_buffer_size = 500, #tune.choice([300, 400, 500, 600, 700]),
        horizon_scale = 0.01, #tune.choice( [0.01, 0.015, 0.02, 0.025, 0.03]), #(0.02, 0.01), # reward then horizon
        reward_scale = 0.02, #tune.choice( [0.01, 0.015, 0.02, 0.025, 0.03]),
        discount_factor = 1.0,
        last_few = 75, #tune.choice([25, 75]),
        desired_reward_dist_beta=1,
        num_grad_steps = 100,#tune.choice([100, 150, 200, 250, 300])
        )
        # TODO: do I need to provide seeds to the buffer like this? 
        
    config.update(constants) 
    config.update(env_params)
    config.update(vars(args))
    config['NODE_SIZE'] = [64,128,128,128] #tune.choice([[32], [32, 32], [32, 64], [32, 64, 64], [32, 64, 64, 64],
        #[64], [64, 64], [64, 128], [64, 128, 128], [64, 128, 128, 128]])

    use_tune = False   

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
    if use_tune:
        game_dir = ''
        run_name = str(np.random.randint(0,1000,1)[0])
    else: 
        # Init save filenames 
        base_game_dir = join(args.logdir, args.gamename)
        if not exists(base_game_dir):
            mkdir(base_game_dir)
        game_dir = join(base_game_dir, 'seed_'+str(args.seed))
        filenames_dict = { bc:join(game_dir, 'model_'+bc+'.tar') for bc in ['best', 'checkpoint'] }
        for dirr in [game_dir]:
            if not exists(dirr):
                mkdir(dirr)

    # Load in the Model, Loggers, etc:
    seed_everything(args.seed)

    # Logging and Checkpointing:
    # have logger and versions work with the seed. 
    if use_tune:
        logger=False 
        every_checkpoint_callback = False 
        callback_list = [TuneReportCallback()]
    else: 
        logger = TensorBoardLogger(game_dir, "logger")
        # have the checkpoint overwrite itself. 
        every_checkpoint_callback = ModelCheckpoint(
            filepath=game_dir,
            save_top_k=1,
            verbose=False ,
            monitor='train_loss',
            mode='min',
            prefix=''
        )
        callback_list = []
    '''best_checkpoint_callback = ModelCheckpoint(
        filepath=filenames_dict['best'],
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
        prefix=''
    )'''

    def run_lightning(config):

        if not Levine_Implementation:
            config['max_buffer_size'] *= env_params['avg_episode_length']
            train_buffer = SortedBuffer(obs_dim=env_params['STORED_STATE_SIZE'], act_dim=env_params['STORED_ACTION_SIZE'], size=config['max_buffer_size'] )
            test_buffer = SortedBuffer(obs_dim=env_params['STORED_STATE_SIZE'], act_dim=env_params['STORED_ACTION_SIZE'], size=config['batch_size']*10)

        model = LightningTemplate(game_dir, config, train_buffer, test_buffer)

        if not args.no_reload:
            # load in: 
            print('loading in from:', game_dir)
            load_name = join(game_dir, 'epoch=1723.ckpt')
            state_dict = torch.load(load_name)['state_dict']
            state_dict = {k[6:]:v for k, v in state_dict.items()}
            model.model.load_state_dict(state_dict)
            print("Loaded in Model state!")

        if args.eval_agent:
            model.eval_agent()

        else: 
            trainer = Trainer(deterministic=True, logger=logger,
                default_root_dir=game_dir, max_epochs=epochs, profiler=False,
                checkpoint_callback = every_checkpoint_callback,
                log_save_interval=1,
                callbacks=callback_list, 
                progress_bar_refresh_rate=0
            )
            trainer.fit(model)

    scheduler = ASHAScheduler(
        time_attr='epoch',
        metric="mean_reward_20_epochs",
        mode="max",
        max_t=epochs,
        grace_period=25,
        reduction_factor=4)

    reporter = CLIReporter(
        metric_columns=["loss", "mean_reward_20_epochs", "epoch"],
        )

    num_samples = 256
    if use_tune:
        tune.run(
            run_lightning,
            name=run_name,
            resources_per_trial={"cpu": 1},
            config=config,
            num_samples=num_samples,
            scheduler=scheduler,
            progress_reporter=reporter,
            verbose=1,
            fail_fast=True )

    else: 
        run_lightning(config)
        
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
    parser.add_argument('--eval_agent', type=bool, default=False,
                        help="Able to eval the agent!")
    args = parser.parse_args()
    main(args)