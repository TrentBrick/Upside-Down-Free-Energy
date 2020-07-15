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
    RingBuffer
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
            epoch=trainer.current_epoch)

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
    epochs = 500
    training_rollouts_total = 20
    training_rollouts_per_worker = training_rollouts_total//args.num_workers
    constants = dict(
        random_action_epochs = 1,
        evaluate_every = 10,
        training_rollouts_total = training_rollouts_total,
        training_rollouts_per_worker = training_rollouts_per_worker,
        num_new_rollouts = args.num_workers*training_rollouts_per_worker,
        antithetic = False,
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
            Levine_Implementation=Levine_Implementation,
            num_grad_steps = 1000
        )
        train_buffer = RingBuffer(obs_dim=env_params['STORED_STATE_SIZE'], act_dim=env_params['STORED_ACTION_SIZE'], size=config['max_buffer_size'])
        test_buffer = RingBuffer(obs_dim=env_params['STORED_STATE_SIZE'], 
            act_dim=env_params['STORED_ACTION_SIZE'], size=config['batch_size']*10)
        
    else: 
        config= dict(
        lr= tune.choice([0.0003, 0.01, 0.001]),
        batch_size = tune.choice([768, 4096]),
        max_buffer_size = tune.choice([100, 500]),
        desire_scalings = (0.02, 0.01), # reward then horizon
        discount_factor = 1.0,
        last_few = tune.choice([25, 75]),
        Levine_Implementation=Levine_Implementation,
        desired_reward_dist_beta=1,
        num_grad_steps = tune.choice([10, 100, 200])
        )
        # TODO: do I need to provide seeds to the buffer like this? 
        
    config.update(constants) 
    config.update(env_params)
    config.update(vars(args))

    config['sparse'] = tune.choice([True, False])
    config['NODE_SIZE'] = tune.choice([32, 128, 256])

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
    game_dir = join(base_game_dir, 'seed_'+str(args.seed))
    filenames_dict = { bc:join(game_dir, 'model_'+bc+'.tar') for bc in ['best', 'checkpoint'] }
    for dirr in [game_dir]:
        if not exists(dirr):
            mkdir(dirr)

    # Load in the Model, Loggers, etc:
    seed_everything(args.seed)

    # Logging and Checkpointing:
    # have logger and versions work with the seed. 
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
            train_buffer = ReplayBuffer(config['max_buffer_size'], args.seed, config['batch_size'], config['num_grad_steps'])
            test_buffer = ReplayBuffer(config['batch_size']*10, args.seed, config['batch_size'], 5)

        model = LightningTemplate(game_dir, config, train_buffer, test_buffer)

        if not args.no_reload:
            # load in: 
            model = LightningTemplate.load_from_checkpoint(filenames_dict['best'])

        trainer = Trainer(deterministic=True, logger=logger,
            default_root_dir=game_dir, max_epochs=epochs, profiler=False,
            checkpoint_callback = every_checkpoint_callback,
            log_save_interval=1,
            callbacks=[TuneReportCallback()], 
            progress_bar_refresh_rate=0
        )
        trainer.fit(model)

    scheduler = ASHAScheduler(
        time_attr='epoch',
        metric="mean_reward",
        mode="max",
        max_t=epochs,
        grace_period=10,
        reduction_factor=3)

    reporter = CLIReporter(
        metric_columns=["loss", "mean_reward", "epoch"],
        )

    num_samples = 10
    tune.run(
        run_lightning,
        resources_per_trial={"cpu": 1},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        verbose=1)
        
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
    args = parser.parse_args()
    main(args)