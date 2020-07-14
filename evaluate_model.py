import gym
from models import UpsdModel, UpsdBehavior
import torch 
import numpy as np 
from envs import get_env_params
from control import Agent 

gamename = 'lunarlander'
game_dir = 'exp_dir/'+gamename #+'/model_checkpoint.tar'
desire_scalings = (0.02, 0.01)

desired_horizon = 230
reward_from_epoch_stats = (250, 10)
Levine_Implementation = False 

agent = Agent(gamename, game_dir, False,  
                Levine_Implementation= Levine_Implementation,
                desired_reward_stats = reward_from_epoch_stats, 
                desired_horizon = desired_horizon, 
                model_version='checkpoint', desire_scalings=desire_scalings)

seed = np.random.randint(0,1e9, 1)[0]
print('seed being used is:', seed)
output = agent.simulate(seed, return_events=False,
                                compute_feef=False,
                                num_episodes=10, 
                                render_mode=True )


