import gym
import scipy.misc
import numpy as np
from random import choice, random, randint

env = gym.make("CarRacing-v0")
# This first render is required !
#env.render()

episodes = 2
steps = 150

def get_action():
    # return choice([[1, 0, 0], [-1, 0, 0], [0, 1, 0]])
    # return [2*random()-1.05, 1, 0]
    action = env.action_space.sample()
    action[0] -= 0.05
    action[1] = 1
    action[2] = 0
    return action

for eps in range(episodes):
    env.reset()
    env.env.viewer.window.dispatch_events()
    r = 0
    for t in range(steps):
        print(t)
        action = get_action()
        obs, reward, done, _ = env.step(action)
        env.env.viewer.window.dispatch_events()
        # env.render()
        r += reward
        