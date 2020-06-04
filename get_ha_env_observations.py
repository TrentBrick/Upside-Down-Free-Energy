from ha_env import make_env
from PIL import Image
import pickle 
import numpy as np 
obs_list = []

env = make_env('carracing')

env.render('rgb_array')
obs = env.reset()
env.viewer.window.dispatch_events()
for i in range(500):

    if i > 490:

        obs_list.append(obs)
    
    obs, reward, done, _ = env.step( np.array([0.1,0.1,0.1]))
    env.viewer.window.dispatch_events()


pickle.dump(obs_list, open('ha_env_images.pkl', 'wb'))
