# pylint: disable=no-member
import gym
from torchvision import transforms
from torchvision.utils import save_image
import torch 
import pickle
transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

env = gym.make('LunarLander-v2')
obs = env.reset()
print(obs.shape)
print('action space', env.action_space.__dict__)
frames = []
states = []
for t in range(1000):
    im_frame = env.render(mode='rgb_array')
    env.viewer.window.dispatch_events()
    observation, reward, done, info = env.step(env.action_space.sample()) # take a random action
    print('obs shape', observation.shape)
    print(t, reward, observation, observation.shape, info, im_frame.shape, 'sample action', env.action_space.sample(), type(env.action_space.sample()))
    if t< 1000:
        im_frame = im_frame[200:800, 200:800, :]
        print('im frame is:', im_frame.shape)
        frames.append(  transform_train(im_frame)  )
        states.append((observation, reward))
    else:
        done=True
    if done: 
        frames = torch.stack(frames)
        print('obs shape', frames.shape)
        save_image(frames, 'testing_scripts/gym_env_imgs.png')
        print('SAVED OBS OUT!')
        pickle.dump(states, open('testing_scripts/gym_states.pkl', 'wb'))
        break
env.close()

