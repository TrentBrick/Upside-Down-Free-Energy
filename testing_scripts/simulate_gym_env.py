import gym
from torchvision import transforms
from torchvision.utils import save_image
import torch 
transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

env = gym.make('Pendulum-v0')
obs = env.reset()
print(obs.shape)
print('action space', env.action_space.__dict__)
obs = []
for t in range(1000):
    im_frame = env.render(mode='rgb_array')
    env.viewer.window.dispatch_events()
    observation, reward, done, info = env.step(env.action_space.sample()) # take a random action
    print(t, reward, observation.shape, im_frame.shape, env.action_space.sample())
    if t< 64:
        obs.append(  transform_train(im_frame)  )
    if done: 
        break
env.close()

obs = torch.stack(obs)
save_image(obs, 'gym_env_imgs.png')