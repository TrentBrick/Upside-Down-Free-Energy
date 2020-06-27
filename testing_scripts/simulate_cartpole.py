import gym
from torchvision import transforms
from torchvision.utils import save_image
import torch 
transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

env = gym.make('CartPole-v0')
env.reset()
obs = []
for t in range(1000):
    im_frame = env.render(mode='rgb_array')
    observation, reward, done, info = env.step(env.action_space.sample()) # take a random action
    print(t, observation.shape, im_frame.shape)
    if t< 64:
        obs.append(  transform_train(im_frame)  )
    if done: 
        break
env.close()

obs = torch.stack(obs)
save_image(obs, 'cartpole_imgs.png')