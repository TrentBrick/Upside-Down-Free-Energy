""" Use the MDRNN to predict the future. """
import torch
from torch.distributions.categorical import Categorical
from torch.nn import functional as F
from torchvision import transforms
from torchvision.utils import save_image
import argparse
from os.path import join, exists
from os import mkdir
import numpy as np
from models import MDRNNCell, VAE, Controller
import gym
import gym.envs.box2d
from utils.misc import ACTION_SIZE, LATENT_SIZE, LATENT_RECURRENT_SIZE, IMAGE_RESIZE_DIM, SIZE
from utils.misc import sample_continuous_policy

## WARNING : THIS SHOULD BE REPLACE WITH PYTORCH 0.5
env = gym.make("CarRacing-v0")
env.reset()
seq_len=1000
time_limit = seq_len
dream_point = 30
device = 'cpu' #torch.device('cuda' if torch.cuda.is_available() else 'cpu')

a_rollout = sample_continuous_policy(env.action_space, seq_len, 1. / 50)

mdir = 'exp_dir'
# load in the VAE and MDRNN: 
vae_file, rnn_file, ctrl_file = \
    [join(mdir, m, 'best.tar') for m in ['vae', 'mdrnn', 'ctrl']]

assert exists(vae_file) and exists(rnn_file),\
    "Either vae or mdrnn is untrained or the file is in the wrong place."

vae_state, rnn_state = [
    torch.load(fname, map_location={'cuda:0': str(device)})
    for fname in (vae_file, rnn_file)]

for m, s in (('VAE', vae_state), ('MDRNN', rnn_state)):
    print("Loading {} at epoch {} "
        "with test loss {}".format(
            m, s['epoch'], s['precision']))

vae = VAE(3, LATENT_SIZE).to(device)
vae.load_state_dict(vae_state['state_dict'])

mdrnn = MDRNNCell(LATENT_SIZE, ACTION_SIZE, LATENT_RECURRENT_SIZE, 5).to(device)
mdrnn.load_state_dict(
    {k.strip('_l0'): v for k, v in rnn_state['state_dict'].items()})

'''controller = Controller(LATENT_SIZE, LATENT_RECURRENT_SIZE, ACTION_SIZE).to(device)

# load controller if it was previously saved
if exists(ctrl_file):
    ctrl_state = torch.load(ctrl_file, map_location={'cuda:0': str(device)})
    print("Loading Controller with reward {}".format(
        ctrl_state['reward']))
    controller.load_state_dict(ctrl_state['state_dict'])'''

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_RESIZE_DIM, IMAGE_RESIZE_DIM)),
    transforms.ToTensor()
])

def rollout(render=False):
    """ Execute a rollout and returns minus cumulative reward.

    Load :params: into the controller and execute a single rollout. This
    is the main API of this class.

    :args params: parameters as a single 1D np array

    :returns: minus cumulative reward
    # Why is this the minus cumulative reward?!?!!?
    """
    print('a rollout dims', len(a_rollout))

    #env.seed(int(rand_env_seed)) # ensuring that each rollout has a differnet random seed. 
    obs = env.reset()

    # This first render is required !
    env.render()

    next_hidden = [
        torch.zeros(1, LATENT_RECURRENT_SIZE).to(device)
        for _ in range(2)]

    cumulative = 0
    i = 0
    rollout_dict = {k:[] for k in ['obs', 'rew', 'act', 'term']}

    obs = transform(obs).unsqueeze(0).to(device)
    mu, logsigma = vae.encoder(obs)
    next_z =  mu + logsigma.exp() * torch.randn_like(mu)

    while True:
        #print(i)

        action = torch.Tensor(a_rollout[i]).to(device).unsqueeze(0)

        #print('into mdrnn',action.shape, next_z.shape, next_hidden[0].shape)

        # commented out reward and done. 
        mus, sigmas, logpi, _, _, next_hidden = mdrnn(action, next_z, next_hidden)

        # decode current z to see what it looks like. 
        recon_obs = vae.decoder(next_z)

        if i>dream_point: 
            if type(obs) != torch.Tensor:
                obs = transform(obs).unsqueeze(0)
            to_save = torch.cat([obs, recon_obs.cpu()], dim=0)
            #print(to_save.shape)
            # .view(args.batch_size*2, 3, IMAGE_RESIZE_DIM, IMAGE_RESIZE_DIM)
            save_image(to_save,
                        join(mdir, 'dream/sample_' + str(i) + '.png'))

        obs, reward, done, _ = env.step(a_rollout[i])
        
        if i < dream_point or np.random.random()>0.95: 
            print('using real obs at point:', i)
            obs = transform(obs).unsqueeze(0).to(device)
            mu, logsigma = vae.encoder(obs)
            next_z =  mu + logsigma.exp() * torch.randn_like(mu)
        else: 
            # sample the next z. 
            g_probs = Categorical(probs=torch.exp(logpi).permute(0,2,1))
            which_g = g_probs.sample()
            #print(logpi.shape, mus.permute(0,2,1)[:,which_g].shape ,mus[:,:,which_g].shape,  which_g, mus.shape )
            #print(mus.squeeze().permute(1,0).shape, which_g.permute(1,0))
            mus_g, sigs_g = torch.gather(mus.squeeze(), 0, which_g), torch.gather(sigmas.squeeze(), 0, which_g)
            #print(mus_g.shape)
            next_z = mus_g + sigs_g * torch.randn_like(mus_g)
            #print(next_z.shape)
        
        #for key, var in zip(['obs', 'rew', 'act', 'term'], [obs,reward, action, done]):
        #    rollout_dict[key].append(var)

        if render:
            env.render()

        cumulative += reward
        if done or i >= time_limit:

            return - cumulative
            
        i += 1


rollout()

'''if __name__ == "__main__":'''