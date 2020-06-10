""" Recurrent model training """
import argparse
from functools import partial
from os.path import join, exists
from os import mkdir
import torch
import torch.nn.functional as f
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import json
from tqdm import tqdm
from utils.misc import save_checkpoint
from utils.misc import ACTION_SIZE, LATENT_SIZE, LATENT_RECURRENT_SIZE, IMAGE_RESIZE_DIM, SIZE
from utils.learning import EarlyStopping
## WARNING : THIS SHOULD BE REPLACED WITH PYTORCH 0.5
from utils.learning import ReduceLROnPlateau

from data.loaders import RolloutSequenceDataset
from models.vae import VAE
from models.mdrnn import MDRNN, gmm_loss

parser = argparse.ArgumentParser("MDRNN training")
parser.add_argument('--logdir', type=str,
                    help="Where things are logged and models are loaded from.")
parser.add_argument('--noreload', action='store_true',
                    help="Do not reload if specified.")
parser.add_argument('--include_reward', action='store_true',
                    help="Add a reward modelisation term to the loss.")
parser.add_argument('--include_terminal', action='store_true',
                    help="Add a terminal modelisation term to the loss.")
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

assert args.include_reward==True, "check to ensure that the reward prediction is included. Comment this out if dont want it. "

# constants
BATCH_SIZE = 48
SEQ_LEN = 256
epochs = 30
conditional=True 

# Loading VAE
vae_file = join(args.logdir, 'vae', 'best.tar')
assert exists(vae_file), "No trained VAE in the logdir..."
state = torch.load(vae_file)
print("Loading VAE at epoch {} "
      "with test error {}".format(
          state['epoch'], state['precision']))

vae = VAE(3, LATENT_SIZE).to(device)
vae.load_state_dict(state['state_dict'])

# Loading model
rnn_dir = join(args.logdir, 'mdrnn')
best_filename = join(rnn_dir, 'best.tar')
checkpoint_filename = join(rnn_dir, 'checkpoint.tar')
logger_filename = join(rnn_dir, 'logger.json')

logger = {k:[] for k in ['train_loss','test_loss']}

if not exists(rnn_dir):
    mkdir(rnn_dir)

mdrnn = MDRNN(LATENT_SIZE, ACTION_SIZE, LATENT_RECURRENT_SIZE, 5, conditional=conditional ).to(device)

optimizer = torch.optim.Adam(mdrnn.parameters(), lr=1e-3)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
earlystopping = EarlyStopping('min', patience=30)

if exists(best_filename) and not args.noreload:
    rnn_state = torch.load(best_filename)
    print("Loading MDRNN at epoch {} "
          "with test error {}".format(
              rnn_state["epoch"], rnn_state["precision"]))
    cur_best = rnn_state["precision"]
    mdrnn.load_state_dict(rnn_state["state_dict"])
    optimizer.load_state_dict(rnn_state["optimizer"])
    scheduler.load_state_dict(rnn_state['scheduler'])
    earlystopping.load_state_dict(rnn_state['earlystopping'])


# Data Loading. Cant use previous transform directly as it is a seq len sized batch of observations!!
transform = transforms.Lambda(
    lambda x: np.transpose(x, (0, 3, 1, 2)) / 255) #why is this necessary?

# note that the buffer sizes are very small. and batch size is even smaller.
# batch size is smaller because each element is in fact 32 observations!
train_loader = DataLoader(
    RolloutSequenceDataset('datasets/carracing', SEQ_LEN, transform, buffer_size=30),
    batch_size=BATCH_SIZE, num_workers=16, shuffle=True, drop_last=True)
test_loader = DataLoader(
    RolloutSequenceDataset('datasets/carracing', SEQ_LEN, transform, train=False, buffer_size=10),
    batch_size=BATCH_SIZE, num_workers=16, drop_last=True)

# TODO: Wasted compute. split into obs and next obs only much later!!
def to_latent(obs, rewards):
    """ Transform observations to latent space.

    :args obs: 5D torch tensor (BATCH_SIZE, SEQ_LEN, Channel Size, SIZE, SIZE)
    
    :returns: (latent_obs, latent_next_obs)
        - latent_obs: 4D torch tensor (BATCH_SIZE, SEQ_LEN, LATENT_SIZE)
    """
    with torch.no_grad():
        # obs shape is: [48, 257, 3, 84, 96] currently. this means each batch when we loop over and use
        # the seq len as the batch is 257 long!
    
        latent_obs = []
        for x, r in zip(obs, rewards): # loop over the batches. 

            # reshaping the image why wasnt this part of the normal transform? cant use transform as it is applying it to a batch of seq len!!
            # 84 becasue of the trimming!
            x = f.upsample(x.view(-1, 3, 84, SIZE), size=IMAGE_RESIZE_DIM, 
                       mode='bilinear', align_corners=True)

            mu, logsigma = vae.encoder(x, r)
            latent_obs.append( mu + (logsigma.exp() * torch.randn_like(mu)) )

        latent_obs = torch.stack(latent_obs)
       
    return latent_obs

def get_loss(latent_obs, action, pres_reward, next_reward, terminal,
             latent_next_obs, include_reward: bool, include_terminal:bool):
    # TODO: I thought for the car racer we werent predicting terminal states 
    # and also in general that we werent predicting the reward of the next state. 
    """ Compute losses.

    The loss that is computed is:
    (GMMLoss(latent_next_obs, GMMPredicted) + MSE(reward, predicted_reward) +
         BCE(terminal, logit_terminal)) / (LATENT_SIZE + 2)
    The LATENT_SIZE + 2 factor is here to counteract the fact that the GMMLoss scales
    approximately linearily with LATENT_SIZE. All losses are averaged both on the
    batch and the sequence dimensions (the two first dimensions).

    :args latent_obs: (BATCH_SIZE, SEQ_LEN, LATENT_SIZE) torch tensor
    :args action: (BATCH_SIZE, SEQ_LEN, ACTION_SIZE) torch tensor
    :args reward: (BATCH_SIZE, SEQ_LEN) torch tensor
    :args latent_next_obs: (BATCH_SIZE, SEQ_LEN, LATENT_SIZE) torch tensor

    :returns: dictionary of losses, containing the gmm, the mse, the bce and
        the averaged loss.
    """
    # set LSTM to batch true instead. This does not affect the loss in any other way as I mean across the seq len and batch anyways. 
    '''latent_obs, action,\
        reward, terminal,\
        latent_next_obs = [arr.transpose(1, 0)
                           for arr in [latent_obs, action,
                                       reward, terminal,
                                       latent_next_obs]]'''
    mus, sigmas, logpi, rs, ds = mdrnn(action, latent_obs, pres_reward)
    gmm = gmm_loss(latent_next_obs, mus, sigmas, logpi) # by default gives mean over all.
    # scale = LATENT_SIZE
    if include_reward:
        mse = f.mse_loss(rs, next_reward.squeeze())
        #scale += 1
    else:
        mse = 0

    if include_terminal:
        bce = f.binary_cross_entropy_with_logits(ds, terminal)
        #scale += 1
    else:
        bce = 0

    loss = (gmm + bce + mse) #/ scale
    return dict(gmm=gmm, bce=bce, mse=mse, loss=loss)

def data_pass(epoch, train, include_reward, include_terminal): # pylint: disable=too-many-locals
    """ One pass through the data """
    if train:
        mdrnn.train()
        loader = train_loader
    else:
        mdrnn.eval()
        loader = test_loader

    loader.dataset.load_next_buffer()

    cum_loss = 0
    cum_gmm = 0
    cum_bce = 0
    cum_mse = 0

    pbar = tqdm(total=len(loader.dataset), desc="Epoch {}".format(epoch))
    for i, data in enumerate(loader):
        obs, action, reward, terminal = [arr.to(device) for arr in data]

        # transform obs
        latent_obs = to_latent(obs, reward)

        #split into previous and next observations:
        latent_next_obs = latent_obs[:,1:,:].clone() #possible BUG: need to ensure these tensors are different to each other. Tensors arent being modified though so should be ok? Test it anyways.
        latent_obs = latent_obs[:,:-1,:]

        next_reward = reward[:, 1:].clone()
        pres_reward = reward[:, :-1]
        
        if train:
            losses = get_loss(latent_obs, action, pres_reward, next_reward,
                              terminal, latent_next_obs, include_reward, include_terminal)

            optimizer.zero_grad()
            losses['loss'].backward()
            # gradient clipping! From Ha. 
            torch.nn.utils.clip_grad_norm_(mdrnn.parameters(), 1.0)
            optimizer.step()
        else:
            with torch.no_grad():
                losses = get_loss(latent_obs, action, pres_reward, next_reward,
                                  terminal, latent_next_obs, include_reward, include_terminal)

        cum_loss += losses['loss'].item()
        cum_gmm += losses['gmm'].item()
        cum_bce += losses['bce'].item() if hasattr(losses['bce'], 'item') else \
            losses['bce']
        # nice. this is better than a try statement and all on one line!
        cum_mse += losses['mse'].item() if hasattr(losses['mse'], 'item') else \
            losses['mse']

        pbar.set_postfix_str("loss={loss:10.6f} bce={bce:10.6f} "
                             "gmm={gmm:10.6f} mse={mse:10.6f}".format(
                                 loss=cum_loss / (i + 1), bce=cum_bce / (i + 1),
                                 gmm=cum_gmm / LATENT_SIZE / (i + 1), mse=cum_mse / (i + 1)))
        pbar.update(BATCH_SIZE)
    pbar.close()
    return cum_loss * BATCH_SIZE / len(loader.dataset) # puts it on a per seq len chunk level. 


train = partial(data_pass, train=True, include_reward=args.include_reward, include_terminal=args.include_terminal)
test = partial(data_pass, train=False, include_reward=args.include_reward, include_terminal=args.include_terminal)

if not cur_best:
    cur_best = None
    
for e in range(epochs):
    train_loss = train(e)
    test_loss = test(e)
    scheduler.step(test_loss)
    earlystopping.step(test_loss)

    logger['train_loss'].append(train_loss)
    logger['test_loss'].append(test_loss)

    # write out the logger. 
    with open(logger_filename, "w") as file:
        json.dump(logger, file)

    is_best = not cur_best or test_loss < cur_best
    if is_best:
        cur_best = test_loss
    
    # TODO: Do I want checkpointing every epoch? How long does saving take? 
    save_checkpoint({
        "state_dict": mdrnn.state_dict(),
        "optimizer": optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'earlystopping': earlystopping.state_dict(),
        "precision": test_loss,
        "epoch": e}, is_best, checkpoint_filename,
                    best_filename)

    if earlystopping.stop:
        print("End of Training because of early stopping at epoch {}".format(e))
        break
