""" Recurrent model training """

# rest of imports are at the bottom. This ensures they are only
# imported if this file is run directly. 
from models.mdrnn import MDRNN, gmm_loss
import torch.nn.functional as f
from utils.misc import sample_mdrnn_latent

def get_loss(mdrnn, latent_obs, latent_next_obs, action, pres_reward, next_reward, terminal,
             include_reward = True, include_terminal = False):
    # TODO: I thought for the car racer we werent predicting terminal states 
    # and also in general that we werent predicting the reward of the next state. 
    """ Compute MDRNN losses.


    The loss that is computed is:
    (GMMLoss(latent_next_obs, GMMPredicted) + MSE(reward, predicted_reward) +
         BCE(terminal, logit_terminal))
    All losses are averaged both on the batch and the 
    sequence dimensions (the two first dimensions).

    :args latent_obs: (BATCH_SIZE, SEQ_LEN, LATENT_SIZE) torch tensor
    :args action: (BATCH_SIZE, SEQ_LEN, ACTION_SIZE) torch tensor
    :args reward: (BATCH_SIZE, SEQ_LEN, 1) torch tensor
    :args latent_next_obs: (BATCH_SIZE, SEQ_LEN, LATENT_SIZE) torch tensor

    :returns: dictionary of losses, containing the gmm, the mse, the bce and
        the averaged loss.
    """
    
    mus, sigmas, logpi, rs, ds = mdrnn(action, latent_obs, pres_reward)

    # find the delta between the next observation and the present one. 
    latent_delta = latent_next_obs - latent_obs
    gmm = gmm_loss(latent_delta, mus, sigmas, logpi) # by default gives mean over all.

    pred_latent_obs = sample_mdrnn_latent(mus, sigmas, logpi, latent_obs)

    print('MSE between predicted and real latent values', 
        f.mse_loss(latent_next_obs, pred_latent_obs))

    if include_reward:
        mse = f.mse_loss(rs, next_reward.squeeze())
    else:
        mse = 0

    if include_terminal:
        bce = f.binary_cross_entropy_with_logits(ds, terminal)

    else:
        bce = 0

    # adding coefficients to make them the same order of magnitude. 
    mse = 100*mse

    loss = (gmm + bce + mse) #/ scale
    return dict(gmm=gmm, bce=bce, mse=mse, loss=loss)

def main(args):

    # makes including the reward a default!
    if args.do_not_include_reward: 
        include_reward = False
    else: 
        include_reward = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # constants
    BATCH_SIZE = 256
    SEQ_LEN = 16
    epochs = 30
    conditional=True 
    cur_best = None

    # Loading VAE
    vae_file = join(args.logdir, 'vae', 'best.tar')
    assert exists(vae_file), "No trained VAE in the logdir..."
    state = torch.load(vae_file)
    print("Loading VAE at epoch {} "
        "with test error {}".format(
            state['epoch'], state['precision']))

    vae = VAE(NUM_IMG_CHANNELS, LATENT_SIZE, conditional=conditional).to(device)
    vae.load_state_dict(state['state_dict'])

    # Loading model
    rnn_dir = join(args.logdir, 'mdrnn')
    best_filename = join(rnn_dir, 'best.tar')
    checkpoint_filename = join(rnn_dir, 'checkpoint.tar')
    logger_filename = join(rnn_dir, 'logger.json')

    logger = {k:[] for k in ['train_loss','test_loss']}

    if not exists(rnn_dir):
        mkdir(rnn_dir)

    mdrnn = MDRNN(LATENT_SIZE, ACTION_SIZE, LATENT_RECURRENT_SIZE, NUM_GAUSSIANS_IN_MDRNN, conditional=conditional ).to(device)

    optimizer = torch.optim.Adam(mdrnn.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    earlystopping = EarlyStopping('min', patience=30)

    if exists(best_filename) and not args.no_reload:
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
        batch_size=BATCH_SIZE, num_workers=10, shuffle=True, drop_last=True)
    test_loader = DataLoader(
        RolloutSequenceDataset('datasets/carracing', SEQ_LEN, transform, train=False, buffer_size=5),
        batch_size=BATCH_SIZE, num_workers=10, shuffle=True, drop_last=True)

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
                losses = get_loss(mdrnn, latent_obs, latent_next_obs, action, pres_reward, next_reward,
                                terminal, include_reward, include_terminal)

                optimizer.zero_grad()
                losses['loss'].backward()
                # gradient clipping! From Ha. 
                torch.nn.utils.clip_grad_norm_(mdrnn.parameters(), 1.0)
                optimizer.step()
            else:
                with torch.no_grad():
                    losses = get_loss(mdrnn, latent_obs, latent_next_obs, action, pres_reward, next_reward,
                                    terminal, include_reward, include_terminal)

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

    train = partial(data_pass, train=True, include_reward=include_reward, include_terminal=args.include_terminal)
    test = partial(data_pass, train=False, include_reward=include_reward, include_terminal=args.include_terminal)
        
    for e in range(epochs):
        print('========== Training run')
        train_loss = train(e)
        print('========== Testing run')
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

if __name__ == '__main__':
    import argparse
    from functools import partial
    from os.path import join, exists
    from os import mkdir
    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms
    import numpy as np
    import json
    from tqdm import tqdm
    from utils.misc import save_checkpoint
    from utils.misc import NUM_GAUSSIANS_IN_MDRNN, NUM_IMG_CHANNELS, ACTION_SIZE, LATENT_SIZE, LATENT_RECURRENT_SIZE, IMAGE_RESIZE_DIM, SIZE
    from utils.learning import EarlyStopping
    ## WARNING : THIS SHOULD BE REPLACED WITH PYTORCH 0.5
    from utils.learning import ReduceLROnPlateau

    from data.loaders import RolloutSequenceDataset
    from models.vae import VAE
    
    parser = argparse.ArgumentParser("MDRNN training")
    parser.add_argument('--logdir', type=str,
                        help="Where things are logged and models are loaded from.")
    parser.add_argument('--no_reload', action='store_true',
                        help="Do not reload if specified.")
    parser.add_argument('--do_not_include_reward', action='store_true',
                        help="Add a reward modelisation term to the loss.")
    parser.add_argument('--include_terminal', action='store_true',
                        help="Add a terminal modelisation term to the loss.")
    args = parser.parse_args()

    main(args)