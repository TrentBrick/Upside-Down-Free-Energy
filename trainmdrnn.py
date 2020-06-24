""" Recurrent model training """

# rest of imports are at the bottom. This ensures they are only
# imported if this file is run directly. 
from models.mdrnn import MDRNN, gmm_loss, MDRNNCell
import torch.nn.functional as f
from utils.misc import sample_mdrnn_latent, generate_samples

def get_loss(mdrnn, latent_obs, latent_next_obs, 
             pres_action, pres_reward, 
             next_reward, terminal, device,
             include_reward = True, include_overshoot=False, 
             include_terminal = False, deterministic=True):
    # TODO: I thought for the car racer we werent predicting terminal states 
    # and also in general that we werent predicting the reward of the next state. 
    """ Compute MDRNN losses.

    The loss that is computed is:
    (GMMLoss(latent_next_obs, GMMPredicted) + MSE(reward, predicted_reward) +
         BCE(terminal, logit_terminal))
    All losses are averaged both on the batch and the 
    sequence dimensions (the two first dimensions).

    :args: 
        - latent_obs: (BATCH_SIZE, SEQ_LEN, LATENT_SIZE) torch tensor
        - latent_next_obs: (BATCH_SIZE, SEQ_LEN, LATENT_SIZE) torch tensor
        - pres_action: (BATCH_SIZE, SEQ_LEN, ACTION_SIZE) torch tensor
        - pres_reward: (BATCH_SIZE, SEQ_LEN, 1) torch tensor
        - next_reward: (BATCH_SIZE, SEQ_LEN, 1) torch tensor
        - terminal: (BATCH_SIZE, SEQ_LEN, 1) torch tensor
    
    :returns: dictionary of losses, containing the gmm, the mse, the bce and
        the averaged loss.
    """

    mse_coef = 10
    mus, sigmas, logpi, rs, ds = mdrnn(pres_action, latent_obs, pres_reward)

    # find the delta between the next observation and the present one. 
    latent_delta = latent_next_obs - latent_obs
    
    if deterministic: 
        mus = mus.squeeze(-2)
        gmm = f.mse_loss(latent_delta, mus)
        pred_latent_obs = mus + latent_obs
    else: 
        gmm = gmm_loss(latent_delta, mus, sigmas, logpi) # by default gives mean over all.
        pred_latent_obs = sample_mdrnn_latent(mus, sigmas, logpi, latent_obs)
    
    print('MSE between predicted and real latent values', 
        f.mse_loss(latent_next_obs, pred_latent_obs))

    overshoot_losses = 0
    if include_overshoot:
        # latent overshooting: 
        overshooting_horizon = 6
        min_memory_adapt_period= 0 # for zero indexing
        stop_ind = latent_obs.shape[1] - overshooting_horizon
        for i in range(min_memory_adapt_period, stop_ind):

            next_hidden = [
                        torch.zeros(1, latent_obs.shape[0], LATENT_RECURRENT_SIZE).to(device)
                        for _ in range(2)]

            #memory adapt up to this point. 
            _, _, _, _, _, next_hidden = mdrnn(pres_action[:, :i, :], 
                                    latent_obs[:, :i, :], pres_reward[:, :i, :], 
                                    last_hidden=next_hidden)

            pred_latent_obs, pred_reward = latent_obs[:, i, :].unsqueeze(1), \
                                            pres_reward[:, i, :].unsqueeze(1)

            for t in range(i, i+overshooting_horizon):
                #print('into mdrnn', pred_latent_obs.shape, pred_reward.shape)
                mus, sigmas, logpi, pred_reward, ds, next_hidden = mdrnn(pres_action[:, t, :].unsqueeze(1), 
                                    pred_latent_obs, pred_reward, last_hidden=next_hidden)
                # get next latent observation
                if deterministic: 
                    #print('latent overshooot', latent_delta[:, t, :].unsqueeze(1).shape, mus.shape)
                    overshoot_loss = f.mse_loss(latent_delta[:, t, :].squeeze(), mus.squeeze())
                    mus = mus.squeeze(-2)
                    #print('pred and mus', pred_latent_obs.shape, mus.shape)
                    pred_latent_obs = mus + pred_latent_obs.squeeze()
                else: 
                    overshoot_loss = gmm_loss(latent_delta[:, t, :].unsqueeze(1), mus, sigmas, logpi )
                    pred_latent_obs = sample_mdrnn_latent(mus, sigmas, logpi, pred_latent_obs)
                pred_latent_obs = pred_latent_obs.unsqueeze(1)
                #print('pred latent', pred_latent_obs.shape)
                pred_reward = pred_reward.squeeze()
                reward_loss = f.mse_loss(pred_reward, next_reward[:,t,:].squeeze())
                overshoot_losses += overshoot_loss+(mse_coef*reward_loss)
                pred_reward = pred_reward.unsqueeze(1).unsqueeze(2)
                   
    if include_reward:
        mse = f.mse_loss(rs, next_reward.squeeze())
    else:
        mse = 0

    if include_terminal:
        bce = f.binary_cross_entropy_with_logits(ds, terminal)
    else:
        bce = 0

    # adding coefficients to make them the same order of magnitude. 
    mse = mse_coef*mse

    loss = (gmm + bce + mse + overshoot_losses) #/ scale
    return dict(gmm=gmm, bce=bce, mse=mse, overshoot=overshoot_losses, loss=loss)

def main(args):

    # makes including the reward a default!
    if args.do_not_include_reward: 
        include_reward = False
    else:
        include_reward = True

    if args.do_not_include_overshoot: 
        include_overshoot=False
    else:
        include_overshoot=True 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # constants
    BATCH_SIZE = 1024
    SEQ_LEN = 12
    epochs = 300
    conditional=True 
    cur_best = None
    make_mdrnn_samples = True 
    if args.probabilistic:
        deterministic = False 
    else: 
        deterministic = True 

    desired_horizon = 30
    num_action_repeats = 5 # number of times the same action is performed repeatedly. 
    # this makes the environment accelerate by this many frames. 
    actual_horizon = desired_horizon//num_action_repeats
    # for plotting example horizons:
    example_length = 11
    assert example_length< SEQ_LEN, "Example length must be smaller."
    memory_adapt_period = example_length - actual_horizon

    samples_dir = join(args.logdir, 'mdrnn', 'samples')
    if not exists(samples_dir):
        mkdir(samples_dir)

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

    logger = {typee+'_'+k:[] for k in ['loss','gmm', 'mse', 'overshoot'] for typee in ['train','test']}

    if not exists(rnn_dir):
        mkdir(rnn_dir)

    mdrnn = MDRNN(LATENT_SIZE, ACTION_SIZE, LATENT_RECURRENT_SIZE, 
            NUM_GAUSSIANS_IN_MDRNN, conditional=conditional, 
            use_lstm=args.use_lstm ).to(device)

    optimizer = torch.optim.Adam(mdrnn.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    earlystopping = EarlyStopping('min', patience=100)

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
        RolloutSequenceDataset('datasets/carracing', SEQ_LEN, transform, buffer_size=200),
        batch_size=BATCH_SIZE, num_workers=10, shuffle=True, drop_last=False)
    test_loader = DataLoader(
        RolloutSequenceDataset('datasets/carracing', SEQ_LEN, transform, train=False, buffer_size=20),
        batch_size=BATCH_SIZE, num_workers=10, shuffle=True, drop_last=False)

    def to_latent(obs, rewards):
        """ Transform observations to latent space.

        :args obs: 5D torch tensor (BATCH_SIZE, SEQ_LEN, Channel Size, SIZE, SIZE)
        
        :returns: (latent_obs, latent_next_obs)
            - latent_obs: 4D torch tensor (BATCH_SIZE, SEQ_LEN, LATENT_SIZE)
        """
        with torch.no_grad():
            # eg obs shape is: [48, 257, 3, 84, 96]. this means each batch when we loop over and use
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

        if epoch !=0:
            # else has already loaded one in on init
            loader.dataset.load_next_buffer()

        cum_loss = 0
        cum_gmm = 0
        cum_bce = 0
        cum_mse = 0
        cum_overshoot = 0
        num_rollouts_shown = 0

        pbar = tqdm(total=len(loader.dataset), desc="Epoch {}".format(epoch))
        for i, data in enumerate(loader):
            obs, action, reward, terminal = [arr.to(device) for arr in data]
            cur_batch_size = terminal.shape[0]
            num_rollouts_shown+= cur_batch_size

            print('current ||| total number of rollouts shown', cur_batch_size, num_rollouts_shown)

            print('===== obs shape is:', obs.shape)

            # transform obs
            latent_obs = to_latent(obs, reward)

            #split into previous and next observations:
            latent_next_obs = latent_obs[:,1:,:].clone() #possible BUG: need to ensure these tensors are different to each other. Tensors arent being modified though so should be ok? Test it anyways.
            latent_obs = latent_obs[:,:-1,:]
            next_reward = reward[:, 1:].clone()
            pres_reward = reward[:, :-1]
            #next_action = action[:, 1:].clone()
            pres_action = action[:, :-1]
            next_terminal = terminal[:,1:]

            if train:
                losses = get_loss(mdrnn, latent_obs, latent_next_obs, pres_action, 
                                pres_reward, next_reward,
                                next_terminal, device, include_reward=include_reward, 
                                include_terminal=include_terminal, 
                                include_overshoot=include_overshoot, deterministic=deterministic)

                optimizer.zero_grad()
                losses['loss'].backward()
                # gradient clipping! From Ha. 
                torch.nn.utils.clip_grad_norm_(mdrnn.parameters(), 1000.0)
                optimizer.step()
            else:
                with torch.no_grad():
                    losses = get_loss(mdrnn, latent_obs, latent_next_obs, pres_action, 
                                    pres_reward, next_reward,
                                    next_terminal, device, include_reward=include_reward, 
                                    include_terminal=include_terminal, 
                                    include_overshoot=include_overshoot, deterministic=deterministic)

            cum_loss += losses['loss'].item() * cur_batch_size
            cum_gmm += losses['gmm'].item() * cur_batch_size
            cum_bce += losses['bce'].item() * cur_batch_size if hasattr(losses['bce'], 'item') else \
                losses['bce']
            # nice. this is better than a try statement and all on one line!
            cum_mse += losses['mse'].item() * cur_batch_size if hasattr(losses['mse'], 'item') else \
                losses['mse']
            cum_overshoot += losses['overshoot'].item() * cur_batch_size if hasattr(losses['overshoot'], 'item') else \
                losses['overshoot']

            pbar.set_postfix_str("loss={loss:10.6f} overshoot={overshoot:10.6f} "
                                "gmm={gmm:10.6f} mse={mse:10.6f} bce={bce:10.6f}".format(
                                    loss=(cum_loss /num_rollouts_shown)/SEQ_LEN, 
                                    overshoot=(cum_overshoot /num_rollouts_shown)/SEQ_LEN,
                                    gmm=(cum_gmm /num_rollouts_shown)/SEQ_LEN, # LatentSIZE
                                    mse=(cum_mse /num_rollouts_shown)/SEQ_LEN,
                                    bce=(cum_bce /num_rollouts_shown)/SEQ_LEN ))

            pbar.update(cur_batch_size)
        pbar.close()
        cum_losses = []
        for c in  [cum_loss, cum_gmm, cum_mse, cum_overshoot]:
            cum_losses.append( (c/num_rollouts_shown)/SEQ_LEN )
        if train: 
            return cum_losses # puts it on a per seq len chunk level. 
        else: 
            for_mdrnn_sampling = [obs[0,:,:,:,:], pres_reward[0,:,:], next_reward[0,:,:], latent_obs[0,:,:], latent_next_obs[0,:,:], pres_action[0,:,:]]
            return cum_losses, for_mdrnn_sampling

    train = partial(data_pass, train=True, include_reward=include_reward, include_terminal=args.include_terminal)
    test = partial(data_pass, train=False, include_reward=include_reward, include_terminal=args.include_terminal)
        
    for e in range(epochs):
        print('========== Training run')
        train_losses = train(e)
        print('========== Testing run')
        test_losses, for_mdrnn_sampling = test(e)
        train_loss = train_losses[0]
        test_loss = test_losses[0]
        scheduler.step(test_loss)
        earlystopping.step(test_loss)

        logger['train_loss'].append(train_loss)
        logger['train_gmm'].append(train_losses[1])
        logger['train_mse'].append(train_losses[2])
        logger['train_overshoot'].append(train_losses[3])
        logger['test_loss'].append(test_loss)
        logger['test_gmm'].append(test_losses[1])
        logger['test_mse'].append(test_losses[2])
        logger['test_overshoot'].append(test_losses[3])

        # write out the logger. 
        with open(logger_filename, "w") as file:
            json.dump(logger, file)

        is_best = not cur_best or test_loss < cur_best
        if is_best:
            cur_best = test_loss
        
        save_checkpoint({
            "state_dict": mdrnn.state_dict(),
            "optimizer": optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'earlystopping': earlystopping.state_dict(),
            "precision": test_loss,
            "epoch": e}, is_best, checkpoint_filename,
                        best_filename)

        # generate examples of MDRNN
        if make_mdrnn_samples: 
            generate_samples( vae, mdrnn, for_mdrnn_sampling, deterministic, 
                            samples_dir, SEQ_LEN, example_length, memory_adapt_period,
                            e, device,
                            make_vae_samples=False,
                            make_mdrnn_samples=make_mdrnn_samples, 
                            transform_obs=True  )

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
    from torchvision.utils import save_image
    import numpy as np
    import json
    from tqdm import tqdm
    from utils.misc import save_checkpoint
    from utils.misc import NUM_GAUSSIANS_IN_MDRNN, NUM_IMG_CHANNELS, ACTION_SIZE, LATENT_SIZE, LATENT_RECURRENT_SIZE, IMAGE_RESIZE_DIM, SIZE
    from utils.learning import EarlyStopping
    from utils.learning import ReduceLROnPlateau

    from data.loaders import RolloutSequenceDataset
    from models.vae import VAE
    
    parser = argparse.ArgumentParser("MDRNN training")
    parser.add_argument('--logdir', type=str,
                        help="Where things are logged and models are loaded from.")
    parser.add_argument('--no_reload', action='store_true',
                        help="Do not reload if specified.")
    parser.add_argument('--use_lstm', action='store_true',
                        help="Use LSTM with hidden state rather than the forward model.")
    parser.add_argument('--probabilistic', action='store_true',
                        help="Have the model be probabilistic. LSTM needs to be true.")
    parser.add_argument('--do_not_include_reward', action='store_true',
                        help="If true doesn't add reward modelisation term to the loss.")
    parser.add_argument('--do_not_include_overshoot', action='store_true',
                        help="If true doesn't add an overshoot modelisation term to the loss.")
    parser.add_argument('--include_terminal', action='store_true',
                        help="Add a terminal modelisation term to the loss.")
    args = parser.parse_args()

    main(args)