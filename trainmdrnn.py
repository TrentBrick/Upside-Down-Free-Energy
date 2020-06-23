""" Recurrent model training """

# rest of imports are at the bottom. This ensures they are only
# imported if this file is run directly. 
from models.mdrnn import MDRNN, gmm_loss, MDRNNCell
import torch.nn.functional as f
from utils.misc import sample_mdrnn_latent

def get_loss(mdrnn, latent_obs, latent_next_obs, 
             pres_action, pres_reward, 
             next_reward, terminal, device,
             include_reward = True, include_overshoot=True, 
             include_terminal = False):
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

    mse_coef = 100
    deterministic = True
    
    mus, sigmas, logpi, rs, ds = mdrnn(pres_action, latent_obs, pres_reward)

    # find the delta between the next observation and the present one. 
    latent_delta = latent_next_obs - latent_obs
    
    if deterministic: 
        gmm = f.mse_loss(latent_delta, mus))
        pred_latent_obs = mus + latent_obs
        print('deterministic outputs', mus.shape, pred_latent_obs.shape, latent_obs.shape)
    else: 
        gmm = gmm_loss(latent_delta, mus, sigmas, logpi) # by default gives mean over all.
        pred_latent_obs = sample_mdrnn_latent(mus, sigmas, logpi, latent_obs)
    
    print('MSE between predicted and real latent values', 
        f.mse_loss(latent_next_obs, pred_latent_obs))

    overshoot_losses = 0
    if include_overshoot:
        # latent overshooting: 
        overshooting_horizon = 6
        min_memory_adapt_period= 4 # for zero indexing
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
                mus, sigmas, logpi, pred_reward, ds, next_hidden = mdrnn(pres_action[:, t, :].unsqueeze(1), 
                                    pred_latent_obs, pred_reward, last_hidden=next_hidden)
                # get next latent observation
                if deterministic: 
                    pred_latent_obs = mus + latent_obs
                else: 
                    pred_latent_obs = sample_mdrnn_latent(mus, sigmas, logpi, pred_latent_obs)
                # log this one
                overshoot_loss = gmm_loss(latent_delta[:, t, :].unsqueeze(1), mus, sigmas, logpi )
                pred_reward = pred_reward.squeeze()
                reward_loss = f.mse_loss(pred_reward, next_reward[:,t,:].squeeze())
                overshoot_losses += overshoot_loss+(mse_coef*reward_loss)
                pred_reward = pred_reward.unsqueeze(1).unsqueeze(2)
                pred_latent_obs = pred_latent_obs.unsqueeze(1)
            
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

    mdrnn = MDRNN(LATENT_SIZE, ACTION_SIZE, LATENT_RECURRENT_SIZE, NUM_GAUSSIANS_IN_MDRNN, conditional=conditional ).to(device)

    optimizer = torch.optim.Adam(mdrnn.parameters(), lr=1e-3)
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

    if make_mdrnn_samples:
        mdrnn_cell = MDRNNCell(LATENT_SIZE, ACTION_SIZE, LATENT_RECURRENT_SIZE, NUM_GAUSSIANS_IN_MDRNN).to(device)
        if exists(best_filename) and not args.no_reload:
            mdrnn_cell.load_state_dict( 
                {k.strip('_l0'): v for k, v in rnn_state["state_dict"].items()})

        transform_for_mdrnn_samples = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((IMAGE_RESIZE_DIM, IMAGE_RESIZE_DIM)),
            transforms.ToTensor(),
            ])

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
                                include_overshoot=include_overshoot)

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
                                    include_overshoot=include_overshoot)

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
                                    loss=cum_loss / (i + 1), 
                                    overshoot=cum_overshoot / (i + 1),
                                    gmm=cum_gmm / LATENT_SIZE / (i + 1), 
                                    mse=cum_mse / (i + 1),
                                    bce=cum_bce / (i + 1) ))

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

            # need to restrict the data to a random segment. Important in cases where 
            # sequence length is too long
            start_sample_ind = np.random.randint(0, SEQ_LEN-example_length,1)[0]
            end_sample_ind = start_sample_ind+example_length

            # ensuring this is the same length as everything else. 
            for_mdrnn_sampling[0] = for_mdrnn_sampling[0][1:, :, :, :]

            last_test_observations, \
            last_test_pres_rewards, last_test_next_rewards, \
            last_test_latent_pres_obs, last_test_latent_next_obs, \
            last_test_pres_action = [var[start_sample_ind:end_sample_ind] for var in for_mdrnn_sampling]

            with torch.no_grad():

                # update MDRNN with the new samples: 
                # TODO: would this update have happened by default?? 
                mdrnn_cell.load_state_dict( 
                    {k.strip('_l0'): v for k, v in mdrnn.state_dict().items()})

                # vae of one in the future
                decoder_mu, decoder_logsigma = vae.decoder(last_test_latent_next_obs, last_test_next_rewards)
                real_next_vae_decoded_observation = decoder_mu.view(decoder_mu.shape[0], 3, IMAGE_RESIZE_DIM, IMAGE_RESIZE_DIM)
                
                # unsqueeze all to make a batch.
                # predict MDRNN one into the future
                #print('shape of actions going into mdrnn', last_test_pres_action.shape)
                mus, sigmas, logpi, rs, ds = mdrnn(last_test_pres_action.unsqueeze(0), last_test_latent_pres_obs.unsqueeze(0), last_test_pres_rewards.unsqueeze(0))
                # squeeze just the first dimension!
                mus, sigmas, logpi, rs, ds = [var.squeeze(0) for var in [mus, sigmas, logpi, rs, ds]]
                print('MDRNN Debugger shouldnt have batch dimensions. :', mus.shape, sigmas.shape, logpi.shape)
                
                print('Eval of MDRNN: Real rewards and latent squeezed or not? :', last_test_next_rewards.squeeze(), last_test_next_rewards.shape, last_test_latent_pres_obs.shape)
                #print('Predicted Rewards:', rs, rs.shape)
                
                pred_latent_obs = sample_mdrnn_latent(mus, sigmas, logpi, last_test_latent_pres_obs)

                print('shape of pred latent obs', pred_latent_obs.shape )

                # squeeze it back again:
                decoder_mu, decoder_logsigma = vae.decoder(pred_latent_obs, rs.unsqueeze(1))
                #print('shape of decoder mu', decoder_mu.shape)
                pred_next_vae_decoded_observation = decoder_mu.view(decoder_mu.shape[0], 3, IMAGE_RESIZE_DIM, IMAGE_RESIZE_DIM)
                
                # need to transform the last_test_observations here: 
                # THIS IS ONLY NEEDED HERE, NOT IN JOINT TRAINING AS IN JOINT ALL THE TRANSFORMS HAVE
                # ALREADY OCCURED. 
                last_test_observations = last_test_observations.permute(0,2,3,1) * 255
                last_test_observations = last_test_observations.cpu().numpy().astype(np.uint8)
                trans_imgs = []
                for i in range(last_test_observations.shape[0]):
                    trans_imgs.append( transform_for_mdrnn_samples(last_test_observations[i,:,:,:]) )
                last_test_observations = torch.stack(trans_imgs)

                #print('trying to save all out', last_test_observations.shape, 
                #    real_next_vae_decoded_observation.shape, 
                #    pred_next_vae_decoded_observation.shape )

                to_save = torch.cat([last_test_observations, 
                    real_next_vae_decoded_observation.cpu(), 
                    pred_next_vae_decoded_observation.cpu()], dim=0)

                print('Generating MDRNN samples of the shape:', to_save.shape)
                save_image(to_save,
                        join(samples_dir, 'next_pred_sample_' + str(e) + '.png'))

                ##################
                #print('predicting the next observation using the full RNN!!!!!!!!!!')

                # Generating multistep predictions from the first latent. 
                horizon_pred_obs = []
                mse_losses = []
                # first latent state. straight from the VAE. 
                
                horizon_next_hidden = [
                    torch.zeros(1, 1, LATENT_RECURRENT_SIZE).to(device)
                    for _ in range(2)]
    
                for t in range(example_length):
                    next_action = last_test_pres_action[t].unsqueeze(0).unsqueeze(0)
                    horizon_next_latent_state = last_test_latent_pres_obs[t].unsqueeze(0).unsqueeze(0)
                    horizon_next_reward = last_test_pres_rewards[t].unsqueeze(0).unsqueeze(0)

                    #print('going into horizon pred', next_action.shape, horizon_next_latent_state.shape, horizon_next_hidden[0].shape, horizon_next_reward.shape)
                    md_mus, md_sigmas, md_logpi, horizon_next_reward, d, horizon_next_hidden = mdrnn(next_action, 
                                horizon_next_latent_state, horizon_next_reward, horizon_next_hidden)
                    horizon_next_reward = horizon_next_reward.unsqueeze(0)
                    #print('going into sample mdrnn latent', md_mus.shape, horizon_next_latent_state.shape)
                    horizon_next_latent_state = sample_mdrnn_latent(md_mus, md_sigmas, md_logpi, horizon_next_latent_state)
                    
                    # mse between this and the real one. 
                    mse_losses.append(  f.mse_loss(last_test_latent_next_obs[t], horizon_next_latent_state) )
                    
                    #print('going into vae', horizon_next_latent_state.shape, horizon_next_reward.shape )
                    decoder_mu, decoder_logsigma = vae.decoder(horizon_next_latent_state.unsqueeze(0), horizon_next_reward.squeeze(0))
                    #print('shape of decoder mu', decoder_mu.shape)
                    pred_next_vae_decoded_observation = decoder_mu.view(decoder_mu.shape[0], 3, IMAGE_RESIZE_DIM, IMAGE_RESIZE_DIM)
                    horizon_pred_obs.append(pred_next_vae_decoded_observation)

                print('===== MSE losses between horizon prediction and real', mse_losses)

                horizon_pred_obs_given_next = torch.stack(horizon_pred_obs).squeeze()

                ##################
                #print('using full RNN!!!')

                # Generating multistep predictions from the first latent. 
                horizon_pred_obs = []
                mse_losses = []
                # first latent state. straight from the VAE. 
                #horizon_next_latent_state = last_test_latent_pres_obs[0].unsqueeze(0).unsqueeze(0)
                #horizon_next_reward = last_test_pres_rewards[0].unsqueeze(0).unsqueeze(0)
                horizon_next_hidden = [
                    torch.zeros(1, 1, LATENT_RECURRENT_SIZE).to(device)
                    for _ in range(2)]
    
                for t in range(example_length):

                    if t< memory_adapt_period:
                        # giving the real observation still 
                        horizon_next_latent_state = last_test_latent_pres_obs[t].unsqueeze(0).unsqueeze(0)
                        horizon_next_reward = last_test_pres_rewards[t].unsqueeze(0).unsqueeze(0)

                    next_action = last_test_pres_action[t].unsqueeze(0).unsqueeze(0)
                    #print('going into horizon pred', next_action.shape, horizon_next_latent_state.shape, horizon_next_hidden[0].shape, horizon_next_reward.shape)
                    md_mus, md_sigmas, md_logpi, horizon_next_reward, d, horizon_next_hidden = mdrnn(next_action, 
                                horizon_next_latent_state, horizon_next_reward, horizon_next_hidden)
                    horizon_next_reward = horizon_next_reward.unsqueeze(0)
                    #print('going into sample mdrnn latent', md_mus.shape, horizon_next_latent_state.shape)
                    horizon_next_latent_state = sample_mdrnn_latent(md_mus, md_sigmas, md_logpi, horizon_next_latent_state)
                    
                    # mse between this and the real one. 
                    mse_losses.append(  f.mse_loss(last_test_latent_next_obs[t], horizon_next_latent_state) )
                    
                    horizon_next_latent_state = horizon_next_latent_state.unsqueeze(0)
                    
                    #print('going into vae', horizon_next_latent_state.shape, horizon_next_reward.shape )
                    decoder_mu, decoder_logsigma = vae.decoder(horizon_next_latent_state, horizon_next_reward.squeeze(0))
                    horizon_next_latent_state = horizon_next_latent_state.unsqueeze(0)
                    #print('shape of decoder mu', decoder_mu.shape)
                    pred_next_vae_decoded_observation = decoder_mu.view(decoder_mu.shape[0], 3, IMAGE_RESIZE_DIM, IMAGE_RESIZE_DIM)
                    horizon_pred_obs.append(pred_next_vae_decoded_observation)

                print('===== MSE losses between horizon prediction and real', mse_losses)

                horizon_pred_obs_full_based = torch.stack(horizon_pred_obs).squeeze()

                #######################

                # Generating multistep predictions from the first latent. 
                horizon_pred_obs = []
                mse_losses = []
                # first latent state. straight from the VAE. 
                #horizon_next_latent_state = last_test_latent_pres_obs[0].unsqueeze(0)
                #horizon_next_reward = last_test_pres_rewards[0].unsqueeze(0)
                horizon_next_hidden = [
                    torch.zeros(1, LATENT_RECURRENT_SIZE).to(device)
                    for _ in range(2)]
                for t in range(example_length):

                    if t< memory_adapt_period:
                        # giving the real observation still 
                        horizon_next_latent_state = last_test_latent_pres_obs[t].unsqueeze(0)
                        horizon_next_reward = last_test_pres_rewards[t].unsqueeze(0)

                    next_action = last_test_pres_action[t].unsqueeze(0)
                    #print('going into horizon pred', next_action.shape, horizon_next_latent_state.shape, horizon_next_hidden[0].shape, horizon_next_reward.shape)
                    md_mus, md_sigmas, md_logpi, horizon_next_reward, d, horizon_next_hidden = mdrnn_cell(next_action, 
                                horizon_next_latent_state, horizon_next_hidden, horizon_next_reward)
                    horizon_next_reward = horizon_next_reward.unsqueeze(0)
                    horizon_next_latent_state = sample_mdrnn_latent(md_mus, md_sigmas, md_logpi, horizon_next_latent_state)
                    
                    # mse between this and the real one. 
                    mse_losses.append(  f.mse_loss(last_test_latent_next_obs[t], horizon_next_latent_state) )
                    horizon_next_latent_state = horizon_next_latent_state.unsqueeze(0)
                    #print('going into vae', horizon_next_latent_state.shape, horizon_next_reward.shape )
                    decoder_mu, decoder_logsigma = vae.decoder(horizon_next_latent_state, horizon_next_reward)
                    #print('shape of decoder mu', decoder_mu.shape)
                    pred_next_vae_decoded_observation = decoder_mu.view(decoder_mu.shape[0], 3, IMAGE_RESIZE_DIM, IMAGE_RESIZE_DIM)
                    horizon_pred_obs.append(pred_next_vae_decoded_observation)

                print('===== MSE losses between horizon prediction and real', mse_losses)

                horizon_pred_obs_cell_based = torch.stack(horizon_pred_obs).squeeze()

                to_save = torch.cat([last_test_observations, real_next_vae_decoded_observation.cpu(), horizon_pred_obs_given_next.cpu(),
                    horizon_pred_obs_full_based.cpu(), horizon_pred_obs_cell_based.cpu() ], dim=0)

                print('Generating MDRNN samples of the shape:', to_save.shape)
                save_image(to_save,
                        join(samples_dir, 'horizon_pred_sample_' + str(e) + '.png'))

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
                        help="If true doesn't add reward modelisation term to the loss.")
    parser.add_argument('--do_not_include_overshoot', action='store_true',
                        help="If true doesn't add an overshoot modelisation term to the loss.")
    parser.add_argument('--include_terminal', action='store_true',
                        help="Add a terminal modelisation term to the loss.")
    args = parser.parse_args()

    main(args)