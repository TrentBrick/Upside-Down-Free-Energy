""" Joint training of  """
import argparse
from functools import partial
from os.path import join, exists
from os import mkdir
import torch
import torch.nn.functional as f
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
import json
from tqdm import tqdm
from joint_utils import generate_rollouts, train_controller
from controller_model import flatten_parameters
from utils.misc import save_checkpoint, load_parameters, flatten_parameters
from utils.misc import RolloutGenerator, ACTION_SIZE, LATENT_SIZE, LATENT_RECURRENT_SIZE, IMAGE_RESIZE_DIM, SIZE
from utils.learning import EarlyStopping
from utils.learning import ReduceLROnPlateau
from ha_es import CMAES
import sys
from data.loaders import RolloutSequenceDataset
from models.vae import VAE
from models.mdrnn import MDRNN, gmm_loss
from models import Controller
import cma
from multiprocessing import cpu_count
from trainvae import loss_function as trainvae_loss_function
from trainmdrnn import get_loss as trainmdrnn_loss_function

def main(args):

    assert args.num_workers <= cpu_count(), "Providing too many workers!" 

    conditional =True
    use_ctrl_pretrain = False # as it is not trained to be conditioned on rewards!
    make_vae_samples = True 
    # used for saving which models are the best based upon their train performance. 
    vae_n_mdrnn_cur_best=None
    ctrl_cur_best_rewards = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    use_feef = False

    # constants
    BATCH_SIZE = 48
    SEQ_LEN = 128 #256
    epochs = 50
    time_limit =1000 # for the rollouts generated
    num_vae_mdrnn_training_rollouts_per_worker = 2

    kl_tolerance=0.5
    kl_tolerance_scaled = torch.Tensor([kl_tolerance*LATENT_SIZE]).to(device)

    include_reward = conditional # this is very important for the conditional 
    include_terminal = False

    model_types = ['ctrl', 'vae', 'mdrnn']
    # Init save filenames 
    joint_dir = join(args.logdir, 'joint')
    filenames_dict = {m+'_'+bc:join(joint_dir, m+'_'+bc+'.tar') for bc in ['best', 'checkpoint'] \
                                                    for m in model_types}
    # make directories if they dont exist
    samples_dir = join(joint_dir, 'samples')
    for dirr in [joint_dir, samples_dir]:
        if not exists(dirr):
            mkdir(dirr)

    logger_filename = join(joint_dir, 'logger.txt')

    # init models
    vae = VAE(3, LATENT_SIZE, conditional).to(device)
    mdrnn = MDRNN(LATENT_SIZE, ACTION_SIZE, LATENT_RECURRENT_SIZE, 5,conditional).to(device)
    controller = Controller(LATENT_SIZE, LATENT_RECURRENT_SIZE, ACTION_SIZE, conditional).to(device)

    # TODO: consider learning these parameters with different optimizers and learning rates for each network. 
    optimizer = torch.optim.Adam(list(vae.parameters())+list(mdrnn.parameters()), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    earlystopping = EarlyStopping('min', patience=30) # NOTE: this needs to be esp high as the epochs are heterogenous buffers!! not all data. 

    #model_variables_dict = dict(controller=controller, vae=vae, mdrnn=mdrnn)
    model_variables_dict_NO_CTRL = dict(vae=vae, mdrnn=mdrnn) # used for when the controller is trained and these models are provided. 
    # Loading in trained models: 

    if not args.no_reload:
        for name, model_var in model_variables_dict_NO_CTRL.items():
        # Loading from previous joint training or from all separate training
        # TODO: enable some to come from pretraining and others to be fresh. 
            if not args.reload_from_pretrain: # dont come from separate pretraining steps. 
                load_file = filenames_dict[name+'_best']
            else: 
                load_file = join(args.logdir, name, 'best.tar')
            if exists(load_file):
                state = torch.load(load_file, map_location={'cuda:0': str(device)})
                #if name != 'ctrl':
                print("Loading model_type {} at epoch {} "
                    "with test error {}".format(name,
                        state['epoch'], state['precision']))

                model_var.load_state_dict(state['state_dict'])

                # load in the training loop states only if all jointly trained together before
                if not args.reload_from_pretrain and name =='mdrnn': 
                    # this info is currently saved with the vae and mdrnn on their own pulling from mdrnn as its currently the last.
                    print(' Loading in training state info (eg optimizer state) from last model in iter list:', name)
                    optimizer.load_state_dict(state["optimizer"])
                    scheduler.load_state_dict(state['scheduler'])
                    earlystopping.load_state_dict(state['earlystopping'])
                    vae_n_mdrnn_cur_best = state['precision']
            else: 
                print('trying to load file at:', load_file, "but couldnt find it so starting fresh")

                for model_var, model_name in zip([vae, mdrnn],['vae', 'mdrnn']):
                    save_checkpoint({
                        "state_dict": model_var.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'earlystopping': earlystopping.state_dict(),
                        "precision": None,
                        "epoch": -1}, True, filenames_dict[model_name+'_checkpoint'],
                                    filenames_dict[model_name+'_best'])

        # load in the controller 
        if use_ctrl_pretrain:
            with open('es_log/carracing.cma.12.64.best.json', 'r') as f:
                ctrl_params = json.load(f)
            print("Loading in the pretrained best controller model, its average eval score was:", ctrl_params[1])
            controller = load_parameters(ctrl_params[0], controller)
            ctrl_cur_best_rewards = ctrl_params[1]

        else: 
            if exists(filenames_dict['ctrl_best']): # loading in the checkpoint
                print('loading in the best controller')
                state = torch.load(filenames_dict['ctrl_best'], map_location={'cuda:0': str(device)})
                ctrl_cur_best_rewards = state['reward']
                ctrl_cur_best_feef = state['feef']
                controller.load_state_dict(state['state_dict'])
                print("Previous best reward was {} and best FEEF {}...".format(ctrl_cur_best_rewards, ctrl_cur_best_feef))
                print("At epoch:", state['epoch'])

    # never need gradients with controller for evo methods. 
    # NOTE: if I stop using evolutionary approach this will need to change. 
    controller.eval()

    # dont need as saving the observations after their transforms in the rollout itself. 
    #transform = transforms.Lambda( lambda x: np.transpose(x, (0, 3, 1, 2)) / 255)

    vae_output_names = ['encoder_mu', 'encoder_logsigma', 'latent_s', 'decoder_mu', 'decoder_logsigma']

    def run_vae(obs, rewards):
        # TODO: update this documentation. 
        """ Transform observations to latent space.

        :args obs: 5D torch tensor (BATCH_SIZE, SEQ_LEN, ACTION_SIZE, SIZE, SIZE)
        
        :returns: (latent_obs, latent_next_obs)
            - latent_obs: 4D torch tensor (BATCH_SIZE, SEQ_LEN, LATENT_SIZE)
        """

        # TODO: make this more pythonic and efficient. shouldnt have to loop over the VAE outputs. 
        vae_res_dict = {n:[] for n in vae_output_names}
        for x, r in zip(obs, rewards):

            # the rollout generator returns observations that have already been resized and VAE transformed
            #x = f.upsample(x.view(-1, 3, 84, SIZE), size=IMAGE_RESIZE_DIM, 
            #               mode='bilinear', align_corners=True)

            vae_outputs = vae(x, r)
            for ind, n in enumerate(vae_output_names):
                vae_res_dict[n].append(vae_outputs[ind])

        # stacking everything.
        for k in vae_res_dict.keys():
            vae_res_dict[k] = torch.stack(vae_res_dict[k])
        
        return vae_res_dict

    # Reconstruction + KL divergence losses summed over all elements and batch
    def vae_loss_function(real_obs, vae_res_dict):
        """ VAE loss function 
        Images (recon_x and x) are: (BATCH_SIZE, SEQ_LEN, NUM_CHANNELS, IMG_RESIZE, IMG_RESIZE)
        mu and logsigma: (BATCH_SIZE, SEQ_LEN, LATENT_SIZE)
        treating time independently here. 
        """

        # flatten the batch and seq length tensors.
        flat_tensors = [ vae_res_dict[k].flatten(end_dim=1) for k in ['encoder_mu', 'encoder_logsigma', 'decoder_mu', 'decoder_logsigma']]
        vae_loss, recon, kld = trainvae_loss_function(real_obs.flatten(end_dim=1), *flat_tensors, kl_tolerance_scaled)
        
        return dict(loss=vae_loss, recon=recon, kld=kld)

    def data_pass(epoch, train): # pylint: disable=too-many-locals
        """ One pass through the data """
        
        if train:
            loader = train_loader
            for model_var in [vae, mdrnn]:
                model_var.train()
            
        else:
            loader = test_loader
            for model_var in [vae, mdrnn]:
                model_var.eval()

        pbar = tqdm(total=len(loader.dataset), desc="Epoch {}".format(epoch))

        cumloss_dict = {n:0 for n in ['loss', 'loss_vae', 'loss_mdrnn','kld', 'recon', 'gmm', 'bce', 'mse']}

        def forward_and_loss():
            # transform obs
            vae_res_dict = run_vae(obs, reward)
            vae_loss_dict = vae_loss_function(obs, vae_res_dict)

            #split into previous and next observations:
            latent_next_obs = vae_res_dict['latent_s'][:,1:,:].clone() #possible BUG: need to ensure these tensors are different to each other. Tensors arent being modified though so should be ok? Test it anyways.
            latent_obs = vae_res_dict['latent_s'][:,:-1,:]

            next_reward = reward[:, 1:,:].clone()
            pres_reward = reward[:, :-1,:]

            mdrnn_loss_dict = trainmdrnn_loss_function(mdrnn, latent_obs, latent_next_obs, action, 
                                pres_reward, next_reward,
                                terminal, include_reward, include_terminal )

            return vae_loss_dict, mdrnn_loss_dict

        for i, data in enumerate(loader):
            obs, action, reward, terminal = [arr.to(device) for arr in data]

            if train:

                vae_loss_dict, mdrnn_loss_dict = forward_and_loss()
                total_loss = vae_loss_dict['loss'] + mdrnn_loss_dict['loss']

                # taking grad step after every batch. 
                optimizer.zero_grad()
                total_loss.backward()
                # TODO: consider adding gradient clipping like Ha.  
                # torch.nn.utils.clip_grad_norm_(list(vae.parameters())+list(mdrnn.parameters()), 100.0)
                optimizer.step()
            else:
                with torch.no_grad():
                    vae_loss_dict, mdrnn_loss_dict = forward_and_loss()
                    #total_loss = vae_loss_dict['loss'] + mdrnn_loss_dict['loss']

            # add to cumulative losses
            for k in cumloss_dict.keys():
                for loss_dict in [vae_loss_dict, mdrnn_loss_dict]:
                    if k in loss_dict.keys():
                        cumloss_dict[k] += loss_dict[k].item() if hasattr(loss_dict[k], 'item') else \
                                                loss_dict[k]
            # separate vae and mdrnn losses: 
            cumloss_dict['loss_vae'] += vae_loss_dict['loss'].item()
            cumloss_dict['loss_mdrnn'] += mdrnn_loss_dict['loss'].item()

            # TODO: make this much more modular. 
            postfix_str = ""
            for k,v in cumloss_dict.items():
                v = v / (i + 1)
                postfix_str+= k+'='+str(round(v,4))+', '
            pbar.set_postfix_str(postfix_str)
            pbar.update(BATCH_SIZE)
        pbar.close()

        # puts losses on a per element level.
        cumloss_dict = {k: (v*BATCH_SIZE) / len(loader.dataset) for k, v in cumloss_dict.items()}

        if train: 
            return cumloss_dict 
        else: 
            return cumloss_dict, obs[0,:,:,:,:], reward[0,:,:]
            # return the last observation and reward to generate the VAE examples. 

    train = partial(data_pass, train=True)
    test = partial(data_pass, train=False)

    # TODO: take in the final sigma from the last epoch. 
    sigma_init=0.1
    num_params = len( flatten_parameters(controller.parameters()) )
    population_size = args.num_workers*args.num_trials_per_worker
    # init here so the learning memory is retained over time. 
    es = CMAES(num_params,
            sigma_init=sigma_init,
            popsize=population_size, 
            init_params= flatten_parameters(controller.parameters()),
            invert_reward=False)

    ################## Main Training Loop ############################

    for e in range(epochs):
        print('====== New Epoch: ',e)
        ## run the current policy with the current VAE and MDRNN
        # TODO: implement memory buffer as data doesnt need to be on policy. 
        # TODO: sample a set of parameters from the controller rather than just using the same one.
        print('====== Generating Rollouts to train the MDRNN and VAE') 
        # TODO: dont feed in similar time sequences, have some gap between them or slicing of them.
        # TODO: get rollouts from the agent CMA-ES evaluation and use them here for training. 
        train_dataset, test_dataset = generate_rollouts(flatten_parameters(controller.parameters()), 
                SEQ_LEN, time_limit, joint_dir, num_rolls_per_worker=num_vae_mdrnn_training_rollouts_per_worker, 
                num_workers=args.num_workers, joint_file_dir=True, transform=None )
                # joint_file_dir is to direct the workers to load from the joint directory which is flat. 
    
        # TODO: ensure these workers are freed up after the vae/mdrnn training is Done. 
        train_loader = DataLoader(train_dataset,
            batch_size=BATCH_SIZE, num_workers=0, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_dataset,
            batch_size=BATCH_SIZE, num_workers=0, drop_last=True)
        print('====== Starting Training of VAE and MDRNN')
        # train VAE and MDRNN. uses partial(data_pass)
        train_loss_dict = train(e)
        print('====== Done Training VAE and MDRNN')
        # returns the last ones in order to produce samples!
        test_loss_dict, last_test_observations, last_test_rewards = test(e)
        print('====== Done Testing VAE and MDRNN')
        scheduler.step(test_loss_dict['loss'])

        # checkpointing the model: 
        # needs to be here so that the policy learning workers below can load in the new parameters.
        is_best = not vae_n_mdrnn_cur_best or test_loss_dict['loss'] < vae_n_mdrnn_cur_best
        if is_best:
            vae_n_mdrnn_cur_best = test_loss_dict['loss']
        for model_var, model_name in zip([vae, mdrnn],['vae', 'mdrnn']):
            save_checkpoint({
                "state_dict": model_var.state_dict(),
                "optimizer": optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'earlystopping': earlystopping.state_dict(),
                "precision": test_loss_dict['loss_'+model_name],
                "epoch": e}, is_best, filenames_dict[model_name+'_checkpoint'],
                            filenames_dict[model_name+'_best'])
        print('====== Done Saving VAE and MDRNN')

        # generating and saving VAE samples
        if make_vae_samples:
            with torch.no_grad():
                # get test samples
                encoder_mu, encoder_logsigma, latent_s, decoder_mu, decoder_logsigma = vae(last_test_observations, last_test_rewards)
                recon_batch = decoder_mu + (decoder_logsigma.exp() * torch.randn_like(decoder_mu))
                recon_batch = recon_batch.view(recon_batch.shape[0], 3, IMAGE_RESIZE_DIM, IMAGE_RESIZE_DIM)
                decoder_mu = decoder_mu.view(decoder_mu.shape[0], 3, IMAGE_RESIZE_DIM, IMAGE_RESIZE_DIM)
                to_save = torch.cat([last_test_observations.cpu(), recon_batch.cpu(), decoder_mu.cpu()], dim=0)
                print('to save shape', to_save.shape)
                save_image(to_save,
                        join(samples_dir, 'sample_' + str(e) + '.png'))
        print('====== Done Generating VAE Samples')

        # TODO: generate MDRNN examples. 

        # train the controller/policy using the updated VAE and MDRNN
        es, best_params, best_feef, best_reward = train_controller(es, flatten_parameters(controller.parameters()), joint_dir, 
            args.gamename, args.num_episodes, args.num_workers, args.num_trials_per_worker,
            args.num_generations_per_epoch, seed_start=None, time_limit=time_limit, use_feef=use_feef )
        print('====== Done Training Controller Samples')

        controller = load_parameters(best_params, controller)

        # checkpointing controller:
        is_best = not ctrl_cur_best_rewards or ctrl_cur_best_rewards > best_reward
        if is_best:
            ctrl_cur_best_rewards = best_reward
        
        save_checkpoint({
            "state_dict": controller.state_dict(),
            "feef": best_feef,
            "reward": best_reward,
            "epoch": e}, is_best, filenames_dict['ctrl_checkpoint'],
                        filenames_dict['ctrl_best'])

        test_loss_dict['best_reward_ctrl'] = best_reward
        test_loss_dict['best_feef_ctrl'] = best_feef
        test_loss_dict['loss'] += best_feef

        # header at the top of logger file
        if not exists(logger_filename): 
            header_string = ""
            for loss_dict, train_or_test in zip([train_loss_dict, test_loss_dict], ['train', 'test']):
                for k in loss_dict.keys():
                    header_string+=train_or_test+'_'+k+' '
            header_string+= '\n'
            with open(logger_filename, "w") as file:
                file.write(header_string) 

        # write out all of the logger losses.
        with open(logger_filename, "a") as file:
            log_string = ""
            for loss_dict in [train_loss_dict, test_loss_dict]:
                for k, v in loss_dict.items():
                    log_string += str(v)+' '
            log_string+= '\n'
            file.write(log_string)
        print('====== Done Writing out to the Logger')
        # accounts for all of the different module losses. 
        earlystopping.step(test_loss_dict['loss'])

        if earlystopping.stop:
            print("End of Training because of early stopping at epoch {}".format(e))
            break

if __name__ =='__main__':
    parser = argparse.ArgumentParser("Joint training")
    parser.add_argument('--logdir', type=str, default='exp_dir',
                        help="Where things are logged and models are loaded from.")
    parser.add_argument('--gamename', type=str, default='carracing',
                        help="What Gym environment to train in.")
    parser.add_argument('--reload_from_pretrain', action='store_true',
                        help="Do not reload if specified.")
    parser.add_argument('--no_reload', action='store_true',
                        help="Do not reload if specified.")

    # Controller training arguments
    parser.add_argument('--num_generations_per_epoch', type=int, help='Number of generations of CMA-ES per epoch.',
                        default=5)
    parser.add_argument('--num_episodes', type=int, default=8 ,help='Number of samples rollouts to evaluate each agent')
    parser.add_argument('--num_trials_per_worker', type=int, default=1, help='Population size.')
    parser.add_argument('--num_workers', type=int, help='Maximum number of workers.',
                        default=16)
    parser.add_argument('--target_return', type=float, help='Stops once the return '
                        'gets above target_return')
    parser.add_argument('--display', action='store_true', help="Use progress bars if "
                        "specified.")
    args = parser.parse_args()
    main(args)