""" Training VAE """
import argparse
from os.path import join, exists
from os import mkdir
import json
import torch
import torch.utils.data
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.utils import save_image
from torch.distributions.normal import Normal
from models.vae import VAE

from utils.misc import save_checkpoint
from utils.misc import LATENT_SIZE, IMAGE_RESIZE_DIM
## WARNING : THIS SHOULD BE REPLACE WITH PYTORCH 0.5
from utils.learning import EarlyStopping
from utils.learning import ReduceLROnPlateau
from data.loaders import RolloutObservationDataset

parser = argparse.ArgumentParser(description='VAE Trainer')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 1000)')
parser.add_argument('--logdir', type=str, help='Directory where results are logged')
parser.add_argument('--noreload', action='store_true',
                    help='Best model is not reloaded if specified')
parser.add_argument('--nosamples', action='store_true',
                    help='Does not save samples during training if specified')


args = parser.parse_args()
cuda = torch.cuda.is_available()

conditional = True

torch.manual_seed(123)
# Fix numeric divergence due to bug in Cudnn
# good whenever the input sizes do not vary. 
# thus finds optimal algorithm for faster runtime
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if cuda else "cpu")

transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_RESIZE_DIM, IMAGE_RESIZE_DIM)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
# random horizontal flip for extra data augmentation, nice. 

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_RESIZE_DIM, IMAGE_RESIZE_DIM)),
    transforms.ToTensor(),
])

# each call to _get_ gives a single observation. 
dataset_train = RolloutObservationDataset('datasets/carracing',
                                          transform_train, train=True)
dataset_test = RolloutObservationDataset('datasets/carracing',
                                         transform_test, train=False)
train_loader = torch.utils.data.DataLoader(
    dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=16)
test_loader = torch.utils.data.DataLoader(
    dataset_test, batch_size=args.batch_size, shuffle=True, num_workers=16)


vae = VAE(3, LATENT_SIZE, conditional=conditional).to(device) # latent size. 
optimizer = optim.Adam(vae.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
earlystopping = EarlyStopping('min', patience=30)

kl_tolerance=0.5
kl_tolerance_scaled = torch.Tensor([kl_tolerance*LATENT_SIZE]).to(device)

logger = {k:[] for k in ['train_loss','test_loss']}

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(real_obs, enc_mu, enc_logsigma, dec_mu, dec_logsigma, kl_tolerance=True):
    """ VAE loss function """
    
    #BCE = F.mse_loss(recon_x, x, size_average=False)
    real_obs = real_obs.view(real_obs.size(0), -1) # flattening all but the batch. 
    log_P_OBS_GIVEN_S = Normal(dec_mu, dec_logsigma.exp()).log_prob(real_obs)
    #print('log p', log_P_OBS_GIVEN_S.shape, log_P_OBS_GIVEN_S[0,:])
    #print('mus and sigmas', dec_mu[0,:], dec_logsigma.exp()[0,:])
    log_P_OBS_GIVEN_S = log_P_OBS_GIVEN_S.sum(dim=-1) #multiply the probabilities within the batch. 

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + 2 * enc_logsigma - enc_mu.pow(2) - (2 * enc_logsigma).exp(), dim=-1)
    #print('kld want to keep batches separate for now', KLD.shape)
    if kl_tolerance:
        assert enc_mu.shape[-1] == LATENT_SIZE, "early debug statement for VAE free bits to work"
        KLD = torch.max(KLD, kl_tolerance_scaled)
    #print('kld POST FREE BITS. want to keep batches separate for now', KLD.shape)
    batch_loss = log_P_OBS_GIVEN_S - KLD
    return - torch.mean(batch_loss), torch.mean(log_P_OBS_GIVEN_S), torch.mean(KLD)  # take expectation across them. 
    # minus sign because we are doing minimization

def train(epoch):
    """ One training epoch. This is the length of the data buffer. """
    # TODO: make one epoch be through all of the data. not just one buffer. 
    vae.train()
    dataset_train.load_next_buffer() # load the underlying dataset new buffer. 
    # TODO: DOESNT THIS IGNORE THE VERY FIRST BUFFER?? OR WOULD THAT ONLY BE IF LENGTH WAS CALLED FIRST?
    # # doesnt really matter as it will cycle back through again at a later period.  
    train_loss = 0
    for batch_idx, data in enumerate(train_loader): # go through whole buffer. 
        obs, rewards = [arr.to(device) for arr in data]
        optimizer.zero_grad()
        encoder_mu, encoder_logsigma, latent_s, decoder_mu, decoder_logsigma = vae(obs, rewards)
        loss, recon_loss, kld_loss = loss_function(obs, encoder_mu, encoder_logsigma, decoder_mu, decoder_logsigma)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tRecon: {:.6f}\tKLD: {:.6f}'.format(
                epoch, batch_idx * len(obs), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item(), recon_loss.item() , kld_loss.item() ))
            # TODO: this is the length of the buffer, not the epoch (if these become separate.)

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))

    return (train_loss / len(train_loader.dataset))

def test():
    """ One test epoch """
    vae.eval()
    dataset_test.load_next_buffer()
    test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            obs, rewards = [arr.to(device) for arr in data]

            encoder_mu, encoder_logsigma, latent_s, decoder_mu, decoder_logsigma = vae(obs, rewards)
            batch_test_loss, test_recon_loss, test_kld_loss = loss_function(obs, encoder_mu, encoder_logsigma, decoder_mu, decoder_logsigma)
            test_loss += batch_test_loss.item()

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

    # returns the last test data batch in order to use this for generating samples!
    return test_loss, obs, rewards

# check vae dir exists, if not, create it
vae_dir = join(args.logdir, 'vae')
if not exists(vae_dir):
    mkdir(vae_dir)
    mkdir(join(vae_dir, 'samples')) # these are examples of the performance. 

best_filename = join(vae_dir, 'best.tar')
checkpoint_filename = join(vae_dir, 'checkpoint.tar')
logger_filename = join(vae_dir, 'logger.json')

reload_file = join(vae_dir, 'best.tar')
if not args.noreload and exists(reload_file):
    state = torch.load(reload_file)
    print("Reloading vae at epoch {}"
          ", with test error {}".format(
              state['epoch'],
              state['precision']))
    vae.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    earlystopping.load_state_dict(state['earlystopping'])

cur_best = None

for epoch in range(1, args.epochs + 1):
    train_loss = train(epoch)
    test_loss, last_test_observations, last_test_rewards = test()
    scheduler.step(test_loss)
    earlystopping.step(test_loss)

    logger['train_loss'].append(train_loss)
    logger['test_loss'].append(test_loss)
    
    # write out the logger. 
    with open(logger_filename, "w") as file:
        json.dump(logger, file)

    # checkpointing
    is_best = not cur_best or test_loss < cur_best
    if is_best:
        cur_best = test_loss

    # why does this currently checkpoint after every single epoch? isnt this slow? 
    # esp as currently every epoch is actually just a buffer size. 
    save_checkpoint({
        'epoch': epoch,
        'state_dict': vae.state_dict(),
        'precision': test_loss,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'earlystopping': earlystopping.state_dict()
    }, is_best, checkpoint_filename, best_filename) # saves to a different checkpoint_filename if it is the best or not. 

    if not args.nosamples:
        with torch.no_grad():
            # get test samples
            encoder_mu, encoder_logsigma, latent_s, decoder_mu, decoder_logsigma = vae(last_test_observations, last_test_rewards)
            recon_batch = decoder_mu + (decoder_logsigma.exp() * torch.randn_like(decoder_mu))
            recon_batch = recon_batch.view(args.batch_size, 3, IMAGE_RESIZE_DIM, IMAGE_RESIZE_DIM)
            #sample = torch.randn(IMAGE_RESIZE_DIM, LATENT_SIZE).to(device) # random point in the latent space.  
            # image reduced size by the latent size. 64 x 32. is this a batch of 64 then?? 
            #sample = vae.decoder(sample).cpu()
            to_save = torch.cat([last_test_observations.cpu(), recon_batch.cpu()], dim=0)
            print(to_save.shape)
            # .view(args.batch_size*2, 3, IMAGE_RESIZE_DIM, IMAGE_RESIZE_DIM)
            save_image(to_save,
                       join(vae_dir, 'samples/sample_' + str(epoch) + '.png'))

    if earlystopping.stop:
        print("End of Training because of early stopping at epoch {}".format(epoch))
        break
