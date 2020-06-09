
import unittest
import torch 
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.utils import save_image
from torch.distributions.normal import Normal
from models.vae import VAE

from utils.misc import save_checkpoint
from utils.misc import LATENT_SIZE, IMAGE_RESIZE_DIM
import numpy as np
class TestVAE(unittest.TestCase):

    vae = VAE(3, LATENT_SIZE, conditional=True).to("cpu")

    
    def test_vae_loss(self):

        #from trainvae import loss_function
        bs=10

        obs = torch.randn((bs,3,64,64))
        enc_mu = torch.randn((bs, 32))
        enc_logsig = torch.Tensor( np.random.random((bs, 32)) )-1
        dec_mu = torch.randn((bs, 3*64*64))
        dec_logsig = torch.Tensor( np.random.random((bs, 3*64*64)) )-1

        loss = loss_function(obs, enc_mu, enc_logsig, dec_mu, dec_logsig)
        print(loss)

    def test_normal_distribution(self):

        mus = torch.Tensor([[1,2,10], [ 20, 22, 30]])
        sigmas = torch.Tensor([[1,1,1], [3,3,3]])
        obs = torch.Tensor([[1,2,9], [20, 22, 28]])
        print('getting the result')
        res = Normal(mus, sigmas).log_prob(obs).exp()
        print('the resulting probs are: ', res)

def loss_function(real_obs, enc_mu, enc_logsigma, dec_mu, dec_logsigma, kl_tolerance=True):
    """ VAE loss function """

    kl_tolerance=0.5
    kl_tolerance_scaled = torch.Tensor([kl_tolerance*LATENT_SIZE]).to("cpu")

    
    #BCE = F.mse_loss(recon_x, x, size_average=False)
    print(real_obs.shape)
    real_obs = real_obs.view(real_obs.size(0), -1) # flattening all but the batch. 
    print('dec mu and logsigma')
    log_P_OBS_GIVEN_S = Normal(dec_mu, dec_logsigma.exp()).log_prob(real_obs)
    print('log p', log_P_OBS_GIVEN_S.shape)
    log_P_OBS_GIVEN_S = log_P_OBS_GIVEN_S.sum(dim=-1) #multiply the probabilities within the batch. 
    print('log p obs', log_P_OBS_GIVEN_S, log_P_OBS_GIVEN_S.shape)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # t
    KLD = -0.5 * torch.sum(1 + 2 * enc_logsigma - enc_mu.pow(2) - (2 * enc_logsigma).exp(), dim=-1)
    print('kld want to keep batches separate for now', KLD, KLD.shape)
    if kl_tolerance:
        assert enc_mu.shape[-1] == LATENT_SIZE, "early debug statement for VAE free bits to work"
        KLD = torch.max(KLD, kl_tolerance_scaled)
    print('kld POST FREE BITS. want to keep batches separate for now', KLD, KLD.shape)
    batch_loss = log_P_OBS_GIVEN_S - KLD
    return - torch.mean(batch_loss) # take expectation across them. 
    # minus sign because we are doing minimization
        


