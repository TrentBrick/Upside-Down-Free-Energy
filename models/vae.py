
"""
Variational encoder model, used as a visual model
for our model of the world.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    """ VAE decoder """
    def __init__(self, img_channels, latent_size, conditional):
        super(Decoder, self).__init__()
        self.conditional = conditional
        self.latent_size = latent_size
        self.img_channels = img_channels

        if self.conditional:

            self.r_cond1 = nn.Linear(latent_size+1, 256)
            self.fc1 = nn.Linear(256, 1024)

        else: 
            self.fc1 = nn.Linear(latent_size, 1024)
        self.deconv1 = nn.ConvTranspose2d(1024, 128, 5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.deconv4 = nn.ConvTranspose2d(32, img_channels, 6, stride=2)

        #self.fc_mu = nn.Linear(3*64*64, 3*64*64)
        self.fc_logsigma = nn.Linear(1024, 3*64*64)

    def forward(self, s, r): # pylint: disable=arguments-differ
        if self.conditional:
            s = torch.cat([s, r], dim=1)
            s = F.relu(self.r_cond1(s))
        s = F.relu(self.fc1(s))
        logsigma = self.fc_logsigma(s)
        s = s.unsqueeze(-1).unsqueeze(-1)
        s = F.relu(self.deconv1(s))
        s = F.relu(self.deconv2(s))
        s = F.relu(self.deconv3(s))
        s = F.sigmoid(self.deconv4(s))
        mu = s.view(s.size(0), -1)

        return mu, logsigma

class Encoder(nn.Module): # pylint: disable=too-many-instance-attributes
    """ VAE encoder """
    def __init__(self, img_channels, latent_size, conditional):
        super(Encoder, self).__init__()
        self.latent_size = latent_size
        self.conditional = conditional
        #self.img_size = img_size
        self.img_channels = img_channels

        self.conv1 = nn.Conv2d(img_channels, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)

        if self.conditional:
            self.r_cond1 = nn.Linear((2*2*256)+1, 256)
            self.r_cond2 = nn.Linear(256, 128)

            self.fc_mu = nn.Linear(128, latent_size)
            self.fc_logsigma = nn.Linear(128, latent_size)
        else: 
            self.fc_mu = nn.Linear(2*2*256, latent_size)
            self.fc_logsigma = nn.Linear(2*2*256, latent_size)


    def forward(self, v, r): # v is the observation# pylint: disable=arguments-differ
        v = F.relu(self.conv1(v))
        v = F.relu(self.conv2(v))
        v = F.relu(self.conv3(v))
        v = F.relu(self.conv4(v))
        v = v.view(v.size(0), -1)

        if self.conditional: 
            x = torch.cat([v,r], dim=1)
            x = F.relu(self.r_cond1(x))
            x = F.relu(self.r_cond2(x))

        mu = self.fc_mu(x)
        logsigma = self.fc_logsigma(x)

        return mu, logsigma

class VAE(nn.Module):
    """ Variational Autoencoder """
    def __init__(self, img_channels, latent_size, conditional=True):
        super(VAE, self).__init__()
        self.encoder = Encoder(img_channels, latent_size, conditional)
        self.decoder = Decoder(img_channels, latent_size, conditional)

    def forward(self, v, r): # pylint: disable=arguments-differ
        
        #r = r.unsqueeze(1)
        #print(v.shape, r.shape)
        encoder_mu, encoder_logsigma = self.encoder(v, r)

        """sigma = encoder_logsigma.exp()
        eps = torch.randn_like(sigma)
        latent_s = eps.mul(sigma).add_(encoder_mu)"""
        latent_s = encoder_mu + (encoder_logsigma.exp() * torch.randn_like(encoder_mu))  
        
        decoder_mu, decoder_logsigma = self.decoder(latent_s, r)
        return encoder_mu, encoder_logsigma, latent_s, decoder_mu, decoder_logsigma
