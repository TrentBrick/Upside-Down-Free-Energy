import torch 
from torch.nn import functional as F
from torchvision import transforms
from torchvision.utils import save_image

# choose the action that minimizes the following reward.
# collect a batch of runs. with different policies. 
# this includes the observation, the actions taken, the VAE mu and sigma, and the next hidden state predictions. 
for rollout in set_of_rollouts: 
    torch.log( p_opi ) - torch.log( p_tilde )

# for p_opi

# should see what the difference in variance is between using a single value and taking an expectation over many. 
# as calculating a single value would be so much faster. 

num_z_samps = 3
num_next_z_samps = 3 

def p_opi():

    '''#Should have access already
    #transform then encode the observations and sample latent zs
    #obs = transform(obs).unsqueeze(0).to(device)
    # batch, channels, image dim, image dim
    #mu, logsigma = vae.encoder(obs)'''

    for _ in range(num_z_samps): # vectorize this somehow 
        latent_z =  mus + logsigmas.exp() * torch.randn_like(mus)

        # predict the next state from this one. 
        # I already have access to the action from the runs generated. 
        # TODO: make both of these run with a batch. DONT NEED TO GENERATE OR PASS AROUND HIDDEN AS A RESULT. 

        md_mus, md_sigmas, md_logpi, r, d, next_hidden = self.mdrnn(actions, latent_z)

        # sample the next latent variable 

        g_probs = Categorical(probs=torch.exp(logpi).permute(0,2,1))
        which_g = g_probs.sample()
        mus_g, sigs_g = torch.gather(mus.squeeze(), 0, which_g), torch.gather(sigmas.squeeze(), 0, which_g)
        #print(mus_g.shape)
        for _ in range(num_next_z_samps):
            next_z = mus_g + sigs_g * torch.randn_like(mus_g)

            hat_obs = self.vae.decoder(next_z)

            # reconstruction loss: 
            BCE = F.mse_loss(hat_obs, obs, size_average=False)

            # multiply all of these probabilities together within a single batch. 

    # average across the rollouts. 

def p_tilde():
    # need to have a prior over the reward signal that is expected at any point. 
    # then need to do bayesian updating of this signal. train this separately. 
    # and use it to compute the probability of the preferences! 
    # ensure it is on the observation and sensory reward channels. 
    # thus it learns both its reward channel AND to stay on the road












    obs = transform(obs)
    mus, vae.encode(obs) 
    

    #

#