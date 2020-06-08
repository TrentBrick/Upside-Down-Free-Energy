"""import torch 
from torch.nn import functional as F
from torchvision import transforms
from torchvision.utils import save_image

num_z_samps = 3
num_next_z_samps = 3 

def p_policy():

    '''#Should have access already
    #transform then encode the observations and sample latent zs
    #obs = transform(obs).unsqueeze(0).to(device)
    # batch, channels, image dim, image dim
    #mu, logsigma = vae.encoder(obs)'''

    total_samples = num_z_samps * num_next_z_samps
    expected_kld = 0
    for _ in range(num_z_samps): # vectorize this somehow 
        latent_z =  mus + logsigmas.exp() * torch.randn_like(mus)

        # predict the next state from this one. 
        # I already have access to the action from the runs generated. 
        # TODO: make both of these run with a batch. DONT NEED TO GENERATE OR PASS AROUND HIDDEN AS A RESULT. 

        md_mus, md_sigmas, md_logpi, r, d, next_hidden = self.mdrnn(actions, latent_z)

        # sample the next latent variable 
        next_r = self.reward_predictor(next_hidden)
        # reward loss 
        reward_surprise = self.reward_likelihood(next_r)

        g_probs = Categorical(probs=torch.exp(logpi).permute(0,2,1))
        which_g = g_probs.sample()
        mus_g, sigs_g = torch.gather(mus.squeeze(), 0, which_g), torch.gather(sigmas.squeeze(), 0, which_g)
        #print(mus_g.shape)
        for _ in range(num_next_z_samps):
            next_z = mus_g + sigs_g * torch.randn_like(mus_g)
            hat_obs = self.vae.decoder(next_z, next_r)
            # reconstruction loss: 
            BCE = F.mse_loss(hat_obs, obs, size_average=False)

            per_time_loss = torch.log(BCE*reward_surprise)

            # can sum across time with these logs. 
            expected_loss += torch.sum(per_time_loss, dim=)
            # multiply all of these probabilities together within a single batch. 

    # average across the rollouts. 

    return expected_loss / total_samples

def p_tilde():
    # for the actual observations need to compute the prob of seeing it and its reward
    # the rollout will also contain the reconstruction loss so: 

    # batch is the observations at different time points. 
    BCE = F.mse_loss(hat_obs, obs, size_average=False)
    # all of the rewards
    reward_surprise = self.reward_likelihood(r)

    per_time_loss = torch.log(BCE*reward_surprise)

    # can sum across time with these logs. 
    expected_loss += torch.sum(per_time_loss, dim=)

    return expected_loss

def feef_loss(data_dict_rollout):

    # choose the action that minimizes the following reward.
    # provided with information from a single rollout 
    # this includes the observation, the actions taken, the VAE mu and sigma, and the next hidden state predictions. 
    # for p_opi
    # should see what the difference in variance is between using a single value and taking an expectation over many. 
    # as calculating a single value would be so much faster. 

        return torch.log( p_policy(data_dict_rollout) ) - torch.log( p_tilde(data_dict_rollout) )

    



"""