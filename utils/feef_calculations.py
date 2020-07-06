# pylint: disable=not-callable
# pylint: disable=no-member
import torch 
from torch.distributions.normal import Normal

def importance_sampling(num_samps, real_obs, latent_s, encoder_mu, 
        encoder_logsigma, cond_reward, delta_prediction=False, pres_latent_s=None):
        """
        Returns a full batch. 
        """

        real_obs = real_obs.view(real_obs.size(0), -1) # flatten all but sequence. 
        log_p_v = torch.zeros(encoder_mu.shape[0]) # batch size

        for _ in range(num_samps):

            decoder_mu, decoder_logsigma = self.vae.decoder(latent_s, cond_reward)

            log_P_OBS_GIVEN_S = Normal(decoder_mu, decoder_logsigma.exp()).log_prob(real_obs)
            log_P_OBS_GIVEN_S = log_P_OBS_GIVEN_S.sum(dim=-1) #multiply the probabilities within the batch. 

            log_P_S = Normal(0.0, 1.0).log_prob(latent_s).sum(dim=-1)
            log_Q_S_GIVEN_X = Normal(encoder_mu, encoder_logsigma.exp()).log_prob(latent_s).sum(dim=-1)

            log_p_v += log_P_OBS_GIVEN_S + log_P_S - log_Q_S_GIVEN_X

        return log_p_v/num_samps


def max_feef():
    # choose the actions that max the observations and rewards. 
    # can sample n planned rollouts:
    # perform_rollout is: 1. sample from encoder. 
    # hiddens and future states conditioned on these. 
    # then but rollout result into decoder to get mus and sigmas. 

    # then using importance sampling 
    # sample from decoder and get observations. compute the prob
    # of the observation. seems kind of weird to be using fictional obs. 
    # but its the same thing planning ahead with future rewards!




def p_policy( hiddens, pred_rewards, encoder_mus, encoder_sigmas, num_s_samps, num_next_encoder_samps, 
        num_importance_samps):

    # rollout_dict values are of the size and shape seq_len, values dimensions. 

    # TODO: break the sequence length into smaller batches. 
    
    total_samples = num_s_samps * num_next_encoder_samps * num_importance_samps
    expected_loss = 0

    for i in range(num_s_samps): # vectorize this somehow 
        latent_s = encoder_mus + (encoder_sigmas * torch.randn_like(encoder_mus))
        # TODO: enable multiple recomputes using perform_rollout.  
        decoded_mus, decoded_sigmas = rssm.decode_sequence_obs(hiddens, latent_s, pred_rewards)
        print('p_policy, shape of latent states', latent_s.shape )
        
        # reward loss 
        log_p_r = self.reward_prior.log_prob(pred_rewards)

        for j in range(num_next_encoder_samps):

            

            next_encoder_sample, mus_g, sigs_g = sample_mdrnn_latent(md_mus, md_sigmas, 
                                    md_logpi, latent_s, 
                                    return_chosen_mus_n_sigs=True)

            # importance sampling which has its own number of iterations: 
            log_p_v = self.importance_sampling(num_importance_samps, next_obs, 
                next_encoder_sample, mus_g, torch.log(sigs_g), 
                next_r.unsqueeze(1))
            
            # can sum across time with these logs. (as the batch is the different time points)
            expected_loss += torch.sum(log_p_v+log_p_r)

    # average across the all of the sample rollouts. 
    return expected_loss / total_samples

def p_tilde( rollout_dict, num_importance_samps, encoder_mus, encoder_sigmas):
    # for the actual observations need to compute the prob of seeing it and its reward
    # the rollout will also contain the reconstruction loss so: 

    pres_rewards = rollout_dict['rewards'][:-1]

    # all of the rewards
    log_p_r = self.reward_prior.log_prob(pres_rewards.squeeze())

    # compute the probability of the visual observations: 
    curr_obs = rollout_dict['obs'][:-1]
    log_p_v = self.importance_sampling(num_importance_samps, curr_obs, encoder_mus, encoder_sigmas, pres_rewards)
    #print('p tilde', log_p_r.shape, log_p_v.shape)
    # can sum across time with these logs. (as the batch is the different time points)
    expected_loss = torch.sum(log_p_v+log_p_r)

    return expected_loss

def feef_loss(model, actions, hidden, state):

    # choose the action that minimizes the following reward.
    # provided with information from a single rollout 
    # this includes the observation, the actions taken, the VAE mu and sigma, and the next hidden state predictions. 
    # for p_opi
    # should see what the difference in variance is between using a single value and taking an expectation over many. 
    # as calculating a single value would be so much faster. 
    print('computing feef loss')

    num_s_samps = 1
    num_next_encoder_samps = 1 
    num_importance_samps = 1 # I think 250 is ideal but probably too slow
    self.reward_prior = Normal(reward_prior_mu,reward_prior_sigma) # treating this as basically a half normal. it should be higher than the max reward available for any run. 

    with torch.no_grad():

        

        log_policy_loss = self.p_policy(prior_mus, prior_sigmas, num_s_samps, num_next_encoder_samps, 
                                        num_importance_samps)
        log_tilde_loss =  self.p_tilde(data_dict_rollout, num_importance_samps, 
                                        encoder_mus, encoder_sigmas )

    return log_policy_loss - log_tilde_loss