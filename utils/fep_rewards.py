# pylint: disable=not-callable
# pylint: disable=no-member

"""
Computes FEP rewards that are environment specific. 
"""

import torch
from torch.optim import Adam
import torch.nn as nn
from torch.distributions import Normal 

class CondDensityModel(nn.Module):
    def __init__(self, node_size, num_states, num_g, act_fn="relu"):
        super().__init__()
        self.num_g = num_g
        self.num_states = num_states
        self.stride = num_g*num_states

        self.act_fn = getattr(torch, act_fn)
        self.fc_1 = nn.Linear(1, node_size)
        self.fc_2 = nn.Linear(node_size, node_size)
        
        self.fc_3 = nn.Linear(node_size, 3*num_states*num_g)

    def forward(self, reward):
        out = self.act_fn(self.fc_1(reward))
        out = self.act_fn(self.fc_2(out))
        out = self.fc_3(out)

        logpi = out[:,:self.stride].view(-1, self.num_g, self.num_states)
        mus = out[:,self.stride:2*self.stride].view(-1, self.num_g, self.num_states)
        logsigmas = out[:,2*self.stride:3*self.stride].view(-1, self.num_g, self.num_states)
        log_probs = torch.log_softmax(logpi, dim=-2)
        return mus, logsigmas, log_probs

    def give_modes(self, reward):
        # gives the mode (also mean) of the highest probability gaussian. 
        mus, logsigmas, log_probs = self.forward(reward)
        which_g = log_probs.argmax(-2)
        mus_g, sigmas_g = torch.gather(mus.squeeze(), 1, which_g.unsqueeze(1)).squeeze(), torch.gather(logsigmas.exp().squeeze(), 1, which_g.unsqueeze(1)).squeeze()
        return mus_g

    def state_probs(self, reward, obs):
        mus, logsigmas, log_probs = self.forward(reward)
        normal_dist = Normal(mus, logsigmas.exp()) # for every gaussian in each latent dimension. 
        g_log_probs = log_probs + normal_dist.log_prob(obs.unsqueeze(-2))
        each_state_probs = torch.logsumexp(g_log_probs, dim=-2) 
        # combine the states and just have the batch. 
        return each_state_probs.sum(-1)

class FepCalculator():

    def __init__(self, algo, gamename ):

        self.include_state_cond = True  
        self.algo = algo

        if gamename=='Pendulum-v0':
            self.r_mu_prior, self.r_sigma_prior = torch.Tensor([0.0]), torch.Tensor([0.5])
            num_g = 2
            num_states = 3
            node_size = 10
            self.num_epochs = 1

        self.model = CondDensityModel(node_size, num_states, num_g)
        self.optimizer = Adam(self.model.parameters(), lr=0.03)

    def _process_inputs(self, rews, obs, path_slice_inds=None):
        # need to get future obs and remove the last reward. corresponds to an unsaved obs. 
        if self.algo=='ppo': # because of finish path which adds an additional value. 
            if self.include_state_cond:
                obs = obs[1:]
                rews = rews[:-1]
                # need to modify to lose some data because the rewards are shorter! 
                path_slice_inds = slice(path_slice_inds[0], path_slice_inds[1] -1 )
            else: 
                path_slice_inds = slice(path_slice_inds[0], path_slice_inds[1])
        elif self.algo=='ddpg': # obs2 is passed in which is the next observation. 
            pass 
        return torch.as_tensor(rews), torch.as_tensor(obs), path_slice_inds

    def train_model(self, rews, obs):
        rews, obs, _ = self._process_inputs(rews, obs)
        rews = rews.unsqueeze(1)
        for epoch in range(self.num_epochs):  # loop over the dataset multiple times
            # zero the parameter gradients
            self.optimizer.zero_grad()
            mus, logsigmas, log_probs = self.model(rews)
            normal_dist = Normal(mus, logsigmas.exp()) # for every gaussian in each latent dimension. 
            g_log_probs = log_probs + normal_dist.log_prob(obs.unsqueeze(-2)) # how far off are the next obs? 
            # sum across the gaussians, need to do so in log space: 
            loss = - torch.logsumexp(g_log_probs, dim=-2).sum(-1).mean()
            loss.backward()
            self.optimizer.step()
        print("loss at end of training conddensity:", loss.item())

    def fep_calculate(self, rews, obs, path_slice_inds=None):
        with torch.no_grad():
            rews, obs, path_slice_inds = self._process_inputs( rews, obs, path_slice_inds)
            reward_neg_surprise = Normal(self.r_mu_prior, self.r_sigma_prior).log_prob(rews)
            #print('rewards neg surprise being returned are:', reward_neg_surprise.shape, reward_neg_surprise[0:5])

            if self.include_state_cond:
                # 0. learn conditional reward distribution. 
                rews = rews.unsqueeze(1)
                # get prob of these obs/states from trained conditional density model
                obs_neg_surprise = self.model.state_probs(rews, obs)
                # 1. have evo priors on the states. 
                # 2. learn the states distribution. 
                #o_mu = torch.Tensor(obs.mean(axis=0))
                #o_sigma = torch.Tensor(np.cov(obs, rowvar=False))
                # obs = torch.distributions.MultivariateNormal(o_mu, o_sigma ).log_prob(obs)

                # combine rewards and obs 
                neg_surprise = reward_neg_surprise+obs_neg_surprise # as they are both log probs. 

            else: 
                neg_surprise = reward_neg_surprise

            # maximizing rewards so dont need to add minus here. 
            if path_slice_inds:
                return neg_surprise, path_slice_inds
            else: 
                return neg_surprise 