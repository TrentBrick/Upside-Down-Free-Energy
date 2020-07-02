# pylint: disable=not-callable
# pylint: disable=no-member

import torch
import torch.nn as nn
from torch.nn import functional as F

class ConvDecoder(nn.Module):
    def __init__(self, hidden_size, state_size, embedding_size, 
            reward_condition, make_sigmas, act_fn="relu"):
        super().__init__()
        self.act_fn = getattr(F, act_fn)
        self.embedding_size = embedding_size
        self.reward_condition = reward_condition
        self.make_sigmas = make_sigmas
        if self.reward_condition: 
            self.fc_rcond1 = nn.Linear(hidden_size + state_size+1, embedding_size)
            self.fc1 = nn.Linear(embedding_size, embedding_size)
        else: 
            self.fc_1 = nn.Linear(hidden_size + state_size, embedding_size)
        if self.make_sigmas:
            self.fc_logsigma1 = nn.Linear(embedding_size, 3*64*64)
        self.conv_1 = nn.ConvTranspose2d(embedding_size, 128, 5, stride=2)
        self.conv_2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.conv_3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.conv_4 = nn.ConvTranspose2d(32, 3, 6, stride=2)

    def forward(self, hidden, state, reward=None):
        if self.reward_condition: 
            out = self.act_fn(self.fc_rcond1(torch.cat([hidden, state, reward], dim=1)))
            out = self.fc_1(out)
        else: 
            out = self.fc_1(torch.cat([hidden, state], dim=1))
        if self.make_sigmas: 
            logsigma = self.fc_logsigma1( self.act_fn(out) )
        out = out.view(-1, self.embedding_size, 1, 1)
        out = self.act_fn(self.conv_1(out))
        out = self.act_fn(self.conv_2(out))
        out = self.act_fn(self.conv_3(out))
        obs = self.conv_4(out)

        if self.make_sigmas: 
            # reshape into image mus shape: 
            logsigma = logsigma.view(obs.shape)
            # TODO: constrain variance for out to distribution based on min and max training data as in Appendix A1 here: https://arxiv.org/pdf/1805.12114.pdf
            return obs, logsigma
        else:
            return obs, torch.zeros_like(obs)

class LinearDecoder(nn.Module):
    def __init__(self, obs_size, hidden_size, state_size, node_size, act_fn="relu"):
        super().__init__()
        self.act_fn = getattr(F, act_fn)
        self.fc_1 = nn.Linear(hidden_size + state_size, node_size)
        self.fc_2 = nn.Linear(node_size, node_size)
        self.fc_3 = nn.Linear(node_size, obs_size)

    def forward(self, hidden, state):
        out = self.act_fn(self.fc_1(torch.cat([hidden, state], dim=1)))
        out = self.act_fn(self.fc_2(out))
        obs = self.fc3(out)
        return obs
