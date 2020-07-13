# pylint: disable=not-callable
# pylint: disable=no-member

import torch
import torch.nn as nn
from torch.nn import functional as F

class UpsdModel(nn.Module):
    """ Using Fig.1 from Reward Conditioned Policies 
        https://arxiv.org/pdf/1912.13465.pdf """
    def __init__(self, state_size, desires_size, 
        action_size, node_size, act_fn="relu"):
        super().__init__()
        self.act_fn = getattr(torch, act_fn)

        # states
        self.state_fc_1 = nn.Linear(state_size, node_size)
        self.state_fc_2 = nn.Linear(node_size, node_size)
        self.state_fc_3 = nn.Linear(node_size, node_size)
        self.state_fc_4 = nn.Linear(node_size, action_size)

        # desires
        self.desire_fc_1 = nn.Linear(desires_size, node_size)
        self.desire_fc_2 = nn.Linear(desires_size, node_size)
        self.desire_fc_3 = nn.Linear(desires_size, node_size)

    def forward(self, state, desires):
        # returns an action
        state = self.act_fn(self.state_fc_1(state))
        d_mod = self.act_fn(self.desire_fc_1(desires))
        state = torch.mul(state, d_mod)

        state = self.act_fn(self.state_fc_2(state))
        d_mod = self.act_fn(self.desire_fc_2(desires))
        state = torch.mul(state, d_mod)

        state = self.act_fn(self.state_fc_3(state))
        d_mod = self.act_fn(self.desire_fc_3(desires))
        state = torch.mul(state, d_mod)

        state = self.state_fc_4(state)
        return state