# pylint: disable=not-callable
# pylint: disable=no-member

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np 

class BackwardModel(nn.Module):
    
    def __init__(self, state_size, output_size, hidden_sizes=[128, 128, 128], 
            act_fn="ReLU"):
        super().__init__()
        self.act_fn = getattr(torch.nn, act_fn)
        hidden_sizes.insert(0, state_size)
        self.layers = nn.ModuleList()
        for j in range(len(hidden_sizes)-1):
            self.layers.append( nn.Linear(hidden_sizes[j], hidden_sizes[j+1])  )
            self.layers.append( self.act_fn() )
        self.layers = nn.Sequential(*self.layers)
        self.output_fc = nn.Linear(hidden_sizes[-1], output_size )

    def forward(self, inp):
        inp = torch.cat(inp, dim=1)
        inp = self.layers(inp)
        inp = self.output_fc(inp)
        # TODO: should I applying any activation function for the output here? 
        return inp





