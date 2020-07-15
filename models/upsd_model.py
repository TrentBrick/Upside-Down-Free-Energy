# pylint: disable=not-callable
# pylint: disable=no-member

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np 

class UpsdBehavior(nn.Module):
    '''
    Behavour function that produces actions based on a state and command.
    NOTE: At the moment I'm fixing the amount of units and layers.
    TODO: Make hidden layers configurable.
    
    Params:
        state_size (int)
        action_size (int)
        hidden_size (int) -- NOTE: not used at the moment
        desire_scalings (List of float)
    '''
    
    def __init__(self, state_size, action_size, hidden_sizes,
            desire_scalings):
        super().__init__()
        
        self.desire_scalings = torch.FloatTensor(desire_scalings)
        
        self.state_fc = nn.Sequential(nn.Linear(state_size, hidden_sizes[0]), 
                                      nn.Tanh())
        
        self.command_fc = nn.Sequential(nn.Linear(2, hidden_sizes[0]), 
                                        nn.Sigmoid())

        self.output_fc = []
        hidden_sizes.append(action_size)
        output_activation= nn.Identity
        activation = nn.ReLU()
        for j in range(len(hidden_sizes)-1):
            act = activation if j < len(sizes)-2 else output_activation
            layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
        self.output_fc = nn.Sequential(*self.output_fc)
        
        '''self.output_fc = nn.Sequential(nn.Linear(hidden_size, hidden_size), 
                                       nn.ReLU(), 
                                       #nn.Dropout(0.2),
                                       nn.Linear(hidden_size, hidden_size), 
                                       nn.ReLU(), 
                                       #nn.Dropout(0.2),
                                       nn.Linear(hidden_size, hidden_size), 
                                       nn.ReLU(), 
                                       nn.Linear(hidden_size, action_size))   '''
    
    def forward(self, state, command):
        '''Forward pass
        
        Params:
            state (List of float)
            command (List of float)
        
        Returns:
            FloatTensor -- action logits
        '''
        #print('entering the model', state.shape, command.shape)
        state_output = self.state_fc(state)
        command_output = self.command_fc(command * self.desire_scalings)
        embedding = torch.mul(state_output, command_output)
        return self.output_fc(embedding)

class UpsdModel(nn.Module):
    """ Using Fig.1 from Reward Conditioned Policies 
        https://arxiv.org/pdf/1912.13465.pdf """
    def __init__(self, state_size, desires_size, 
        action_size, node_size, desire_scalings = None, act_fn="relu"):
        super().__init__()
        self.act_fn = getattr(torch, act_fn)
        if desire_scalings is not None: 
            self.desire_scalings = torch.FloatTensor(desire_scalings)
        else: 
            self.desire_scalings = desire_scalings

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

        #print("inputs for forward", state.shape, desires.shape)

        if self.desire_scalings:
            desires *= self.desire_scalings

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