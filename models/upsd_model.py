# pylint: disable=not-callable
# pylint: disable=no-member

import torch
import torch.nn as nn
from torch.nn import functional as F

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
    
    def __init__(self, 
                 state_size, 
                 desires_size,
                 action_size, 
                 hidden_size, 
                 desire_scalings = [1, 1],
                 device='cpu'):
        super().__init__()
        
        self.desire_scalings = torch.FloatTensor(desire_scalings).to(device)
        
        self.state_fc = nn.Sequential(nn.Linear(state_size, 64), 
                                      nn.Tanh())
        
        self.command_fc = nn.Sequential(nn.Linear(2, 64), 
                                        nn.Sigmoid())
        
        self.output_fc = nn.Sequential(nn.Linear(64, 128), 
                                       nn.ReLU(), 
    #                                  nn.Dropout(0.2),
                                       nn.Linear(128, 128), 
                                       nn.ReLU(), 
    #                                  nn.Dropout(0.2),
                                       nn.Linear(128, 128), 
                                       nn.ReLU(), 
                                       nn.Linear(128, action_size))
        
        self.to(device)
        
    
    def forward(self, state, command):
        '''Forward pass
        
        Params:
            state (List of float)
            command (List of float)
        
        Returns:
            FloatTensor -- action logits
        '''
        state_output = self.state_fc(state)
        command_output = self.command_fc(command * self.desire_scalings)
        embedding = torch.mul(state_output, command_output)
        return self.output_fc(embedding)

    def sample_action(self, state, command):
        logits = self.forward(state, command)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        return dist.sample().item()

    def greedy_action(self, state, command):
        logits = self.forward(state, command)
        probs = F.softmax(logits, dim=-1)
        return np.argmax(probs.detach().cpu().numpy())
