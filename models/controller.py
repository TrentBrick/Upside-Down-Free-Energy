""" Define controller """
import torch
import torch.nn as nn

class Controller(nn.Module):
    """ Controller """
    def __init__(self, latents, recurrents, actions, gamename='carracing', conditional=True):
        super().__init__()
        self.gamename = gamename
        self.conditional = conditional
        num_inputs = latents + recurrents
        if conditional: 
            num_inputs += 1 # for the reward
        self.fc = nn.Linear(num_inputs, actions)

    def forward(self, *inputs):
        cat_in = torch.cat(inputs, dim=1)
        out = self.fc(cat_in)

        if self.gamename == 'carracing':
            # order is direction, speed and brakes.
            # -1 to 1. then 0 to 1 and 0 to 1. 
            # following the approach from https://github.com/hardmaru/WorldModelsExperiments/blob/master/carracing/model.py 
            out = torch.tanh(out)
            out[0,1] = (out[0,1]+1)/2.0 # this converts tanh to sigmoid
            out[0,2] = torch.clamp(out[0,2], min=0.0, max=1.0) # this makes it more likely that we dont break. 

            return out
