""" Test gmm loss """
import unittest
import sys
sys.path.append("../models/mdrnn.py")
import torch
import torch.nn as nn
import torch.nn.functional as f
from tqdm import tqdm
from torch.distributions.categorical import Categorical
from mdrnn import gmm_loss

class TestGMM(unittest.TestCase):
    """ Test GMMs """
    def test_gmm_loss(self):
        """ Test case 1 """
        n_samples = 10000

        means1 = torch.Tensor([0., 1., -1.]) 
        stds1 = torch.Tensor([.03, .02, .1])
        pi1 = torch.Tensor([.2, .3, .5])

        cat_dist = Categorical(pi1)
        indices = cat_dist.sample((n_samples,)).long()
        rands = torch.randn(n_samples, 1).squeeze()
        samples1 = means1[indices] + rands * stds1[indices]

        print('samples1', samples1.shape, means1[indices].shape, rands.shape)

        means2 = torch.Tensor([0., 1., 1.])
        stds2 = torch.Tensor([.05, .1, .03])
        pi2 = torch.Tensor([.7, .2, .1])

        cat_dist = Categorical(pi2)
        indices = cat_dist.sample((n_samples,)).long()
        rands = torch.randn(n_samples, 1).squeeze()
        samples2 = means2[indices] + rands * stds2[indices]

        samples = torch.cat([samples1.unsqueeze(1), samples2.unsqueeze(1)], dim=1)
        print(samples.shape)
        #samples = samples1.unsqueeze(1)

        class _model(nn.Module):
            def __init__(self, gaussians):
                super().__init__()
                self.means = nn.Parameter(torch.Tensor(1, gaussians, 2).normal_())
                self.pre_stds = nn.Parameter(torch.Tensor(1, gaussians, 2).normal_())
                self.pi = nn.Parameter(torch.Tensor(1, gaussians, 2).normal_())

            def forward(self, *inputs):
                return self.means, torch.exp(self.pre_stds), f.softmax(self.pi, dim=1)

        model = _model(3)
        optimizer = torch.optim.Adam(model.parameters(),lr=0.05)

        iterations = 50000
        log_step = iterations // 10
        pbar = tqdm(total=iterations)
        cum_loss = 0
        for i in range(iterations):
            batch = samples[torch.LongTensor(128).random_(0, n_samples)]
            m, s, p = model.forward()
            #print(batch.shape, m.shape, s.shape, p.shape)
            loss = gmm_loss(batch, m, s, torch.log(p))
            cum_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix_str("avg_loss={:10.6f}".format(
                cum_loss / (i + 1)))
            pbar.update(1)
            if i % log_step == log_step - 1:
                print(m)
                print(s)
                print(p)
