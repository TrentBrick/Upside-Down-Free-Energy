# pylint: disable=not-callable
# pylint: disable=no-member

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.utils import save_image
from os.path import join

class Planner(nn.Module):
    def __init__(self,
        model,
        action_size,
        actions_low_n_high,
        init_cem_params,
        plan_horizon=12,
        optim_iters=10,
        num_particles=1000,
        k_top=100, discount_factor=0.9,
        return_plan_images=False, compute_feef=False ):
        super().__init__()
        self.model = model
        self.action_size = action_size
        self.actions_low = actions_low_n_high[0]
        self.actions_high = actions_low_n_high[1]
        self.plan_horizon = plan_horizon
        self.optim_iters = optim_iters
        self.num_particles = num_particles
        self.k_top = k_top
        self.discount_factor = discount_factor
        self.return_plan_images = return_plan_images
        self.compute_feef = compute_feef

        # TODO: actually use these for the sampling. 
        self.init_action_mean = init_cem_params[0]
        self.init_action_std = init_cem_params[1]
    
    def constrain_actions(self, actions):
        """ Ensures actions sampled from the gaussians are within the game bounds."""
        for ind, (l, h) in enumerate(zip(self.actions_low, self.actions_high)):
            actions[:,:,ind] = torch.clamp(actions[:,:,ind], min=l, max=h)
        return actions

    def forward(self, hidden, state, timepoint=1):
        """ (batch_size, dim) """
        batch_size = hidden.size(0)
        hidden_size = hidden.size(1)
        state_size = state.size(1)

        """ (batch_size * num_particles, hidden_size) """
        hidden = hidden.unsqueeze(dim=1)
        hidden = hidden.expand(batch_size, self.num_particles, hidden_size)
        hidden = hidden.reshape(-1, hidden_size)

        """ (batch_size * num_particles, state_size) """
        state = state.unsqueeze(dim=1)
        state = state.expand(batch_size, self.num_particles, state_size)
        state = state.reshape(-1, state_size)

        """ (plan_horizon, batch_size, 1, action_size) """
        action_mean = self.init_action_mean.repeat(self.plan_horizon, batch_size, 1, 1).to(hidden.device)
        action_std_dev = self.init_action_std.repeat(self.plan_horizon, batch_size, 1, 1).to(hidden.device)
        assert list(action_mean.shape) == [self.plan_horizon, batch_size, 1, self.action_size] ,'shape of action mean is wrong'
        '''action_mean = torch.zeros(
            self.plan_horizon, batch_size, 1, self.action_size, device=hidden.device
        )
        action_std_dev = torch.ones(
            self.plan_horizon, batch_size, 1, self.action_size, device=hidden.device
        )'''

        for _ in range(self.optim_iters):
            """ (plan_horizon, batch_size * num_particles, action_size) """
            epsilon = torch.randn(
                self.plan_horizon,
                batch_size,
                self.num_particles,
                self.action_size,
                device=action_mean.device,
            )
            actions = action_mean + action_std_dev * epsilon
            actions = actions.view(
                self.plan_horizon, batch_size * self.num_particles, self.action_size
            )

            # constrain actions: 
            actions = self.constrain_actions(actions)

            if self.compute_feef:
                # compute feef: 
                feefs = feef_loss(self.model, actions, hidden, state)

            else: 
                """ (plan_horizon, batch_size * num_particles, dim) """
                rollout = self.model.perform_rollout(actions, hidden=hidden, state=state)
                
                """ (plan_horizon * batch_size * num_particles, dim) """
                _hiddens = rollout["hiddens"].view(-1, hidden_size)
                _states = rollout["prior_states"].view(-1, state_size)

                """ (batch_size, num_particles) """
                returns = self.model.decode_reward(_hiddens, _states)
                returns = returns.view(self.plan_horizon, -1)
                # discounting: 
                returns = returns*torch.pow(torch.Tensor([self.discount_factor]), torch.arange(0,self.plan_horizon).float()).unsqueeze(1)
                returns = returns.sum(dim=0)
                returns = returns.reshape(batch_size, self.num_particles)

            action_mean, action_std_dev = self._fit_gaussian(
                actions, returns, batch_size
            )

        if self.return_plan_images:

            best_rew, best_actions_ind = torch.max(returns, dim=1)
            worst_rew, worst_actions_ind = torch.min(returns, dim=1)
            print('From this planning iteration, best reward is:', best_rew, 'worst reward is:', worst_rew)

            _states = _states.view(self.plan_horizon, batch_size, self.num_particles, -1)
            _hiddens = _hiddens.view(self.plan_horizon, batch_size, self.num_particles, -1)
            obs = []
            for ind in [best_actions_ind, worst_actions_ind]:
                decoded_mus, decoded_logsigmas = self.model.decode_sequence_obs(_hiddens[:,:,ind,:], _states[:,:,ind,:])
                temp_obs = decoded_mus.view(decoded_mus.shape[0],3, 64, 64).cpu()
                obs.append(temp_obs)
                # need to append and then save these out. 

            obs = torch.stack(obs)
            save_image(obs,
                    join('exp_dir/simulations/', 'planning_best_n_worst'+str(timepoint)+'.png'))
            return action_mean[0].squeeze(dim=1)

        else:
            return action_mean[0].squeeze(dim=1)

    def _fit_gaussian(self, actions, returns, batch_size):
        _, topk = returns.topk(self.k_top, dim=1, largest=True, sorted=False)

        topk += self.num_particles * torch.arange(
            0, batch_size, dtype=torch.int64, device=topk.device
        ).unsqueeze(dim=1)

        # does not average actions across the horizon!
        best_actions = actions[:, topk.view(-1)].reshape(
            self.plan_horizon, batch_size, self.k_top, self.action_size
        )
        action_mean, action_std_dev = (
            best_actions.mean(dim=2, keepdim=True),
            best_actions.std(dim=2, unbiased=False, keepdim=True),
        )
        return action_mean, action_std_dev

