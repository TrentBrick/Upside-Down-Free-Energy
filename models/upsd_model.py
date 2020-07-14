# pylint: disable=not-callable
# pylint: disable=no-member

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical
import pytorch_lightning as pl 
import numpy as np 

class LightningTemplate(pl.LightningModule):

    def __init__(self, config, env_params):
        super(LightningTemplate).__init__()

        self.Levine_Implementation = config['Levine_Implementation']
        self.config = config
        self.env_params = env_params

        if self.Levine_Implementation:
            self.model = UpsdModel(env_params['STORED_STATE_SIZE'], 
            env_params['desires_size'], 
            env_params['ACTION_SIZE'], 
            env_params['NODE_SIZE'], desire_scalings=config['desire_scalings'])
        else: 
            self.model = UpsdBehavior( env_params['STORED_STATE_SIZE'], 
                env_params['ACTION_SIZE'], 
                env_params['NODE_SIZE'], config['desire_scalings'] )

    def forward(self,state, command):
        return self.model(state, command)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['lr'])
        return optimizer 

    def training_step(self, batch, batch_idx):

        obs, obs2, act, rew, terminal, terminal_rew, time = batch['obs'].squeeze(0), batch['obs2'].squeeze(0), batch['act'].squeeze(0), batch['rew'].squeeze(0), batch['terminal'].squeeze(0), batch['terminal_rew'].squeeze(0), batch['time'].squeeze(0)

        if not self.Levine_Implementation: 
            desires = torch.cat([rew.unsqueeze(1), time.unsqueeze(1)], dim=1)
        pred_action = self.model.forward(obs, desires)
        if not self.env_params['continuous_actions']:
            #pred_action = torch.sigmoid(pred_action)
            act = act.squeeze().long()
        pred_loss = self._pred_loss(pred_action, act, continous=self.training_endenv_params['continuous_actions'])
        if self.config['Levine_Implementation'] and self.config['weight_loss']:
            #print('weights used ', torch.exp(rew/desired_reward_dist_beta))
            pred_loss = pred_loss*torch.exp(terminal_rew/self.config.desired_reward_dist_beta)
        pred_loss = pred_loss.mean(dim=0)
        #if return_for_model_sampling: 
        #    return pred_loss, (rew[0:5], pred_action[0:5], pred_action[0:5].argmax(-1), act[0:5])
        
        logs = {"train_loss": pred_loss}
        return {'loss':pred_loss, 'log':logs}

    def _pred_loss(self, pred_action, real_action, continous=True):
        if continous:
            # add a sigmoid activation layer.: 
            return F.mse_loss(pred_action, real_action ,reduction='none').sum(dim=1)
        else: 
            return F.cross_entropy(pred_action, real_action, reduction='none')

    def validation_step(self, batch, batch_idx):
        train_dict = self.training_step(batch, batch_idx)
        # rename 
        train_dict['log']['val_loss'] = train_dict['log'].pop('train_loss')

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {
                "avg_val_loss": avg_loss,
                "log": tensorboard_logs
                }

    '''# evaluate every .... 
    if e%evaluate_every==0:
            print('======= Evaluating the agent')
            seed = np.random.randint(0, 1e9, 1)[0]
            cum_rewards, finish_times = agent.simulate(seed, num_episodes=5, greedy=True)
            print('Evaluation, mean reward:', np.mean(cum_rewards), 'mean horizon length:', np.mean(finish_times))
            print('===========================')

    if make_vae_samples:
            generate_model_samples( model, for_upsd_sampling, 
                            samples_dir, SEQ_LEN, env_params['IMAGE_RESIZE_DIM'],
                            example_length,
                            memory_adapt_period, e, device, 
                            make_vae_samples=make_vae_samples,
                            make_mdrnn_samples=False, 
                            transform_obs=False  )
            print('====== Done Generating Samples')'''

    def sample_action(self, state, command):
        logits = self.forward(state, command)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        return dist.sample().detach()

    def greedy_action(self, state, command):
        logits = self.forward(state, command)
        probs = F.softmax(logits, dim=-1)
        return np.argmax(probs.detach())

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
    
    def __init__(self, state_size, action_size, hidden_size,
            desire_scalings):
        super().__init__()
        
        self.desire_scalings = torch.FloatTensor(desire_scalings)
        
        self.state_fc = nn.Sequential(nn.Linear(state_size, 64), 
                                      nn.Tanh())
        
        self.command_fc = nn.Sequential(nn.Linear(2, 64), 
                                        nn.Sigmoid())
        
        self.output_fc = nn.Sequential(nn.Linear(64, 128), 
                                       nn.ReLU(), 
                                       #nn.Dropout(0.2),
                                       nn.Linear(128, 128), 
                                       nn.ReLU(), 
                                       #nn.Dropout(0.2),
                                       nn.Linear(128, 128), 
                                       nn.ReLU(), 
                                       nn.Linear(128, action_size))   
    
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