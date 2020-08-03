# pylint: disable=no-member
from models import UpsdModel, UpsdBehavior
import torch
import torch.nn.functional as F 
from torch.distributions import Normal, Categorical
import pytorch_lightning as pl 
from control import Agent 
import numpy as np
from torch.utils.data import DataLoader, RandomSampler, BatchSampler
from utils import save_checkpoint, generate_model_samples, \
    generate_rollouts, write_logger, ReplayBuffer, \
    RingBuffer, combine_single_worker

class LightningTemplate(pl.LightningModule):

    def __init__(self, game_dir, config, train_buffer, test_buffer):
        super().__init__()

        self.game_dir = game_dir
        self.Levine_Implementation = config['Levine_Implementation']
        self.config = config
        self.train_buffer = train_buffer
        self.test_buffer = test_buffer
        self.mean_reward_over_20_epochs = []

        if self.config['use_Levine_model']:
            self.model = UpsdModel(self.config['STORED_STATE_SIZE'], 
            self.config['desires_size'], 
            self.config['ACTION_SIZE'], 
            self.config['hidden_sizes'], desires_scalings=None, 
            desire_states=self.config['desire_states'])
        else: 
            # concatenate all of these lists together. 
            desires_scalings = [config['reward_scale']]+[config['horizon_scale']]+ config['state_scale']
            self.model = UpsdBehavior( self.config['STORED_STATE_SIZE'], 
                self.config['desires_size'],
                self.config['ACTION_SIZE'], 
                self.config['hidden_sizes'], 
                desires_scalings,
                desire_states=self.config['desire_states'] )
 
        # log the hparams. 
        if self.logger:
            self.logger.experiment.add_hparams(config)

        # start filling up the buffer.
        output = self.collect_rollouts(num_episodes=self.config['num_rand_action_rollouts']) 
        self.add_rollouts_to_buffer(output)
    
    def eval_agent(self):
        self.desired_horizon = 285
        self.desired_reward_stats = (319, 1)
        self.desired_state = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1., 1.]
        print('Desired Reward and Horizon are:', self.desired_horizon, self.desired_reward_stats, 
        self.desired_state)
        self.current_epoch = self.config['random_action_epochs']+1
        output = self.collect_rollouts(num_episodes=100, greedy=True, render=True  ) 

    def forward(self,state, command):
        return self.model(state, command)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['lr'])
        return optimizer 

    def collect_rollouts(self, greedy=False, 
            num_episodes=None, render=False):
        if self.current_epoch<self.config['random_action_epochs']:
            agent = Agent(self.config['gamename'], 
                take_rand_actions=True,
                discount_factor=self.config['discount_factor'])
        else: 
            agent = Agent(self.config['gamename'], 
                model = self.model, 
                Levine_Implementation= self.Levine_Implementation,
                desired_reward_stats = self.desired_reward_stats, 
                desired_horizon = self.desired_horizon,
                desired_state = self.desired_state,
                desired_reward_dist_beta=self.config['desired_reward_dist_beta'],
                discount_factor=self.config['discount_factor'])
        
        seed = np.random.randint(0, 1e9, 1)[0]
        print('seed used for agent simulate:', seed )
        output = agent.simulate(seed, return_events=True,
                                num_episodes=num_episodes,
                                greedy=greedy, render_mode=render)

        return output

    def add_rollouts_to_buffer(self, output):
        
        train_data =output[2][:-1]
        test_data = [output[2][-1]]
        reward_losses, termination_times = output[0], output[1]

        # modify the training data how I want to now while its in a list of rollouts. 
        # dictionary of items with lists inside of each rollout. 
        # add data to the buffer. 
        self.train_buffer.add_rollouts(train_data)
        self.test_buffer.add_rollouts(test_data)

        if self.Levine_Implementation:
            self.desired_horizon = None
            self.desired_reward_stats = (np.mean(reward_losses), np.std(reward_losses))
            self.desired_state = np.unique(self.train_buffer.final_obs).mean(axis=0) # take the mean or sample from everything. 
        else: 
            # TODO: return mean and std to sample from the desired states. 
            last_few_mean_returns, last_few_std_returns, self.desired_horizon, self.desired_state = self.train_buffer.get_desires(last_few=self.config['last_few'])
            self.desired_reward_stats = (last_few_mean_returns, last_few_std_returns)

        self.mean_reward_rollouts = np.mean(reward_losses)
        self.mean_reward_over_20_epochs.append( self.mean_reward_rollouts)

        if self.logger:
            self.logger.experiment.add_scalar("mean_reward", np.mean(reward_losses), self.global_step)
            self.logger.experiment.add_scalars('rollout_stats', {"std_reward":np.std(reward_losses),
                "max_reward":np.max(reward_losses), "min_reward":np.min(reward_losses)}, self.global_step)
            
            to_write = {
                "reward":self.desired_reward_stats[0]
                    }
            if self.desired_horizon: 
                to_write["horizon"]=self.desired_horizon
            self.logger.experiment.add_scalars('desires', to_write, self.global_step)
            self.logger.experiment.add_scalar("steps", self.train_buffer.total_num_steps_added, self.global_step)

    def on_epoch_end(self):
        # create new rollouts using stochastic actions. 
        output = self.collect_rollouts(num_episodes=self.config['training_rollouts_per_worker'])
        # process the data/add to the buffer.
        self.add_rollouts_to_buffer(output)

        # evaluate the agents
        if self.current_epoch % self.config['eval_every']==0:
            output = self.collect_rollouts(greedy=True, num_episodes=self.config['eval_episodes'])
            reward_losses = output[0]
            self.logger.experiment.add_scalar("eval_mean", np.mean(reward_losses), self.global_step)

    def training_step(self, batch, batch_idx):
        # run training on this data
        if self.Levine_Implementation: 
            # TODO: input final obs here. it will be fine as the model knows when to ignore it. 
            obs, act, rew = batch['obs'], batch['act'], batch['cum_rew']
            desires = [rew.unsqueeze(1), None]
        else:
            obs, final_obs, act, rew, horizon = batch['obs'], batch['final_obs'], batch['act'], batch['cum_rew'], batch['horizon']
            if not self.config['sparse']: 
                rew = batch['rew']
                # need to uncomment form the Sorted Buffer if want back. 
            desires = [rew.unsqueeze(1), horizon.unsqueeze(1), final_obs]
        #print(desires[0].shape, desires[1].shape, desires[2].shape, desires[2] )
        pred_action = self.model.forward(obs, desires)
        if not self.config['continuous_actions']:
            #pred_action = torch.sigmoid(pred_action)
            act = act.squeeze().long()
        pred_loss = self._pred_loss(pred_action, act)
        if self.config['Levine_Implementation'] and self.config['weight_loss']:
            #print('weights used ', torch.exp(rew/desired_reward_dist_beta))
            pred_loss = pred_loss*torch.exp(rew/self.config['desired_reward_dist_beta'])
        pred_loss = pred_loss.mean(dim=0)
        #if return_for_model_sampling: 
        #    return pred_loss, (rew[0:5], pred_action[0:5], pred_action[0:5].argmax(-1), act[0:5])
        
        logs = {"train_loss": pred_loss}
        return {'loss':pred_loss, 'log':logs}

    def _pred_loss(self, pred_action, real_action):
        if self.config['continuous_actions']:
            # add a sigmoid activation layer.: 
            return F.mse_loss(pred_action, real_action ,reduction='none').sum(dim=1)
        else: 
            return F.cross_entropy(pred_action, real_action, reduction='none')

    def validation_step(self, batch, batch_idx):
        train_dict = self.training_step(batch, batch_idx)
        # rename
        train_dict['log']['val_loss'] = train_dict['log'].pop('train_loss')
        return train_dict['log'] 

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {
                "avg_val_loss": avg_loss,
                "log": tensorboard_logs
                }

    def train_dataloader(self):
        bs = BatchSampler( RandomSampler(self.train_buffer, replacement=True, 
                    num_samples= self.config['num_grad_steps']*self.config['batch_size']  ), 
                    batch_size=self.config['batch_size'], drop_last=False )
        return DataLoader(self.train_buffer, batch_sampler=bs)
    
    def val_dataloader(self):
        bs = BatchSampler( RandomSampler(self.test_buffer, replacement=True, 
                    num_samples= self.config['num_val_batches']*self.config['batch_size']  ), 
                    batch_size=self.config['batch_size'], drop_last=False )
        return DataLoader(self.test_buffer, batch_sampler=bs)

    '''
    if make_vae_samples:
            generate_model_samples( model, for_upsd_sampling, 
                            samples_dir, SEQ_LEN, self.config['IMAGE_RESIZE_DIM'],
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