# pylint: disable=no-member
from models import UpsdModel, UpsdBehavior
import torch
import torch.nn.functional as F 
from torch.distributions import Normal, Categorical
import pytorch_lightning as pl 
from control import Agent 
import numpy as np
from torch.utils.data import DataLoader
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
        self.cum_iters_generated = 0
        self.mean_reward_over_20_epochs = []

        if self.Levine_Implementation:
            self.model = UpsdModel(self.config['STORED_STATE_SIZE'], 
            self.config['desires_size'], 
            self.config['ACTION_SIZE'], 
            self.config['NODE_SIZE'], desire_scalings=config['desire_scalings'])
        else: 
            self.model = UpsdBehavior( self.config['STORED_STATE_SIZE'], 
                self.config['ACTION_SIZE'], 
                self.config['NODE_SIZE'], config['desire_scalings'] )

        # start filling up the buffer.
        output = self.collect_rollouts() 
        self.add_rollouts_to_buffer(output)

    def forward(self,state, command):
        return self.model(state, command)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['lr'])
        return optimizer 

    def collect_rollouts(self):
        if self.current_epoch<self.config['random_action_epochs']:
            agent = Agent(self.config['gamename'], self.game_dir, 
                take_rand_actions=True,
                discount_factor=self.config['discount_factor'])
        else: 
            agent = Agent(self.config['gamename'], self.game_dir, 
                model = self.model, 
                Levine_Implementation= self.Levine_Implementation,
                desired_reward_stats = self.reward_from_epoch_stats, 
                desired_horizon = self.desired_horizon,
                desired_reward_dist_beta=self.config['desired_reward_dist_beta'],
                discount_factor=self.config['discount_factor'])
        
        seed = np.random.randint(0, 1e9, 1)[0]
        output = agent.simulate(seed, return_events=True,
                                compute_feef=True ,
                                num_episodes=self.config['training_rollouts_per_worker'])

        return output

    def add_rollouts_to_buffer(self, output):
        if self.Levine_Implementation: 
            train_data = combine_single_worker(output[2][:-1], 1 )
            test_data = {k:[v]for k, v in output[2][-1].items()}
        else: 
            train_data =output[2][:-1]
            test_data = [output[2][-1]]
        reward_losses, termination_times, feef_losses = output[0], output[1], output[3]

        # modify the training data how I want to now while its in a list of rollouts. 
        # dictionary of items with lists inside of each rollout. 
        # add data to the buffer. 
        iters_generated = 0
        if self.Levine_Implementation: 
            self.train_buffer.add_rollouts(train_data)
            self.test_buffer.add_rollouts(test_data)
        else: 
            for r in range(len(train_data)):
                self.train_buffer.add_sample(  train_data[r]['obs'], train_data[r]['act'], 
                    train_data[r]['rew'], termination_times[r] )
                iters_generated+= termination_times[r]

            for r in range(len(test_data)):
                self.test_buffer.add_sample(  test_data[r]['obs'], test_data[r]['act'], 
                    test_data[r]['rew'], len(test_data[r]['terminal']) )
        self.cum_iters_generated+= iters_generated

        if self.Levine_Implementation:
            self.desired_horizon = 99999 
            self.reward_from_epoch_stats = (np.mean(reward_losses), np.std(reward_losses))
        else: 
            last_few_mean_returns, last_few_std_returns, self.desired_horizon  = self.train_buffer.get_desires(last_few=self.config['last_few'])
            self.reward_from_epoch_stats = (last_few_mean_returns, last_few_std_returns)

        self.mean_reward_rollouts = np.mean(reward_losses)
        self.mean_reward_over_20_epochs.append( self.mean_reward_rollouts)

        if self.logger:
            self.logger.experiment.add_scalars('rollout_results', {"mean_reward":np.mean(reward_losses), "std_reward":np.std(reward_losses),
                "max_reward":np.max(reward_losses), "min_reward":np.min(reward_losses)}, self.global_step)
            self.logger.experiment.add_scalars('desires', {
                "desired_reward":self.reward_from_epoch_stats[0],
                "desired_horizon":self.desired_horizon}, self.global_step)
            self.logger.experiment.add_scalar("steps", self.cum_iters_generated, self.global_step)

    def on_epoch_start(self):
        output = self.collect_rollouts()
        # process the data/add to the buffer.
        self.add_rollouts_to_buffer(output)

    def training_step(self, batch, batch_idx):
        # run training on this data
        if self.Levine_Implementation: 
            obs, obs2, act, rew, terminal, terminal_rew, time = batch['obs'].squeeze(0), batch['obs2'].squeeze(0), batch['act'].squeeze(0), batch['rew'].squeeze(0), batch['terminal'].squeeze(0), batch['terminal_rew'].squeeze(0), batch['time'].squeeze(0)
        else: 
            obs, act, rew, time = batch['obs'].squeeze(0), batch['act'].squeeze(0), batch['rew'].squeeze(0), batch['time'].squeeze(0)
            # this is actually delta time. 
        
        if not self.Levine_Implementation: 
            desires = torch.cat([rew.unsqueeze(1), time.unsqueeze(1)], dim=1)
        pred_action = self.model.forward(obs, desires)
        if not self.config['continuous_actions']:
            #pred_action = torch.sigmoid(pred_action)
            act = act.squeeze().long()
        pred_loss = self._pred_loss(pred_action, act)
        if self.config['Levine_Implementation'] and self.config['weight_loss']:
            #print('weights used ', torch.exp(rew/desired_reward_dist_beta))
            pred_loss = pred_loss*torch.exp(terminal_rew/self.config.desired_reward_dist_beta)
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
        return DataLoader(self.train_buffer, batch_size=1)
    
    def val_dataloader(self):
        return DataLoader(self.test_buffer, batch_size=1)

    '''# evaluate every .... 
    if e%evaluate_every==0:
            print('======= Evaluating the agent')
            seed = np.random.randint(0, 1e9, 1)[0]
            cum_rewards, finish_times = agent.simulate(seed, num_episodes=5, greedy=True)
            print('Evaluation, mean reward:', np.mean(cum_rewards), 'mean horizon length:', np.mean(finish_times))
            print('===========================')

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