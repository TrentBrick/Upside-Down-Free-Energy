# pylint: disable=no-member
from models import UpsdModel, UpsdBehavior, AdvantageModel
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

    def __init__(self, game_dir, hparams, train_buffer, test_buffer):
        super().__init__()

        '''if hparams['use_advantage']:
            self.training_step = self.training_step_multi_model
            self.optimizer_step = self.optimizer_step_multi
        else: 
            self.training_step = self.training_step_single_model'''

        self.game_dir = game_dir
        self.Levine_Implementation = hparams['Levine_Implementation']
        self.hparams = hparams
        self.train_buffer = train_buffer
        self.test_buffer = test_buffer
        self.mean_reward_over_20_epochs = []

        if self.hparams['use_Levine_model']:
            self.model = UpsdModel(self.hparams['STORED_STATE_SIZE'], 
            self.hparams['desires_size'], 
            self.hparams['ACTION_SIZE'], 
            self.hparams['hidden_sizes'], desires_scalings=None, 
            desire_states=self.hparams['desire_states'])
        else: 
            # concatenate all of these lists together. 
            desires_scalings = [hparams['reward_scale']]+[hparams['horizon_scale']]+ hparams['state_scale']
            self.model = UpsdBehavior( self.hparams['STORED_STATE_SIZE'], 
                self.hparams['desires_size'],
                self.hparams['ACTION_SIZE'], 
                self.hparams['hidden_sizes'], 
                desires_scalings,
                desire_states=self.hparams['desire_states'] )

        if self.hparams['use_advantage']:
            self.advantage_model = AdvantageModel(self.hparams['STORED_STATE_SIZE'] )
        else: 
            self.advantage_model = None 
        # log the hparams. 
        if self.logger:
            print("Adding the hparams to the logger!!!")
            self.logger.experiment.add_hparams(hparams)

        # start filling up the buffer.
        output = self.collect_rollouts(num_episodes=self.hparams['num_rand_action_rollouts']) 
        self.add_rollouts_to_buffer(output)
    
    def eval_agent(self):
        self.desired_horizon = 285
        self.desired_reward_stats = (319, 1)
        self.desired_state = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1., 1.]
        print('Desired Horizon and Rewards are:', self.desired_horizon, self.desired_reward_stats, 
        self.desired_state)
        self.current_epoch = self.hparams['random_action_epochs']+1
        output = self.collect_rollouts(num_episodes=100, greedy=True, render=True  ) 

    def forward(self,state, command):
        return self.model(state, command)

    '''def optimizer_step_multi(self, current_epoch, batch_nb, optimizer, 
        optimizer_i, second_order_closure=None, on_tpu=False, 
        using_native_amp=False, using_lbfgs=False):

        if optimizer_i == 1: # for the 2nd optimizer which is the 
            # advantage function, only update it every 
            # 5th step, ie 200 grad updates each round. 
            if batch_nb % 5 == 0:
                optimizer.step()
                optimizer.zero_grad()
        else: 
            optimizer.step()
            optimizer.zero_grad()'''

    def configure_optimizers(self):
        if self.hparams['use_advantage']:
            opt = torch.optim.Adam( list(self.model.parameters())+list(self.advantage_model.parameters()) , lr=self.hparams['lr'])
        else:
            opt = torch.optim.Adam(self.model.parameters(), lr=self.hparams['lr'])
        return opt

    def collect_rollouts(self, greedy=False, 
            num_episodes=None, render=False):
        if self.current_epoch<self.hparams['random_action_epochs']:
            agent = Agent(self.hparams['gamename'], 
                take_rand_actions=True,
                discount_factor=self.hparams['discount_factor'])
        else: 
            agent = Agent(self.hparams['gamename'], 
                model = self.model, 
                Levine_Implementation= self.Levine_Implementation,
                desired_reward_stats = self.desired_reward_stats, 
                desired_horizon = self.desired_horizon,
                desired_state = self.desired_state,
                #beta_reward_weighting=self.hparams['beta_reward_weighting'],
                discount_factor=self.hparams['discount_factor'], 
                advantage_model=self.advantage_model,
                td_lambda=self.hparams['td_lambda'])
        
        seed = np.random.randint(0, 1e9, 1)[0]
        print('seed used for agent simulate:', seed )
        output = agent.simulate(seed, return_events=True,
                                num_episodes=num_episodes,
                                greedy=greedy, render_mode=render)

        return output

    def add_rollouts_to_buffer(self, output):
        
        train_data =output[3][:-1]
        test_data = [output[3][-1]]
        reward_losses, discounted_rewards, termination_times = output[0], output[1], output[2]

        # modify the training data how I want to now while its in a list of rollouts. 
        # dictionary of items with lists inside of each rollout. 
        # add data to the buffer. 
        self.train_buffer.add_rollouts(train_data)
        self.test_buffer.add_rollouts(test_data)

        if self.Levine_Implementation:
            self.desired_horizon = None
            # Beta is 1. Otherwise would appear in the log sum exp here. 
            self.desired_reward_stats = (np.log(np.sum(np.exp(discounted_rewards))), np.std(discounted_rewards))
            self.desired_state = np.unique(self.train_buffer.final_obs).mean(axis=0) # take the mean or sample from everything. 
        else: 
            # TODO: return mean and std to sample from the desired states. 
            last_few_mean_returns, last_few_std_returns, self.desired_horizon, self.desired_state = self.train_buffer.get_desires(last_few=self.hparams['last_few'])
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
        output = self.collect_rollouts(num_episodes=self.hparams['training_rollouts_per_worker'])
        # process the data/add to the buffer.
        self.add_rollouts_to_buffer(output)

        # evaluate the agents
        if self.current_epoch % self.hparams['eval_every']==0:
            output = self.collect_rollouts(greedy=True, num_episodes=self.hparams['eval_episodes'])
            reward_losses = output[0]
            self.logger.experiment.add_scalar("eval_mean", np.mean(reward_losses), self.global_step)

    def training_step(self, batch, batch_idx, run_eval=False):
        # run training on this data
        if self.Levine_Implementation: 
            # TODO: input final obs here. it will be fine as the model knows when to ignore it. 
            obs, obs2, act, rew = batch['obs'], batch['obs2'], batch['act'], batch['rew']
            if self.hparams['desire_states']:
                raise Exception("Still need to implement this. ")
            else: 
                desires = [rew.unsqueeze(1), None]
        else:
            obs, final_obs, act, rew, horizon = batch['obs'], batch['final_obs'], batch['act'], batch['rew'], batch['horizon']
            desires = [rew.unsqueeze(1), horizon.unsqueeze(1), final_obs]
            #print(desires[0].shape, desires[1].shape, desires[2].shape, desires[2] )

        if self.hparams['use_advantage']:
            if batch_idx%self.hparams['val_func_update_iterval']==0: 
                pred_vals = self.advantage_model.forward(obs2).squeeze()
                if self.hparams['norm_advantage']:
                    rew_norm = (rew - rew.mean()) / rew.std()
                    # compute loss for advantage model.
                    adv_loss = F.mse_loss(pred_vals, rew_norm ,reduction='none').mean(dim=0)
                else: 
                    print('adv loss: pred vs real. ', pred_vals[0], rew[0])
                    adv_loss = F.mse_loss(pred_vals, rew ,reduction='none').mean(dim=0)
            else: 
                with torch.no_grad():
                    # just get the predictions but without a gradient. 
                    # could have stored and computed this inside the buffer. 
                    pred_vals = self.advantage_model.forward(obs2).squeeze()

            # detach for use in the desires
            # TD-lambda is the reward value. 
            adv = rew - pred_vals.detach()
            if self.hparams['norm_advantage']:
                adv = (adv - adv.mean()) / adv.std()
            # set it as a desire. 
            print("advantage in desire:", adv[0])
            desires[0] = adv.unsqueeze(1)

        pred_action = self.model.forward(obs, desires)

        if not self.hparams['continuous_actions']:
            #pred_action = torch.sigmoid(pred_action)
            act = act.squeeze().long()
        pred_loss = self._pred_loss(pred_action, act)
        if self.hparams['Levine_Implementation'] and self.hparams['weight_loss']:
            #print('weights used ', torch.exp(rew/beta_reward_weighting))
            loss_weighting = torch.clamp( torch.exp(rew/self.hparams['beta_reward_weighting']), max=self.hparams['max_loss_weighting'])
            print('loss weights post clamp and their rewards', loss_weighting[-1], rew[-1])
            pred_loss = pred_loss*loss_weighting
        pred_loss = pred_loss.mean(dim=0)
        logs = {"policy_loss": pred_loss}

        if self.hparams['use_advantage'] and batch_idx%self.hparams['val_func_update_iterval']==0:
            pred_loss += adv_loss 
            logs["advantage_loss"] = adv_loss
            
        return {'loss':pred_loss, 'log':logs}

    def _pred_loss(self, pred_action, real_action):
        if self.hparams['continuous_actions']:
            # add a sigmoid activation layer.: 
            return F.mse_loss(pred_action, real_action ,reduction='none').sum(dim=1)
        else: 
            return F.cross_entropy(pred_action, real_action, reduction='none')

    def validation_step(self, batch, batch_idx):
        batch_idx=0 # so that advantage_val_loss is always called. 
        train_dict = self.training_step(batch, batch_idx)
        train_dict['log']['policy_val_loss'] = train_dict['log'].pop('policy_loss')
        if self.hparams['use_advantage']:
            train_dict['log']['advantage_val_loss'] = train_dict['log'].pop('advantage_loss')
        return train_dict['log'] 

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["policy_val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {
                "avg_val_loss": avg_loss,
                "log": tensorboard_logs
                }

    def train_dataloader(self):
        bs = BatchSampler( RandomSampler(self.train_buffer, replacement=True, 
                    num_samples= self.hparams['num_grad_steps']*self.hparams['batch_size']  ), 
                    batch_size=self.hparams['batch_size'], drop_last=False )
        return DataLoader(self.train_buffer, batch_sampler=bs)
    
    def val_dataloader(self):
        bs = BatchSampler( RandomSampler(self.test_buffer, replacement=True, 
                    num_samples= self.hparams['num_val_batches']*self.hparams['batch_size']  ), 
                    batch_size=self.hparams['batch_size'], drop_last=False )
        return DataLoader(self.test_buffer, batch_sampler=bs)

    '''
    if make_vae_samples:
            generate_model_samples( model, for_upsd_sampling, 
                            samples_dir, SEQ_LEN, self.hparams['IMAGE_RESIZE_DIM'],
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