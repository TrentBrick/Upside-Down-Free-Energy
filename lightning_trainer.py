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

        '''if hparams['desire_advantage']:
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

        # init the desired stats. 
        self.desired_reward_stats = (-10000000, -10000000)

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

        if self.hparams['desire_advantage']:
            self.advantage_model = AdvantageModel(self.hparams['STORED_STATE_SIZE'] )
        else: 
            self.advantage_model = None 
        # log the hparams. 
        if self.logger:
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

    def configure_optimizers(self):
        if self.hparams['desire_advantage']:
            opt = torch.optim.Adam( list(self.model.parameters())+list(self.advantage_model.parameters()) , lr=self.hparams['lr'])
        else:
            opt = torch.optim.Adam(self.model.parameters(), lr=self.hparams['lr'])
        return opt

    def collect_rollouts(self, greedy=False, 
            num_episodes=None, render=False):
        if self.current_epoch<self.hparams['random_action_epochs']:
            agent = Agent(self.hparams['gamename'], 
                take_rand_actions=True,
                discount_factor=self.hparams['discount_factor'],
                advantage_model=self.advantage_model,
                td_lambda=self.hparams['td_lambda'])
        else: 
            agent = Agent(self.hparams['gamename'], 
                model = self.model, 
                Levine_Implementation= self.Levine_Implementation,
                desired_reward_stats = self.desired_reward_stats, 
                desired_horizon = self.desired_horizon,
                desired_state = self.desired_state,
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
        reward_losses, to_desire, termination_times = output[0], output[1], output[2]

        print("termination times for rollouts are:", np.mean(termination_times), termination_times)
        print("first advantages from these rollouts:", np.mean(to_desire), to_desire)
        # modify the training data how I want to now while its in a list of rollouts. 
        # dictionary of items with lists inside of each rollout. 
        # add data to the buffer. 
        self.train_buffer.add_rollouts(train_data)
        self.test_buffer.add_rollouts(test_data)

        if self.Levine_Implementation:
            self.desired_horizon = None
            # Beta is 1. Otherwise would appear in the log sum exp here. 
            if self.hparams['desire_advantage']:
                # doing all of this inside of training step where I am 
                # already computing the advantage!
                pass
            else: 
                # TODO: get all of the starting discounted values from the whole buffer. 
                # not just the most recent values.
                '''if self.hparams['desire_cum_rew']: 
                    self.desired_reward_stats = [np.max(self.train_buffer.cum_rew[:self.train_buffer.size]), np.std(self.train_buffer.cum_rew[:self.train_buffer.size])]
                else: '''
                self.desired_reward_stats = [np.max(to_desire), np.std(to_desire)]

                if self.hparams['desire_mu_minus_std']:
                    self.desired_reward_stats[0] = self.desired_reward_stats[0]-self.desired_reward_stats[1]

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

        if self.Levine_Implementation and self.hparams['desire_advantage']:  
            # reset the desired stats. Important for Levine use advantage. 
            # do so after saving whatever had appeared before. 
            self.desired_reward_stats = (-10000000, -10000000)

    def on_epoch_end(self):
        # create new rollouts using stochastic actions. 
        output = self.collect_rollouts(num_episodes=self.hparams['training_rollouts_per_worker'])
        # process the data/add to the buffer.
        self.add_rollouts_to_buffer(output)

        # evaluate the agents where greedy actions are taken. 
        if self.current_epoch % self.hparams['eval_every']==0:
            output = self.collect_rollouts(greedy=True, num_episodes=self.hparams['eval_episodes'])
            reward_losses = output[0]
            self.logger.experiment.add_scalar("eval_mean", np.mean(reward_losses), self.global_step)

    def training_step(self, batch, batch_idx):
        # run training on this data
        if self.Levine_Implementation: 
            # TODO: input final obs here. it will be fine as the model knows when to ignore it. 
            # desire here is the discounted rewards to go. 
            obs, act = batch['obs'], batch['act']
            if self.hparams['desire_cum_rew']:
                des = batch['cum_rew']
            else: 
                des = batch['desire']
            
            if self.hparams['desire_states']:
                raise Exception("Still need to implement this. ")
            else: 
                desires = [des.unsqueeze(1), None]
        else:
            # desire here is the reward to go. 
            obs, final_obs, act, des, horizon = batch['obs'], batch['final_obs'], batch['act'], batch['desire'], batch['horizon']
            desires = [des.unsqueeze(1), horizon.unsqueeze(1), final_obs]
            #print(desires[0].shape, desires[1].shape, desires[2].shape, desires[2] )

        if self.hparams['desire_advantage']:
            if batch_idx%self.hparams['val_func_update_iterval']==0: 

                if self.hparams['use_lambda_td']:
                    # randomly sample indices from the buffer
                    # TODO: set the number of indices to sample from here. 
                    # NOTE: the number of values going into the NN will be changing. 
                    idxs = np.random.randint(0, self.train_buffer.size, 4)

                    obs_paths, td_lambda_paths = [], []
                    for idx in idxs: 
                        path_obs, path_rew = self.train_buffer.retrieve_path(idx)
                        path_obs = self.advantage_model.forward(path_obs).squeeze()
                        # compute TD lambda for this path: 
                        if len(path_obs.shape)==0:
                            path_obs = path_obs.unsqueeze(0)
                            td_lambda_target = path_rew
                        else: 
                            td_lambda_target = self.advantage_model.calculate_lambda_target(path_obs.detach(), path_rew,
                                                                self.hparams['discount_factor'], 
                                                                self.hparams['td_lambda'])
                        obs_paths.append(path_obs)
                        td_lambda_paths.append(td_lambda_target)
                    obs_paths = torch.cat(obs_paths, dim=0)
                    td_lambda_paths = torch.cat(td_lambda_paths, dim=0)

                    #pred_vals = self.advantage_model.forward(obs_paths).squeeze()
                    adv_loss = F.mse_loss(obs_paths, td_lambda_paths, reduction='none').mean(dim=0)

                    # to use for the calcs below. 
                    with torch.no_grad(): pred_vals = self.advantage_model.forward(obs).squeeze()

                else: 
                    pred_vals = self.advantage_model.forward(obs).squeeze()
                    # need to compute all of the TD lambda losses right here. 
                    #print('adv loss: pred vs real. ', pred_vals[0], des[0])
                    # des here is TD lambda as modified and set by the agent. 
                    adv_loss = F.mse_loss(pred_vals, des, reduction='none').mean(dim=0)
            else: 
                with torch.no_grad(): pred_vals = self.advantage_model.forward(obs).squeeze()

            # need to compute this here to use the most up to date V(s)
            # des is the rewards to go. 
            des = des - pred_vals.detach() # this is the advantage.

            # clamping it to prevent the desires and advantages 
            # from being too high. 
            if self.hparams['clamp_adv_to_max']:
                des = torch.clamp(des, max=50)

            desires[0] = des.unsqueeze(1)

            # set the desired rewards here 
            max_adv = float(des.max().numpy())
            if max_adv>= self.desired_reward_stats[0]:
                self.desired_reward_stats = ( max_adv, float(des.std().numpy()) )
                print("new max adv desired mu and std are:", self.desired_reward_stats)

        pred_action = self.model.forward(obs, desires)

        if not self.hparams['continuous_actions']:
            #pred_action = torch.sigmoid(pred_action)
            act = act.squeeze().long()

        pred_loss = self._pred_loss(pred_action, act)
        if self.hparams['Levine_Implementation'] and self.hparams['exp_weight_losses']:
            print("pre weight norm", des[-1])
            des = (des - des.mean()) / des.std()
            #print('weights used ', torch.exp(des/beta_reward_weighting))
            loss_weighting = torch.clamp( torch.exp(des/self.hparams['beta_reward_weighting']), max=self.hparams['max_loss_weighting'])
            print('loss weights post clamp', loss_weighting[-1], 'the reward itself.', des[-1])
            pred_loss = pred_loss*loss_weighting
        pred_loss = pred_loss.mean(dim=0)
        logs = {"policy_loss": pred_loss}

        # learn the advantage function too by adding it to the loss if this is the correct iteration. 
        if self.hparams['desire_advantage'] and batch_idx%self.hparams['val_func_update_iterval']==0:
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
        if self.hparams['desire_advantage']:
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