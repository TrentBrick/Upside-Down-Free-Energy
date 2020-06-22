import numpy as np
import torch

# copying planner from execute_environment.py 
def calc_rew(actions, timestep):
    #print('actions in CALC REW shape', actions.shape)
    #print('calc rew optimal action', optimal_actions[timestep][0])
    rewards =  - torch.nn.functional.mse_loss(actions, optimal_actions[timestep], reduction='none' )
    #print('calc rew min and max reward actions', actions[torch.argmax(rewards),0] , actions[torch.argmin(rewards),0] )
    #print('negative mses', rewards, rewards.shape, rewards.sum(dim=-1).shape)
    #print('calc rew max reward and min', torch.max(rewards), torch.min(rewards))
    rewards = rewards.sum(dim=-1)
    #print(rewards.shape)
    return rewards

def planner(starting_time):
    # predicts into the future up to horizon. 
    # returns the immediate action that will lead to the largest
    # cumulative reward

    # starting CEM from scratch each time 
    cem_mus, cem_sigmas = init_cem_params[0], init_cem_params[1]

    for cem_iter in range(cem_iters):

        all_particles_cum_rewards = torch.zeros((planner_n_particles))
        all_particles_sequential_actions = torch.zeros((planner_n_particles, horizon, cem_mus.shape[0]))

        for t in range(0, horizon):
            # get the actual reward
            ens_action = sample_cross_entropy_method(cem_mus, cem_sigmas) 
            ens_reward = calc_rew(ens_action,starting_time+t)

            # store these cumulative rewards and action
            all_particles_cum_rewards += (discount_factor**t)*ens_reward
            all_particles_sequential_actions[:, t, :] = ens_action

        cem_mus, cem_sigmas = update_cross_entropy_method(all_particles_sequential_actions, 
                                    all_particles_cum_rewards, cem_mus, cem_sigmas)

    # choose the best next action out of all of them. 
    best_actions_ind = torch.argmax(all_particles_cum_rewards)
    best_action = all_particles_sequential_actions[best_actions_ind, 0, :]
    print("After ITER:", cem_iter, 'cem params are:', cem_mus, cem_sigmas)
    print('best sequence of actions at time:', starting_time, all_particles_sequential_actions[best_actions_ind, :, :])
    #print('best action is:', best_action)
    return best_action.unsqueeze(0)

def sample_cross_entropy_method(cem_mus, cem_sigmas):
    actions = torch.distributions.Normal(cem_mus, cem_sigmas).sample([planner_n_particles])
    #print('sample cross ent actions:', actions[:,0])
    # constrain these actions:
    actions = constrain_actions(actions)
    #print('actions from sample cross entropy method:' ,actions.shape)
    return actions

def update_cross_entropy_method(all_actions, rewards, cem_mus, cem_sigmas):
    # for carracing we have 3 independent gaussians
    #print('update cross entropy, rewards', rewards.shape)
    smoothing = 0.5
    vals, inds = torch.topk(rewards, k_top, sorted=False )
    elite_actions = all_actions[inds]
    #print('elite actions shape', elite_actions.shape, elite_actions, vals )

    num_elite_actions = k_top*horizon 

    new_mu = elite_actions.sum(dim=(0,1))/num_elite_actions
    new_sigma = torch.sqrt(torch.sum( (elite_actions - new_mu)**2, dim=(0,1))/num_elite_actions)
    cem_mus = smoothing*new_mu + (1-smoothing)*(cem_mus) 
    cem_sigmas = smoothing*new_sigma+(1-smoothing)*(cem_sigmas )
    return cem_mus, cem_sigmas

def constrain_actions(out):
    out[:,0] = torch.clamp(out[:,0], min=-1.0, max=1.0)
    out[:,1] = torch.clamp(out[:,1], min=0.0, max=1.0)
    out[:,2] = torch.clamp(out[:,2], min=0.0, max=1.0)
    #print('after all processing', out)
    return out

optimal_actions = [ torch.Tensor([0.9, 0.8,0.6]), 
                    torch.Tensor([0.4, 0.9,0.1]),
                    torch.Tensor([-0.2, 0.6,0.4]),
                    torch.Tensor([0.4, 0.4,0.95]),
                    torch.Tensor([0.4, 0.4,0.95]),
                    torch.Tensor([-0.95, 0.1,0.1])  ]
# reward corresponds to mse with the optimal action. 

# test if the reward function is working: 
proposed_actions = torch.stack([ torch.Tensor([0.9, 0.8,0.6]), 
                    torch.Tensor([0.4, 0.9,0.1]),
                    torch.Tensor([-0.8, 0.05,0.9]),
                    ])
print('proposed actions to ensure the MSE loss is working', proposed_actions)
for i in range(3):
    print('rewards are:', calc_rew(proposed_actions, i) )

init_cem_params = ( torch.Tensor([0,0.5,0.5]), torch.Tensor([0.5,0.7,0.5]) )

cem_mus, cem_sigmas = init_cem_params[0], init_cem_params[1]
horizon = 3
cem_iters =30
planner_n_particles = 2000
discount_factor=1.0
k_top = int(planner_n_particles*0.1)

# test if the planner is working
for i in range(1):
    best_action = planner(i)
    print('best action at time:', i, 'is:', best_action)
    print('the real underlying action to match is:', optimal_actions[i])
    print('============================')
