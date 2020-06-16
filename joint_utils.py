
import numpy as np
import sys 
from os.path import join, exists
from os import mkdir, unlink, listdir, getpid
from bisect import bisect
import torch
import torch.utils.data
from torchvision import transforms
import gym 
from utils.misc import ACTION_SIZE, LATENT_RECURRENT_SIZE, LATENT_SIZE
import time
import cma
from controller_model import Models, load_parameters, flatten_parameters
from ray.util.multiprocessing import Pool
#from multiprocessing import Pool 

class GeneratedDataset(torch.utils.data.Dataset):
    """ """
    def __init__(self, transform, data, seq_len): 
        self._transform = transform
        self.data = data
        self._cum_size = [0]
        self._buffer_index = 0
        self._seq_len = seq_len

        # set the cum size tracker by iterating through the data:
        for d in self.data['terminal']:
            #print(d, 'whats being added to cum size')
            self._cum_size += [self._cum_size[-1] +
                                   (len(d)-self._seq_len)]

    def __getitem__(self, i): # kind of like the modulo operator but within rollouts of batch size. 
        # binary search through cum_size
        rollout_index = bisect(self._cum_size, i) - 1 # because it finds the index to the right of the element. 
        # within a specific rollout. will linger on one rollout for a while
        seq_index = i - self._cum_size[rollout_index] # references the previous file length. so normalizes to within this file's length. 
        
        obs_data = self.data['obs'][rollout_index][seq_index:seq_index + self._seq_len + 1]
        if self._transform:
            obs_data = self._transform(obs_data.astype(np.float32))
        action = self.data['actions'][rollout_index][seq_index+1:seq_index + self._seq_len + 1]
        #action = action.astype(np.float32)
        reward, terminal = [self.data[key][rollout_index][seq_index:
                                      seq_index + self._seq_len + 1]#.astype(np.float32)
                            for key in ('rewards', 'terminal')]
        # this transform is already being done and saved. 
        #reward = np.expand_dims(reward, 1)
        return obs_data, action, reward.unsqueeze(1), terminal
        
    def __len__(self):
        return self._cum_size[-1]

def combine_worker_rollouts(inp, seq_len, dim=1):
    """Combine data across the workers and their rollouts (making each rollout a separate batch)
    I currently only care about the data_dict_list. This is a list of each rollout: 
    each one has the dictionary keys: 'obs', 'rewards', 'actions', 'terminal'"""
    print('RUNNING COMBINE WORKER')

    first_iter = True
    for worker_rollouts in inp: 
        for rollout_data_dict in worker_rollouts[dim]: # this is pulling from a list!
            # this rollout is too small so it is being ignored. 
            # getting one of the keys from the dictionary
            if len(rollout_data_dict[list(rollout_data_dict.keys())[0]])-seq_len <= 0:
                print('!!!!!! Combine_worker_rollouts is ignoring rollout of length', len(rollout_data_dict[list(rollout_data_dict.keys())[0]]), 'for being smaller than the sequence length')
                continue

            if first_iter:
                combo_dict = {k:[v] for k, v in rollout_data_dict.items()} 
                first_iter = False
            else: 
                for k, v in combo_dict.items():
                    v.append(rollout_data_dict[k])

    #combo_dict = {k:torch.stack(v) for k, v in combo_dict.items()}
    return combo_dict

def generate_rollouts_using_planner(cem_params, horizon, num_action_repeats, planner_n_particles, seq_len, 
    time_limit, logdir, num_rolls_per_worker=2, num_workers=16, transform=None, joint_file_dir=True): # this makes 32 pieces of data.

    # 10% of the rollouts to use for test data. 
    ten_perc = np.floor(num_workers*0.1)
    if ten_perc == 0.0:
        ten_perc=1
    ninety_perc = int(num_workers- ten_perc)

    # generate a random number to give as a random seed to each pool worker. 
    rand_ints = np.random.randint(0, 1e9, num_workers)

    worker_data = []
    #NOTE: currently not using joint_file_directory here (this is used for each worker to know to pull files from joint or from a subsection)
    for i in range(num_workers):
        worker_data.append( (cem_params, horizon, num_action_repeats, planner_n_particles, rand_ints[i], num_rolls_per_worker, time_limit, logdir, True ) ) # compute FEEF.

    #res = ray.get( [worker.remote() ] )
    with Pool(processes=num_workers) as pool:
        all_worker_outputs = pool.map(worker, worker_data) 

    res = []
    for ind, worker_output in enumerate(all_worker_outputs): 
        res.append(worker_output[0])
        if ind==0:
            cem_mus = worker_output[1]  
            cem_sigmas = worker_output[2]
        else:
            cem_mus += worker_output[1] 
            cem_sigmas += worker_output[2]
    # averaging across the workers for the new CEM parameters. 
    cem_mus /= num_workers
    cem_sigmas /= num_workers

    # cem_smoothing: 
    alpha_smoothing=0.2
    cem_mus = cem_mus*alpha_smoothing + (1-alpha_smoothing)*cem_params[0]
    cem_sigmas = cem_sigmas*alpha_smoothing + (1-alpha_smoothing)*cem_params[1]

    # res is a list with tuples for each worker containing: reward_list, data_dict_list, t_list
    print('===== Done with pool of workers')

    # get all of the feef losses aggregated across the different workers: 
    feef_losses, reward_list = [], []
    for worker_rollouts in res:
        for ind, li in zip([3, 0], [feef_losses, reward_list]): # output order from simulate.
            li += worker_rollouts[ind]

    return (cem_mus, cem_sigmas), GeneratedDataset(transform, combine_worker_rollouts(res[:ninety_perc], seq_len, dim=2), seq_len),  \
                GeneratedDataset(transform, combine_worker_rollouts(res[ninety_perc:], seq_len, dim=2), seq_len), \
                feef_losses, reward_list

#@ray.remote
def worker(inp): # run lots of rollouts 
    cem_params, horizon, num_action_repeats, planner_n_particles, seed, num_episodes, max_len, logdir, compute_feef = inp
    gamename = 'carracing'
    model = Models(gamename, 1000, mdir = logdir, conditional=True, 
            return_events=True, use_old_gym=False, joint_file_dir=True,
            planner_n_particles = planner_n_particles, cem_params=cem_params, 
            horizon=horizon, num_action_repeats=num_action_repeats)

    return model.simulate(train_mode=True, render_mode=False, 
            num_episode=num_episodes, seed=seed, max_len=max_len, 
            compute_feef=compute_feef), model.cem_mus, model.cem_sigmas


'''def train_controller(es, curr_best_ctrl_params, logdir, gamename, num_episodes, num_workers, 
    num_trials_per_worker, num_generations, seed_start=None, time_limit=1000, use_feef=True ):

    population_size = num_workers*num_trials_per_worker

    start_time = int(time.time())
    sprint("training", gamename)
    sprint("population", population_size)
    sprint("num_worker", num_workers)
    sprint("num_trials_per_worker", num_trials_per_worker)
    sys.stdout.flush()

    if not seed_start: 
        seed_start = np.random.randint(0,1e9, size=1)[0]
    seeder = Seeder(seed_start)

    filename_hist = join(logdir, 'ctrl_logger.txt')

    # making header for the history saving. 
    # the append writing operation is important here so memory is stored 
    # across the epochs of VAE and MDRNN training
    if not exists(filename_hist): 
        header_string = ""
        for k in ['generation', 'time', 'avg_rew', 'min_rew', 'max_rew',
                'std_rew', 'dist_std', 'mean_run_time', 'max_run_time',
                'avg_feef', 'min_feef', 'max_feef', 'std_feef']:
            header_string+=k+' '
        header_string+= '\n'
        with open(filename_hist, "w") as file:
            file.write(header_string)  

    t = 0
    antithetic=True
    history = []

    max_len = time_limit # max time steps (-1 means ignore)
    best_reward = -1000000000

    for generation in range(num_generations):
        t += 1

        #sprint('asking for solutions in master')
        solutions = es.ask()
        #sprint('============================= solutions from es.ask', solutions.shape)
        #sprint('getting seeds')
        if antithetic:
            seeds = seeder.next_batch(int(es.popsize/2))
            seeds = seeds+seeds
        else:
            seeds = seeder.next_batch(es.popsize)

        # start Pool and get rewards
        ###########################
        # generate a random number to give as a random seed to each pool worker. 
        #rand_ints = np.random.randint(0, 1e9, population_size)

        # TODO: ultimately these processes should live on to surive intializiation between CMA-ES rounds and so 
        # each worker can try different sets of parameters. But for now I am not doing this. 
        worker_data = []
        for i in range(population_size):
            worker_data.append( (solutions[i], seeds[i], num_episodes, max_len, logdir, True) ) # compute_feef=True

        with Pool(processes=num_workers) as pool:
            res = pool.map(worker, worker_data)

        # aggregating results for each worker by averaging the results of each of their episodes
        reward_list, times_taken, feef_losses = [], [], []
        for worker_rollouts in res:
            for ind, li in enumerate([reward_list, times_taken, feef_losses]):
                li.append( np.mean(worker_rollouts[ind]) )

        #for li in [reward_list, times_taken, feef_losses]:
        #    li = np.asarray(li)

        print('done with pool')
        ###########################

        mean_time_step = int(np.mean(times_taken)*100)/100. # get average time step
        max_time_step = int(np.max(times_taken)*100)/100.
        r_max = int(np.max(reward_list)*100)/100.
        r_min = int(np.min(reward_list)*100)/100.
        avg_reward = int(np.mean(reward_list)*100)/100. 
        std_reward = int(np.std(reward_list)*100)/100. 
        # TODO: implement the same rounding as above? 
        feef_max = np.max(feef_losses)
        feef_min = np.min(feef_losses)
        avg_feef = np.mean(feef_losses)
        std_feef = np.std(feef_losses)

        curr_time = int(time.time()) - start_time

        h = (t, curr_time, avg_reward, r_min, r_max, 
            std_reward, int(es.rms_stdev()*100000)/100000., 
            mean_time_step+1., int(max_time_step)+1., avg_feef, feef_min, feef_max, 
            std_feef )

        history.append(h)

        if use_feef:
            es.tell(feef_losses) # dont need to be converted to maximization because I set it to be a min now!
        else: 
            es.tell(-np.asarray(reward_list))


        if r_max > best_reward:
            best_reward=r_max 

        sprint('================================',gamename, h)

    #with open(filename_hist, 'wt') as out:
    #    res = json.dump(history, out, sort_keys=False, indent=0, separators=(',', ':'))

    with open(filename_hist, "a") as file:
        log_string = ""
        for h in history:
            for v in h:
                log_string += str(v)+' '
            log_string+= '\n'
        file.write(log_string)

    if use_feef:
        index_min = np.argmin(feef_losses)
        model_params = solutions[index_min]
    else: 
        index_max = np.argmax(reward_list)
        model_params = solutions[index_max]
        #best_reward = np.max(reward_list)
    best_feef = np.min(feef_losses)
    


    #model_params = es.result[0] # best solution of all time
    #best_feef = es.result[1] # best feef reward of all time
    #best_curr_feef = es.result[2] # best feef of the current batch

    return es, model_params, best_feef, best_reward
'''