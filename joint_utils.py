
import numpy as np
from os.path import join, exists
from os import mkdir, unlink, listdir, getpid
from multiprocessing import Pool 
from bisect import bisect
import torch
import torch.utils.data
from torchvision import transforms
import gym 
from utils.misc import RolloutGenerator, ACTION_SIZE, LATENT_RECURRENT_SIZE, LATENT_SIZE
from ha_es import CMAES
import time
import cma
from controller_model import Models, load_parameters, flatten_parameters

class GeneratedDataset(torch.utils.data.Dataset): # pylint: disable=too-few-public-methods
    def __init__(self, transform, data, seq_len): 
        self._transform = transform
        self.data = data
        self._cum_size = None
        self._buffer_index = 0
        self._seq_len = seq_len

        # set the cum size tracker by iterating through the data:
        for d in self.data['term']:
            self._cum_size += [self._cum_size[-1] +
                                   (d.shape[0]-self._seq_len)]

    def __getitem__(self, i): # kind of like the modulo operator but within rollouts of batch size. 
        # binary search through cum_size
        rollout_index = bisect(self._cum_size, i) - 1 # because it finds the index to the right of the element. 
        # within a specific rollout. will linger on one rollout for a while
        seq_index = i - self._cum_size[rollout_index] # references the previous file length. so normalizes to within this file's length. 
        
        obs_data = self.data['obs'][rollout_index][seq_index:seq_index + self._seq_len + 1]
        if self._transform:
            obs_data = self._transform(obs_data.astype(np.float32))
        action = self.data['act'][rollout_index][seq_index+1:seq_index + self._seq_len + 1]
        action = action.astype(np.float32)
        reward, terminal = [self.data[key][rollout_index][seq_index:
                                      seq_index + self._seq_len + 1].astype(np.float32)
                            for key in ('rew', 'term')]
        # this transform is already being done and saved. 
        #reward = np.expand_dims(reward, 1)
        return obs_data, action, reward, terminal
        
    def __len__(self):
        return self._cum_size[-1]

def combine_worker_rollouts(inp, dim=2):
    """Combine data across the workers and their rollouts (making each rollout a separate batch)
    I currently only care about the data_dict_list. This is a list of each rollout: 
    each one has the dictionary keys: 'obs', 'rewards', 'actions', 'terminal'"""
    first_iter = True
    for worker_rollouts in inp: 
        for rollout_data_dict in worker_rollouts[dim]:
            if first_iter:
                combo_dict = rollout_data_dict
                first_iter = False
            else: 
                combo_dict = {k:v.append(rollout_data_dict[k]) for k, v in combo_dict.items()}
    
    #combo_dict = {k:torch.stack(v) for k, v in combo_dict.items()}
    return combo_dict

def generate_rollouts(ctrl_params, seq_len, 
    time_limit, logdir, num_rolls=2, num_workers=16, transform=None): # this makes 32 pieces of data.

    # 10% of the rollouts to use for test data. 
    ninety_perc = ten_perc - np.floor(num_workers*0.1)

    # generate a random number to give as a random seed to each pool worker. 
    rand_ints = np.random.randint(0, 1e9, num_workers)

    worker_data = []
    for i in range(num_workers):
        worker_data.append( (ctrl_params, rand_ints[i], num_rolls, time_limit, logdir, False ) ) # compute FEEF.

    with Pool(processes=num_workers) as pool:
        res = pool.map(worker, worker_data) 
    # res is a list with tuples for each worker containing: reward_list, data_dict_list, t_list
    print('done with pool')
    
    return GeneratedDataset(transform, combine_worker_rollouts(res[:ninety_perc], dim=2) , seq_len),  \
                GeneratedDataset(transform, combine_worker_rollouts(res[ninety_perc:], dim=2), seq_len)

def worker(inp): # run lots of rollouts 
    ctrl_params, seed, num_episodes, max_len, logdir, compute_feef = inp
    print('worker has started')
    gamename = 'carracing'
    model = Models(gamename, 1000, mdir = logdir, conditional=True, 
            return_events=True, use_old_gym=False)

    model.make_env(seed)

    return model.simulate(ctrl_params, train_mode=True, render_mode=False, 
            num_episode=num_episodes, seed=seed, max_len=max_len, compute_feef=compute_feef)
    


def sprint(*args):
    print(*args) # if python3, can do print(*args)
    sys.stdout.flush()

class Seeder:
    def __init__(self, init_seed=0):
        np.random.seed(init_seed)
        self.limit = np.int32(2**31-1)
    def next_seed(self):
        result = np.random.randint(self.limit)
        return result
    def next_batch(self, batch_size):
        result = np.random.randint(self.limit, size=batch_size).tolist()
        return result

def train_controller(logdir, gamename, num_episodes, num_workers, num_trials_per_worker, seed_start=27, time_limit=1000 ):

    population_size = num_workers*num_trials_per_worker

    start_time = int(time.time())
    sprint("training", gamename)
    sprint("population", population_size)
    sprint("num_worker", num_workers)
    sprint("num_trials_per_worker", num_trials_per_worker)
    sys.stdout.flush()

    seeder = Seeder(seed_start)

    filebase = join(logdir, 'ctrl_')
    filename = filebase+'.json'
    filename_log = filebase+'.log.json'
    filename_hist = filebase+'.hist.json'
    filename_hist_best = filebase+'.hist_best.json'
    filename_best = filebase+'.best.json'

    controller = Controller(LATENT_SIZE, LATENT_RECURRENT_SIZE, ACTION_SIZE, conditional=True).to(self.device)

    sigma_init=0.1
    num_params = len( flatten_parameters(controller.parameters()) )

    es = CMAES(num_params,
            sigma_init=sigma_init,
            popsize=population_size)

    t = 0
    history = []
    history_best = []
    eval_log = []
    best_reward_eval = 0
    best_model_params_eval = None

    max_len = time_limit # max time steps (-1 means ignore)

    while True:
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
        rand_ints = np.random.randint(0, 1e9, population_size)

        # TODO: ultimately these processes should live on to surive intializiation between CMA-ES rounds and so 
        # each worker can try different sets of parameters. But for now I am not doing this. 
        worker_data = []
        for i in range(population_size):
            worker_data.append( (solutions[i], rand_ints[i], num_episodes, max_len, logdir, True) ) # compute_feef=True

        with Pool(processes=num_workers) as pool:
            res = pool.map(worker, worker_data)

        reward_list, times_taken, feef_losses = [], [], []
        for worker_rollouts in res:
            for ind, listt in enumerate([reward_list, times_taken, feef_losses]):
                listt.append( np.mean(worker_rollouts[ind] )

        for listt in [reward_list, times_taken, feef_losses]:
            listt = np.asarray(listt)

        print('done with pool')
        ###########################

        mean_time_step = int(np.mean(times_taken)*100)/100. # get average time step
        max_time_step = int(np.max(times_taken])*100)/100.
        r_max = int(np.max(reward_list)*100)/100.
        r_min = int(np.min(reward_list)*100)/100.
        avg_reward = int(np.mean(reward_list)*100)/100. 
        std_reward = int(np.std(reward_list)*100)/100. 
        # TODO: implement the same rounding as above? 
        fem_max = np.max(feef_losses)
        fem_min = np.min(feef_losses)
        avg_fem = np.mean(feef_losses)
        std_fem = np.std(feef_losses)

        curr_time = int(time.time()) - start_time

        h = (t, curr_time, avg_reward, r_min, r_max, 
                std_reward, int(es.rms_stdev()*100000)/100000., 
                mean_time_step+1., int(max_time_step)+1., fem_min, fem_max, 
                avg_fem, std_fem  )

        history.append(h)

        es.tell(feef_losses)

        es_solution = es.result()
        model_params = es_solution[0] # best historical solution
        reward = es_solution[1] # best reward
        curr_reward = es_solution[2] # best of the current batch

        # NOTE: update the model parameters here. Why are they quantized and set here? 
        #model.controller = load_parameters(np.array(model_params).round(4), model.controller)
        
        if cap_time_mode:
            max_len = 2*int(mean_time_step+1.0)
        else:
            max_len = -1

        with open(filename, 'wt') as out:
            res = json.dump([np.array(es.current_param()).round(4).tolist()], out, sort_keys=True, indent=2, separators=(',', ': '))

        with open(filename_hist, 'wt') as out:
            res = json.dump(history, out, sort_keys=False, indent=0, separators=(',', ':'))

        sprint('================================',gamename, h)

        if (t == 1):
            best_reward_eval = avg_reward
        if (t % eval_steps == 0): # evaluate on actual task at hand

            prev_best_reward_eval = best_reward_eval
            model_params_quantized = np.array(es.current_param()).round(4)
            reward_eval = evaluate_batch(model_params_quantized, max_len=-1)
            model_params_quantized = model_params_quantized.tolist()
            improvement = reward_eval - best_reward_eval
            eval_log.append([t, reward_eval, model_params_quantized])
            with open(filename_log, 'wt') as out:
                res = json.dump(eval_log, out)
            if (len(eval_log) == 1 or reward_eval > best_reward_eval):
                best_reward_eval = reward_eval
                best_model_params_eval = model_params_quantized
            else:
                if retrain_mode:
                    sprint("reset to previous best params, where best_reward_eval =", best_reward_eval)
                    es.set_mu(best_model_params_eval)
            with open(filename_best, 'wt') as out:
                # TODO: save out the FEM score that came with this
                res = json.dump([best_model_params_eval, best_reward_eval, best_fem_eval], out, sort_keys=True, indent=0, separators=(',', ': '))

            # dump history of best
            curr_time = int(time.time()) - start_time
            best_record = [t, curr_time, "improvement", improvement, "curr", reward_eval, "prev", prev_best_reward_eval, "best", best_reward_eval]
            history_best.append(best_record)
            with open(filename_hist_best, 'wt') as out:
                res = json.dump(history_best, out, sort_keys=False, indent=0, separators=(',', ':'))

            sprint("improvement", t, improvement, "curr", reward_eval, "prev", prev_best_reward_eval, "best", best_reward_eval)


if __name__ == '__main__':

    import json 

    with open('es_log/carracing.cma.12.64.best.json', 'r') as f:
        ctrl_params = json.load(f)

    transform = transforms.Lambda(
        lambda x: np.transpose(x, (0, 3, 1, 2)) / 255)

    print(len(ctrl_params[0]))

    generate_rollouts(ctrl_params[0], transform, 30, 1000, 
        'exp_dir', num_rolls = 4, num_workers = 1 )




