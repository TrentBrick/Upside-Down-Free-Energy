from ray.util.multiprocessing import Pool
#from multiprocessing import Pool 
import copy
from control import Agent
import numpy as np

def set_seq_and_batch_vals(inp, batch_size_to_seq_len_multiple, dim=1):

    # get all of the rollout lengths: 
    rollout_lengths = []
    for worker_rollouts in inp:
        # iterate through each of the rollouts within the list of rollouts inside this worker.
        for rollout_data_dict in worker_rollouts[dim]: # this is pulling from a list!
            rollout_lengths.append( len(rollout_data_dict[list(rollout_data_dict.keys())[0]]))

    rollout_lengths = np.asarray(rollout_lengths)
    print('Mean length', rollout_lengths.mean(), 'Std', rollout_lengths.std())
    print('Min length', rollout_lengths.min(), 'Max length', rollout_lengths.max())
    print('10% quantile', np.quantile(rollout_lengths, 0.1))
    # set the new sequence length for the next rollouts.: 
    print('dynamically updating sequence length')
    SEQ_LEN = int(np.quantile(rollout_lengths, 0.1))-1 # number of sequences in a row used during training
    # needs one more than this for the forward predictions. 
    BATCH_SIZE = batch_size_to_seq_len_multiple//SEQ_LEN

    return SEQ_LEN, BATCH_SIZE

def combine_worker_rollouts(inp, seq_len, dim=1):
    """Combine data across the workers and their rollouts (making each rollout a separate element in a batch)
    
    :args:
        - inp: list. containing outputs of each worker. Assumes that each worker output inside this list is itself a list (or tuple)
        - seq_len: int. given to ensure each rollout is equal to or longer. 
        - dim: int. for a given worker, the element in its returned list that corresponds to the rollouts to be combined.

    :returns:
        - dictionary of rollouts organized by element type (eg. actions, pixel based observations). 
            Each dictionary key corresponds to a list of full rollouts. 
            Each full rollout has its first dimension corresponding to time. 
    """
    
    print('RUNNING COMBINE WORKER')
    first_iter = True
    # iterate through the worker outputs. 
    for worker_rollouts in inp: 
        # iterate through each of the rollouts within the list of rollouts inside this worker.
        for rollout_data_dict in worker_rollouts[dim]: # this is pulling from a list!
            if len(rollout_data_dict[list(rollout_data_dict.keys())[0]])-seq_len <= 0:
                # this rollout is too small so it is being ignored. 
                # getting one of the keys from the dictionary in a dict agnostic way.
                # NOTE: assumes that all keys in the dictionary correspond to lists of the same length
                print('!!!!!! Combine_worker_rollouts is ignoring rollout of length', len(rollout_data_dict[list(rollout_data_dict.keys())[0]]), 'for being smaller than the sequence length')
                continue

            if first_iter:
                # init the combo dictionary with lists for each key. 
                combo_dict = {k:[v] for k, v in rollout_data_dict.items()} 
                first_iter = False
            else: 
                # append items to the list for each key
                for k, v in combo_dict.items():
                    v.append(rollout_data_dict[k])

    return combo_dict


def worker(inp): # run lots of rollouts 

    """ Parses its input before creating an environment simulator instance
    and using it to generate the desired number of rollouts. These rollouts
    save and return the observations, rewards, and FEEF calculations."""

    gamename, decoder_reward_condition, logdir, \
        training_rollouts_per_worker, planner_n_particles, \
        cem_iters, discount_factor, \
        compute_feef, antithetic, seed = inp
    
    agent = Agent(gamename, logdir, decoder_reward_condition,
            planner_n_particles = planner_n_particles, 
            cem_iters=cem_iters, 
            discount_factor=discount_factor)

    return agent.simulate(return_events=True,
            compute_feef=compute_feef,
            num_episodes=training_rollouts_per_worker, seed=seed, antithetic=antithetic)


def generate_rollouts_using_planner(num_workers, batch_size_to_seq_len_multiple, worker_package): 

    """ Uses ray.util.multiprocessing Pool to create workers that each 
    run environment rollouts using planning with CEM (Cross Entropy Method). 
    Each worker (that runs code in execute_environment.py) passes back the 
    rollouts which have already been transformed (done for the planner to work). 
    These are split between train and test. The FEEF (Free energy of the expected future)
    and cumulative rewards from the rollout are also passed back. 

    Arguments and returns should be self explanatory. 
    All inputs are ints, strings or bools except 'init_cem_params' 
    which is a tuple/list containing two tensors. One for mus and other
    for sigmas. 
    """

    # 10% of the rollouts to use for test data. 
    ten_perc = np.floor(num_workers*0.1)
    if ten_perc == 0.0:
        ten_perc=1
    ninety_perc = int(num_workers-ten_perc)

    # generate a random number to give as a random seed to each pool worker. 
    # TODO: add antithetic sampling. 
    rand_ints = np.random.randint(0, 1e9, num_workers)

    worker_data = []
    #NOTE: currently not using joint_file_directory here (this is used for each worker to know to pull files from joint or from a subsection)
    for i in range(num_workers):

        package_w_rand = copy.copy(worker_package)
        package_w_rand.append(rand_ints[i])

        worker_data.append( tuple(package_w_rand)  )

    # deploy all of the workers and wait for results. 
    with Pool(processes=num_workers) as pool:
        res = pool.map(worker, worker_data) 

    # res is a list with tuples for each worker containing: reward_list, data_dict_list, t_list
    print('===== Done with pool of workers')

    # get all of the feef losses aggregated across the different workers: 
    feef_losses, reward_list = [], []
    for worker_rollouts in res:
        for ind, li in zip([3, 0], [feef_losses, reward_list]): # output order from simulate.
            li += worker_rollouts[ind]

    # TODO: make this more efficient. 
    seq_len, batch_size = set_seq_and_batch_vals(res, batch_size_to_seq_len_multiple, dim=2)

    print("Number of rollouts being given to test:", len(res[ninety_perc:]))

    return seq_len, batch_size, combine_worker_rollouts(res[:ninety_perc], seq_len, dim=2), \
                combine_worker_rollouts(res[ninety_perc:], seq_len, dim=2), \
                feef_losses, reward_list