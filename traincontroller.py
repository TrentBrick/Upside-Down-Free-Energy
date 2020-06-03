"""
Training a linear controller on latent + recurrent state
with CMAES.

This is a bit complex. num_workers slave threads are launched
to process a queue filled with parameters to be evaluated.
"""
import argparse
import sys
from os.path import join, exists
from os import mkdir, unlink, listdir, getpid
from time import sleep
from torch.multiprocessing import Process, Queue,  cpu_count
import torch
import cma
from models import Controller
from tqdm import tqdm
import numpy as np
from utils.misc import RolloutGenerator, ACTION_SIZE, LATENT_RECURRENT_SIZE, LATENT_SIZE
from utils.misc import load_parameters
from utils.misc import flatten_parameters

def slave_routine(p_queue, r_queue, e_queue, p_index, 
    rand_int, time_limit, logdir, tmp_dir, 
    model_variables_dict, return_events):

    cmd = ['xvfb-run', '-a -s', '"-screen 0 1400x900x24 +extension RANDR"']
    cmd += ['--server-num={}'.format(p_index + 1)]
    cmd += ["python3", "-m", "run_workers", "--dir",
            tdir, "--rollouts", str(rpt), "--policy", args.policy,
            "--rand_seed", str(rand_seed)]
    cmd = " ".join(cmd)
    print(cmd)
    call(cmd, shell=True)

def train_controller(args, model_variables_dict=None, return_events=False):

################################################################################
#                           Thread routines                                    #
################################################################################

    # Max number of workers. M

    # multiprocessing variables
    n_samples = args.n_samples
    pop_size = args.pop_size
    num_workers = min(args.max_workers, n_samples * pop_size)
    assert num_workers <= cpu_count(), "Fewer CPUs than the number of workers assigned!!!"
    time_limit = 1000

    # create tmp dir if non existent and clean it if existent
    tmp_dir = join(args.logdir, 'tmp')
    if not exists(tmp_dir):
        mkdir(tmp_dir)
    else:
        for fname in listdir(tmp_dir):
            unlink(join(tmp_dir, fname))

    # create ctrl dir if non exitent
    ctrl_dir = join(args.logdir, 'ctrl')
    if not exists(ctrl_dir):
        mkdir(ctrl_dir)

    ################################################################################
    #                Define queues and start workers                               #
    ################################################################################
    p_queue = Queue()
    r_queue = Queue()
    e_queue = Queue() 

    # generate a random number to give as a random seed to each process. 
    rand_ints = np.random.randint(0, 1e9 ,num_workers)
    print('spinning up workers!')
    for p_index in range(num_workers):
        Process(target=slave_routine, args=(p_queue, r_queue, e_queue, p_index, 
            rand_ints[p_index], time_limit, args.logdir, tmp_dir,model_variables_dict, return_events)).start()

    ################################################################################
    #                           Evaluation                                         #
    ################################################################################
    def evaluate(solutions, results, rollouts=100):
        """ Give current controller evaluation.

        Evaluation is minus the cumulated reward averaged over rollout runs.

        :args solutions: CMA set of solutions
        :args results: corresponding results
        :args rollouts: number of rollouts

        :returns: minus averaged cumulated reward
        """
        index_min = np.argmin(results)
        best_guess = solutions[index_min]
        restimates = []

        for s_id in range(rollouts):
            p_queue.put((s_id, best_guess))

        print("Evaluating...")
        for _ in tqdm(range(rollouts)):
            while r_queue.empty():
                sleep(.1)
            restimates.append(r_queue.get()[1])

        return best_guess, np.mean(restimates), np.std(restimates)

    ################################################################################
    #                           Launch CMA                                         #
    ################################################################################
    controller = Controller(LATENT_SIZE, LATENT_RECURRENT_SIZE, ACTION_SIZE)  # dummy instance

    # define current best and load parameters
    cur_best = None
    ctrl_file = join(ctrl_dir, 'best.tar')
    print("Attempting to load previous best...")
    if exists(ctrl_file):
        state = torch.load(ctrl_file, map_location={'cuda:0': 'cpu'})
        cur_best = - state['reward']
        controller.load_state_dict(state['state_dict'])
        print("Previous best was {}...".format(-cur_best))

    parameters = controller.parameters()
    # parameters to be optimized, starting sigmas, population size. 
    es = cma.CMAEvolutionStrategy(flatten_parameters(parameters), 0.1,
                                {'popsize': pop_size})

    epoch = 0
    log_step = 10
    while not es.stop():
        print('epoch:', epoch)
        if cur_best is not None and - cur_best > args.target_return:
            print("Already better than target, breaking...")
            break

        r_list = [0] * pop_size  # result list
        # get sets of parameters from the distribution. 
        solutions = es.ask()

        # push parameters to queue
        print('pushing parameters to queue')
        for s_id, s in enumerate(solutions):
            for _ in range(n_samples):
                p_queue.put((s_id, s))

        # retrieve results
        print('waiting to retrieve results')
        if args.display:
            pbar = tqdm(total=pop_size * n_samples)
        for _ in range(pop_size * n_samples):
            while r_queue.empty():
                sleep(.1)
            r_s_id, r = r_queue.get()
            r_list[r_s_id] += r / n_samples
            if args.display:
                pbar.update(1)
        if args.display:
            pbar.close()
        print('all results retrieved!')

        # updates the population parameters based upon the results from the simulation. 
        es.tell(solutions, r_list)
        # display all of the parameters on a single line. 
        es.disp()

        # evaluation and saving
        if epoch % log_step == log_step - 1:
            best_params, best, std_best = evaluate(solutions, r_list)
            print("Current evaluation: {}".format(best))
            if not cur_best or cur_best > best:
                cur_best = best
                print("Saving new best with value {}+-{}...".format(-cur_best, std_best))
                load_parameters(best_params, controller)
                torch.save(
                    {'epoch': epoch,
                    'reward': - cur_best,
                    'state_dict': controller.state_dict()},
                    join(ctrl_dir, 'best.tar'))
            if - best > args.target_return:
                print("Terminating controller training with value {}...".format(best))
                break

        epoch += 1

    es.result_pretty()
    e_queue.put('EOP') # ends all of the processes! 

    
if __name__ == '__main__':

    # parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, help='Where everything is stored.')
    parser.add_argument('--n-samples', type=int, help='Number of samples used to obtain '
                        'return estimate.')
    parser.add_argument('--pop-size', type=int, help='Population size.')
    parser.add_argument('--target-return', type=float, help='Stops once the return '
                        'gets above target_return')
    parser.add_argument('--display', action='store_true', help="Use progress bars if "
                        "specified.")
    parser.add_argument('--max-workers', type=int, help='Maximum number of workers.',
                        default=64)
    args = parser.parse_args()

    train_controller(args)

