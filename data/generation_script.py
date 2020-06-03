"""
Encapsulate generate data to make it parallel
"""
from os import makedirs
from os.path import join
import argparse
import multiprocessing
from multiprocessing import Pool
from subprocess import call
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--rollouts', type=int, help="Total number of rollouts.")
parser.add_argument('--threads', type=int, help="Number of threads")
parser.add_argument('--rootdir', type=str, help="Directory to store rollout "
                    "directories of each thread")
parser.add_argument('--policy', type=str, choices=['brown', 'white'],
                    help="Directory to store rollout directories of each thread",
                    default='brown')
args = parser.parse_args()

#args.threads = multiprocessing.cpu_count()

rpt = args.rollouts // args.threads + 1

# getting random seeds for each of the threads
rand_seeds = np.random.randint(0,1000000000,args.threads)
input_tuple = [(ind, r) for ind, r in enumerate(rand_seeds)]

def _threaded_generation(input_tuple):
    i = input_tuple[0]
    rand_seed = input_tuple[1]
    tdir = join(args.rootdir, 'thread_{}'.format(i))
    makedirs(tdir, exist_ok=True)
    cmd = ['xvfb-run', '-a -s', '"-screen 0 1400x900x24 +extension RANDR"']
    cmd += ['--server-num={}'.format(i + 1)]
    cmd += ["python3", "-m", "data.carracing", "--dir",
            tdir, "--rollouts", str(rpt), "--policy", args.policy,
            "--rand_seed", str(rand_seed)]
    cmd = " ".join(cmd)
    print(cmd)
    call(cmd, shell=True)
    return True

with Pool(args.threads) as p:
    p.map(_threaded_generation, input_tuple)
