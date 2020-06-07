"""
Generating data from the CarRacing gym environment.
!!! DOES NOT WORK ON TITANIC, DO IT AT HOME, THEN SCP !!!
"""
import argparse
from os.path import join, exists
import gym
import numpy as np
from utils.misc import sample_continuous_policy

def generate_data(rollouts, data_dir, noise_type, rand_seed, dont_trim_controls): # pylint: disable=R0914
    """ Generates data """
    assert exists(data_dir), "The data directory does not exist..."

    np.random.seed(rand_seed)

    env = gym.make("CarRacing-v0")
    seq_len = 1000

    for i in range(rollouts):
        rand_env_seed = np.random.randint(0,10000000,1)[0]
        env.seed(int(rand_env_seed))
        env.reset()
        env.env.viewer.window.dispatch_events()
        # sample all of the actions in one go for this rollout. 
        if noise_type == 'white':
            a_rollout = [env.action_space.sample() for _ in range(seq_len)]
        elif noise_type == 'brown':
            a_rollout = sample_continuous_policy(env.action_space, seq_len, 1. / 50)

        s_rollout = []
        r_rollout = []
        d_rollout = []

        t = 0
        while True:
            action = a_rollout[t]
            t += 1
            s, r, done, _ = env.step(action)
            env.env.viewer.window.dispatch_events() # needed for a bug in the rendering. 
            
            if not dont_trim_controls:
                s = s[:84]
            
            s_rollout += [s]
            r_rollout += [r]
            d_rollout += [done]
            if done:
                print("> End of rollout {}, {} frames...".format(i, len(s_rollout)))
                np.savez(join(data_dir, 'rollout_{}'.format(i)),
                         observations=np.array(s_rollout),
                         rewards=np.array(r_rollout),
                         actions=np.array(a_rollout),
                         terminals=np.array(d_rollout))
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rollouts', type=int, help="Number of rollouts")
    parser.add_argument('--dir', type=str, help="Where to place rollouts")
    parser.add_argument('--policy', type=str, choices=['white', 'brown'],
                        help='Noise type used for action sampling.',
                        default='brown')
    parser.add_argument('--rand_seed', type=int, help="Random seed here")
    parser.add_argument('--dont_trim_controls', action='store_true',
                    help='Best model is not reloaded if specified')
    args = parser.parse_args()

    print('the status of trimming the controls from the image is: ', args.dont_trim_controls)
    
    generate_data(args.rollouts, args.dir, args.policy, args.rand_seed, args.dont_trim_controls)
