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

    def run_rollout():
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
        sim_rewards = []

        t = 0
        while True:
            action = a_rollout[t]

            for _ in range(num_action_repeats):
                s, r, done, _ = env.step(action)
                #env.env.viewer.window.dispatch_events() # needed for a bug in the rendering with the old gym environment.  
                
                if not dont_trim_controls:
                    s = s[:84]

                # this ends the rollout early. 
                if len(sim_rewards) <30:
                    sim_rewards.append(r)
                else: 
                    sim_rewards.pop(0)
                    sim_rewards.append(r)
                    #print('lenght of sim rewards',  len(sim_rewards),round(sum(sim_rewards),3))
                    if round(sum(sim_rewards), 3) == -3.0:
                        done=True 

                if done:
                    s_rollout += [s]
                    r_rollout += [r]
                    d_rollout += [done]
                    print("> End of rollout {}, {} frames...".format(i, len(s_rollout)))
                    np.savez(join(data_dir, 'rollout_{}'.format(i)),
                            observations=np.array(s_rollout),
                            rewards=np.array(r_rollout),
                            actions=np.array(a_rollout),
                            terminals=np.array(d_rollout))
                    return
                
                t += 1

            s_rollout += [s]
            r_rollout += [r]
            d_rollout += [done]

    assert exists(data_dir), "The data directory does not exist..."

    np.random.seed(rand_seed)

    env = gym.make("CarRacing-v0")
    seq_len = 1000
    # TODO: make this a command line argument in generation_script.py
    num_action_repeats = 5
    print('The action repeat number is:', num_action_repeats)

    for i in range(rollouts):
        run_rollout() # calls a separate function so I can have "return"
        # at the end rather than "break" in order to stop the rollout at the right time.
        

            
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
