import torch 

def get_env_params(gamename):

    if gamename == "carracing":
        env_params = {
            'env_name': 'CarRacing-v0',
            'desired_horizon': 18,
            'num_action_repeats': 3,
            'time_limit':1000, # max time limit for the rollouts generated
            'NUM_IMG_CHANNELS': 3,
            'ACTION_SIZE': 3,
            'init_cem_params': ( torch.Tensor([0.,0.7,0.]), 
                        torch.Tensor([0.5,0.7,0.3]) ),
            'LATENT_SIZE': 32, 
            'LATENT_RECURRENT_SIZE': 512,
            'EMBEDDING_SIZE': 256,
            'NODE_SIZE': 256,
            'IMAGE_RESIZE_DIM': 64,
            'IMAGE_DEFAULT_SIZE': 96,
            # top, bottom, left, right
            # can set to equal None if dont want any trimming. 
            'trim_shape': (0,84,0,96), 
            # buffer to keep track of rewards and cut 
            # rollout early if it is less than or equal to the 
            # stop_early_value
            #'stop_early_buf_size': 50,
            #'stop_early_value': -5.0

        }

        env_params['actual_horizon'] = env_params['desired_horizon']//env_params['num_action_repeats']

    elif gamename == "cartpole":

        env_params = {
            'desired_horizon': 30,
            'num_action_repeats': 3,
            'NUM_IMG_CHANNELS': 3,
            'ACTION_SIZE': 3,
            'init_cem_params': ( torch.Tensor([0.,0.,0.]), 
                torch.Tensor([1.,1.,1.]) ),
            'LATENT_SIZE': 32, 
            'LATENT_RECURRENT_SIZE': 512,
            'IMAGE_RESIZE_DIM': 64,
            'IMAGE_DEFAULT_SIZE': 96,
        }

    else: 
        raise ValueError("Don't know what this gamename (environment) is!")

    return env_params