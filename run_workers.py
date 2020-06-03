


def slave_routine(p_queue, r_queue, e_queue, p_index, 
    rand_int, time_limit, logdir, tmp_dir, 
    model_variables_dict, return_events):
    """ Thread routine.

    Threads interact with p_queue, the parameters queue, r_queue, the result
    queue and e_queue the end queue. They pull parameters from p_queue, execute
    the corresponding rollout, then place the result in r_queue.

    Each parameter has its own unique id. Parameters are pulled as tuples
    (s_id, params) and results are pushed as (s_id, result).  The same
    parameter can appear multiple times in p_queue, displaying the same id
    each time.

    As soon as e_queue is non empty, the thread terminate.

    When multiple gpus are involved, the assigned gpu is determined by the
    process index p_index (gpu = p_index % n_gpus).

    :args p_queue: queue containing couples (s_id, parameters) to evaluate
    :args r_queue: where to place results (s_id, results)
    # NOTE: This seems like a weird way to terminate... 
    :args e_queue: as soon as not empty, terminate
    :args p_index: the process index
    """

    # init random seed here. 
    np.random.seed(rand_int)
    torch.manual_seed(rand_int)

    # init routine
    if torch.cuda.device_count() >0:
        gpu = p_index % torch.cuda.device_count()
        device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu')
    else: 
        device = 'cpu'

    # redirect streams 
    # this is cool! 
    #sys.stdout = open(join(tmp_dir, str(getpid()) + '.out'), 'a')
    #sys.stderr = open(join(tmp_dir, str(getpid()) + '.err'), 'a')

    with torch.no_grad():
        if model_variables_dict:
            r_gen = RolloutGenerator(device, time_limit, 
                return_events=return_events, give_models=model_variables_dict)
        else: 
            r_gen = RolloutGenerator(device, time_limit, mdir=logdir)

        while e_queue.empty():
            if p_queue.empty():
                sleep(.1)
            else:
                rand_env_seed = np.random.randint(0,1e9,1)[0]
                print('random seed being used by worker', p_index, 'for this rollout is:', rand_env_seed)
            
                s_id, params = p_queue.get()
                r_queue.put((s_id, r_gen.rollout(params, rand_env_seed)))