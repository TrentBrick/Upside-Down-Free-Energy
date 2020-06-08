from multiprocessing import Pool 

def worker(inp):
    return [1,2,3], [5,6,7]

if __name__ == '__main__':
    num_workers = 2
    worker_data = [0]*num_workers

    with Pool(processes=num_workers) as pool:
        res = pool.map(worker, worker_data)
    print(res)

    
