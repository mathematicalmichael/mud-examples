#!/usr/env/bin python
import os
# os.environ['OMP_NUM_THREADS'] = '1'
from mud_examples.poisson import evaluate_and_save_poisson as wrapper 
import numpy as np
from fenics import set_log_level

set_log_level(40) # ERROR=40
# from mpi4py import MPI
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()

if __name__=='__main__':

    import argparse

    parser = argparse.ArgumentParser(description="Poisson Problem")
    parser.add_argument('-n', '--num', default = 10, type=int,
                       help="Number of samples")
    parser.add_argument('-o', '--outfile', default='results',
                       help="Output filename (no extension)")
    parser.add_argument('-i', '--input-dim', default=1, type=int)
    parser.add_argument('-d', '--dist', default='u', help='Distribution. `n` (normal), `u` (uniform, default)')
    args = parser.parse_args()

    num_samples = args.num
    dist = args.dist
    outfile = args.outfile
    inputdim = args.input_dim

    if dist == 'n':  # N(0,1)
        randsamples = np.random.randn(num_samples, inputdim)
    elif dist == 'u':  # U(-4, 0)
        randsamples = -4*np.random.rand(num_samples, inputdim)
    else:
        raise ValueError("Improper distribution choice, use `n` (normal), `u` (uniform)")

    # indexed list of samples we will evaluate through our poisson model
    sample_seed_list = list(zip(range(num_samples), randsamples))

    results = []
    for sample in sample_seed_list:
        r = wrapper(sample, outfile)
        results.append(r)
#         print(results)

    import pickle
    pickle.dump(results, open(f'{outfile}.pkl','wb'))
