import numpy as np

from mud_examples.plotting import plot_experiment_measurements, plot_experiment_equipment
from mud_examples.plotting import plot_poisson_solution


import pickle
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.backend = 'Agg'
matplotlib.rcParams['figure.figsize'] = 10,10
matplotlib.rcParams['font.size'] = 16

from ode import main_ode
from pde import main_pde

def main(args):
    np.random.seed(args.seed)
    example       = args.example
    num_trials   = args.num_trials
    fsize        = args.fsize
    linewidth    = args.linewidth
    seed         = args.seed
    inputdim     = args.input_dim
    save         = args.save
    alt          = args.alt
    bayes        = args.bayes
    tolerances   = list(np.sort([ float(t) for t in args.sensor_tolerance ]))
    if len(tolerances) == 0: tolerances = [0.1]

    if example == 'pde':
        measurements = list(np.sort([ int(n) for n in args.num_measure ]))
        if len(measurements) == 0:
            measurements = [100]
    else:
        time_ratios  = list(np.sort([ float(r) for r in args.ratio_measure ]))
        if len(time_ratios) == 0:
            time_ratios = [1.0]

    print("Running...")
    if example == 'pde':
        lam_true = 3.0
        res = main_pde(num_trials=num_trials,
                         fsize=fsize,
                         seed=seed,
                         lam_true=lam_true,
                         tolerances=tolerances,
                         input_dim=inputdim,
                         alt=alt, bayes=bayes,
                         measurements=measurements)
        if inputdim == 1:
            plot_poisson_solution(res=res, measurements=measurements,
                     fsize=fsize, prefix=f'pde_{inputdim}D/' + example, lam_true=lam_true, save=save)
        else:
            # somehow get P passed here or handle plotting above..
            pass

        if len(measurements) > 1:
            plot_experiment_measurements(measurements, res,
                                         f'pde_{inputdim}D/' + example, fsize,
                                         linewidth, save=save)
        if len(tolerances) > 1:
            plot_experiment_equipment(tolerances, res,
                                      f'pde_{inputdim}D/' + example, fsize,
                                      linewidth, save=save)
    elif example == 'ode':
        lam_true = 0.5
        res = main_ode(num_trials=num_trials,
                         fsize=fsize,
                         seed=seed,
                         lam_true=lam_true,
                         tolerances=tolerances,
                         alt=alt, bayes=bayes,
                         time_ratios=time_ratios)

        if len(time_ratios) > 1:
            plot_experiment_measurements(time_ratios, res,
                                         'ode/' + example,
                                         fsize, linewidth,
                                         save=save, legend=True)

        if len(tolerances) > 1:
            plot_experiment_equipment(tolerances, res,
                                      'ode/' + example, fsize, linewidth,
                                      title=f"Variance of MUD Error\nfor t={1+2*np.median(time_ratios):1.3f}s",
                                      save=save)
    ##########


    if args.save:
        with open('results.pkl', 'wb') as f:
            pickle.dump(res, f)


######


if __name__ == "__main__":
    import argparse
    desc = """
        Examples
        """

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-e', '--example',       default='ode', type=str)
    parser.add_argument('-m', '--num-measure',   default=[],  action='append')
    parser.add_argument('-r', '--ratio-measure', default=[],  action='append')
    parser.add_argument('--num-trials',    default=20,    type=int)
    parser.add_argument('-t', '--sensor-tolerance',  default=[0.1], action='append')
    parser.add_argument('-s', '--seed',          default=21)
    parser.add_argument('-lw', '--linewidth',    default=5)
    parser.add_argument('-i', '--input-dim',     default=2, type=int)
    parser.add_argument('--fsize',               default=32, type=int)
    parser.add_argument('--bayes', action='store_true')
    parser.add_argument('--alt', action='store_true')
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()
    main(args)

