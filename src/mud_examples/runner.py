# -*- coding: utf-8 -*-
#!/usr/env/bin python

import argparse
import logging
import os
import sys
from pathlib import Path
import pickle


import numpy as np

from mud_examples.plotting import plot_experiment_measurements, plot_experiment_equipment
from mud_examples.plotting import plot_scalar_poisson_summary


import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.backend = 'Agg'
matplotlib.rcParams['figure.figsize'] = 10,10
matplotlib.rcParams['font.size'] = 16

from mud_examples.ode import main_ode
from mud_examples.pde import main_pde



__author__ = "Mathematical Michael"
__copyright__ = "Mathematical Michael"
__license__ = "mit"
from mud_examples import __version__
from mud import __version__ as __mud_version__

_logger = logging.getLogger(__name__) # TODO: make use of this instead of print


def parse_args(args):
    """Parse command line parameters

    Args:
      args ([str]): command line parameters as list of strings

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    
    desc = """
        Examples
        """

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-e', '--example',       default='ode', type=str)
    parser.add_argument('-m', '--num-measure',   default=[20, 100],  type=int, nargs='+')
    parser.add_argument('-r', '--ratio-measure', default=[1],  type=float, nargs='+')
    parser.add_argument('--num-trials',    default=20,    type=int)
    parser.add_argument('-t', '--sensor-tolerance',  default=[0.1], type=float, action='append')
    parser.add_argument('-s', '--seed',          default=21)
    parser.add_argument('-lw', '--linewidth',    default=5)
    parser.add_argument('--fsize',               default=32, type=int)
    parser.add_argument('--bayes', action='store_true')
    parser.add_argument('--alt', action='store_true')
    parser.add_argument('--save', action='store_true')

    parser.add_argument(
        "--version",
        action="version",
        version=f"mud_examples {__version__}, mud {__mud_version__}")
#     parser.add_argument('-n', '--num_samples',
#         dest="num",
#         help="Number of samples",
#         default=100,
#         type=int,
#         metavar="INT")
    parser.add_argument('-i', '--input_dim',
        dest="input_dim",
        help="Dimension of input space (default=2).",
        default=2,
        type=int,
        metavar="INT")
    parser.add_argument('-d', '--distribution',
        dest="dist",
        help="Distribution. `n` (normal), `u` (uniform, default)",
        default='u',
        type=str,
        metavar="STR")
#     parser.add_argument('-b', '--beta-params',
#         dest="beta_params",
#         help="Parameters for beta distribution. Overrides --distribution. (default = 1 1 )",
#         default=None,
#         nargs='+',
#         type=float,
#         metavar="FLOAT FLOAT")
    parser.add_argument('-p', '--prefix',
        dest="prefix",
        help="Output filename prefix (no extension)",
        default='results',
        type=str,
        metavar="STR")
    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO",
        action="store_const",
        const=logging.INFO)
    parser.add_argument(
        "-vv",
        "--very-verbose",
        dest="loglevel",
        help="set loglevel to DEBUG",
        action="store_const",
        const=logging.DEBUG)
    return parser.parse_args(args)


def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(level=loglevel, stream=sys.stdout,
                        format=logformat, datefmt="%Y-%m-%d %H:%M:%S")



def main(args):
    """
    Main entrypoint for example-generation
    """
    args = parse_args(args)
    setup_logging(args.loglevel)
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
    prefix       = args.prefix
    dist         = args.dist

    tolerances   = list(np.sort([ float(t) for t in args.sensor_tolerance ]))

    if example == 'pde':
        measurements = list(np.sort([ int(n) for n in args.num_measure ]))
        if len(measurements) == 0:
            measurements = [100]
    else:
        time_ratios  = list(np.sort([ float(r) for r in args.ratio_measure ]))
        if len(time_ratios) == 0:
            time_ratios = [1.0]

    _logger.info("Running...")
    if example == 'pde':
        lam_true = 3.0
        res = main_pde(num_trials=num_trials,
                         fsize=fsize,
                         seed=seed,
                         lam_true=lam_true,
                         tolerances=tolerances,
                         input_dim=inputdim,
                         alt=alt, bayes=bayes,
                         dist=dist, prefix=prefix,
                         measurements=measurements)
        if inputdim == 1:  # TODO: roll this plotting into main_pde, handle w/o fenics?
            plot_scalar_poisson_summary(res=res, measurements=measurements,
                     fsize=fsize, prefix=f'pde_{inputdim}D/' + example, lam_true=lam_true, save=save)
        else:
            # solution / sensors plotted by main_pde method
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


def run():
    """Entry point for console_scripts
    """
    main(sys.argv[1:])


def run_pde():
    """Recreates Poisson figures in MUD paper.
    """
    run_cmd = """--example pde --bayes --save \
    --num-trials 20 -m 20 100 250 500 750 1000
    """.replace('    ','').replace('\n','').split(' ')
    main(run_cmd + sys.argv[1:])


def run_ode():
    """Recreates Poisson figures in MUD paper.
    """
    run_cmd = """-v --example ode --bayes --save \
    --num-trials 20 -r 0.01 0.05 0.1 0.25 0.5 1 -t 0.1
    """.replace('    ','').replace('\n','').split(' ')
    main(run_cmd + sys.argv[1:])


def run_all():
    run_ode()
    run_pde()

############################################################


if __name__ == "__main__":
    run()
