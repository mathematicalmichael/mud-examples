# -*- coding: utf-8 -*-
#!/usr/env/bin python
import argparse
import logging
import os
import sys
# from mud_examples.runner import setup_logging
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['figure.figsize'] = 10,10
plt.rcParams['font.size'] = 24

__author__ = "Mathematical Michael"
__copyright__ = "Mathematical Michael"
__license__ = "mit"
from mud_examples import __version__
from mud import __version__ as __mud_version__
from mud_examples.helpers import check_dir
_logger = logging.getLogger(__name__) # TODO: make use of this instead of print

from mud_examples.plotting import plot_2d_contour_example

def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(level=loglevel, stream=sys.stdout,
                        format=logformat, datefmt="%Y-%m-%d %H:%M:%S")


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


def main(args):
    """
    Main entrypoint for example-generation
    """
    args = parse_args(args)
    setup_logging(args.loglevel)
    np.random.seed(args.seed)
#     example       = args.example
#     num_trials   = args.num_trials
#     fsize        = args.fsize
#     linewidth    = args.linewidth
#     seed         = args.seed
#     inputdim     = args.input_dim
#     save         = args.save
#     alt          = args.alt
#     bayes        = args.bayes
#     prefix       = args.prefix
#     dist         = args.dist

    presentation = False
    save = True

    if not presentation:
        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['font.family'] = 'STIXGeneral'
    fdir = 'contours'
    check_dir(fdir)
    lam_true = np.array([0.7, 0.3])
    initial_mean = np.array([0.25, 0.25])
    A = np.array([[1, 1]])
    b = np.zeros((1,1))

    experiments = {}
    
    # data mismatch
    experiments['data_mismatch'] = {}
    experiments['data_mismatch']['fig_title'] = f'{fdir}/data_mismatch_contour.png'
    experiments['data_mismatch']['data_check'] = True
    experiments['data_mismatch']['full_check'] = False
    experiments['data_mismatch']['tk_slide'] = 0
    experiments['data_mismatch']['pr_slide'] = 0
    
    # tikonov regularization
    experiments['tikonov'] = {}
    experiments['tikonov']['fig_title'] = f'{fdir}/tikonov_contour.png'
    experiments['tikonov']['tk_slide'] = 1
    experiments['tikonov']['pr_slide'] = 0
    experiments['tikonov']['data_check'] = False
    experiments['tikonov']['full_check'] = False
    
    # modified regularization
    experiments['modified'] = {}
    experiments['modified']['fig_title'] = f'{fdir}/consistent_contour.png'
    experiments['modified']['tk_slide'] = 1
    experiments['modified']['pr_slide'] = 1
    experiments['modified']['data_check'] = False
    experiments['modified']['full_check'] = False
    
    # map point
    experiments['classical'] = {}
    experiments['classical']['fig_title'] = f'{fdir}/classical_solution.png'
    experiments['classical']['tk_slide'] = 1
    experiments['classical']['pr_slide'] = 0
    experiments['classical']['data_check'] = True
    experiments['classical']['full_check'] = True
    
    # comparison
    experiments['compare'] = {}
    experiments['compare']['fig_title'] = f'{fdir}/map_compare_contour.png'
    experiments['compare']['data_check'] = True
    experiments['compare']['full_check'] = True
    experiments['compare']['tk_slide'] = 1
    experiments['compare']['pr_slide'] = 0
    experiments['compare']['comparison'] = True
    experiments['compare']['cov_01'] = -0.5

    for ex in experiments.values():
        tk_slide = ex.get('tk_slide')
        pr_slide = ex.get('pr_slide')
        data_check = ex.get('data_check')
        cov_01 = ex.get('cov_01', -0.25)
        cov_11 = ex.get('cov_11', 0.5)
        obs_std = ex.get('obs_std', 0.5)
        fig_title = ex.get('fig_title')
        full_check = ex.get('full_check', True)
        data_check = ex.get('data_check', True)
        numr_check = ex.get('numr_check', False)
        comparison = ex.get('comparison', False)

        plot_2d_contour_example(A=A, b=b, save=save,
                        param_ref=lam_true,
                        compare=comparison,
                        cov_01=cov_01,
                        cov_11=cov_11, 
                        initial_mean=initial_mean,
                        alpha=tk_slide,
                        omega=pr_slide,
                        show_full=full_check,
                        show_data=data_check,
                        show_est=numr_check,
                        obs_std = obs_std,
                        figname=fig_title)


def run():
    """Entry point for console_scripts
    """
    main(sys.argv[1:])

############################################################


if __name__ == "__main__":
    run()
