# -*- coding: utf-8 -*-
#!/usr/env/bin python
import logging
import os
import sys
# from mud_examples.runner import setup_logging
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import null_space

plt.rcParams['figure.figsize'] = 10,10
plt.rcParams['font.size'] = 24

__author__ = "Mathematical Michael"
__copyright__ = "Mathematical Michael"
__license__ = "mit"
from mud_examples import __version__
from mud import __version__ as __mud_version__

_logger = logging.getLogger(__name__) # TODO: make use of this instead of print

from mud_examples.helpers import make_2d_unit_mesh, check_dir, parse_args
from mud.norm import full_functional, norm_input, norm_data, norm_predicted
from matplotlib import cm
from mud.funs import mud_sol, map_sol


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
    experiments['data_mismatch']['out_file'] = f'{fdir}/data_mismatch_contour.png'
    experiments['data_mismatch']['data_check'] = True
    experiments['data_mismatch']['full_check'] = False
    experiments['data_mismatch']['tk_reg'] = 0
    experiments['data_mismatch']['pr_reg'] = 0
    
    # tikonov regularization
    experiments['tikonov'] = {}
    experiments['tikonov']['out_file'] = f'{fdir}/tikonov_contour.png'
    experiments['tikonov']['tk_reg'] = 1
    experiments['tikonov']['pr_reg'] = 0
    experiments['tikonov']['data_check'] = False
    experiments['tikonov']['full_check'] = False
    
    # modified regularization
    experiments['modified'] = {}
    experiments['modified']['out_file'] = f'{fdir}/consistent_contour.png'
    experiments['modified']['tk_reg'] = 1
    experiments['modified']['pr_reg'] = 1
    experiments['modified']['data_check'] = False
    experiments['modified']['full_check'] = False
    
    # map point
    experiments['classical'] = {}
    experiments['classical']['out_file'] = f'{fdir}/classical_solution.png'
    experiments['classical']['tk_reg'] = 1
    experiments['classical']['pr_reg'] = 0
    experiments['classical']['data_check'] = True
    experiments['classical']['full_check'] = True
    
    # mud point
    experiments['consistent'] = {}
    experiments['consistent']['out_file'] = f'{fdir}/consistent_solution.png'
    experiments['consistent']['tk_reg'] = 1
    experiments['consistent']['pr_reg'] = 1
    experiments['consistent']['data_check'] = True
    experiments['consistent']['full_check'] = True

    # comparison
    experiments['compare'] = {}
    experiments['compare']['out_file'] = f'{fdir}/map_compare_contour.png'
    experiments['compare']['data_check'] = True
    experiments['compare']['full_check'] = True
    experiments['compare']['tk_reg'] = 1
    experiments['compare']['pr_reg'] = 0
    experiments['compare']['comparison'] = True
    experiments['compare']['cov_01'] = -0.5

    for ex in experiments:
        _logger.info(f"Running {ex}")
        config = experiments[ex]
        out_file = config.get('out_file', 'latest_figure.png')
        tk_reg = config.get('tk_reg', 1)
        pr_reg = config.get('pr_reg', 1)
        cov_01 = config.get('cov_01', -0.25)
        cov_11 = config.get('cov_11', 0.5)
        obs_std = config.get('obs_std', 0.5)
        full_check = config.get('full_check', True)
        data_check = config.get('data_check', True)
        numr_check = config.get('numr_check', False)
        comparison = config.get('comparison', False)

        contour_example(A=A, b=b, save=save,
                        param_ref=lam_true,
                        compare=comparison,
                        cov_01=cov_01,
                        cov_11=cov_11, 
                        initial_mean=initial_mean,
                        alpha=tk_reg,
                        omega=pr_reg,
                        show_full=full_check,
                        show_data=data_check,
                        show_est=numr_check,
                        obs_std = obs_std,
                        figname=out_file)


def run():
    """Entry point for console_scripts
    """
    main(sys.argv[1:])

############################################################




def contour_example(A=np.array([[1, 1]]), b=np.zeros([1, 1]),  # noqa: C901
                            cov_11=0.5, cov_01=-0.25,
                            initial_mean=np.array([0.25, 0.25]),
                            alpha=1, omega=1, obs_std=1,
                            show_full=True, show_data=True,
                            show_est=False, param_ref=None, compare=False,
                            fsize=42, figname='latest_figure.png', save=False,
                            ):
    """
    alpha: float in [0, 1], weight of Tikhonov regularization
    omega: float in [0, 1], weight of Modified regularization
    """
    # mesh for plotting
    N, r = 250, 1
    X, Y, XX = make_2d_unit_mesh(N, r)
    inputs = XX

    std_of_data = [obs_std]
    obs_cov = np.diag(std_of_data)
    observed_data_mean = np.array([[1]])
    initial_cov = np.array([[1, cov_01], [cov_01, cov_11]])

    assert np.all(np.linalg.eigvals(initial_cov) > 0)

    z = full_functional(A, XX, b, initial_mean, initial_cov, observed_data_mean, observed_cov=obs_cov)
    zp = norm_predicted(A, XX, initial_mean, initial_cov)
    zi = norm_input(XX, initial_mean, initial_cov)
    zd = norm_data(A, XX, b, observed_data_mean, observed_cov=obs_cov)
    # sanity check that all arguments passed correctly:
    assert np.linalg.norm(z - zi - zd + zp) < 1E-8

    # plotting contours
    z = (alpha * zi + zd - omega * zp)
    mud_a = np.argmin(z)
    map_a = np.argmin(alpha * zi + zd)

    # get mud/map points from minimal values on mesh
    mud_pt = inputs[mud_a, :]
    map_pt = inputs[map_a, :]

    msize = 500
    ls = (np.linalg.pinv(A) @ observed_data_mean.T).ravel()
    if show_data:
        plt.contour(inputs[:, 0].reshape(N, N),
                    inputs[:, 1].reshape(N, N),
                    (zd).reshape(N, N), 25,
                    cmap=cm.viridis, alpha=0.5, vmin=0, vmax=4)
        plt.axis('equal')

        s = np.linspace(-2 * r, 2 * r, 10)

        if A.shape[0] < A.shape[1]:
            # nullspace through least-squares
            null_line = null_space(A) * s + ls.reshape(-1, 1)
            plt.plot(null_line[0, :], null_line[1, :],
                     label='Solution Contour',
                     lw=2, color='xkcd:red')
            if not show_full:
                plt.annotate('Solution Contour', (0.1, 0.9),
                             fontsize=fsize, backgroundcolor="w")

    if show_full:
        plt.contour(inputs[:, 0].reshape(N, N),
                    inputs[:, 1].reshape(N, N),
                    z.reshape(N, N), 50,
                    cmap=cm.viridis, alpha=1.0)
    elif alpha + omega > 0:
        plt.contour(inputs[:, 0].reshape(N, N),
                    inputs[:, 1].reshape(N, N),
                    (alpha * zi - omega * zp).reshape(N, N), 100,
                    cmap=cm.viridis, alpha=0.25)
    plt.axis('equal')


    if alpha + omega > 0:
        plt.scatter(initial_mean[0], initial_mean[1],
                    label='Initial Mean',
                    color='k', s=msize)
        if not show_full:
            plt.annotate('Initial Mean',
                         (initial_mean[0] + 0.001 * fsize, initial_mean[1] - 0.001 * fsize),
                         fontsize=fsize, backgroundcolor="w")
        else:
            if param_ref is not None:
                plt.scatter(param_ref[0], param_ref[1],
                    label='$\\lambda^\\dagger$',
                    color='k', s=msize, marker='*')
            if compare:
                plt.annotate('Truth',
                         (param_ref[0] + 0.0005 * fsize, param_ref[1] + 0.0005 * fsize),
                         fontsize=fsize, backgroundcolor="w")

        show_mud = omega > 0 or compare

        if show_full:
            # scatter and line from origin to least squares
            plt.scatter(ls[0], ls[1],
                        label='Least Squares',
                        color='xkcd:blue',
                        marker='d', s=msize, zorder=10)
            plt.plot([0, ls[0]], [0, ls[1]],
                     color='xkcd:blue',
                     marker='d', lw=1, zorder=10)
            plt.annotate('Least Squares',
                         (ls[0] - 0.001 * fsize, ls[1] + 0.001 * fsize),
                         fontsize=fsize, backgroundcolor="w")

            if show_est:  # numerical solutions
                if omega > 0:
                    plt.scatter(mud_pt[0], mud_pt[1],
                                label='min: Tk - Un', color='xkcd:sky blue',
                                marker='o', s=3 * msize, zorder=10)
                if (alpha > 0 and omega != 1):
                    plt.scatter(map_pt[0], map_pt[1],
                                label='min: Tk', color='xkcd:blue',
                                marker='o', s=3 * msize, zorder=10)

            if (alpha > 0 and omega != 1):  # analytical MAP point
                map_pt_eq = map_sol(A, b, observed_data_mean,
                                    initial_mean, initial_cov,
                                    data_cov=obs_cov, w=alpha)
                plt.scatter(map_pt_eq[0], map_pt_eq[1],
                            label='MAP', color='xkcd:orange',
                            marker='x', s=msize, lw=10, zorder=10)

                if compare:  # second map point has half the regularization strength
                    plt.annotate('MAP$_{\\alpha}$',
                                 (map_pt_eq[0] - 0.004 * fsize, map_pt_eq[1] - 0.002 * fsize),
                                 fontsize=fsize, backgroundcolor="w")

                else:
                    plt.annotate('MAP$_{\\alpha}$',
                                 (map_pt_eq[0] + 0.0001 * fsize, map_pt_eq[1] - 0.002 * fsize),
                                 fontsize=fsize, backgroundcolor="w")

            if show_mud:  # analytical MUD point
                mud_pt_eq = mud_sol(A, b, observed_data_mean,
                                    initial_mean, initial_cov)
                plt.scatter(mud_pt_eq[0], mud_pt_eq[1],
                            label='MUD', color='xkcd:brown',
                            marker='*', s=2 * msize, lw=5, zorder=10)
                plt.annotate('MUD',
                             (mud_pt_eq[0] + 0.001 * fsize, mud_pt_eq[1] - 0.001 * fsize),
                             fontsize=fsize, backgroundcolor="w")

        if A.shape[0] < A.shape[1]:
            # want orthogonal nullspace, function gives one that is already normalized
            v = null_space(A @ initial_cov)
            v = v[::-1]  # in 2D, we can just swap entries and put a negative sign in front of one
            v[0] = - v[0]

            if show_full and show_mud:
                # grid search to find upper/lower bounds of line being drawn.
                # importance is the direction, image is nicer with a proper origin/termination
                s = np.linspace(-1, 1, 1000)
                new_line = (v.reshape(-1, 1) * s) + initial_mean.reshape(-1, 1)
                mx = np.argmin(np.linalg.norm(new_line - initial_mean.reshape(-1, 1), axis=0))
                mn = np.argmin(np.linalg.norm(new_line - mud_pt_eq.reshape(-1, 1), axis=0))
                plt.plot(new_line[0, mn:mx], new_line[1, mn:mx], lw=1, label='projection line', c='k')
        elif show_full:
            plt.plot([initial_mean[0], ls[0]],
                     [initial_mean[1], ls[1]],
                     lw=1, label='Projection Line', c='k')

    #     print(p)

    plt.axis('square')
    plt.axis([0, r, 0, r])
#     plt.legend(fontsize=fsize)
    plt.xticks(fontsize=0.75 * fsize)
    plt.yticks(fontsize=0.75 * fsize)
    plt.tight_layout()
    if save:
        if '/' in figname:
            fdir = ''.join(figname.split('/')[:-1])
            check_dir(fdir)
        plt.savefig(figname, dpi=300)
        plt.close()

#     plt.title('Predicted Covariance: {}'.format((A@initial_cov@A.T).ravel() ))
    # plt.show()


if __name__ == "__main__":
    run()
