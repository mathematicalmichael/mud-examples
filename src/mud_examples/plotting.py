# -*- coding: utf-8 -*-
#!/usr/env/bin python

import logging

from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.gridspec as gridspec
plt.rcParams['figure.figsize'] = 10,10
plt.rcParams['font.size'] = 16
# import matplotlib
# matplotlib.rcParams['mathtext.fontset'] = 'stix'
# matplotlib.rcParams['font.family'] = 'STIXGeneral'
# matplotlib.backend = 'Agg'

import numpy as np
from mud.util import null_space
from mud.funs import mud_sol, map_sol
from mud.norm import full_functional, norm_input, norm_data, norm_predicted
from mud.plot import make_2d_unit_mesh
from mud_examples.helpers import check_dir

_logger = logging.getLogger(__name__) # TODO: make use of this instead of print
_mpl_logger = logging.getLogger('matplotlib')
_mpl_logger.setLevel(logging.WARNING)


def fit_log_linear_regression(input_values, output_values):
    if len(np.unique(output_values)) == 1:
        _logger.info("Log Linear Regression: All values identical.")
        return np.array(output_values), 0
    x, y = np.log10(input_values), np.log10(output_values)
    X, Y = np.vander(x, 2), np.array(y).reshape(-1, 1)
    slope, intercept = (np.linalg.pinv(X) @ Y).ravel()
    regression_line = 10**(slope * x + intercept)
    return regression_line, slope


def plot_2d_contour_example(A=np.array([[1, 1]]), b=np.zeros([1, 1]),  # noqa: C901
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

#     plt.title('Predicted Covariance: {}'.format((A@initial_cov@A.T).ravel() ))
    # plt.show()


def plot_decay_solution(solutions, model_generator, sigma, prefix,
                        time_vector, lam_true, qoi_true, end_time=3, fsize=32, save=True):
    alpha_signal = 0.2
    alpha_points = 0.6
#     num_meas_plot_list = [25, 50, 400]

    print("Plotting decay solution.")
    for num_meas_plot in solutions:
        filename = f'{prefix}_{num_meas_plot}_reference_solution.png'
        plt.rcParams['figure.figsize'] = 25, 10
        _ = plt.figure()  # TODO: proper figure handling with `fig`

        plotting_mesh = np.linspace(0, end_time, 1000 * end_time)
        plot_model = model_generator(plotting_mesh, lam_true)
        true_response = plot_model()  # no args evaluates true param

        # true signal
        plt.plot(plotting_mesh, true_response, lw=5, c='k', alpha=1, label="True Signal, $\\xi \\sim N(0, \\sigma^2)$")

        # observations
        np.random.seed(11)
        annotate_height = 0.82
        u = qoi_true + np.random.randn(len(qoi_true)) * sigma
        plot_num_measure = num_meas_plot
        plt.scatter(time_vector[:plot_num_measure], u[:plot_num_measure], color='k', marker='.', s=250, alpha=alpha_points, label=f'{num_meas_plot} Sample Measurements')
        plt.annotate("$ \\downarrow$ Observations begin", (0.95, annotate_height), fontsize=fsize)
    #     plt.annotate("$\\downarrow$ Possible Signals", (0,annotate_height), fontsize=fsize)

        # sample signals
        num_sample_signals  = 100
        alpha_signal_sample = 0.15
        alpha_signal_mudpts = 0.45
        _true_response = plot_model(np.random.rand())  # uniform(0,1) draws from parameter space
        plt.plot(plotting_mesh, _true_response, lw=2, c='k', alpha=alpha_signal_sample, label='Predictions from Initial Density')
        for i in range(1, num_sample_signals):
            _true_response = plot_model(np.random.rand())  # uniform(0,1) draws from parameter space
            plt.plot(plotting_mesh, _true_response, lw=1, c='k', alpha=alpha_signal_sample)

        # error bars
        sigma_label = f"$\\pm3\\sigma \\qquad\\qquad \\sigma^2={sigma**2:1.3E}$"
        plt.plot(plotting_mesh[1000:], true_response[1000:] + 3 * sigma, ls='--', lw=3, c='xkcd:black', alpha=1)
        plt.plot(plotting_mesh[1000:], true_response[1000:] - 3 * sigma, ls='--', lw=3, c='xkcd:black', alpha=1, label=sigma_label)
        plt.plot(plotting_mesh[:1000], true_response[:1000] + 3 * sigma, ls='--', lw=3, c='xkcd:black', alpha=alpha_signal)
        plt.plot(plotting_mesh[:1000], true_response[:1000] - 3 * sigma, ls='--', lw=3, c='xkcd:black', alpha=alpha_signal)

        # solutions / samples
        mud_solutions = solutions[num_meas_plot]
        plt.plot(plotting_mesh, plot_model(mud_solutions[0][0]), lw=3, c='xkcd:bright red', alpha=alpha_signal_mudpts, label=f'{len(mud_solutions)} Estimates with $N={num_meas_plot:3d}$')
        for _lam in mud_solutions[1:]:
            _true_response = plot_model(_lam[0])
            plt.plot(plotting_mesh, _true_response, lw=3, c='xkcd:bright red', alpha=alpha_signal_mudpts)

        plt.ylim([0, 0.9])
        plt.xlim([0, end_time + .05])
        plt.ylabel('Response', fontsize=60)
        plt.xlabel('Time', fontsize=60)
        plt.xticks(fontsize=fsize)
        plt.yticks(fontsize=fsize)
        # legend ordering has a mind of its own, so we format it to our will
        # plt.legend(fontsize=fsize, loc='upper right')
        handles, labels = plt.gca().get_legend_handles_labels()
        order = [4, 0, 2, 1, 3]
        plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], fontsize=fsize, loc='upper right')
        plt.tight_layout()
        if save:
            fdir = filename.split('/')[0]
            check_dir(fdir)
            plt.savefig(filename, bbox_inches='tight')
        # plt.show()


def plot_experiment_equipment(tolerances, res, prefix, fsize=32, linewidth=5,
                              title="Variance of MUD Error", save=True):
    print("Plotting experiments involving equipment differences...")
    plt.figure(figsize=(10, 10))
    for _res in res:
        _prefix, _in, _rm, _re = _res
        regression_err_mean, slope_err_mean, regression_err_vars, slope_err_vars, sd_means, sd_vars, num_sensors = _re
        plt.plot(tolerances, regression_err_mean, label=f"{_prefix:10s} slope: {slope_err_mean:1.4f}", lw=linewidth)
        plt.scatter(tolerances, sd_means, marker='x', lw=20)

    plt.yscale('log')
    plt.xscale('log')
    plt.Axes.set_aspect(plt.gca(), 1)
    plt.ylim(2E-3, 2E-2)
    # plt.ylabel("Absolute Error", fontsize=fsize)
    plt.xlabel('Tolerance', fontsize=fsize)
    plt.legend()
    plt.title(f"Mean of MUD Error for N={num_sensors}", fontsize=1.25 * fsize)
    if save:
        plt.savefig(f'{prefix}_convergence_mud_std_mean.png', bbox_inches='tight')
    # plt.show()

    plt.figure(figsize=(10, 10))
    for _res in res:
        _prefix, _in, _rm, _re = _res
        regression_err_mean, slope_err_mean, regression_err_vars, slope_err_vars, sd_means, sd_vars, num_sensors = _re
        plt.plot(tolerances, regression_err_vars, label=f"{_prefix:10s} slope: {slope_err_vars:1.4f}", lw=linewidth)
        plt.scatter(tolerances, sd_vars, marker='x', lw=20)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(2E-5, 2E-4)
    plt.Axes.set_aspect(plt.gca(), 1)
    # plt.ylabel("Absolute Error", fontsize=fsize)
    plt.xlabel('Tolerance', fontsize=fsize)
    plt.legend()
    plt.title(title, fontsize=1.25 * fsize)
    if save:
        plt.savefig(f'{prefix}_convergence_mud_std_var.png', bbox_inches='tight')
    # plt.show()


def plot_experiment_measurements(measurements, res, prefix, fsize=32, linewidth=5, xlabel='Number of Measurements', save=True, legend=False):
    print("Plotting experiments involving increasing # of measurements.")
    plt.figure(figsize=(10, 10))
    for _res in res:
        _prefix, _in, _rm, _re = _res
        regression_mean, slope_mean, regression_vars, slope_vars, means, variances = _rm
        plt.plot(measurements[:len(regression_mean)], regression_mean, label=f"{_prefix:10s} slope: {slope_mean:1.4f}", lw=linewidth)
        plt.scatter(measurements[:len(means)], means, marker='x', lw=20)
    plt.xscale('log')
    plt.yscale('log')
    plt.Axes.set_aspect(plt.gca(), 1)
    plt.ylim(0.9 * min(means), 1.3 * max(means))
    plt.ylim(2E-3, 2E-1)
    plt.xlabel(xlabel, fontsize=fsize)
    if legend:
        plt.legend(fontsize=fsize * 0.8)
    # plt.ylabel('Absolute Error in MUD', fontsize=fsize)
    plt.title("$\\mathrm{\\mathbb{E}}(|\\lambda^\\mathrm{MUD} - \\lambda^\\dagger|)$", fontsize=1.25 * fsize)
    if save:
        plt.savefig(f'{prefix}_convergence_obs_mean.png', bbox_inches='tight')
    # plt.show()

    plt.figure(figsize=(10, 10))
    for _res in res:
        _prefix, _in, _rm, _re = _res
        regression_mean, slope_mean, regression_vars, slope_vars, means, variances = _rm
        plt.plot(measurements[:len(regression_vars)], regression_vars, label=f"{_prefix:10s} slope: {slope_vars:1.4f}", lw=linewidth)
        plt.scatter(measurements[:len(variances)], variances, marker='x', lw=20)
    plt.xscale('log')
    plt.yscale('log')
    plt.Axes.set_aspect(plt.gca(), 1)
#     if not len(np.unique(variances)) == 1:
#         plt.ylim(0.9 * min(variances), 1.3 * max(variances))
    plt.ylim(1E-5, 1E-3)
    plt.xlabel(xlabel, fontsize=fsize)
    if legend:
        plt.legend(fontsize=fsize * 0.8)
    # plt.ylabel('Absolute Error in MUD', fontsize=fsize)
    plt.title("$\\mathrm{Var}(|\\lambda^\\mathrm{MUD} - \\lambda^\\dagger|)$", fontsize=1.25 * fsize)
    if save:
        _logger.info("Saving measurement experiments.")
        plt.savefig(f'{prefix}_convergence_obs_var.png', bbox_inches='tight')
    else:
        plt.show()


def plot_scalar_poisson_summary(res, measurements, prefix, lam_true, fsize=32, save=False):
    from fenics import plot as _plot
    from mud_examples.poisson import poissonModel # function evaluation (full response surface)

    _logger.info("Fenics plotting for 1D example: Plotting surface...")
    for _res in res:
        _prefix, _in, _rm, _re = _res
        lam, qoi, sensors, qoi_true, experiments, solutions = _in
        gamma = lam
        plot_num_measure = min(100, max(measurements))
        raveled_input = np.repeat(gamma, qoi.shape[1])
        raveled_output = qoi.reshape(-1)
        x = raveled_input
        y = raveled_output

        fig = plt.figure(figsize=(10,8))
        gs = gridspec.GridSpec(3, 3)
        ax_main = plt.subplot(gs[1:3, :2])
        # ax_xDist = plt.subplot(gs[0, :2],sharex=ax_main)
        ax_yDist = plt.subplot(gs[1:3, 2],sharey=ax_main)

        a = np.argsort(gamma)
        slopes = []

        # ax_main.plot(x,y,marker='.')
        for idx in range(plot_num_measure):
            ax_main.plot(gamma[a], qoi[a,idx], c='k',
                     label=f'sensor {idx}: (%.2f, %.2f)'%(sensors[idx,0], sensors[idx,1]),
                     lw=1, alpha=0.1)
            slopes.append(qoi[a[-1],idx] - qoi[a[0],idx])
        sa = np.argsort(slopes)
        slopes = np.array(slopes)
        ranked_slopes = slopes[sa]

        xlabel_text = "$\\lambda$"
        # ylabel_text = "$u(x_i, \lambda)$"
        ylabel_text = "Measurement\nResponse"
        ax_main.axes.set_xlabel(xlabel_text, fontsize=fsize)
        ax_main.axes.set_ylabel(ylabel_text, fontsize=fsize)
        ax_main.axes.set_ylim((-1.25,0.5))
        # ax_main.axes.set_title('Sensitivity of Measurements', fontsize=1.25*fsize)
        ax_main.axvline(3)

        ax_yDist.hist(qoi_true, bins=np.linspace(-1.25,0.5,35), orientation='horizontal', align='mid')
        # ax_yDist.set(xlabel='count')
        ax_yDist.tick_params(labelleft=False, labelbottom=False)
        if save:
            plt.savefig(f'{_prefix}_qoi_response.png', bbox_inches='tight')
        #plt.show()

        plt.figure(figsize=(10,10))
        plt.title("Sensitivity of\nMeasurement Locations", fontsize=1.25*fsize)
        plt.hist(ranked_slopes, bins=np.linspace(-1.25,0,25), density=True)
        plt.xlabel("Slope", fontsize=fsize)
        if save:
            plt.savefig(f'{_prefix}_sensitivity_qoi.png', bbox_inches='tight')
        else:
            plt.show()

        ##########

        plt.figure(figsize=(10,10))
        num_sensitive  = 20
        most_sensitive = sa[sa < 100][0:num_sensitive]
        _logger.info(f"{num_sensitive} most sensitive sensors in first 100: {most_sensitive}")
        _plot(poissonModel(lam_true))
        for i in range(min(100, max(measurements))):
            plt.scatter(sensors[i,0], sensors[i,1], c='w', s=200)
            if i in most_sensitive:
                plt.scatter(sensors[i,0], sensors[i,1], c='y', s=100)
        #     plt.annotate(f"{i+1:02d}", (sensors[i,0]-0.0125, sensors[i,1]-0.01), alpha=1, fontsize=0.35*fsize)
        # plt.title('Reference solution', fontsize=1.25*fsize)
        plt.xlabel('$x_1$', fontsize=fsize)
        plt.ylabel('$x_2$', fontsize=fsize)
        if save:
            plt.savefig(f'{_prefix}_reference_solution.png', bbox_inches='tight')
        else:
            plt.show()
