from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
from mud.util import null_space
from mud.funs import mud_sol, map_sol
from mud.norm import full_functional, norm_input, norm_data, norm_predicted
from mud.plot import make_2d_unit_mesh


def fit_log_linear_regression(input_values, output_values):
    x, y = np.log10(input_values), np.log10(output_values)
    X, Y = np.vander(x, 2), np.array(y).reshape(-1, 1)
    slope, intercept = (np.linalg.pinv(X) @ Y).ravel()
    regression_line = 10**(slope * x + intercept)
    return regression_line, slope


def plot_2d_contour_example(A=np.array([[1, 1]]), b=np.zeros([1, 1]),  # noqa: C901
                            cov_11=0.5, cov_01=-0.25,
                            initial_mean=np.array([0.25, 0.25]),
                            alpha=1, omega=1, obs_std=1,
                            show_full=True, show_data=True, show_est=False,
                            fsize=42, figname='latest_figure.png', save=False):
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
                plt.annotate('MAP',
                             (map_pt_eq[0] + 0.001 * fsize, map_pt_eq[1] - 0.001 * fsize),
                             fontsize=fsize, backgroundcolor="w")
            if omega > 0:  # analytical MUD point
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

            if show_full and omega > 0:
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
        plt.savefig(figname, dpi=300)

#     plt.title('Predicted Covariance: {}'.format((A@initial_cov@A.T).ravel() ))
    # plt.show()


def plot_decay_solution(solutions, model_generator, sigma, prefix,
                        time_vector, lam_true, qoi_true, end_time=3, fsize=32, test=False):
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
        plt.plot(plotting_mesh, plot_model(mud_solutions[0][0]), lw=3, c='xkcd:bright red', alpha=alpha_signal_mudpts, label=f'{len(mud_solutions)} Updated Solutions for N={num_meas_plot}')
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
        if not test:
            plt.savefig(filename, bbox_inches='tight')
        # plt.show()
