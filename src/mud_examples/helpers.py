import numpy as np


def fit_log_linear_regression(input_values, output_values):
    x, y = np.log10(input_values), np.log10(output_values)
    X, Y = np.vander(x, 2), np.array(y).reshape(-1, 1)
    slope, intercept = (np.linalg.pinv(X) @ Y).ravel()
    regression_line = 10**(slope * x + intercept)
    return regression_line, slope


def experiment_measurements(fun, num_measurements, sd, num_trials, seed=21):
    """
    Fixed sensors, varying how much data is incorporated into the solution.
    """
    experiments = {}
    solutions = {}
    for ns in num_measurements:
        discretizations = []
        mud_solutions = []
        for t in range(num_trials):
            np.random.seed(seed + t)
            _d = fun(sd=sd, num_obs=ns)
            discretizations.append(_d)
            mud_solutions.append(_d.mud_point())
        experiments[ns] = discretizations
        solutions[ns] = mud_solutions

    return experiments, solutions


def extract_statistics(solutions, reference_value):
    num_sensors_plot_conv = solutions.keys()
    means = []
    variances = []
    for ns in num_sensors_plot_conv:
        mud_solutions = solutions[ns]
        num_trials = len(mud_solutions)
        err = [np.linalg.norm(m - reference_value) for m in mud_solutions]
        assert len(err) == num_trials
        mean_mud_sol = np.mean(err)
        var_mud_sol = np.var(err)
        means.append(mean_mud_sol)
        variances.append(var_mud_sol)

    return means, variances
