#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

import numpy as np

_logger = logging.getLogger(__name__)


def extract_statistics(solutions, reference_value):
    num_sensors_plot_conv = solutions.keys()
    means = []
    variances = []
    for ns in num_sensors_plot_conv:
        _logger.debug(f'Extracting stats for {ns} measurements.')
        mud_solutions = solutions[ns]
        num_trials = len(mud_solutions)
        err = [np.linalg.norm(m - reference_value) for m in mud_solutions]
        assert len(err) == num_trials
        mean_mud_sol = np.mean(err)
        var_mud_sol = np.var(err)
        means.append(mean_mud_sol)
        variances.append(var_mud_sol)

    return means, variances


def fit_log_linear_regression(input_values, output_values):
    if len(np.unique(output_values)) == 1:
        _logger.info("Log Linear Regression: All values identical.")
        return np.array(output_values), 0
    x, y = np.log10(input_values), np.log10(output_values)
    X, Y = np.vander(x, 2), np.array(y).reshape(-1, 1)
    slope, intercept = (np.linalg.pinv(X) @ Y).ravel()
    regression_line = 10**(slope * x + intercept)
    return regression_line, slope