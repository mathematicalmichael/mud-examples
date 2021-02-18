# -*- coding: utf-8 -*-
#!/usr/env/bin python

import importlib
import logging
import os
import types

import numpy as np
from mud.funs import mud_sol, map_sol

_logger = logging.getLogger(__name__)


def check_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

class LazyLoader(types.ModuleType):

  def __init__(self, module_name='utensor_cgen', submod_name=None):
    self._module_name = '{}{}'.format(
      module_name,
      submod_name and '.{}'.format(submod_name) or ''
    )
    self._mod = None
    super(LazyLoader, self).__init__(self._module_name)

  def _load(self):
    if self._mod is None:
      self._mod = importlib.import_module(
        self._module_name
      )
    return self._mod

  def __getattr__(self, attrb):
    try:
        return getattr(self._load(), attrb)
    except ModuleNotFoundError:
        pass

  def __dir__(self):
    return dir(self._load())

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
        _logger.debug(f'Measurement experiment. Num measurements: {ns}')
        discretizations = []
        estimates = []
        for t in range(num_trials):
            np.random.seed(seed + t)
            _d = fun(sd=sd, num_obs=ns)
            estimate = _d.estimate()
            discretizations.append(_d)
            estimates.append(estimate)
        experiments[ns] = discretizations
        solutions[ns] = estimates

    return experiments, solutions


def experiment_equipment(fun, num_measure, sd_vals, num_trials, reference_value):
    """
    Fixed number of sensors, varying the quality of equipment.
    """
    sd_err = []
    sd_var = []
    for sd in sd_vals:
        _logger.debug(f'Equipment Experiment. Std Dev: {sd}')
        temp_err = []
        for t in range(num_trials):
            _d = fun(sd=sd, num_obs=num_measure)
            estimate = _d.estimate()
            temp_err.append(np.linalg.norm(estimate - reference_value))
        sd_err.append(np.mean(temp_err))
        sd_var.append(np.var(temp_err))

    return sd_err, sd_var


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


def compare_mud_map_pin(A, b, d, mean, cov):
    mud_pt = mud_sol(A, b, d, mean, cov)
    map_pt = map_sol(A, b, d, mean, cov)
    pin_pt = (np.linalg.pinv(A)@(d-b)).reshape(-1,1)
    return mud_pt, map_pt, pin_pt


def transform_rank_list(lam_ref, A, b, rank):
    """
    A is a list here. We sum the first `rank` elements of it
    to return a matrix with the desired rank.
    """
    _A = sum(A[0:rank])
    _b = b
    _d = _A@lam_ref + _b
    assert np.linalg.matrix_rank(_A) == rank, "Unexpected rank mismatch"
    return _A, _b, _d


def transform_dim_out(lam_ref, A, b, dim):
    if isinstance(A, list) or isinstance(A, tuple):
        raise AttributeError("A must be a matrix, not a list or tuple.")

    _A = A[:dim, :]
    _b = b[:dim, :]
    _d = _A@lam_ref + _b
    return _A, _b, _d


def compare_linear_sols_rank_list(lam_ref, A, b,
                             alpha=1, mean=None, cov=None):
    """
    Input and output dimensions fixed, varying rank 1..dim_output.
    """
    
    return compare_linear_sols(transform_rank_list, lam_ref, A, b, alpha, mean, cov)


def compare_linear_sols_dim(lam_ref, A, b,
                            alpha=1, mean=None, cov=None):
    """
    Input dimension fixed, varying output dimension.
    """
    return compare_linear_sols(transform_dim_out, lam_ref, A, b, alpha, mean, cov)


def compare_linear_sols(transform, lam_ref, A, b,
                            alpha=1, mean=None, cov=None):
    """
    Input dimension fixed, varying according to the output
    of the anonymous function `transform`'s return.
    """
    sols = {}
    if isinstance(alpha, list) or isinstance(alpha, tuple):
        alpha_list = alpha
    else:
        alpha_list = [alpha]

    if mean is None:
        mean = np.zeros(A.shape[1])
    
    if cov is None:
        cov = np.eye(A.shape[1])

    print("alpha = {}".format(alpha_list))
    if isinstance(A, list):  # svd approach returns list
        dim_output = A[0].shape[0]
    else:
        dim_output = A.shape[0]

    for alpha in alpha_list:
        sols[alpha] = []
        for _out in range(1, dim_output+1, 1):
            _A, _b, _d = transform(lam_ref, A, b, _out)
            _mud, _map, _pin = compare_mud_map_pin(_A, _b, _d, mean, alpha*cov)
            sols[alpha].append((_mud, _map, _pin))

    return sols
