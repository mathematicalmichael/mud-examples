import logging
import os

_logger = logging.getLogger(__name__)


import numpy as np
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.backend = 'Agg'
matplotlib.rcParams['figure.figsize'] = 10,10
matplotlib.rcParams['font.size'] = 16

from mud.funs import mud_problem, map_problem
from mud.util import std_from_equipment

from mud_examples.models import generate_spatial_measurements as generate_sensors_pde
from mud_examples.helpers import experiment_measurements, extract_statistics, experiment_equipment, check_dir
from mud_examples.plotting import fit_log_linear_regression

import mud_examples.poisson as ps  # lazy loads fenics

def main_pde(num_trials=20,
             tolerances=[0.1],
             measurements=[5, 20, 50, 100, 250, 500],
             fsize=32,
             seed=21,
             lam_true=3.0,
             input_dim=2, dist='u',
             prefix='results',
             alt=True, bayes=True):
    """

    >>> from mud_examples.pde import main_pde
    >>> res = main_pde(num_trials=10)
    Attempt run for measurements = [5, 20, 50, 100, 250, 500]
    Running example: mud
    Running example: mud-alt
    Running example: map
    """
    print(f"Attempt run for measurements = {measurements}")
    res = []
    num_measure = max(measurements)

    sd_vals     = [ std_from_equipment(tolerance=tol, probability=0.99) for tol in tolerances ]
    sigma       = sd_vals[-1] # sorted, pick largest
    _logger.info(f'Using std. dev {sigma}')
    example_list = [ 'mud' ]
    if alt:
        example_list.append('mud-alt')
    if bayes:
        example_list.append('map')

    for example in example_list:
        print(f"Running example: {example}")
        P = ps.pdeProblem()
        # in 1d this is a change in sensor location
        # in ND, change in how we partition sensors (vertical vs horizontal)
        fdir = f'pde_{input_dim}D' # expectation from make_reproducible_without_fenics
        check_dir(fdir)
        # mud and mud alt have same sensors in higher dimensional examples
        # in 1d, the alternative approach is to change sensor placement, which requires
        # loading a separate file.
        if example == 'mud-alt' and input_dim == 1:
            fname = f'{fdir}/ref_alt_{prefix}{input_dim}{dist}.pkl'
            try:
                P.load(fname)
            except FileNotFoundError:
                # attempt to load xml results from disk.
                fname = ps.make_reproducible_without_fenics('mud-alt', lam_true, input_dim=1,
                                                    num_samples=None, num_measure=num_measure,
                                                    prefix=prefix, dist=dist)
                P.load(fname)
            wrapper = P.mud_scalar()
            ps.plot_without_fenics(fname, num_sensors=100, num_qoi=1, example=example)
        else:
            fname = f'{fdir}/ref_{prefix}{input_dim}{dist}.pkl'
            try:
                P.load(fname)
            except FileNotFoundError:
                
                try: # available data in package
                    _logger.info("Trying packaged data.")
                    fname = 'data/' + fname
                    P.load(fname)
#                     curdir = os.getcwd().split('/')[-1]
#                     if curdir == 'scripts':
#                         raise FileNotFoundError("already within scripts directory.")
#                     _logger.warning("Attempting from scripts directory.")
#                     fname = f'scripts/{fname}'
#                     P.load(fname)
                except FileNotFoundError:
                    _logger.info("Failed to load requested data from disk or packaged datasets.")
                    fname = ps.make_reproducible_without_fenics('mud', lam_true, input_dim=input_dim,
                                                        num_samples=None, num_measure=num_measure,
                                                        prefix=prefix, dist=dist)
                    try:
                        P.load(fname)
                    except FileNotFoundError as e:
                        raise(e)
                        _logger.critical("Exiting program")
                        break

            # plots show only one hundred sensors to avoid clutter
            if example == 'mud-alt':
                wrapper = P.mud_vector_vertical()
                ps.plot_without_fenics(fname, num_sensors=100, mode='ver',
                                       num_qoi=input_dim, example=example)
            elif example == 'mud':
                wrapper = P.mud_vector_horizontal()
                ps.plot_without_fenics(fname, num_sensors=100, mode='hor',
                                       num_qoi=input_dim, example=example)
            elif example == 'map':
                wrapper = P.map_scalar(log=True)
                ps.plot_without_fenics(fname, num_sensors=100,
                                       num_qoi=input_dim, example=example)
        # adjust measurements to account for what we actually have simulated
        measurements = np.array(measurements)
        measurements = list(measurements[measurements <= P.qoi.shape[1]])
        _logger.info("Increasing Measurements Study")
        _logger.info(f"Will run simulations for N={measurements}")
        experiments, solutions = experiment_measurements(num_measurements=measurements,
                                                 sd=sigma,
                                                 num_trials=num_trials,
                                                 seed=seed,
                                                 fun=wrapper)

        means, variances = extract_statistics(solutions, lam_true)
        regression_mean, slope_mean = fit_log_linear_regression(measurements, means)
        regression_vars, slope_vars = fit_log_linear_regression(measurements, variances)

        ##########

        num_sensors = min(100, num_measure)
        if len(tolerances) > 1:
            _logger.info("Increasing Measurement Precision Study")
            sd_means, sd_vars = experiment_equipment(num_trials=num_trials,
                                                  num_measure=num_sensors,
                                                  sd_vals=sd_vals,
                                                  reference_value=lam_true,
                                                  fun=wrapper)

            regression_err_mean, slope_err_mean = fit_log_linear_regression(tolerances, sd_means)
            regression_err_vars, slope_err_vars = fit_log_linear_regression(tolerances, sd_vars)
            _re = (regression_err_mean, slope_err_mean,
                   regression_err_vars, slope_err_vars,
                   sd_means, sd_vars, num_sensors)
        else:
            _re = None  # hack to avoid changing data structures for the time being

        _in = (P.lam, P.qoi, P.sensors, P.qoi_ref, experiments, solutions)
        _rm = (regression_mean, slope_mean, regression_vars, slope_vars, means, variances)
        res.append((example, _in, _rm, _re))

        if input_dim > 1:
            if example == 'mud':
                P.plot_initial()
            for m in measurements:
                P.plot_solutions(solutions, m, example=example)
#             P.plot_solutions(solutions, 100, example=example, save=True)

    return res