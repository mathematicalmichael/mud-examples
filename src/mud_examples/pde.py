

import numpy as np
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.backend = 'Agg'
matplotlib.rcParams['figure.figsize'] = 10,10
matplotlib.rcParams['font.size'] = 16

import poisson as ps  # TODO: move this to mud_examples
from mud.funs import mud_problem, map_problem
from mud.util import std_from_equipment

from mud_examples.models import generate_spatial_measurements as generate_sensors_pde
from mud_examples.helpers import experiment_measurements, extract_statistics, experiment_equipment
from mud_examples.plotting import fit_log_linear_regression

from mud_examples.datasets import load_poisson

def main_pde(num_trials=5,
             tolerances=[],
             measurements=[],
             fsize=32,
             seed=21,
             lam_true=3.0,
             input_dim=2,
             domain=[[-4,0]], alt=False, bayes=False):
    print(f"Will run simulations for N={measurements}")
    res = []
    num_measure = max(measurements)

    sd_vals     = [ std_from_equipment(tolerance=tol, probability=0.99) for tol in tolerances ]
    sigma       = sd_vals[-1] # sorted, pick largest
    example_list = [ 'mud' ]
    if alt:
        example_list.append('mud-alt')
    if bayes:
        example_list.append('map')

    for example in example_list:
        print(f"Example: {example}")
        P = ps.pdeProblem()
        # in 1d this is a change in sensor location
        # in ND, change in how we partition sensors (vertical vs horizontal)
        if example == 'mud-alt' and input_dim == 1:
            fname = 'pde_{input_dim}D/mud-alt_ref.pkl'
            try:
                P.load(fname)
            except FileNotFoundError:
                # attempt to load xml results from disk.
                ps.make_reproducible_without_fenics('mud-alt', lam_true, input_dim=1, num_samples=None, num_measure=num_measure)
                P.load(fname)
            wrapper = P.mud_scalar()
            ps.plot_without_fenics(fname, num_sensors=100, num_qoi=1, example=example)
        else:
            fname = f'pde_{input_dim}D/mud_ref.pkl'
            try:
                P.load(fname)
            except FileNotFoundError:
                ps.make_reproducible_without_fenics('mud', lam_true, input_dim=input_dim, num_samples=None, num_measure=num_measure)
                P.load(fname)

            if example == 'mud-alt':
                wrapper = P.mud_vector_vertical()
                ps.plot_without_fenics(fname, num_sensors=100, mode='ver',
                                       num_qoi=input_dim, example=example)
            else:
                wrapper = P.mud_vector_horizontal()
                ps.plot_without_fenics(fname, num_sensors=100, mode='hor',
                                       num_qoi=input_dim, example=example)

        # adjust measurements to account for what we actually have simulated
        measurements = np.array(measurements)
        measurements = list(measurements[measurements <= P.qoi.shape[1]])
        print("Increasing Measurements Study")
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
            print("Increasing Measurement Precision Study")
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
            P.plot_initial()
            P.plot_solutions(solutions, 20, example=example)
            P.plot_solutions(solutions, 100, example=example, save=True)

    return res