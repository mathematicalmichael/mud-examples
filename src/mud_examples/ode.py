import logging

_logger = logging.getLogger(__name__)

from mud.funs import mud_problem, map_problem
from mud.util import std_from_equipment

from mud_examples.models import generate_spatial_measurements as generate_sensors_pde
from mud_examples.helpers import experiment_measurements, extract_statistics, experiment_equipment
from mud_examples.plotting import plot_decay_solution
from mud_examples.plotting import fit_log_linear_regression
from mud_examples.models import generate_decay_model
from mud_examples.models import generate_temporal_measurements as generate_sensors_ode

import numpy as np
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.backend = 'Agg'
matplotlib.rcParams['figure.figsize'] = 10,10
matplotlib.rcParams['font.size'] = 16


def main_ode(num_trials=20,
             fsize=32,
             seed=21,
             lam_true=0.5,
             domain=[[0,1]],
             tolerances=[0.1],
             time_ratios=[0.01, 0.05, 0.1, 0.25, 0.5, 1],
             alt=False, bayes=True):
    """
    
    
    >>> from mud_examples.ode import main_ode
    >>> res = main_ode(num_trials=5, time_ratios=[0.01, 0.1, 1])
    Will run simulations for %T=[0.01, 0.1, 1]
    Running example: mud
    Measurements: [2, 20, 200]
    Plotting decay solution.
    Running example: map
    Measurements: [2, 20, 200]
    Plotting decay solution.
    """
    res = []
    print(f"Will run simulations for %T={time_ratios}")
    sd_vals      = [ std_from_equipment(tolerance=tol, probability=0.99) for tol in tolerances ]
    sigma        = sd_vals[-1] # sorted, pick largest
    t_min, t_max = 1, 3
    example_list = [ 'mud' ]
    if alt:
        example_list.append('mud-alt')

    if bayes:
        example_list.append('map')

    for example in example_list:
        print(f"Running example: {example}")
        if example == 'ode-mud-alt':
            sensors = generate_sensors_ode(measurement_hertz=200, start_time=t_min, end_time=t_max)
        else:
            sensors = generate_sensors_ode(measurement_hertz=100, start_time=t_min, end_time=t_max)

        measurements = [ int(np.floor(len(sensors)*r)) for r in time_ratios ]
        print(f"Measurements: {measurements}")
#         times        = [ sensors[m-1] for m in measurements ]
        num_measure = max(measurements)

        model    = generate_decay_model(sensors, lam_true)
        qoi_true = model()  # no args evaluates true param
        np.random.seed(seed)
        lam = np.random.rand(int(1E3)).reshape(-1,1)
        qoi = model(lam)

        if example == 'ode-map':
            def wrapper(num_obs, sd):
                return map_problem(domain=domain, lam=lam, qoi=qoi,
                                   sd=sd, qoi_true=qoi_true, num_obs=num_obs)
        else:
            def wrapper(num_obs, sd):
                return mud_problem(domain=domain, lam=lam, qoi=qoi,
                                   sd=sd, qoi_true=qoi_true, num_obs=num_obs)


        _logger.info("Increasing Measurements Quantity Study")
        experiments, solutions = experiment_measurements(num_measurements=measurements,
                                                 sd=sigma,
                                                 num_trials=num_trials,
                                                 seed=seed,
                                                 fun=wrapper)

        means, variances = extract_statistics(solutions, lam_true)
        regression_mean, slope_mean = fit_log_linear_regression(time_ratios, means)
        regression_vars, slope_vars = fit_log_linear_regression(time_ratios, variances)

        ##########

        num_sensors = num_measure
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

        # TO DO clean all this up
        _in = (lam, qoi, sensors, qoi_true, experiments, solutions)
        _rm = (regression_mean, slope_mean, regression_vars, slope_vars, means, variances)
        example_name = '-'.join(example.split('-')[1:]).upper()
        res.append((example_name, _in, _rm, _re))

        # TODO check for existence of save directory, grab subset of measurements properly.
        plot_decay_solution(solutions, generate_decay_model, fsize=fsize,
                            end_time=t_max, lam_true=lam_true, qoi_true=qoi_true,
                            sigma=sigma, time_vector=sensors, prefix='ode/' + example)

    return res
