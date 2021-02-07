import numpy as np

from mud.plot import make_2d_unit_mesh
from mud.util import std_from_equipment

import pickle

from mud_examples.plotting import plot_experiment_measurements, plot_experiment_equipment
from mud_examples.plotting import plot_decay_solution, plot_poisson_solution


from mud_examples.plotting import fit_log_linear_regression
from mud_examples.models import generate_decay_model
from mud_examples.models import generate_temporal_measurements as generate_sensors_ode
from mud_examples.models import generate_spatial_measurements as generate_sensors_pde
from mud_examples.datasets import load_poisson

from mud.funs import mud_problem, map_problem
from mud_examples.helpers import experiment_measurements, extract_statistics, experiment_equipment

import pickle
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.backend = 'Agg'
matplotlib.rcParams['figure.figsize'] = 10,10
matplotlib.rcParams['font.size'] = 16

### PDE ###
def main_pde(model_list, num_trials=5,
             tolerances=[],
             measurements=[],
             fsize=32,
             seed=21,
             lam_true=3.0,
             domain=[[1,5]], alt=False):
    print(f"Will run simulations for S={measurements}")
    res = []
    num_measure = max(measurements)

    sd_vals     = [ std_from_equipment(tolerance=tol, probability=0.99) for tol in tolerances ]
    sigma       = sd_vals[-1] # sorted, pick largest
    example_list = [ 'pde' ]
    if alt:
        example_list.append('pde-alt')

    for example in example_list:
        print(f"Example: {example}")
        if example == 'pde-alt':
            sensors = generate_sensors_pde(num_measure, ymax=0.95, xmax=0.25)
        else:
            sensors = generate_sensors_pde(num_measure, ymax=0.95, xmax=0.95)

        # TODO clean this up ... if you cannot import it, load from file.
        from poisson import poisson_sensor_model
        qoi_true = poisson_sensor_model(sensors, gamma=lam_true, nx=36, ny=36) # wrapper around `poisson`
        lam, qoi = load_poisson(sensors, model_list)

        with open(f'{example}_data.pkl', 'wb') as f:
            pickle.dump({'qoi_true': qoi_true, 'sensors': sensors, 'lam': lam, 'qoi': qoi}, f)

        def mud_wrapper(num_obs, sd):
            return mud_problem(domain=domain, lam=lam, qoi=qoi, sd=sd, qoi_true=qoi_true, num_obs=num_obs)

        print("Increasing Measurements Study")
        experiments, solutions = experiment_measurements(num_measurements=measurements,
                                                 sd=sigma,
                                                 num_trials=num_trials,
                                                 seed=seed,
                                                 fun=mud_wrapper)

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
                                                  fun=mud_wrapper)

            regression_err_mean, slope_err_mean = fit_log_linear_regression(tolerances, sd_means)
            regression_err_vars, slope_err_vars = fit_log_linear_regression(tolerances, sd_vars)
            _re = (regression_err_mean, slope_err_mean,
                   regression_err_vars, slope_err_vars,
                   sd_means, sd_vars, num_sensors)
        else:
            _re = None  # hack to avoid changing data structures for the time being

        _in = (lam, qoi, sensors, qoi_true, experiments, solutions)
        _rm = (regression_mean, slope_mean, regression_vars, slope_vars, means, variances)
        res.append((example, _in, _rm, _re))
    return res


### ODE ###
def main_ode(num_trials,
             fsize=32,
             seed=21,
             lam_true=0.5,
             domain=[[0,1]],
             tolerances=[0.1],
             time_ratios=[1], alt=False, bayes=False):
    res = []
    print(f"Will run simulations for %T={time_ratios}")
    sd_vals      = [ std_from_equipment(tolerance=tol, probability=0.99) for tol in tolerances ]
    sigma        = sd_vals[-1] # sorted, pick largest
    t_min, t_max = 1, 3
    example_list = [ 'ode-mud' ]
    if alt:
        example_list.append('ode-mud-alt')

    if bayes:
        example_list.append('ode-map')

    for example in example_list:
        print(f"Example: {example}")
        if example == 'ode-mud-alt':
            sensors = generate_sensors_ode(measurement_hertz=200, start_time=t_min, end_time=t_max)
        else:
            sensors = generate_sensors_ode(measurement_hertz=100, start_time=t_min, end_time=t_max)

        measurements = [ int(np.floor(len(sensors)*r)) for r in time_ratios ]
        print(f"Measurements: {measurements}")
#         times        = [ sensors[m-1] for m in measurements ]
        num_measure = max(measurements)

        model    = generate_decay_model(sensors, lam_true)
        qoi_true = model() # no args evaluates true param
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


        print("Increasing Measurements Quantity Study")
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


def main(args):
    np.random.seed(args.seed)
    example       = args.example
    num_trials   = args.num_trials
    fsize        = args.fsize
    linewidth    = args.linewidth
    seed         = args.seed
    inputdim     = args.input_dim
    save         = args.save
    tolerances   = list(np.sort([ float(t) for t in args.sensor_tolerance ]))
    if len(tolerances) == 0: tolerances = [0.1]

    if example == 'pde':
        measurements = list(np.sort([ int(n) for n in args.num_measure ]))
        if len(measurements) == 0:
            measurements = [100]
    else:
        time_ratios  = list(np.sort([ float(r) for r in args.ratio_measure ]))
        if len(time_ratios) == 0:
            time_ratios = [1.0]

    print("Running...")
    if example == 'pde':
        lam_true = 3.0
        model_list  = pickle.load(open(f'res{inputdim}u.pkl', 'rb'))
        res = main_pde(model_list, num_trials=num_trials,
                         fsize=fsize,
                         seed=seed,
                         lam_true=lam_true,
                         tolerances=tolerances,
                         measurements=measurements)

        plot_poisson_solution(res=res, measurements=measurements,
                     fsize=fsize, prefix=f'pde_{inputdim}D/' + example, lam_true=lam_true, save=save)

        if len(measurements) > 1:
            plot_experiment_measurements(measurements, res,
                                         f'pde_{inputdim}D/' + example, fsize,
                                         linewidth, save=save)
        if len(tolerances) > 1:
            plot_experiment_equipment(tolerances, res,
                                      f'pde_{inputdim}D/' + example, fsize,
                                      linewidth, save=save)
    elif example == 'ode':
        lam_true = 0.5
        res = main_ode(num_trials=num_trials,
                         fsize=fsize,
                         seed=seed,
                         lam_true=lam_true,
                         tolerances=tolerances,
                         time_ratios=time_ratios, bayes=True)

        if len(time_ratios) > 1:
            plot_experiment_measurements(time_ratios, res,
                                         'ode/' + example,
                                         fsize, linewidth,
                                         save=save, legend=True)

        if len(tolerances) > 1:
            plot_experiment_equipment(tolerances, res,
                                      'ode/' + example, fsize, linewidth,
                                      title=f"Variance of MUD Error\nfor t={1+2*np.median(time_ratios):1.3f}s",
                                      save=save)
    ##########


    if args.save:
        with open('results.pkl', 'wb') as f:
            pickle.dump(res, f)


######


if __name__ == "__main__":
    import argparse
    desc = """
        Examples
        """

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-e', '--example',       default='ode', type=str)
    parser.add_argument('-m', '--num-measure',   default=[],  action='append')
    parser.add_argument('-r', '--ratio-measure', default=[],  action='append')
    parser.add_argument('--num-trials',    default=20,    type=int)
    parser.add_argument('-t', '--sensor-tolerance',  default=[0.1], action='append')
    parser.add_argument('-s', '--seed',          default=21)
    parser.add_argument('-lw', '--linewidth',    default=5)
    parser.add_argument('-i', '--input-dim',     default=1, type=int)
    parser.add_argument('--fsize',               default=32, type=int)
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()
    main(args)

