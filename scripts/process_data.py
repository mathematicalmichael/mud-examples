#!/usr/bin/env python3
from mud_examples.poisson import make_reproducible_without_fenics as mf

num_sensors = 500

dist = "u"
for input_dim in [2, 5, 11]:
    fname = mf(input_dim=input_dim, sample_dist=dist, num_measure=num_sensors)
    print(fname)

dist = "n"
input_dim = 2
for tol in [0.9999, 0.99, 0.95]:
    fname = mf(input_dim=input_dim, sample_dist=dist, tol=tol, num_measure=num_sensors)
    print(fname)
