#!/bin/sh
python mud_problem.py --save --example ode --num-trials 20 \
	-r 0.01 -r 0.05 -r 0.1 -r 0.25 -r 0.5 -r 1
