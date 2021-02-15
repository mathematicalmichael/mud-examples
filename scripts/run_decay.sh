#!/bin/sh
mud_examples --example ode --bayes --num-trials 20 \
	-r 0.01 0.05 0.1 0.25 0.5 1
