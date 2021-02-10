#!/bin/sh
python mud_problem.py --save --example pde --alt --num-trials 20 \
	-m 5 -m 10 -m 20 -m 50 -m 100 -m 250 -m 500 -m 1000
