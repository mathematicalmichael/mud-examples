#!/bin/sh
NUM_SAMPLES=1000
DIST=n
DIM=2
for TOL in 95 99 9999; do
    echo "Running with Normal for Tolerance=0.${TOL}."
    generate_poisson_data -v -s "$NUM_SAMPLES" -d "$DIST" -t 0.${TOL} -i "$DIM" -m -2.0 || break
done
