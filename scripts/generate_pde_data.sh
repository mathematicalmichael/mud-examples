#!/bin/sh
NUM_SAMPLES=1000
FILE_PREFIX=results
DIST=n
DIM=2
for TOL in 95 99 9999; do
    echo "Running for Dim=${dim}."
    generate_poisson_data -v -s "$NUM_SAMPLES" -d "$DIST" -t 0.${TOL} -i "$DIM" -m -2.0 || break
done

DIST=u
for DIM in 2 5 11; do
    echo "Running for Dim=${dim}."
    generate_poisson_data -v -s "$NUM_SAMPLES -d "$DIST" -i "$DIM" || break
done
