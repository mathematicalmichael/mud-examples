#!/bin/sh
NUM_SAMPLES=1000
DIST=u
for DIM in 2 5 11; do
    echo "Running with Uniform for Dim=${DIM}."
    generate_poisson_data -v -s "$NUM_SAMPLES" -d "$DIST" -i "$DIM" || break
done
