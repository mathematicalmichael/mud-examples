#!/bin/sh
NUM_SAMPLES=1000
TOL=95
# FILE_PREFIX=${TOL}results
# DIST=n
FILE_PREFIX=results
DIST=u
for dim in 2 ; do
    echo "Running for Dim=${dim}."
    generate_poisson_data -v -n ${NUM_SAMPLES} -d ${DIST} -t 0.${TOL} -i ${dim} -p ${FILE_PREFIX}
done
