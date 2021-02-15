#!/bin/sh
NUM_SAMPLES=1
FILE_PREFIX=res
INITIALDIST=n
for dim in 2 ; do
    echo "Running for Dim=${dim}."
    generate_poisson_data -v -n ${NUM_SAMPLES} -d ${INITIALDIST} -i ${dim} -p ${FILE_PREFIX}
done
