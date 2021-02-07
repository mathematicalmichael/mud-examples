#!/bin/sh
NUM_SAMPLES=1000
FILE_PREFIX=res
INITIALDIST=u
for dim in 2 ; do
    echo "Running for Dim=${dim}. Saving to ${FILE_PREFIX}${dim}"
    python run_poisson.py -n ${NUM_SAMPLES} -d ${INITIALDIST} -i ${dim} -o ${FILE_PREFIX}${dim}${INITIALDIST}
done
