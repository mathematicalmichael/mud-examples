#!/bin/sh
mkdir -p mud_figures/
echo "Running all examples using default docker entry-point"
docker run --rm -i -v $(pwd)/mud_figures:/work mudex $@
