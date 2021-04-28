#!/bin/sh
mkdir -p /tmp/mud_figures/
echo "Running all examples using default docker entry-point"
docker run --rm -i -v /tmp/mud_figures:/work mudex $@
