#!/bin/sh
mkdir -p mud_figures/
docker run --rm -i -v $(pwd)/mud_figures:/work mudex $@
