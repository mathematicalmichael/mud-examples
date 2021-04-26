#!/bin/sh
mkdir -p mud_figures/
docker run --rm -ti -v $(pwd)/mud_figures:/work mudex $@
