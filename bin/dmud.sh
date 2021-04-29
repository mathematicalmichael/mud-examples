#!/bin/sh
mkdir -p /tmp/mud_figures/
docker run --rm -it -v /tmp/mud_figures:/work/figures mudex $@
