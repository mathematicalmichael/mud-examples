[![PyPI version](https://img.shields.io/pypi/v/mud-examples)](https://pypi.org/project/mud-examples/)
![unit testing workflow](https://github.com/mathematicalmichael/mud-examples/actions/workflows/main.yml/badge.svg)
![example workflow](https://github.com/mathematicalmichael/mud-examples/actions/workflows/examples.yml/badge.svg)
![build workflow](https://github.com/mathematicalmichael/mud-examples/actions/workflows/build.yml/badge.svg)
![publish workflow](https://github.com/mathematicalmichael/mud-examples/actions/workflows/publish.yml/badge.svg)
![docker workflow](https://github.com/mathematicalmichael/mud-examples/actions/workflows/docker.yml/badge.svg)
[![docs](https://readthedocs.org/projects/mud-examples/badge/?version=stable)](https://mud-examples.readthedocs.io/en/stable/?badge=stable)
[![black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![coveralls](https://coveralls.io/repos/github/mathematicalmichael/mud-examples/badge.svg?branch=main)](https://coveralls.io/github/mathematicalmichael/mud-examples?branch=main)
[![downloads](https://static.pepy.tech/personalized-badge/mud-examples?period=total&units=abbreviation&left_color=gray&right_color=blue&left_text=downloads)](https://pepy.tech/project/mud-examples)

# MUD-Examples
## Examples for _Existence, Uniqueness, and Convergence of Parameter Estimates with Maximal Updated Densities_

Authors: Troy Butler & Michael Pilosov

# Installation
For Python 3.7-3.12:

```sh
pip install mud-examples
```

To reproduce the results in Michael's thesis, use `mud-examples==0.1`. However, this comes with `mud==0.0.28`.
Newer versions should still produce the same figures.

TeX is recommended (but not required):

```
apt-get install -yqq \
    texlive-base \
    texlive-latex-base \
    texlive-latex-extra \
    texlive-fonts-recommended \
    texlive-fonts-extra \
    texlive-science \
    latexmk \
    dvipng \
    cm-super
```


# Quickstart

Generate all of the figures the way they are referenced in the paper:
```sh
mud_run_all
```
The above is equivalent to running all of the examples sequentially:

```sh
mud_run_inv
mud_run_lin
mud_run_ode
mud_run_pde
```

# Usage

The `mud_run_X` scripts all call the same primary entrypoint, which you can call with the console script `mud_examples`.

Here are two examples:
```sh
mud_examples --example ode
```

```sh
mud_examples --example lin
```

and so on. (More on this later, once argparsing is better handled, they might just be entrypoints to the modules themselves rather than a central `runner.py`, which really only exists to compare several experiments, so perhaps it warrants renaming to reflect that).
