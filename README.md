[![arXiv](https://img.shields.io/badge/arXiv-2203.11229-f9f107.svg)](https://arxiv.org/abs/2203.11229) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<p align="center">
  <img width="260" src="https://raw.githubusercontent.com/maxmahlke/mcfa/main/gfx/logo_mcfa.png">
</p>

This `python` package implements the Mixtures of Common Factor Analyzers model
introduced by [Baek+ 2010](https://ieeexplore.ieee.org/document/5184847). It
uses [tensorflow](https://www.tensorflow.org/) to implement a stochastic
gradient descent, which allows for model training without prior imputation of
missing data. The interface resembles the [sklearn](https://scikit-learn.org/stable/) model API.

# Documentation

Refer to the `docs/documentation.ipynb` for the documentation and
`docs/4d_gaussian.ipynb` for an example application.

# Install

Install from PyPi using `pip`:

     $ pip install mcfa

The minimum required `python` version is 3.8.

# Alternatives

- [EMMIXmfa](https://github.com/suren-rathnayake/EMMIXmfa) in `R`
- [Casey+ 2019](https://github.com/andycasey/mcfa) in `python`

Compared to this implementation, Casey+ 2019 use an EM-algorithm instead of a
stochastic gradient descent. This requires the imputation of the missing values
**before** the model training. On the other hand, there are more initialization
routines the lower space loadings and factors available in the Casey+ 2019 implementation.
