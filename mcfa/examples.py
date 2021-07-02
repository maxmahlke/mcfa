#!/usr/bin/env python
""" Simulation studies to illustrate MCFA."""

import os

# Run this example on the CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mcfa import MCFA


def sample_3d(N):
    """Draw points from two 3D Gaussian distributions."""

    mu1 = np.array([0, 0, 0])
    mu2 = np.array([2, 2, 6])

    cov1 = np.array(
        [
            [4, -1.8, -1],
            [-1.8, 2, 0.9],
            [-1, 0.9, 2],
        ]
    )
    cov2 = np.array(
        [
            [4, 1.8, 0.8],
            [1.8, 2, 0.5],
            [0.8, 0.5, 2],
        ]
    )
    cluster1 = np.random.multivariate_normal(mu1, cov1, size=int(np.ceil(N / 2)))
    cluster2 = np.random.multivariate_normal(mu2, cov2, size=int(np.floor(N / 2)))
    data = np.append(cluster1, cluster2, axis=0)
    return data


def baek_2010_section_5(dropout=0.0):
    """Fit the MCFA model to the example application in Baek+ 2010.

    Parameters
    ==========
    dropout : float
        Fraction of points to randomly replace by NaN. Default is 0.
    """
    seed = 37  # set seed for reproducibility
    np.random.seed(seed)

    epochs = 100  # number of training epochs

    g = 2  # number of mixture components
    q = 2  # number of latent dimensions

    # ------
    # Define the underlying multivariate Gaussian distributions
    N = 500  # number of random samples drawn

    data = sample_3d(N)

    # Dropout data if specified - nice one-liner from
    # https://stackoverflow.com/a/32182680/6162422
    if dropout:
        data.ravel()[
            np.random.choice(data.size, int(data.size * dropout), replace=False)
        ] = np.nan

    # Some points may have had all observations removed - we resample those
    if any(np.isnan(data).all(axis=1)):
        data[np.isnan(data).all(axis=1)] = sample_3d(
            len(data[np.isnan(data).all(axis=1)])
        )
    # ------
    # Train the MCFA model
    model = MCFA(n_components=g, n_factors=q)

    # Fit the model parameters using ML
    model.fit(
        data,
        n_epochs=500,
        learning_rate=3e-4,
        frac_validation=0.15,
        converge_epochs=10,
        batch_size=4,
    )

    # Compute the cluster probabilities of the observations
    model.transform(data)

    if dropout:
        # Impute the missing data based on the observation's assigned clusters
        model.impute()

    # ------
    # Plot the model properties
    model.plot_data_space()
    model.plot_latent_loadings()
    model.plot_latent_space()
    model.plot_loss()


if __name__ == "__main__":

    # Compute example of Baek+ 2010, section 5
    baek_2010_section_5(dropout=0.0)

    # Compute example of Baek+ 2010, section 5, randomly removing 30% of the observations
    baek_2010_section_5(dropout=0.2)
