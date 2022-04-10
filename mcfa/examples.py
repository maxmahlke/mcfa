#!/usr/bin/env python
""" Simulation studies to illustrate MCFA."""

import os
import sys

# Run this example on the CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import matplotlib.pyplot as plt
import numpy as np

from mcfa import MCFA


def gaussians_example(dropout=0.0, bic_icl=False):
    """Fit the MCFA model to simulated data from 3 4D-Gaussian distributions.

    Parameters
    ----------
    dropout : float
        Fraction of points to randomly replace by NaN. Default is 0.
    bic_icl : bool
        Compute the BIC/ICL scores for multiple model fits for model selection.
    """

    n_epochs = 200  # number of training epochs

    p = 6  # number of data features
    g = 3  # number of mixture components
    q = 2  # number of latent dimensions

    # ------
    # Define the underlying multivariate Gaussian distributions
    N = 200  # number of random samples drawn

    # Define the means and covariances of the three Gaussians
    # These are drawn randomly but reproducibly as we set certain seeds
    seed = 37  # set seed for reproducibility
    np.random.seed(seed)

    mu1 = np.random.rand(p) * 5  # * 5 just for a nicer scale, random.rand -> [0, 1)
    cov1 = np.random.rand(p, p)
    cov1 = np.dot(cov1, cov1.T)  # For any matrix A, A*A is positive semidefinite

    seed = 17
    np.random.seed(seed)
    mu2 = np.random.rand(p) * 5
    cov2 = np.random.rand(p, p)
    cov2 = np.dot(cov2, cov2.T)

    seed = 4
    np.random.seed(seed)
    mu3 = np.random.rand(p) * 5
    cov3 = np.random.rand(p, p)
    cov3 = np.dot(cov3, cov3.T)

    # Sample N points from the distributions
    cluster1 = np.random.multivariate_normal(mu1, cov1, size=N)
    cluster2 = np.random.multivariate_normal(mu2, cov2, size=N)
    cluster3 = np.random.multivariate_normal(mu3, cov3, size=N)

    data = np.concatenate([cluster1, cluster2, cluster3], axis=0)

    # Add missing data if 'drouput' is non-zero - nice one-liner from
    # https://stackoverflow.com/a/32182680/6162422
    if dropout:
        data.ravel()[
            np.random.choice(data.size, int(data.size * dropout), replace=False)
        ] = np.nan

    # Some points may have had all observations removed - we remove those
    data = data[~np.isnan(data).all(axis=1)]

    # Compute the BIC/ICL scores for many model fits if selected
    if bic_icl:
        _compute_bic_icl(data)
        sys.exit()

    # ------
    # Train the MCFA model
    model = MCFA(n_components=g, n_factors=q)

    # Fit the model parameters using ML
    model.fit(
        data,
        n_epochs=n_epochs,
        learning_rate=1e-4,
        frac_validation=0.15,
        converge_epochs=20,
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


def _compute_bic_icl(data):
    """Fit the MCFA model with multiple parameters and plot the BIC/ICL scores for
    model selection.

    Parameters
    ==========
    data : np.ndarray
        The data to cluster.
    """

    n_components = range(1, 7)

    LS = {2: "-", 3: "--", 4: "-.", 5: ":"}

    fig, ax = plt.subplots()

    for q in [2, 3, 4, 5]:

        bic = []
        icl = []

        for g in n_components:

            model = MCFA(n_components=g, n_factors=q)

            model.fit(
                data,
                n_epochs=10,
                learning_rate=3e-4,
                frac_validation=0.0,
                converge_epochs=100,
                batch_size=40,
            )

            model.transform(data)

            bic.append(model.bic())
            icl.append(model.icl())

        ax.plot(n_components, bic, label=f"BIC - q={q}", c="blue", ls=LS[q])
        ax.plot(n_components, icl, label=f"ICL - q={q}", c="red", ls=LS[q])

    ax.set(xlabel="N Components", ylabel="BIC/ICL")
    ax.legend(frameon=False)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":

    # Compute example of three 4D-Gaussians in 2D/3D-latent space
    # gaussians_example()

    # Compute the BIC and ICL scores for the 4D-Gaussian samples
    gaussians_example(bic_icl=True, dropout=0.3)

    # Compute example of three 4D-Gaussians in 2D/3d-latent space with 20% missing data
    # gaussians_example(dropout=0.3)
