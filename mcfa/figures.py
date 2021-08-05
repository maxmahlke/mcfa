#!/usr/bin/env python

"""Functions to plot the data and latent space of the MCFA models."""

import colorcet
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

CLUSTER_COLORS = np.array(colorcet.glasbey_category10)
CLUSTER_MARKERS = np.array([".", "x", "^", "s", "+"])


def plot_data_space(model):
    """Plot the model training data, colour-coded by cluster if the model has
    been trained.

    Parameters
    ==========
    model : core.MCFA
        An MCFA model instance.

    Notes
    =====
    Datapoints with at least one dimension imputed are displayed with half opacity.
    """

    # Create triangular figure of 2d projections of dimension pairs
    fig, axes = plt.subplots(ncols=model.p - 1, nrows=model.p - 1, figsize=(16, 10))

    # Get the cluter moments in data space
    clusters_mean, clusters_cov, _, _ = model._compute_cluster_moments()
    clusters_mean = clusters_mean.numpy()

    for i in range(model.p - 1):
        for j in range(model.p - 1):

            ax = axes[i, j]
            px = j  # index of data feature to plot on x-axis
            py = i + 1  # index of data feature to plot on y-axis

            if px >= py:
                ax.axis("off")
                continue

            # Plot the training data
            for k in range(model.n_components):
                ax.scatter(
                    model.Y[np.where(model.clusters == k), px],
                    model.Y[np.where(model.clusters == k), py],
                    marker=CLUSTER_MARKERS[k],
                    color=CLUSTER_COLORS[k],
                )

            # ------
            # Add data points with imputed values
            if hasattr(model, "Y_imp"):

                for k in range(model.n_components):
                    ax.scatter(
                        model.Y_imp[np.where(model.clusters == k), px],
                        model.Y_imp[np.where(model.clusters == k), py],
                        marker=CLUSTER_MARKERS[k],
                        color=CLUSTER_COLORS[k],
                        alpha=0.5,
                    )

            # ------
            # Add the modeled clusters
            for k in range(model.n_components):

                ax.scatter(
                    clusters_mean[k, px],
                    clusters_mean[k, py],
                    color="black",
                    marker="x",
                )

                confidence_ellipse(
                    mean=clusters_mean[k][[px, py]],
                    cov=clusters_cov[k][np.ix_([px, py], [px, py])],
                    ax=ax,
                    edgecolor="black",
                    ls="--",
                )

            ax.set(xlabel=f"{px + 1}", ylabel=f"{py + 1}")

    fig.suptitle("Data Space")
    plt.tight_layout()
    plt.show()


def plot_latent_space(model):
    """Plot the model data and clusters in latent space.

    Parameters
    ==========
    model : core.MCFA
        An MCFA model instance.

    Notes
    =====
    Datapoints with at least one dimension imputed are displayed with half opacity.
    """

    # Create triangular figure of 2d projections of dimension pairs
    fig, axes = plt.subplots(
        ncols=model.n_factors - 1, nrows=model.n_factors - 1, figsize=(12, 7.5)
    )

    # Compute the cluster moments and factor scores in latent spaces
    _, _, clusters_mean, clusters_cov = model._compute_cluster_moments()
    Z, Zmean, Zclust = model._compute_factor_scores()

    for i in range(model.n_factors - 1):
        for j in range(model.n_factors - 1):

            if model.n_factors > 2:
                ax = axes[i, j]
            else:
                ax = axes
            dx = j  # index of data feature to plot on x-axis
            dy = i + 1  # index of data feature to plot on y-axis

            if dx >= dy:

                if dx == 1 and dy == 1:
                    A = model.W.numpy()
                    D, J = A.shape
                    xi = np.arange(D)

                    for j in range(J):
                        ax.plot(xi, A.T[j], "-", label=str(j + 1))

                    ax.axhline(0, ls=":", c="#000000", zorder=-1, lw=0.5)

                    ax.legend(frameon=False)
                    ax.set(xlim=(-0.5, D - 0.5), title="Latent Loadings")
                    continue

                ax.axis("off")
                continue

            for k in range(model.n_components):

                idx_cluster = np.where(model.clusters == k)[0]

                # Plot the factor scores by cluster
                # Points with imputed dimensions are plotted with alpha=0.5
                ax.scatter(
                    Zclust[idx_cluster, dx],
                    Zclust[idx_cluster, dy],
                    marker=CLUSTER_MARKERS[k],
                    color=CLUSTER_COLORS[k],
                    s=30,
                    alpha=[
                        0.5 if np.any(np.isnan(model.Y[ind])) else 1
                        for ind in idx_cluster
                    ],
                )

                # Add the cluster confidence ellipse
                mean = clusters_mean[k][[dx, dy]]

                ax.scatter(
                    mean[0],
                    mean[1],
                    color="black",
                    marker="x",
                )

                confidence_ellipse(
                    mean=mean,
                    cov=clusters_cov[k][np.ix_([dx, dy], [dx, dy])],
                    ax=ax,
                    edgecolor="black",
                    ls="--",
                )

            ax.set(xlabel=f"{dx + 1}", ylabel=f"{dy + 1}")

    fig.suptitle("Latent Space")
    plt.tight_layout()
    plt.show()


def plot_latent_loadings(model, block=True):
    """Plot the latent loadings of the trained model."""
    fig, ax = plt.subplots(figsize=(16, 10))

    A = model.W.numpy()
    D, J = A.shape
    xi = np.arange(D) + 1

    for j in range(J):
        ax.plot(xi, A.T[j], "-", label=str(j + 1))

    ax.axhline(0, ls=":", c="#000000", zorder=-1, lw=0.5)

    ax.legend(frameon=False)
    ax.set(xlim=(0.5, D + 0.5))
    ax.set_xticks(xi)

    fig.suptitle("Latent Loadings")
    fig.tight_layout()
    plt.show()


def plot_loss(model):
    """Plot the loss of the training and the validation sets over the model training."""

    fig, ax = plt.subplots()

    ax.plot(model.loss_training, ls="-", label="Training")
    ax.plot(model.loss_validation, ls="-", label="Validation")

    ax.legend(frameon=False)
    ax.set(xlabel="Epochs", ylabel="Loss", title="Loss")

    plt.tight_layout()
    plt.show()


# ------
# Helper function from matplotliib
def confidence_ellipse(mean, cov, ax, n_std=1.0, facecolor="none", **kwargs):
    """
    Create a plot of the covariance confidence ellipse of `x` and `y`

    See how and why this works: https://carstenschelp.github.io/2018/09/14/Plot_Confidence_Ellipse_001.html

    This function has made it into the matplotlib examples collection:
    https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html#sphx-glr-gallery-statistics-confidence-ellipse-py

    Or, once matplotlib 3.1 has been released:
    https://matplotlib.org/gallery/index.html#statistics

    I update this gist according to the version there, because thanks to the matplotlib community
    the code has improved quite a bit.
    Parameters
    ----------
    x, y : array_like, shape (n, )
        Input data.
    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.
    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.
    Returns
    -------
    matplotlib.patches.Ellipse
    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    if len(mean) != 2:
        print("mean must have length 2")
    if cov.shape != (2, 2):
        print("cov must be a 2x2 matrix")

    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = mpl.patches.Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs,
    )

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    mean_x, mean_y = mean

    scale_x = np.sqrt(cov[0, 0]) * n_std

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std

    transf = (
        mpl.transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)
