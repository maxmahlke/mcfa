#!/usr/bin/env python

"""Functions to plot the data and latent space of the MCFA models."""

from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

CLUSTER_MARKERS = np.array([".", "x", "^", "s", "+"])


def plot_data_space(model, Y, clusters=np.array([])):
    """Plot the model training data, colour-coded by cluster if the model has
    been trained.

    Parameters
    ----------
    model : core.MCFA
        An MCFA model instance.
    Y : np.ndarray
        The data to plot.
    clusters : np.ndarray
        The cluster assignments of the data. Optional, default is empty array.

    Notes
    -----
    Datapoints with at least one dimension imputed are displayed with half opacity.
    """

    # Create triangular figure of 2d projections of dimension pairs
    fig, axes = plt.subplots(ncols=model.p - 1, nrows=model.p - 1, figsize=(16, 10))

    # Get the cluster moments in data space
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
                    Y[clusters == k, px] if clusters.size else Y[:, px],
                    Y[clusters == k, py] if clusters.size else Y[:, py],
                    marker=CLUSTER_MARKERS[k],
                )

            # ------
            # Add data points with imputed values
            if hasattr(model, "Y_imp"):

                for k in range(model.n_components):
                    ax.scatter(
                        model.Y_imp[np.where(clusters == k), px],
                        model.Y_imp[np.where(clusters == k), py],
                        marker=CLUSTER_MARKERS[k],
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
                    facecolor="none",
                    edgecolor="black",
                    ls="--",
                )

            ax.set(xlabel=f"{px + 1}", ylabel=f"{py + 1}")

    fig.suptitle("Data Space")
    plt.tight_layout()
    plt.show()


def plot_latent_space(model, Z, clusters, mask_imputed=None):
    """Plot the model data and clusters in latent space.

    Parameters
    ----------
    model : core.MCFA
        A trained MCFA model instance.
    Z : np.ndarray
        The latent scores of the data.
    clusters : np.ndarray
        The cluster assignments of the data.
    mask_imputed : np.ndarray
        A mask of the input data Y where the cell is True if the value was
        observed and False if it is missing. Used to change opacity of imputed scores.
        Default is None, disabling the opacity-change.
    """

    # Create triangular figure of 2d projections of dimension pairs
    fig, axes = plt.subplots(
        ncols=model.n_factors - 1, nrows=model.n_factors - 1, figsize=(12, 7.5)
    )

    # Compute the cluster moments and factor scores in latent spaces
    _, _, clusters_mean, clusters_cov = model._compute_cluster_moments()

    for i in range(model.n_factors - 1):
        for j in range(model.n_factors - 1):

            if model.n_factors > 2:
                ax = axes[i, j]
            else:
                ax = axes
            dx = j  # index of data feature to plot on x-axis
            dy = i + 1  # index of data feature to plot on y-axis

            if dx >= dy:
                ax.axis("off")
                continue

            for k in range(model.n_components):

                idx_cluster = np.where(clusters == k)[0]

                # Plot the factor scores by cluster
                # Points with imputed dimensions are plotted with alpha=0.5
                ax.scatter(
                    Z[idx_cluster, dx],
                    Z[idx_cluster, dy],
                    marker=CLUSTER_MARKERS[k],
                    s=30,
                    alpha=[0.5 if np.any(mask_imputed) else 1 for ind in idx_cluster]
                    if mask_imputed is not None
                    else 1,
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
                    facecolor="none",
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


def confidence_ellipse(mean, cov, ax, n_std=2.0, facecolor="none", **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Adapted from https://matplotlib.org/3.5.0/gallery/statistics/confidence_ellipse.html

    Parameters
    ----------
    mean : np.ndarray
        The mean of the dataset in the two dimensions to plot.

    cov : np.ndarray
        The covariance of the dataset in the two dimensions to plot.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """

    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs,
    )

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x, mean_y = mean

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std

    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)
