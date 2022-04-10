#!/usr/bin/env python

"""Functions to plot the data and latent space of the MCFA models."""

import colorcet
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

CLUSTER_COLORS = np.array(colorcet.glasbey_category10)
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
                    color=CLUSTER_COLORS[k],
                )

            # ------
            # Add data points with imputed values
            if hasattr(model, "Y_imp"):

                for k in range(model.n_components):
                    ax.scatter(
                        model.Y_imp[np.where(clusters == k), px],
                        model.Y_imp[np.where(clusters == k), py],
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
                    color=CLUSTER_COLORS[k],
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


def confidence_ellipse(mean, cov, ax, probability=0.95, **kwargs):
    """Plot the confidence ellipse at a given level of the covariance matrix.

    Parameters
    ----------
    mean : np.ndarray
        The mean coordinates of the data.
    cov : np.ndarray
        The covariance matrix of the data
    ax : matplotlib.axis.Axis
        The axis instance to add the ellipse to.
    probability : float
        The probability level at which to draw the ellipse. Default is 0.95.

    Returns
    -------
    matplotlib.axis.Axis
        The axis instance with the ellipse added.

    Notes
    -----
    Further keyword arguments are passed to the matplotlib.patches.Ellipse
    instance and can be used to change the ellipse appearance.
    """

    # ------
    # Compute error ellipse axes and rotation
    eigenvalues, eigenvector = np.linalg.eigh(cov)

    # ensure orientation of eigenvector in line with angle definition
    for i, vector in enumerate(eigenvector):
        if vector[0] < 0:
            eigenvector[i] *= -1

    t = np.linspace(0, 2 * np.pi, 100)
    a = np.sqrt(eigenvalues[0] * chi2(df=2).ppf(probability))
    b = np.sqrt(eigenvalues[1] * chi2(df=2).ppf(probability))

    t_rot = -angle_between([1, 0], eigenvector[0])

    ellipse = np.array([a * np.cos(t), b * np.sin(t)])

    # 2-D rotation matrix
    rotation = np.array(
        [[np.cos(t_rot), -np.sin(t_rot)], [np.sin(t_rot), np.cos(t_rot)]]
    )
    ell_rot = np.zeros((2, ellipse.shape[1]))

    for i in range(ellipse.shape[1]):
        ell_rot[:, i] = np.dot(rotation, ellipse[:, i])

    elli = Ellipse(mean, 2 * a, 2 * b, np.degrees(t_rot), **kwargs)
    return ax.add_patch(elli)


def angle_between(vector1, vector2):
    """Returns the angle in radians between two vectors."""
    vector1 = vector1 / np.linalg.norm(vector1)
    vector2 = vector2 / np.linalg.norm(vector2)
    return np.arccos(np.clip(np.dot(vector1, vector2), -1.0, 1.0))
