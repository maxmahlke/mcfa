#!/usr/bin/env python

"""
Implementation of a Mixture of Common Factor Analyzers model with
Stochastic Gradient Descent training.
"""

import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
import scipy
from sklearn import decomposition, mixture
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm
import pyppca

import mcfa.figures

# Suppress tensorflow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ------
# Tensforflow device settings
tf.config.set_soft_device_placement(True)

# CPUs
tf.config.threading.set_inter_op_parallelism_threads(0)
tf.config.threading.set_intra_op_parallelism_threads(0)

# GPU
if len(tf.config.list_physical_devices("GPU")):
    gpu = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(device=gpu[0], enable=True)

# Eager execution is good for debugging tensorflow but slow
# tf.config.run_functions_eagerly(True)


def ensure_numpy_array(array):
    """Ensure that array is numpy ndarray instead of tensforflow Tensor object.

    Parameters
    ----------
    array : np.ndarray, tf.Tensor
        Array to convert.

    Returns
    -------
    np.ndarray
        Converted arrays.
    """
    if isinstance(array, np.ndarray):
        return array
    elif isinstance(array, tf.Tensor):
        return array.numpy()
    else:
        raise TypeError(
            f"Passed array is of type '{type(array)}', expected np.ndarray or tf.Tensor."
        )


class MCFA:
    """Mixture of Common Factor Analyzers model implementation."""

    def __init__(self, n_components, n_factors):
        """Initialize the MCFA model instance.

        Parameters
        ----------
        n_components : int
            The number of mixture components
        n_factors : int
            The number of latent dimensions / factors
        """
        self.n_components = int(n_components)
        self.n_factors = int(n_factors)

    def fit(
        self,
        Y,
        n_epochs,
        frac_validation=0.1,
        learning_rate=3e-4,
        init_factors="pca",
        batch_size=32,
    ):
        """Initialize the model parameters and train them with the passed data.

        Parameters
        ----------
        Y : np.ndarray
            The observed data to cluster. May contain missing values.
        n_epochs : int
            The number of training epochs
        frac_validation : float
            The fraction of the data used for validation during the training.
            Default is 0.1, i.e. 10%.
        learning_rate : float
            The learning rate passed to the gradient descent optimizer. Default is 3e-4.
        init_factors : str
            The method to initialize the latent factors with. Choose from ['pca', 'ppca']. Default is 'pca'.
        batch_size : int
            Batch size of training subsets to use during SGD. Larger is faster but less fine.
        """
        if isinstance(Y, pd.DataFrame):
            Y = Y.values

        self.Y = Y.astype(np.float64)
        self.N, self.p = self.Y.shape
        self.n_epochs = int(n_epochs)
        self.frac_validation = float(frac_validation)
        self.learning_rate = float(learning_rate)
        self.init_factors = init_factors
        self.batch_size = int(batch_size)

        # Quick tests of the dataset
        if not np.all(self.Y):
            warnings.warn(
                "The dataset contains elements which are equal to zero. Zero is used to flag missing values. "
                "These values will be ignored during the training."
            )
        if not any(np.all(~np.isnan(self.Y), axis=1)):
            warnings.warn(
                "The dataset does not contain a complete row of observations. This is not supported as the model "
                " initialization with PCA will not work. Consider implementing PPCA as initialization method."
            )
            sys.exit()

        # Initialize the clustering model
        self._initialize_model()

        # Train the model
        self._train_model()

        # Orthonormalize the solution
        self.orthonormalize()

    def transform(self, Y, which="zmean"):
        """Transform samples from data into latent space.

        Parameters
        ----------
        Y : np.ndarray
            The data samples to transform of shape N x p
        which : str
            Which type of clustering score should be returned. See Baek* 2011, section 4.
            Choose from ['z', 'zclust', 'zmean']. Default is 'zmean'.

        Returns
        -------
        np.ndarray
            If which is 'z': The factor scores [Z] of shape N x q
            If which is 'zclust':The factor scores if the observation belongs to the most probable cluster [Zclust].
            If which is 'zmean': The factor scores if the observation belongs to all clusters following tau [Zmean].

        Notes
        -----
        Computation based on Section 4 of Baek+ 2011.
        The code is based on github.com/andycasey/mcfa
        """

        Y = Y - self.mu
        N, p = Y.shape

        # Empty arrays to be filled later
        Z = np.zeros((self.n_components, N, self.n_factors))
        gamma = np.zeros((self.n_components, p, self.n_factors))

        D_inv = np.diag(1.0 / tf.math.softplus(self.Psi))
        I = np.eye(self.n_factors)

        # The cholesky decomposition and softplus application are undone in the
        # cluster moments function
        _, _, Xi, Omega = self._compute_cluster_moments()
        W = self.W.numpy()

        # Compute the scores per cluster
        for k in range(self.n_components):

            # Check if covariance matrix is ill-coniditioned
            if np.linalg.cond(Omega[k, :, :]) > 1e4:
                epsilon = 1e-5
                Omega[k] = Omega[k] + I * epsilon

            # using the Woodbury Identity for matrix inversion
            C = scipy.linalg.solve(
                scipy.linalg.solve(Omega[k, :, :], I) + W.T @ D_inv @ W, I
            )
            gamma[k, :, :] = (D_inv - D_inv @ W @ C @ W.T @ D_inv) @ W @ Omega[k, :, :]

            Z[k, :, :] = (
                np.repeat(Xi[k, :], N).reshape((self.n_factors, N)).T
                + (Y - (W @ Xi[k, :].T).T) @ gamma[k, :, :]
            )

        # Compute the cluster-specific scores
        cluster = self.predict(Y)
        tau = self.predict_proba(Y)
        cluster = np.argmax(tau, axis=1)

        Zclust = np.zeros((N, self.n_factors))
        Zmean = np.zeros((N, self.n_factors))

        for i in range(N):
            Zclust[i] = Z[cluster[i], i, :]
            Zmean[i] = Z[:, i].T @ tau[i]

        if which == "z":
            return Z
        elif which == "zmean":
            return Zmean
        elif which == "zclust":
            return Zclust

    def inverse_transform(self, latent_scores):
        """Compute the data given the latent scores."""

    def predict_proba(self, Y):
        """Compute the responsibility matrix tau, giving the probability for each sample to
        belong to any component in the model.

        Parameters
        ----------
        Y : np.ndarray
            The observed data to score. Must have the same features as the data used to train
            the model, of shape N x p

        Returns
        -------
        np.ndarray
            The responsibility matrix tau, of shape N x g
        """
        if isinstance(Y, pd.DataFrame):
            Y = Y.values

        _, p = Y.shape

        assert (
            p == self.p
        ), f"The number of passed input features ({p}) has to be equal to the number of learned input features ({self.p})."

        # Compute responsibility matrix
        tau = np.zeros((len(Y), self.n_components))
        for i, sample in enumerate(Y):
            tau[i, :] = self._cluster_incomplete(sample, np.isfinite(sample))

        return tau

    def predict(self, Y):
        """Compute the most likely latent component for each sample in Y.

        Parameters
        ----------
        Y : np.ndarray
            The observed data to score. Must have the same features as the data used to train
            the model, of shape N x p

        Returns
        -------
        np.ndarray
            Array containing the most probable cluster for each sample, of shape N
        """
        tau = self.predict_proba(Y)

        # Attribute for convenient access
        clusters = tf.argmax(tau, axis=1).numpy()
        return clusters

    # ------
    # Model initialization functions
    def _initialize_model(self):
        """Initialize the latent space and cluster properties."""

        # mu is initialized on the observed data per feature
        self.mu = np.nanmean(self.Y, axis=0).reshape([self.p])

        # Initialize the latent space
        if self.init_factors == "pca":
            self._initialize_by_pca()
        elif self.init_factors == "ppca":
            self._initialize_by_ppca()

        # Initialize the cluster components
        self._initialize_by_gmm()

        # Define the model variables using the init values
        self.pi = tf.Variable(self.pi, dtype=tf.float32, name="pi")
        self.mu = tf.Variable(self.mu, dtype=tf.float32, name="mu")
        self.Xi = tf.Variable(self.Xi, dtype=tf.float32, name="Xi")
        self.W = tf.Variable(self.W, dtype=tf.float32, name="W")
        self.Omega = tf.Variable(self.Omega, dtype=tf.float32, name="Omega")
        self.Psi = tf.Variable(self.Psi, dtype=tf.float32, name="Psi")

        # Create list of trainable parameters for gradient descent
        self.theta = [self.Xi, self.pi, self.W, self.Omega, self.Psi, self.mu]

    def _initialize_by_pca(self):
        """Initialize the latent space parameters W and Psi with PCA
        on the complete-case data."""

        # PCA on the complete data for the latent factors
        Y_complete = self.Y[~np.isnan(self.Y).any(axis=1)]
        pca = decomposition.PCA(n_components=self.n_factors)

        # Initial values of factor scores
        self.Z = pca.fit_transform(Y_complete)

        # Initial values of W
        self.W = pca.components_.T

        # Initial values of the component specific error terms
        self.Psi = tfp.math.softplus_inverse(
            pca.noise_variance_.astype(np.float32)
        ) * tf.ones([self.p])

    def _initialize_by_ppca(self):
        """Initialize the latent space parameters W and Psi with PPCA
        on the entire dataset."""

        data_init = self.Y.copy()
        pca_loadings, ss, M, pca_scores, data_imputed = pyppca.ppca(
            data_init, self.n_factors, dia=False
        )

        # Initial values of factor scores
        self.Z = pca_scores

        # Initial values of W
        self.W = pca_loadings

        # Initial values of the component specific error terms
        noise_variance = np.ones([self.p]) * 0.001

        self.Psi = tfp.math.softplus_inverse(
            noise_variance.astype(np.float32)
        ) * tf.ones([self.p])

    def _initialize_by_gmm(self):
        """Initialize the cluster parameters pi, xi, and omega using GMM in
        the initialized latent space."""

        # GMM on PCA loadings for cluster initialisation
        mix = mixture.GaussianMixture(
            n_components=self.n_components, covariance_type="full", random_state=17
        )
        mix.fit(self.Z)

        # Initial value of latent covariance Omega (Cholesky
        # decomposition for computational efficiency)
        Omega = tf.linalg.cholesky(mix.covariances_.astype(np.float32))
        self.Omega = tf.linalg.set_diag(
            Omega, tfp.math.softplus_inverse(tf.linalg.diag_part(Omega))
        )

        self.pi = np.log(mix.weights_ + 0.01)
        self.Xi = mix.means_.reshape((self.n_components, self.n_factors))

    def _split_datasets(self):
        """Splits the input data into training and validation subsets, each containing batches
        of constant size and with equal missingness pattern among observations in a single batch.
        """

        split = np.random.rand(self.N) < self.frac_validation

        data_training = self.Y[~split]
        data_validation = self.Y[split]

        self.validation_split = split

        # Now we create the subsets of equal missingness patterns for each dataset
        for i, data in enumerate([data_training, data_validation]):

            # The datasets for training cannot contain NaNs, so we fill them with 0
            # These will be filtered out later with the missingness masks
            data_0 = data.copy()
            data_0[np.isnan(data_0)] = 0

            # Missingness patterns are differentiated by hash functions
            missing = pd.DataFrame(data=np.isfinite(data))

            missing["hash"] = missing.apply(
                lambda row: hash("".join([str(col) for col in row])), axis=1
            )
            missing = missing.reset_index()

            # Build list of data subsets with equal missingness pattern.
            for j, hash_ in enumerate(set(missing["hash"])):

                # Create subset with the same missingness pattern
                data_batch = data_0[missing.loc[missing.hash == hash_].index, :]

                # Pad them with all zero observations to the correct batch size
                # This allows to build a single Dataset, which is much faster than training
                # with multiple datasets
                if len(data_batch) % self.batch_size != 0:
                    data_batch = np.vstack(
                        (
                            data_batch,
                            np.zeros(
                                (
                                    self.batch_size - len(data_batch) % self.batch_size,
                                    self.p,
                                )
                            ),
                        )
                    )

                # Stack the batchs on top of each other to construct a dataset
                subsets = data_batch if j == 0 else np.vstack([subsets, data_batch])

            if i == 0:
                data_training = subsets
            else:
                data_validation = subsets

        # Construct the tf.data.Dataset instances from the input data
        self.training_set = (
            tf.data.Dataset.from_tensor_slices(data_training)
            .batch(self.batch_size, num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE)
        )

        # The validation set does not need to be shuffled, and we can pass is as a single batch
        self.validation_set = (
            tf.data.Dataset.from_tensor_slices(data_validation)
            .batch(self.batch_size, num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE)
        )

    # ------
    # Model training functions
    def _train_model(self):
        """Perform stochastic gradient descent to optimize the model parameters."""

        # Split into training:validation
        self._split_datasets()

        # ------
        # Training

        # Record the loss of the training and the validation set during training
        self.loss_training = []
        self.loss_validation = []

        # And train
        self._optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        progress = tqdm(range(self.n_epochs), desc="Training ", unit="epoch")
        for epoch in progress:

            # LL per dataset and training batch
            ll_train = []
            ll_valid = []

            for batch_training in self.training_set:

                # Give a lower probability of training with batches with low completeness
                missing_bins = len(np.where(batch_training.numpy()[0, :] == 0)[0])
                skip_probability = missing_bins / self.p

                if np.random.uniform(0, 1) <= skip_probability:
                    continue
                loss_training = self._train_step(batch_training)
                ll_train.append(tf.reduce_mean(loss_training))

            for batch_validation in self.validation_set:
                loss_validation = -self._log_likelihood_incomplete(batch_validation)
                ll_valid.append(tf.reduce_mean(loss_validation))

            self.loss_training.append(tf.reduce_mean(ll_train))
            self.loss_validation.append(tf.reduce_mean(ll_valid))

    @tf.function
    def _train_step(self, batch):
        """Execute a gradient-descent step.

        Parameters
        ----------
        batch : tf.data.Dataset
            The observation[s] to train the model on. Same missingness pattern between
            the observations, possibly padded by all-zero rows.

        Returns
        -------
        float
            The loss of the training batch.
        """
        with tf.GradientTape() as tape:
            loss = -self._log_likelihood_incomplete(batch)

        # Adam iteration
        gradients = tape.gradient(loss, self.theta)
        self._optimizer.apply_gradients(zip(gradients, self.theta))
        return loss

    @tf.function
    def _log_likelihood_incomplete(self, obs):
        """Compute the log likelihood of a single incomplete observation.

        Parameters
        ----------
        obs : np.ndarray
            A single sample of observations. May contain NaNs.
        m : np.ndarray of bools
            The data mask.

        Returns
        -------
        np.ndarray
            The log-likelihood of the observation.
        """

        # Drop all-zero rows which were added for size padding
        abs_sum_per_row = tf.reduce_sum(tf.abs(obs), 1)
        mask = tf.not_equal(abs_sum_per_row, tf.zeros(shape=(1, 1), dtype=tf.float64))

        obs = tf.boolean_mask(obs, mask[0], axis=0)
        m = tf.cast(obs[0], dtype=bool)  # mask of observed features in this band
        p_observed = tf.shape(tf.where(m))[0]  # number of observed features

        Omega = tf.linalg.set_diag(
            self.Omega,
            tf.math.softplus(tf.linalg.diag_part(self.Omega)),
        )

        if self.n_components > 1:
            pi = tf.squeeze(tf.nn.softmax(self.pi))
        else:
            pi = tf.nn.softmax(self.pi)
        W_missing = tf.reshape(
            tf.squeeze(tf.gather(params=self.W, indices=tf.where(m), axis=0)),
            [p_observed, self.n_factors],
        )
        W_missing_Omega = tf.matmul(W_missing, Omega)

        # Equations 22, 23 in Baek+ 2010
        mean = self.mu[m] + tf.matmul(self.Xi, W_missing, transpose_b=True)

        # Same for noise_variance_softplus
        # Equation  6 not  11
        Sigma = tf.matmul(
            W_missing_Omega, W_missing_Omega, transpose_b=True
        ) + tf.linalg.diag(tf.math.softplus(self.Psi[m]))

        dis = tfp.distributions.MultivariateNormalTriL(
            loc=mean, scale_tril=tf.linalg.cholesky(Sigma)
        )

        gm = tfp.distributions.MixtureSameFamily(
            mixture_distribution=tfp.distributions.Categorical(probs=pi),
            components_distribution=dis,
        )

        masked_data = tf.boolean_mask(obs, m, axis=1)
        return gm.log_prob(tf.cast(masked_data, dtype=np.float32))

    @tf.function
    def _cluster_incomplete(self, obs, m):
        """Cluster (incomplete) data point. Low level, use cluster_data instead.

        Parameters
        ----------
        obs : np.ndarray
            A single sample of observations. May contain NaNs.
        m : np.ndarray of bools
            The data mask.
        d : int
            The number of dimensions of the latent space.
        model : tuple
            The current model parameters.

        Returns
        -------
        np.ndarray of floats
            The probability of belonging to each cluster. Sums to 1.
        """
        obs = tf.squeeze(obs)
        m = tf.squeeze(m)
        p_observed = tf.shape(tf.where(m))[0]  # number of observed features
        Omega = tf.linalg.set_diag(
            self.Omega,
            tf.math.softplus(tf.linalg.diag_part(self.Omega)),
        )
        pi = tf.squeeze(tf.nn.softmax(self.pi))
        W_missing = tf.reshape(
            tf.squeeze(tf.gather(params=self.W, indices=tf.where(m), axis=0)),
            [p_observed, self.n_factors],
        )
        W_missing_Omega = tf.matmul(W_missing, Omega)
        mean = self.mu[m] + tf.matmul(self.Xi, W_missing, transpose_b=True)
        Sigma = tf.matmul(
            W_missing_Omega, W_missing_Omega, transpose_b=True
        ) + tf.linalg.diag(tf.math.softplus(self.Psi[m]))

        dis = tfp.distributions.MultivariateNormalTriL(
            loc=mean, scale_tril=tf.linalg.cholesky(Sigma)
        )
        return tf.nn.softmax(
            dis.log_prob(tf.cast(obs[m], dtype=np.float32)) + tf.math.log(pi)
        )

    # ------
    # Helper functions
    def to_file(self, path):
        """Write the trained model instance to file.

        Parameters
        ----------
        path : str
            The path to the pickled MCFA parameters dictionary.
        """

        parameters = {}

        for param in ATTRS_TO_STORE:
            parameters[param] = getattr(self, param)

        with open(path, "wb") as file_:
            pickle.dump(parameters, file_)

    def plot_data_space(self, *args, **kwargs):
        """Plot the data and clusters in the original space."""
        mcfa.figures.plot_data_space(self, *args, **kwargs)

    def plot_latent_space(self, *args, **kwargs):
        """Plot the data and clusters in the latent space."""
        mcfa.figures.plot_latent_space(self, *args, **kwargs)

    def plot_latent_loadings(self):
        """Plot the latent loadings."""
        mcfa.figures.plot_latent_loadings(self)

    def plot_loss(self):
        """Plot the training and validation loss."""
        mcfa.figures.plot_loss(self)

    # ------
    # Functions for data and cluster parameter computation
    def _compute_cluster_moments(self):
        """Compute the cluster means and covariances in data and in latentspace.

        Returns
        -------
        np.ndarrays
            The cluster means in data space
        np.ndarrays
            The cluster covariances in data space
        np.ndarrays
            The cluster means in latent space
        np.ndarrays
            The cluster covariances in latent space
        """

        # Get the mean and stddev of the clusters
        mean_data = self.mu + tf.matmul(self.Xi, self.W, transpose_b=True).numpy()

        # Undo the Cholesky decomposition
        Omega = tf.linalg.set_diag(
            self.Omega,
            tf.math.softplus(tf.linalg.diag_part(self.Omega)),
        ).numpy()

        Omega = np.array([omega_cluster @ omega_cluster.T for omega_cluster in Omega])

        W_Omega = tf.matmul(self.W, Omega)
        cov_data = (
            tf.matmul(W_Omega, W_Omega, transpose_b=True)
            + tf.linalg.diag(tf.math.softplus(self.Psi))
        ).numpy()

        # Cluster moments in latent space are trained model parameters
        mean_latent = self.Xi.numpy()  # ensure_numpy_array(self.Xi)
        cov_latent = Omega

        return (
            mean_data,
            cov_data,
            mean_latent,
            cov_latent,
        )

    def impute(self, Y):
        """Impute the missing values given the clusters.

        Parameters
        ----------
        Y : np.ndarray
            The observed data to impute.

        Returns
        -------
        np.ndarray
            The imputed data.

        Notes
        -----
        The imputed data is available as Y_imp attribute.
        This follows Wang 2013, Equ. 10.
        """

        _, p = Y.shape
        Y_imp = Y.copy()
        tau = self.predict_proba(Y)

        _, Sigma, _, _ = self._compute_cluster_moments()

        for i in np.argwhere(np.isnan(Y).any(axis=1)):
            # Start with a single sample with missing values
            i = i[0]
            y = Y[i].copy()  # cp because we are assigning 5 below

            # Build auxiliary matrices O and M
            O = np.eye(p, p)[~np.isnan(y)]
            M = np.eye(p, p)[np.isnan(y)]

            # Build conditional cluster moments
            xi = self.Xi[np.argmax(tau[i])]
            sigma = Sigma[np.argmax(tau[i])]

            mu_i_o = O @ self.W.numpy() @ xi.numpy().T + self.mu.numpy()[~np.isnan(y)]
            mu_i_m = M @ self.W.numpy() @ xi.numpy().T + self.mu.numpy()[np.isnan(y)]

            sigma_i_oo = O @ sigma @ O.T
            sigma_i_om = O @ sigma @ M.T
            sigma_i_mo = sigma_i_om.T

            y[np.isnan(y)] = 5  # otherwise matrix multiplication fails
            yo = O @ y

            ym = mu_i_m + sigma_i_mo @ np.linalg.inv(sigma_i_oo) @ (yo - mu_i_o)
            y = O.T @ yo + M.T @ ym
            Y_imp[i] = y

        return Y_imp

    def number_of_parameters(self):
        """Compute the number of free model parameters. Required for
        the Bayesian Information criterion.

        Returns
        -------
        int
            The number of free parameters.

        Notes
        -----
        The computation follows Baek+ 2010, Equ. 13.
        The implementation is from the Casey+ 2019 mcfa implementation.
        """
        p = self.Y.shape[1]
        q, g = self.n_factors, self.n_components

        return int((g - 1) + p + q * (p + g) + (g * q * (q + 1)) / 2 - q**2)

    def bic(self, Y):
        """Compute the Bayesian Information Criterion of the model given the
        data.

        Returns
        -------
        float
            The BIC value of the model given the data. A larger value indicates
            a better model fit.
        """
        N, _ = Y.shape
        ll = []

        for batch in self.training_set:
            loss = self._log_likelihood_incomplete(batch)
            ll.append(tf.reduce_sum(loss))
        for batch in self.validation_set:
            loss = self._log_likelihood_incomplete(batch)
            ll.append(tf.reduce_sum(loss))

        log_likelihood = sum(ll)

        N, _ = np.atleast_2d(self.Y).shape
        return 2 * log_likelihood - np.log(N) * self.number_of_parameters()

    def icl(self, Y):
        """Compute the Integrated Completed Likelihood of the model given the data.

        Returns
        -------
        float
            The ICL value of the model given the data. A larger value indicates
            a better model fit.
        """
        entropy = 0
        tau = self.predict_proba(Y)

        for zi in tau:
            for g in range(self.n_components):

                zig = zi[g] if zi[g] != 0 else 1e-18
                entropy -= zig * np.log(zig)

        icl = self.bic(Y) - entropy
        return icl

    def orthonormalize(self):
        """Orthonormalize the latent loadings following Baek+ 2010, appendix."""

        CTC = self.W.numpy().T @ self.W.numpy()
        C = np.linalg.cholesky(CTC).T

        # Replace W by WC^-1
        self.W = self.W @ np.linalg.inv(C)

        # Update Xi and Omega
        self.Xi = (C @ self.Xi.numpy().T).T
        self.Omega = np.array([C @ omega_i @ C.T for omega_i in self.Omega.numpy()])


def from_file(path):
    """Read a trained model instance from file.

    Parameters
    ----------
    path : str
        The path to the pickled MCFA parameters dictionary.

    Returns
    -------
    mcfa.MCFA
        The MCFA model instance with the parameters instantiated from file.
    """

    with open(path, "rb") as file_:
        parameters = pickle.load(file_)

    # Hyperparameters
    model = MCFA(
        n_components=parameters["n_components"], n_factors=parameters["n_factors"]
    )

    # And learned model parameters
    for param in ATTRS_TO_STORE:
        setattr(model, param, parameters[param])

    return model


# Model attributes which are (de)serialized upon model loading / saving
ATTRS_TO_STORE = [
    "n_components",
    "n_factors",
    "p",
    "Xi",
    "pi",
    "W",
    "Omega",
    "Psi",
    "mu",
    "loss_training",
    "loss_validation",
]
