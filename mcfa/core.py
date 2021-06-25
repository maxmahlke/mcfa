#!/usr/bin/env python

"""
Implementation of a Mixture of Common Factor Analyzers model with
Stochastic Gradient Descent training.
"""

import numpy as np
import pandas as pd
import statsmodels
import scipy
from sklearn import decomposition, mixture, cluster
import tensorflow as tf
import os

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

tf.config.threading.set_inter_op_parallelism_threads(0)
tf.config.threading.set_intra_op_parallelism_threads(0)
tf.config.set_soft_device_placement(True)
if len(tf.config.list_physical_devices("GPU")):
    gpu = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(device=gpu[0], enable=True)

import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tqdm import tqdm
from rich import print as rprint

import mcfa.figures

# Eager execution is good for debugging tensorflow but slower
tf.config.run_functions_eagerly(False)


class MCFA:
    """Mixture of Common Factor Analyzers model implementation."""

    def __init__(self, n_components, n_factors):
        """Initialize the MCFA model instance.

        Parameters
        ==========
        n_components : int
            The number of mixture components
        n_factors : int
            The number of latent dimensions / factors
        """
        self.n_components = int(n_components)
        self.n_factors = int(n_factors)

        # Check for availability of GPU
        if len(tf.config.list_physical_devices("GPU")):
            rprint("\n[green]Running on GPU.[/green]")
        else:
            rprint("\n[red]Not running on GPU.[/red]")

    def fit(
        self,
        Y,
        n_epochs,
        frac_validation=0.1,
        learning_rate=3e-4,
        convergence=0.01,
        batch_size=32,
    ):
        """Fit the MCFA model instance to the passed data.

        Parameters
        ==========
        Y : np.ndarray
            The observed data to cluster. May contain missing values.
        n_epochs : int
            The number of training epochs
        frac_validation : float
            The fraction of the data used for validation during the training.
            Default is 0.1, i.e. 10%.
        learning_rate : float
            The learning rate passed to the gradient descent optimizer. Default is 3e-4.
        convergence : float
            Maximum relative change of loss function over five epochs before it is considered
            to have converged.
        batch_size : int
            Batch size of training subsets to use during SGD. Larger is faster but less fine.
        """
        self.Y = Y
        self.N, self.p = self.Y.shape
        self.n_epochs = int(n_epochs)
        self.frac_validation = frac_validation
        self.learning_rate = learning_rate
        self.convergence = convergence
        self.batch_size = batch_size

        # Initialising the clustering model
        self.initialize_model()

        # Train the model
        self.train_model()

    def transform(self, Y=None):
        """Cluster the passed data given the learned model parameters.

        Parameters
        ==========
        Y : np.ndarray
            The data to cluster. Must have the same features as the data used to train
            the model.

        Notes
        =====
        The assigned clusters and the responsibility matrix tau are accessible as model
        parameters (model.tau and model.clusters_assigned).
        """

        # If no data is passed, training data is clustered
        # TODO Introduce data_trained and data_clustered
        if Y is not None:
            self.Y = Y

        # Compute responsibility matrix
        self.tau = np.zeros((len(self.Y), self.n_components))

        for i, sample in enumerate(tqdm(Y, desc="Clustering")):
            self.tau[i, :] = self.__cluster_incomplete(sample, np.isfinite(sample))

        # Attribute for convenient access
        self.clusters_assigned = tf.argmax(self.tau, axis=1)

    def train_model(self):
        """Perform stochastic gradient descent to optimize the model parameters."""
        self.__optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        # Divide data into subsets of equal missingness pattern
        self.create_training_subsets()

        # Record the loss of the training and the validation set during training
        self.training_loss = []
        self.validation_loss = []

        # And train
        progress = tqdm(range(self.n_epochs), desc="Training")
        for epoch in progress:

            training_batch_loss = []
            validation_batch_loss = []

            # Each subset with unique missingness mask is passed separately
            for subset in self.training_subsets:

                # Each subset contains the data points and the corresponding missingness mask
                # in batches
                for d, m in subset:

                    training_batch_loss.append(tf.reduce_mean(self.__train_step(d, m)))

                # Now compute the loss over the validation set
                # for subset in self.validation_subsets:

                # Each subset contains the data points and the corresponding missingness mask
                for d, m in subset:
                    validation_batch_loss.append(
                        tf.reduce_mean(
                            -self.log_likelihood_incomplete(
                                d, m
                            )  # the loss is the average negative log likelihood
                        )
                    )

            # Record the loss over this training epoch
            self.training_loss.append(tf.reduce_mean(training_batch_loss))
            self.validation_loss.append(tf.reduce_mean(validation_batch_loss))

            # Check convergence
            if epoch > 10:
                if np.mean(self.validation_loss[-10:-5]) <= np.mean(
                    self.validation_loss[-5:]
                ):
                    progress.set_description("Converged")
                    break

    @tf.function
    def __train_step(self, d, m):
        """Execute a gradient-descent step.

        Parameters
        ==========
        d : np.ndarray
            The observation[s] to train the model on.
        m : np.ndarray
            Array containing the missingness pattern of d. Must be
            the same for each row of d.
        """
        with tf.GradientTape() as tape:
            # the gradient tape saves all the step that needs
            # to be saved for automatic differentiation

            # try iterating over points in batch here

            loss = -self.log_likelihood_incomplete(
                d, m
            )  # the loss is the negative log likelihood

        # Adam iteration
        gradients = tape.gradient(loss, self.theta)
        self.__optimizer.apply_gradients(zip(gradients, self.theta))
        return loss

    def create_training_subsets(self):
        """Creates subsets with equal missingness patterns for training and validation."""

        # Split into training and validation data
        split = np.random.rand(self.N) < self.frac_validation

        self.data_training = self.Y[~split]
        self.data_validation = self.Y[split]

        # Now we create the subsets of equal missingness patterns for each dataset
        for i, data in enumerate([self.data_training, self.data_validation]):

            # The datasets for training cannot contain NaNs, so we fill them with 0
            # These will be filtered out later with the missingness masks
            data_0 = data.copy()
            data_0[np.isnan(data)] = 0

            # Missingness patterns are differentiated by hash functions
            missing = pd.DataFrame(data=np.isfinite(data))

            missing["hash"] = missing.apply(
                lambda row: hash("".join([str(col) for col in row])), axis=1
            )

            # We're sorting the datasets by most to least complete for each epoch.
            # If it instead should be random, then construct a set of the hashes when
            # iterating over the epochs.

            # The sorting is done by counting the number of True per row in each mask.
            # missingness_patterns then contains the sorted hash values for the patterns
            missing = missing.set_index("hash")
            sorted_hashes = pd.unique(
                missing[missing].count(axis=1).sort_values(ascending=False).index
            )
            missing = missing.reset_index()

            # Build list of data subsets with equal missingness pattern
            subsets = []

            for hash_ in sorted_hashes:

                # Construct a list of missingness patterns in the data
                mask = missing.loc[missing.hash == hash_, :].drop(columns=["hash"])

                if len(mask) == 1:
                    mask = np.array(mask)

                # Create subset with the same missingness pattern
                subset = data_0[missing.loc[missing.hash == hash_].index, :]

                # Convert to training data
                train_data = (
                    tf.data.Dataset.from_tensor_slices((subset, mask))
                    .shuffle(
                        len(subset), reshuffle_each_iteration=True if i == 0 else False
                    )  # validation set does not need to be shuffled
                    .batch(
                        self.batch_size if i == 0 else len(data),
                        num_parallel_calls=tf.data.AUTOTUNE,
                    )  # validation set can be one batch
                    .prefetch(tf.data.AUTOTUNE)
                )

                subsets.append(train_data)

            if i == 0:
                self.training_subsets = subsets
            else:
                self.validation_subsets = subsets

    def initialize_model(self):
        """Initialize the latent space and cluster properties."""

        # mu is initialized on the observed data per feature
        self.mu = np.nanmean(self.Y, axis=0).reshape([self.p])

        # Initialize the latent space
        self.initialize_by_pca()

        # Initialize the cluster components
        self.initialize_by_gmm()

        # Define the model variables using the init values
        self.pi = tf.Variable(self.pi, dtype=tf.float32, name="pi")
        self.mu = tf.Variable(self.mu, dtype=tf.float32, name="mu")
        self.Xi = tf.Variable(self.Xi, dtype=tf.float32, name="Xi")
        self.W = tf.Variable(self.W, dtype=tf.float32, name="W")
        self.Omega = tf.Variable(self.Omega, dtype=tf.float32, name="Omega")
        self.Psi = tf.Variable(self.Psi, dtype=tf.float32, name="Psi")

        # Create list of trainable parameters for gradient descent
        self.theta = [self.Xi, self.pi, self.W, self.Omega, self.Psi, self.mu]

    def initialize_by_pca(self):
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

    def initialize_by_gmm(self):
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

    @tf.function
    def log_likelihood_incomplete(self, obs, m):
        """Compute the log likelihood of a single incomplete observation.

        Parameters
        ==========
        obs : np.ndarray
            A single sample of observations. May contain NaNs.
        m : np.ndarray of bools
            The data mask.

        Returns
        =======
        np.ndarray
            The log-likelihood of the observation.
        """

        obs = tf.squeeze(obs)
        # m is mask of missing values. We only need the first entry
        m = tf.squeeze(m[0])
        # because model parameters are(d, 1) and m is (d, n).
        # If we assume the same missingness pattern,
        # we can only pass m[0]
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

        # Equations 22, 23 in Baek+ 2010
        mean = self.mu[m] + tf.matmul(self.Xi, W_missing, transpose_b=True)

        # Same for noise_variance_softplus
        # Equation  6 not  11
        Sigma = tf.matmul(
            W_missing_Omega, W_missing_Omega, transpose_b=True
        ) + tf.linalg.diag(tf.math.softplus(self.Psi[m]))

        dis = tfd.MultivariateNormalTriL(loc=mean, scale_tril=tf.linalg.cholesky(Sigma))

        gm = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=pi), components_distribution=dis
        )
        try:
            masked_data = tf.boolean_mask(obs, m, axis=1)
        except ValueError:
            masked_data = tf.boolean_mask(obs, m, axis=None)
        return gm.log_prob(tf.cast(masked_data, dtype=np.float32))

    @tf.function
    def __cluster_incomplete(self, obs, m):
        """Cluster (incomplete) data point. Low level, use cluster_data instead.

        Paramaeters
        ===========
        obs : np.ndarray
            A single sample of observations. May contain NaNs.
        m : np.ndarray of bools
            The data mask.
        d : int
            The number of dimensions of the latent space.
        model : tuple
            The current model parameters.

        Returns
        =======
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

        dis = tfd.MultivariateNormalTriL(loc=mean, scale_tril=tf.linalg.cholesky(Sigma))
        # dis = tfd.MultivariateNormalFullCovariance(loc=mean, covariance_matrix=Sigma)

        return tf.nn.softmax(
            dis.log_prob(tf.cast(obs[m], dtype=np.float32)) + tf.math.log(pi)
        )

    # ------
    # Helper functions for plotting
    def plot_data_space(self):
        """Plot the data and clusters in the original space."""
        mcfa.figures.plot_data_space(self)

    def plot_latent_space(self):
        """Plot the data and clusters in the latent space."""
        mcfa.figures.plot_latent_space(self)

    def plot_latent_loadings(self):
        """Plot the latent loadings."""
        mcfa.figures.plot_latent_loadings(self)

    # ------
    # Functions for data and cluster parameter computation
    def compute_cluster_moments(self):
        """Compute the cluster properties in data and in latentspace.

        Returns
        =======
        Return the data and latent space means and covariances.
        """

        mu = self.mu
        Xi = self.Xi
        W = self.W
        Omega = self.Omega
        Psi = self.Psi

        # Get the mean and stddev of the clusters
        mean_data = mu + tf.matmul(Xi, W, transpose_b=True).numpy()

        Omega = tf.linalg.set_diag(
            Omega,
            tf.math.softplus(tf.linalg.diag_part(Omega)),
        ).numpy()

        # undo chol decomposition
        Omega = np.array([omega_cluster @ omega_cluster.T for omega_cluster in Omega])

        W_Omega = tf.matmul(W, Omega)
        cov_data = (
            tf.matmul(W_Omega, W_Omega, transpose_b=True)
            + tf.linalg.diag(tf.math.softplus(Psi))
        ).numpy()

        # Cluster moments in latent space are trained model parameters
        mean_latent = Xi.numpy()
        cov_latent = Omega

        return (
            mean_data,
            cov_data,
            mean_latent,
            cov_latent,
        )

    def compute_factor_scores(self, imputed=False):
        """Compute the factor scores of the observations.

        Computation based on Section 4 of Baek+ 2011.

        Parameters
        ==========
        imputed : bool
        """
        mu = self.mu
        Xi = self.Xi
        W = self.W
        Omega = self.Omega
        Psi = self.Psi
        tau = self.tau

        W = W.numpy()

        Z = np.zeros((self.n_components, self.N, self.n_factors))

        gamma = np.zeros((self.n_components, self.p, self.n_factors))
        inv_D = np.diag(1.0 / tf.math.softplus(Psi))

        I = np.eye(self.n_factors)

        # The cholesky decomposition and softplus application are undone in the
        # cluster moments function
        _, _, xi, Omega = self.compute_cluster_moments()

        Y = self.Y if not imputed else self.Y_imp

        Y = Y - mu

        for k in range(self.n_components):

            # check if covariance matrix is ill-coniditioned
            if np.linalg.cond(Omega[k, :, :]) > 1e6:
                print(
                    f"Cluster {k} has few members; the covariance matrix is ill-conditioned."
                )
                epsilon = 1e-5
                Omega[k] = Omega[k] + I * epsilon

            # using the Woodbury Identity for matrix inversion
            C = scipy.linalg.solve(
                scipy.linalg.solve(Omega[k, :, :], I) + W.T @ inv_D @ W, I
            )
            gamma[k, :, :] = (inv_D - inv_D @ W @ C @ W.T @ inv_D) @ W @ Omega[k, :, :]
            Z[k, :, :] = (
                np.repeat(xi[[k], :], self.N).reshape((self.n_factors, self.N)).T
                + (Y - (W @ xi[[k], :].T).T) @ gamma[k, :, :]
            )

        cluster = np.argmax(tau, axis=1)

        Zclust = np.zeros((self.N, self.n_factors))
        Zmean = np.zeros((self.N, self.n_factors))

        # Compute (i) Uclust and (ii) Umean: Scores if the ith datapoint belongs to
        # (i) a specific cluster or (ii) all clusters according to tau
        for i in range(self.N):
            Zclust[i] = Z[cluster[i], i, :]
            Zmean[i] = Z[:, i].T @ tau[i]

        return Z, Zmean, Zclust

    def impute(self):
        """Impute missing values following Wang 2013, Equ. 10"""
        self.Y_imp = self.Y.copy()

        for i in np.argwhere(np.isnan(self.Y).any(axis=1)):
            # Start with a single sample with missing values
            i = i[0]
            y = self.Y[i].copy()  # cp because we are assigning 5 below

            # Build auxiliary matrices O and M
            O = np.eye(self.p, self.p)[~np.isnan(y)]
            M = np.eye(self.p, self.p)[np.isnan(y)]

            # Build conditional cluster moments

            _, Sigma, _, _ = self.compute_cluster_moments()

            xi = self.Xi.numpy()[np.argmax(self.tau[i])]
            sigma = Sigma[np.argmax(self.tau[i])]

            mu_i_o = O @ self.W.numpy() @ xi + self.mu.numpy()[~np.isnan(y)]
            mu_i_m = M @ self.W.numpy() @ xi + self.mu.numpy()[np.isnan(y)]

            sigma_i_oo = O @ sigma @ O.T
            sigma_i_om = O @ sigma @ M.T
            sigma_i_mo = sigma_i_om.T

            y[np.isnan(y)] = 5  # otherwise matrix multiplication fails
            yo = O @ y

            ym = mu_i_m + sigma_i_mo @ np.linalg.inv(sigma_i_oo) @ (yo - mu_i_o)
            y = O.T @ yo + M.T @ ym
            self.Y_imp[i] = y
