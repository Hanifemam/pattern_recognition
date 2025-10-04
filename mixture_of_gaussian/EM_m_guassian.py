import pandas as pd
import numpy as np


class MixtureOfGaussian:
    def __init__(self, component_size=2):
        self.data = pd.read_csv(
            "/Users/hanifemamgholizadeh/Desktop/patter_recognition/data/m_g.csv"
        )
        self.component_size = component_size
        self.mu, self.cov, self.pi = self.initalization()

    def initalization(self):
        N, D = self.data.shape
        K = self.component_size

        # pi initialized to uniform distribution
        pi = np.ones(K) / K

        # mu is selected as values of #K sampled from data
        indices = np.random.choice(N, K, replace=False)  # <-- K, not D
        mu = self.data.iloc[indices].values  # <-- ndarray (K, D)

        # cov is initialized as identity
        cov = np.array([np.eye(D) for _ in range(K)])

        return mu, cov, pi

    def E_step(self):
        N = self.data.shape[0]
        K = self.component_size
        hidden_posterior = np.zeros((N, K))
        mu, cov, pi = self.mu, self.cov, self.pi
        X = self.data.values  # <-- ndarray (N, D)
        for ind, x in enumerate(X):
            normalization = self.compute_normalization(x, mu, cov, pi)  # <-- pass all
            for k in range(K):
                hidden_posterior[ind, k] = (
                    self.compute_prior_likelihood(x, mu[k], cov[k], pi[k])
                    / normalization
                )
        return hidden_posterior

    def compute_prior_likelihood(self, x, mu, cov, pi):
        cov_det = np.linalg.det(cov)
        cov_inv = np.linalg.inv(cov)
        D = x.shape[0]
        # Gaussian density constant
        norm_const = 1.0 / ((2 * np.pi) ** (D / 2) * np.sqrt(cov_det))
        # Quadratic form with matrix multiplications
        quad = (x - mu).T @ cov_inv @ (x - mu)
        density = norm_const * np.exp(-0.5 * quad)  # <-- multiply, not divide
        return pi * density

    def compute_normalization(self, x, mu, cov, pi):
        normalization = 0.0
        D = x.shape[0]
        for comp_mu, comp_cov, comp_pi in zip(mu, cov, pi):
            cov_det = np.linalg.det(comp_cov)
            cov_inv = np.linalg.inv(comp_cov)
            norm_const = 1.0 / ((2 * np.pi) ** (D / 2) * np.sqrt(cov_det))
            quad = (x - comp_mu).T @ cov_inv @ (x - comp_mu)
            density = norm_const * np.exp(-0.5 * quad)
            normalization += comp_pi * density
        return normalization


print(MixtureOfGaussian().E_step())
