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
        indices = np.random.choice(N, K, replace=False)
        mu = self.data.iloc[indices].values  # <-- ndarray (K, D)

        # cov is initialized as identity
        cov = np.array([np.eye(D) for _ in range(K)])

        return mu, cov, pi

    def E_step(self):
        N = self.data.shape[0]
        K = self.component_size
        hidden_posterior = np.zeros((N, K))
        mu, cov, pi = self.mu, self.cov, self.pi
        X = self.data.values
        for ind, x in enumerate(X):
            normalization = self.compute_normalization(x, mu, cov, pi)
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

    def M_step(self):
        hidden_posterior = self.E_step()
        N_K = hidden_posterior.sum(axis=0)  # (K, )
        X = self.data.values
        self.new_mu(X, hidden_posterior, N_K)
        self.new_cov(X, hidden_posterior, N_K)
        self.new_pi(N_K)
        return self.mu, self.cov, self.pi

    def new_mu(self, X, hidden_posterior, N_K):
        K = self.component_size
        self.mu = (hidden_posterior.T @ X) / N_K[:, None]

    def new_cov(self, X, hidden_posterior, N_K):
        K = self.component_size
        N, D = X.shape
        mu = self.mu  # (K, D)
        cov_list = []
        for k in range(K):
            Xc = X - mu[k]  # (N, D)
            W = hidden_posterior[:, k][:, None]  # (N, 1)
            Sk = (Xc * W).T @ Xc / N_K[k]  # (D, D)
            Sk.flat[:: D + 1] += 1e-6
            cov_list.append(Sk)
        self.cov = np.stack(cov_list, axis=0)  # (K, D, D)

    def new_pi(self, N_K):
        N = self.data.shape[0]
        self.pi = N_K * (1 / N)


m_of_g = MixtureOfGaussian()
print(m_of_g.E_step())
print(m_of_g.M_step())
