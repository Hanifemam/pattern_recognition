import pandas as pd
import numpy as np


class MixtureOfGaussian:
    def __init__(self, component_size=2):
        self.data = pd.read_csv(
            "/Users/hanifemamgholizadeh/Desktop/patter_recognition/data/m_g.csv"
        )
        self.component_size = component_size

    def initalization(self):
        N, D = self.data.shape
        K = self.component_size

        # pi initialized and uniform distrubution
        pi = np.ones(self.component_size) / K

        # mu is selected as values of #K sampled from data
        indices = np.random.choice(N, D, replace=False)
        mu = self.data[indices]

        # cov is initialized as identity
        cov = np.array([np.eye(D) for _ in range(K)])

        return mu, cov, pi
