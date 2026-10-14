"""数据生成"""
from dataclasses import dataclass
import numpy as np


@dataclass
class DataConfig:
    n: int = 100000
    k_n: int = 500
    d: int = 10
    mu: list = None

    def __post_init__(self):
        if self.mu is None:
            self.mu = [0.8] + [0.0] * (self.d - 1)


def generate_gaussian_data(cfg: DataConfig, seed: int):
    rng = np.random.default_rng(seed)
    n = cfg.n
    k_n = cfg.k_n
    d = cfg.d
    mu = np.array(cfg.mu)

    n_maj = n - k_n
    X_maj = rng.normal(0, 1, size=(n_maj, d))
    y_maj = np.zeros(n_maj)

    X_min = rng.normal(mu, 1, size=(k_n, d))
    y_min = np.ones(k_n)

    X = np.vstack([X_maj, X_min])
    y = np.concatenate([y_maj, y_min])

    idx = rng.permutation(n)
    return X[idx], y[idx], mu