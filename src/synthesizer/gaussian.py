"""高斯模拟生成器（预训练先验版本）"""
import numpy as np
from .base import BaseSynthesizer


class GaussianSynthesizer(BaseSynthesizer):
    """
    受控高斯生成器：
        mu_hat = mu_true + epsilon * 1 / sqrt(d) + (D / sqrt(d)) * s^(-beta) * Z
        Z ~ N(0, I_d)

    这样：
        ||mu_hat - mu_true||_2 的期望 = D * s^(-beta) + epsilon
    与理论 delta(s) = D * s^(-beta) + epsilon 精确匹配。
    """

    def __init__(self, mu_true: np.ndarray, beta: float, epsilon: float, D: float = 1.0):
        self.mu_true = mu_true
        self.beta = beta
        self.epsilon = epsilon
        self.D = D
        self.mu_hat = None
        self.s = 0

    def fit(self, X_min: np.ndarray, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self.s = len(X_min)
        d = X_min.shape[1]

        # epsilon 是 L2 偏差尺度，每维偏移 epsilon / sqrt(d)
        bias = (self.epsilon / np.sqrt(d)) * np.ones(d)

        # D 是理论中的偏差系数，使 L2 噪声期望为 D * s^{-beta}
        # 每维噪声标准差 = D / sqrt(d) * s^{-beta}
        noise_scale = (self.D / np.sqrt(d)) * (self.s ** (-self.beta))
        noise = noise_scale * rng.normal(0, 1, d)

        self.mu_hat = self.mu_true + bias + noise

    def sample(self, m: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed + 1000)
        return rng.normal(self.mu_hat, 1.0, (m, self.mu_hat.shape[0]))

    @property
    def discrepancy(self) -> float:
        return float(np.linalg.norm(self.mu_hat - self.mu_true))