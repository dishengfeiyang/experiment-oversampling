"""高斯混合模型生成器，用于真实数据实验"""
import numpy as np
from sklearn.mixture import GaussianMixture
from .base import BaseSynthesizer


class GMMSynthesizer(BaseSynthesizer):
    """
    在 s 个真实少数类样本上拟合 GMM，采样 m 个合成样本。
    成分数根据样本量自适应：s 很小时退化为 1 个成分。
    """

    def __init__(self, n_components=None, random_state=42):
        self.n_components = n_components
        self.random_state = random_state
        self.gmm = None
        self.fitted = False
        self._discrepancy = np.nan

    def _choose_components(self, s, d):
        if self.n_components is not None:
            return self.n_components
        # 样本少时少用成分，避免拟合失败
        if s < 2 * d:
            return 1
        return min(3, max(1, s // (2 * d)))

    def fit(self, X_min, seed=None):
        s, d = X_min.shape
        n_comp = self._choose_components(s, d)
        rng = seed if seed is not None else self.random_state
        self.gmm = GaussianMixture(
            n_components=n_comp,
            covariance_type="full",
            random_state=rng,
            reg_covar=1e-3,
        )
        self.gmm.fit(X_min)
        self.fitted = True

    def sample(self, m, seed=None):
        if not self.fitted:
            raise RuntimeError("Synthesizer not fitted")
        X_syn, _ = self.gmm.sample(m)
        return X_syn

    @property
    def discrepancy(self):
        return self._discrepancy

    def set_discrepancy(self, value):
        self._discrepancy = value