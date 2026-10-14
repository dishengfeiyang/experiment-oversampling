"""生成器基类"""
from abc import ABC, abstractmethod
import numpy as np


class BaseSynthesizer(ABC):
    """所有生成器的抽象基类"""

    @abstractmethod
    def fit(self, X_min: np.ndarray, seed: int) -> None:
        pass

    @abstractmethod
    def sample(self, m: int, seed: int) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def discrepancy(self) -> float:
        pass