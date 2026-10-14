"""目标函数 J_n(s, m)"""
import numpy as np


def compute_c_n(alpha: float = 0.05, M: int = 13, B: float = 1.0) -> float:
    """
    对应风险界中的：
        2B * sqrt( log(2M/alpha) / 2 )
    """
    return 2.0 * B * np.sqrt(np.log(2.0 * M / alpha) / 2.0)


def compute_J(s: int, m: int, k_n: int, n: int,
              beta: float, epsilon: float, D: float,
              c_n: float = None, w0: float = 0.5, w1: float = 0.5,
              alpha: float = 0.05, M: int = 13, B: float = 1.0) -> float:
    """
    J_n(s, m) = w1 * (m/N) * Delta(s) + c_n * sqrt(w0^2/(n-k) + w1^2/N)
    N = k_n - s + m
    特殊情形：
      s = 0：不使用生成器，必须 m = 0，N = k_n
    """
    if c_n is None:
        c_n = compute_c_n(alpha=alpha, M=M, B=B)

    if s == 0:
        if m != 0:
            return np.inf
        N = k_n
        sampling = c_n * np.sqrt((w0 ** 2) / (n - k_n) + (w1 ** 2) / N)
        return sampling

    if s < 0 or s > k_n:
        return np.inf

    N = k_n - s + m
    if N <= 0:
        return np.inf

    delta = D * (s ** (-beta)) + epsilon
    bias = w1 * (m / N) * delta
    sampling = c_n * np.sqrt((w0 ** 2) / (n - k_n) + (w1 ** 2) / N)
    return bias + sampling