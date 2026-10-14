"""最优分配求解"""
import numpy as np
from .objective import compute_J, compute_c_n


def find_optimal_s_fixed_m(k_n, n, m, beta, epsilon, D, c_n=None,
                           w0=0.5, w1=0.5, s_grid=None,
                           alpha=0.05, M=13, B=1.0):
    if c_n is None:
        c_n = compute_c_n(alpha=alpha, M=M, B=B)
    if s_grid is None:
        s_grid = np.arange(0, k_n + 1, max(1, k_n // 200))
    best_s, best_J = 0, np.inf
    for s in s_grid:
        J = compute_J(s, m, k_n, n, beta, epsilon, D, c_n, w0, w1)
        if J < best_J:
            best_J = J
            best_s = s
    return int(best_s), float(best_J)


def find_optimal_m_fixed_s(s, k_n, n, beta, epsilon, D, c_n=None,
                           w0=0.5, w1=0.5, m_grid=None, m_max=None,
                           alpha=0.05, M=13, B=1.0):
    if c_n is None:
        c_n = compute_c_n(alpha=alpha, M=M, B=B)
    if s == 0:
        return 0, compute_J(0, 0, k_n, n, beta, epsilon, D, c_n, w0, w1)

    M_upper = n - 2 * k_n + s
    if m_max is not None:
        M_upper = min(M_upper, m_max)
    if M_upper < 0:
        return 0, np.inf
    if m_grid is None:
        m_grid = np.linspace(0, M_upper, 100).astype(int)

    best_m, best_J = 0, np.inf
    for m in m_grid:
        J = compute_J(s, m, k_n, n, beta, epsilon, D, c_n, w0, w1)
        if J < best_J:
            best_J = J
            best_m = m
    return int(best_m), float(best_J)


def find_theoretical_optimal(k_n, n, beta, epsilon, D, c_n=None,
                             w0=0.5, w1=0.5, s_grid=None,
                             m_grid_size=100, m_max=None,
                             alpha=0.05, M=13, B=1.0):
    if c_n is None:
        c_n = compute_c_n(alpha=alpha, M=M, B=B)
    if s_grid is None:
        s_grid = np.arange(0, k_n + 1, max(1, k_n // 100))

    best_s, best_m, best_J = 0, 0, np.inf

    # 先考虑 s=0, m=0
    J0 = compute_J(0, 0, k_n, n, beta, epsilon, D, c_n, w0, w1)
    if J0 < best_J:
        best_J = J0
        best_s, best_m = 0, 0

    for s in s_grid:
        if s == 0:
            continue
        M_upper = n - 2 * k_n + s
        if m_max is not None:
            M_upper = min(M_upper, m_max)
        if M_upper < 0:
            continue
        m_grid = np.linspace(0, M_upper, m_grid_size).astype(int)
        for m in m_grid:
            J = compute_J(s, m, k_n, n, beta, epsilon, D, c_n, w0, w1)
            if J < best_J:
                best_J = J
                best_s, best_m = int(s), int(m)

    return best_s, best_m, float(best_J)