"""快速诊断 J(s, m) 是否存在内部最优"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.allocation.objective import compute_J, compute_c_n


def scan(n, k_n, beta, epsilon, D, m_max_factor, m_max_mode="k_n",
         alpha=0.05, M=13, B=1.0, w0=0.5, w1=0.5):
    c_n = compute_c_n(alpha=alpha, M=M, B=B)
    if m_max_mode == "n":
        m_max = int(m_max_factor * n)
    else:
        m_max = int(m_max_factor * k_n)

    best = (np.inf, 0, 0)
    J0 = compute_J(0, 0, k_n, n, beta, epsilon, D, c_n, w0, w1)
    best = (J0, 0, 0)

    m_grid = np.unique(np.concatenate([
        [0], np.arange(1, m_max + 1, max(1, m_max // 50))
    ]))

    for s in range(1, k_n + 1):
        for m in m_grid:
            J = compute_J(s, m, k_n, n, beta, epsilon, D, c_n, w0, w1)
            if J < best[0]:
                best = (J, s, m)

    return c_n, m_max, best


def main():
    # ===== 在这里改参数 =====
    n = 100000
    k_n = 100
    D = 3.0
    m_max_mode = "k_n"       # 或 "n"
    m_max_factors = [0.4]    # 支持列表

    beta_grid = [0.8, 1.0, 1.2]
    epsilon_grid = [0.0, 0.05, 0.1]
    # ========================

    print("=" * 100)
    print(f"n={n}, k_n={k_n}, D={D}, m_max_mode={m_max_mode}")
    print("=" * 100)

    header = (
        f"{'mfac':>6} {'beta':>5} {'eps':>6} {'c_n':>7} {'m_max':>7} "
        f"{'J*':>9} {'s*':>4} {'m*':>7} {'s*/k_n':>8} {'J(0,0)':>9}"
    )
    print(header)
    print("-" * len(header))

    for mf in m_max_factors:
        for beta in beta_grid:
            for eps in epsilon_grid:
                c_n, m_max, best = scan(
                    n, k_n, beta, eps, D, mf, m_max_mode
                )
                J_star, s_star, m_star = best
                J0 = compute_J(0, 0, k_n, n, beta, eps, D, c_n)
                ratio = s_star / k_n
                flag = " <-- 内部最优" if 0 < s_star < k_n else ""
                print(
                    f"{mf:>6.2f} {beta:>5.2f} {eps:>6.2f} {c_n:>7.4f} "
                    f"{m_max:>7d} {J_star:>9.4f} {s_star:>4d} {m_star:>7d} "
                    f"{ratio:>8.2f} {J0:>9.4f}{flag}"
                )
    print()


if __name__ == "__main__":
    main()