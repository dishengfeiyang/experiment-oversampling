from .base import BaseMethod, AllocationResult
from ..allocation.optimization import (
    find_theoretical_optimal,
    find_optimal_s_fixed_m,
)
from ..allocation.objective import compute_c_n


class TheoreticalOptimal(BaseMethod):
    name = "theoretical_optimal"

    def __init__(
        self,
        beta,
        epsilon,
        D,
        c_n=None,
        w0=0.5,
        w1=0.5,
        alpha=0.05,
        M=13,
        B=1.0,
        force_m_max=False,
    ):
        self.beta = beta
        self.epsilon = epsilon
        self.D = D
        self.c_n = c_n
        self.w0 = w0
        self.w1 = w1
        self.alpha = alpha
        self.M = M
        self.B = B
        self.force_m_max = force_m_max

    def allocate(self, k_n, n, m_max=None, c_n=None, **kwargs):
        if c_n is None:
            c_n = self.c_n
        if c_n is None:
            c_n = compute_c_n(alpha=self.alpha, M=self.M, B=self.B)

        if self.force_m_max and m_max is not None and m_max > 0:
            # 固定 m = m_max，只优化 s
            s_n, _ = find_optimal_s_fixed_m(
                k_n=k_n,
                n=n,
                m=m_max,
                beta=self.beta,
                epsilon=self.epsilon,
                D=self.D,
                c_n=c_n,
                w0=self.w0,
                w1=self.w1,
                alpha=self.alpha,
                M=self.M,
                B=self.B,
            )
            m_n = m_max
        else:
            s_n, m_n, _ = find_theoretical_optimal(
                k_n=k_n,
                n=n,
                beta=self.beta,
                epsilon=self.epsilon,
                D=self.D,
                c_n=c_n,
                w0=self.w0,
                w1=self.w1,
                m_max=m_max,
                alpha=self.alpha,
                M=self.M,
                B=self.B,
            )

        return AllocationResult(s_n=s_n, m_n=m_n, use_real_minority=True)