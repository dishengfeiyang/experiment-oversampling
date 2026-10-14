from .base import BaseMethod, AllocationResult
from ..allocation.optimization import find_optimal_m_fixed_s
from ..allocation.objective import compute_c_n


class FixedSOptimalM(BaseMethod):
    def __init__(self, s_ratio, beta, epsilon, D, c_n=None,
                 w0=0.5, w1=0.5, alpha=0.05, M=13, B=1.0,
                 force_m_max=False):
        self.s_ratio = s_ratio
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
        self.name = f"fixed_s_{s_ratio}_optimal_m"

    def allocate(self, k_n, n, m_max=None, c_n=None, **kwargs):
        if c_n is None:
            c_n = self.c_n
        if c_n is None:
            c_n = compute_c_n(alpha=self.alpha, M=self.M, B=self.B)

        s_n = int(self.s_ratio * k_n)

        if self.force_m_max and m_max is not None and m_max > 0:
            # 固定 m = m_max，只改变 s
            m_n = m_max
        elif s_n == 0:
            m_n = 0
        else:
            m_n, _ = find_optimal_m_fixed_s(
                s=s_n, k_n=k_n, n=n,
                beta=self.beta, epsilon=self.epsilon, D=self.D,
                c_n=c_n, w0=self.w0, w1=self.w1,
                m_max=m_max,
            )
        return AllocationResult(s_n=s_n, m_n=m_n, use_real_minority=True)