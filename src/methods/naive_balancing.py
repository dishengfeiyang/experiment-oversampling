from .base import BaseMethod, AllocationResult
from ..allocation.objective import compute_J, compute_c_n


class NaiveBalancing(BaseMethod):
    def __init__(self, tau: float):
        self.tau = tau
        self.name = f"naive_balancing_{tau}"

    def allocate(self, k_n: int, n: int, m_max: int = None, **kwargs) -> AllocationResult:
        s_n = int(self.tau * k_n)
        m_n = n - 2 * k_n + s_n
        if m_n < 0:
            m_n = 0
        if m_max is not None and m_n > m_max:
            m_n = m_max
        return AllocationResult(s_n=s_n, m_n=m_n, use_real_minority=True)


class NaiveBalancingOptimalS(BaseMethod):
    name = "naive_balancing_optimal_s"

    def __init__(self, beta, epsilon, D, c_n=None, w0=0.5, w1=0.5,
                 alpha=0.05, M=13, B=1.0):
        self.beta = beta
        self.epsilon = epsilon
        self.D = D
        self.c_n = c_n
        self.w0 = w0
        self.w1 = w1
        self.alpha = alpha
        self.M = M
        self.B = B

    def allocate(self, k_n: int, n: int, m_max: int = None,
                 c_n: float = None, **kwargs) -> AllocationResult:
        # 优先使用调用时传入的 c_n，其次使用初始化时的 c_n，
        # 最后回退到理论默认值 2B * sqrt(log(2M/alpha)/2)
        if c_n is None:
            c_n = self.c_n
        if c_n is None:
            c_n = compute_c_n(alpha=self.alpha, M=self.M, B=self.B)

        # 在 s 网格上搜索，对每个 s 用平衡所需的 m，再算 J，选最小。
        best_s, best_m, best_J = 0, 0, float('inf')
        step = max(1, k_n // 100)
        for s in range(0, k_n + 1, step):
            m = n - 2 * k_n + s
            if m < 0:
                m = 0
            if m_max is not None and m > m_max:
                m = m_max
            J = compute_J(s, m, k_n, n, self.beta, self.epsilon, self.D,
                          c_n, self.w0, self.w1)
            if J < best_J:
                best_J = J
                best_s, best_m = s, m
        return AllocationResult(s_n=best_s, m_n=best_m, use_real_minority=True)