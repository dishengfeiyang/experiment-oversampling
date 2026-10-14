from .base import BaseMethod, AllocationResult


class CompleteReplacement(BaseMethod):
    def __init__(self, s_ratio: float, name_suffix: str = ""):
        self.s_ratio = s_ratio
        self.name = (
            f"complete_replacement_{name_suffix}"
            if name_suffix
            else f"complete_replacement_{s_ratio}"
        )

    def allocate(self, k_n: int, n: int, m_max: int = None, **kwargs) -> AllocationResult:
        s_n = int(self.s_ratio * k_n)
        m_n = n - k_n
        if m_max is not None and m_n > m_max:
            m_n = m_max
        return AllocationResult(s_n=s_n, m_n=m_n, use_real_minority=False)