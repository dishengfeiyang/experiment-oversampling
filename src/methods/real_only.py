from .base import BaseMethod, AllocationResult


class RealOnly(BaseMethod):
    name = "real_only"

    def allocate(self, k_n: int, n: int, **kwargs) -> AllocationResult:
        return AllocationResult(s_n=0, m_n=0, use_real_minority=True)