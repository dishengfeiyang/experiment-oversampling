"""方法基类"""
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class AllocationResult:
    s_n: int
    m_n: int
    use_real_minority: bool = True


class BaseMethod(ABC):
    name: str = "base"

    @abstractmethod
    def allocate(self, k_n: int, n: int, **kwargs) -> AllocationResult:
        pass