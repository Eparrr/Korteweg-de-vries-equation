from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class KDVSolution:
    x: np.ndarray
    t: np.ndarray
    u: np.ndarray


class KDVSolver(ABC):
    def __init__(self, L: float, Nx: int, T: float, Nt: int, u0: np.ndarray):
        self.L = L
        self.Nx = Nx
        self.T = T
        self.Nt = Nt
        self.u0 = u0

    @abstractmethod
    def solve(self) -> KDVSolution:
        raise NotImplementedError()
