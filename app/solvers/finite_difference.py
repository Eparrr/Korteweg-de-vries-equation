import numpy as np

from .base import KDVSolver, KDVSolution
from .utils import _fd_step


class FiniteDifferenceSolver(KDVSolver):
    def solve(self) -> KDVSolution:
        L, Nx, T, Nt = self.L, self.Nx, self.T, self.Nt
        x = np.linspace(0, L, Nx, endpoint=False)
        dx = x[1] - x[0]
        dt = T / Nt

        u_old = self.u0.copy()
        u_new = _fd_step(u_old, dx, dt)

        u_arr = np.empty((Nt + 1, Nx))
        u_arr[0], u_arr[1] = u_old, u_new
        t_arr = np.linspace(0, T, Nt + 1)

        for n in range(1, Nt):
            next_u = _fd_step(u_new, dx, dt)
            u_old, u_new = u_new, next_u
            u_arr[n + 1] = u_new

        return KDVSolution(x=x, t=t_arr, u=u_arr)
