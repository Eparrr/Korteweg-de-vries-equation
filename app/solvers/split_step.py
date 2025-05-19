import numpy as np
from scipy.fft import fft, ifft, fftfreq

from .base import KDVSolver, KDVSolution
from .utils import _dealias_mask


class SplitStepSolver(KDVSolver):
    def __init__(self, *args, dealias: bool = True, **kw):
        super().__init__(*args, **kw)
        self.dealias = dealias

    def solve(self) -> KDVSolution:
        L, Nx, T, Nt = self.L, self.Nx, self.T, self.Nt
        x = np.linspace(0.0, L, Nx, endpoint=False)
        dx = x[1] - x[0]
        dt = T / Nt

        k = 2.0 * np.pi * fftfreq(Nx, d=dx)
        mask = _dealias_mask(Nx) if self.dealias else None

        u_hat = fft(self.u0)
        half_lin = np.exp(1j * k**3 * dt / 2.0)

        u_arr = np.empty((Nt + 1, Nx))
        t_arr = np.linspace(0.0, T, Nt + 1)
        u_arr[0] = np.real(ifft(u_hat))

        for n in range(Nt):
            u_hat *= half_lin
            if mask is not None:
                u_hat *= mask

            u = ifft(u_hat)
            u_sq_hat = fft(u**2)
            if mask is not None:
                u_sq_hat *= mask
            u_hat -= dt * 3j * k * u_sq_hat

            u_hat *= half_lin
            if mask is not None:
                u_hat *= mask

            if not np.isfinite(u_hat).all():
                raise RuntimeError()

            u_arr[n + 1] = np.real(ifft(u_hat))

        return KDVSolution(x, t_arr, u_arr)
