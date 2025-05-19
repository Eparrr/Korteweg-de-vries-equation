import numpy as np
from scipy.fft import fft, ifft, fftfreq

from .base import KDVSolver, KDVSolution
from .utils import _dealias_mask


class SpectralRK4Solver(KDVSolver):
    def __init__(self, *args, dealias: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.dealias = dealias

    def _rhs(self, u_hat, k, mask):
        if self.dealias:
            u_hat = u_hat * mask
        u = ifft(u_hat)
        nonlinear = 3j * k * fft(u**2)
        if self.dealias:
            nonlinear *= mask
        linear = 1j * k**3 * u_hat
        return linear - nonlinear

    def solve(self) -> KDVSolution:
        L, Nx, T, Nt = self.L, self.Nx, self.T, self.Nt
        x = np.linspace(0, L, Nx, endpoint=False)
        dx = x[1] - x[0]
        dt = T / Nt
        k = 2 * np.pi * fftfreq(Nx, d=dx)
        mask = _dealias_mask(Nx) if self.dealias else None

        u_hat = fft(self.u0)

        u_arr = np.empty((Nt + 1, Nx))
        t_arr = np.linspace(0, T, Nt + 1)
        u_arr[0] = self.u0

        for n in range(Nt):
            k1 = self._rhs(u_hat, k, mask)
            k2 = self._rhs(u_hat + 0.5 * dt * k1, k, mask)
            k3 = self._rhs(u_hat + 0.5 * dt * k2, k, mask)
            k4 = self._rhs(u_hat + dt * k3, k, mask)
            u_hat += dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            if not np.isfinite(u_hat).all():
                raise RuntimeError()
            u_arr[n + 1] = np.real(ifft(u_hat))
        return KDVSolution(x, t_arr, u_arr)
