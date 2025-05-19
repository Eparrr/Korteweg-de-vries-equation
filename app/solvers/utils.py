import numpy as np


def _dealias_mask(N: int) -> np.ndarray:
    mask = np.zeros(N, dtype=bool)
    k = N // 3
    mask[:k] = mask[N - k :] = True
    return mask


def _fd_step(u: np.ndarray, dx: float, dt: float) -> np.ndarray:
    up1, um1 = np.roll(u, -1), np.roll(u, 1)
    up2, um2 = np.roll(u, -2), np.roll(u, 2)

    ux = (up1 - um1) / (2 * dx)
    uxxx = (-up2 + 2 * up1 - 2 * um1 + um2) / (2 * dx**3)

    return u + dt * (-6 * u * ux - uxxx)
