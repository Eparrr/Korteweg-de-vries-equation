from __future__ import annotations

import numpy as np

from numpy.typing import ArrayLike
from scipy.special import ellipj, ellipk

__all__ = [
    "soliton",
    "two_solitons",
    "gaussian",
    "random_gaussians",
    "step",
    "cnoidal_wave",
]


def soliton(
    x: ArrayLike, A: float = 2.0, *, x0: float | None = None, L: float | None = None
) -> np.ndarray:
    x = np.asarray(x)
    if x0 is None:
        x0 = (L or x[-1]) / 4
    c = np.sqrt(A / 2.0)
    return A * (1.0 / np.cosh(c * (x - x0))) ** 2


def two_solitons(
    x: ArrayLike,
    *,
    A1: float = 3.0,
    A2: float = 1.0,
    gap: float = 0.15,
    L: float | None = None,
) -> np.ndarray:
    x = np.asarray(x)
    L = L or x[-1]
    x_fast = 0.25 * L - 0.5 * gap * L
    x_slow = x_fast + gap * L

    return soliton(x, A=A1, x0=x_fast, L=L) + soliton(x, A=A2, x0=x_slow, L=L)


def gaussian(
    x: ArrayLike,
    A: float = 1.0,
    sigma: float = 1.0,
    *,
    x0: float | None = None,
    L: float | None = None,
) -> np.ndarray:
    x = np.asarray(x)
    if x0 is None:
        x0 = 0.5 * (L or x[-1])
    return A * np.exp(-((x - x0) ** 2) / (2 * sigma**2))


def random_gaussians(
    x: ArrayLike, n: int = 8, *, seed: int | None = None
) -> np.ndarray:
    x = np.asarray(x)
    rng = np.random.default_rng(seed)
    u = np.zeros_like(x)
    L = x[-1]
    for _ in range(n):
        amp = rng.uniform(0.3, 2.0)
        width = rng.uniform(0.3, 1.5)
        x0 = rng.uniform(0, L)
        u += amp * np.exp(-((x - x0) ** 2) / (2 * width**2))
    return u


def step(
    x: ArrayLike,
    uL: float = 2.0,
    uR: float = 0.0,
    *,
    x0: float | None = None,
    L: float | None = None,
) -> np.ndarray:
    x = np.asarray(x)
    if x0 is None:
        x0 = 0.5 * (L or x[-1])
    return np.where(x < x0, uL, uR)


def cnoidal_wave(
    x: ArrayLike, m: float = 0.9, *, L: float | None = None, A: float = 1.0
) -> np.ndarray:
    x = np.asarray(x)
    L = L or x[-1]
    K = ellipk(m)
    k = 2 * K / L
    sn, *_ = ellipj(k * x, m)
    return A * (2 * m * sn**2 - 1)
