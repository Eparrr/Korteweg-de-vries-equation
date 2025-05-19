from typing import Callable, Type

import numpy as np

from .base import KDVSolver
from .spectral import SpectralRK4Solver
from .finite_difference import FiniteDifferenceSolver
from .split_step import SplitStepSolver

SOLVER_REGISTRY: dict[str, Type[KDVSolver]] = {
    "spectral_rk4": SpectralRK4Solver,
    "finite_difference": FiniteDifferenceSolver,
    "split_step": SplitStepSolver,
}


def create_solver(
    method: str, ic_func: Callable, ic_params: dict, **sim_params
) -> KDVSolver:
    if method not in SOLVER_REGISTRY:
        raise ValueError(f"Unknown method '{method}'")

    x = np.linspace(0, sim_params["L"], sim_params["Nx"], endpoint=False)
    u0 = ic_func(x, **ic_params)
    return SOLVER_REGISTRY[method](u0=u0, **sim_params)
