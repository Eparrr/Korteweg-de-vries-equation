from typing import Any

from pydantic import BaseModel, Field


class SimulationRequest(BaseModel):
    method: str = "spectral_rk4"
    L: float = 50.0
    Nx: int = 256
    T: float = 4.0
    Nt: int = 8000
    ic_name: str = "soliton"
    ic_params: dict[str, Any] = Field(default_factory=dict)


class SimulationResponse(BaseModel):
    x: list[float]
    t: list[float]
    u: list[list[float]]
