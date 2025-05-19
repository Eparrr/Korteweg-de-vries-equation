from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .schemas import SimulationRequest, SimulationResponse
from .solvers import create_solver, SOLVER_REGISTRY
from . import initial_conditions as ic_mod

if hasattr(ic_mod, "__all__"):
    IC_REGISTRY = {name: getattr(ic_mod, name) for name in ic_mod.__all__}
else:
    IC_REGISTRY = {
        name: obj
        for name, obj in vars(ic_mod).items()
        if callable(obj) and not name.startswith("_")
    }

app = FastAPI()


@app.get("/meta")
def meta():
    return JSONResponse(
        {
            "methods": list(SOLVER_REGISTRY.keys()),
            "initial_conditions": list(IC_REGISTRY.keys()),
        }
    )


@app.post("/simulate", response_model=SimulationResponse)
def simulate(req: SimulationRequest):
    if req.ic_name not in IC_REGISTRY:
        raise HTTPException(400)

    solver = create_solver(
        method=req.method,
        ic_func=IC_REGISTRY[req.ic_name],
        ic_params=req.ic_params,
        L=req.L,
        Nx=req.Nx,
        T=req.T,
        Nt=req.Nt,
    )
    sol = solver.solve()

    return SimulationResponse(
        x=sol.x.tolist(),
        t=sol.t.tolist(),
        u=[row.tolist() for row in sol.u],
    )
