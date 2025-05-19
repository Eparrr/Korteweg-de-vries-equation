from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable

import ipywidgets as w
import numpy as np
import plotly.graph_objects as go
import requests
from IPython.display import clear_output, display

API_URL = os.getenv("KDV_API", "http://127.0.0.1:8000/simulate")
META_URL = API_URL.rsplit("/", 1)[0] + "/meta"


def _fetch_solution(
    *,
    method: str,
    ic: str,
    L: float,
    Nx: int,
    T: float,
    Nt: int,
    ic_params: Dict[str, Any],
):
    payload = dict(
        method=method, L=L, Nx=Nx, T=T, Nt=Nt, ic_name=ic, ic_params=ic_params
    )
    r = requests.post(API_URL, json=payload, timeout=max(30, int(T * 10)))
    r.raise_for_status()
    d = r.json()
    return np.asarray(d["x"]), np.asarray(d["t"]), np.asarray(d["u"])


def _build_fig(
    x: np.ndarray, t: np.ndarray, u: np.ndarray, *, max_frames: int = 600, title: str
) -> go.Figure:
    frames = []
    idx = []

    if t.size > 0 and u.ndim > 1 and u.shape[0] == t.size:
        stride = max(1, len(t) // max_frames)
        idx = list(range(0, len(t), stride))
        frames = [
            go.Frame(
                name=str(i_frame), data=[go.Scatter(x=x, y=u[i_frame], mode="lines")]
            )
            for i_frame in idx
        ]
    elif t.size == 0 and u.size == 0 and x.size == 0:
        pass
    elif u.ndim == 1 and x.size == u.size:
        pass
    elif u.ndim > 1 and u.shape[0] > 0 and u.shape[1] == x.size:
        pass

    u_min_val = u.min() if u.size > 0 else 0
    u_max_val = u.max() if u.size > 0 else 1
    pad = 0.05 * (u_max_val - u_min_val or 1.0)

    layout = go.Layout(
        title=title,
        xaxis=dict(
            title="x",
            range=[x.min() if x.size > 0 else 0, x.max() if x.size > 0 else 1],
        ),
        yaxis=dict(title="u(x,t)", range=[u_min_val - pad, u_max_val + pad]),
        margin=dict(l=60, r=40, t=90, b=60),
    )

    if frames and idx:
        layout.updatemenus = [
            dict(
                type="buttons",
                direction="left",
                x=0.1,
                y=1.15,
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                fromcurrent=True,
                                frame=dict(duration=50, redraw=True),
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                mode="immediate",
                                frame=dict(duration=0),
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                ],
            )
        ]
        layout.sliders = [
            dict(
                x=0.1,
                len=0.9,
                currentvalue=dict(prefix="t = "),
                steps=[
                    dict(
                        label=f"{t[i_slider]:.2f}",
                        method="animate",
                        args=[
                            [str(i_slider)],
                            dict(
                                mode="immediate",
                                frame=dict(duration=0, redraw=True),
                                transition=dict(duration=0),
                            ),
                        ],
                    )
                    for i_slider in idx
                ],
            )
        ]

    if u.ndim > 1 and u.shape[0] > 0:
        initial_y = u[0]
    elif u.ndim == 1:
        initial_y = u
    else:
        initial_y = np.array([])

    if x.size == 0:
        initial_x = np.array([])
    else:
        initial_x = x

    return go.Figure(
        data=[go.Scatter(x=initial_x, y=initial_y, mode="lines")],
        layout=layout,
        frames=frames,
    )


def visualize(
    *,
    L: float = 50.0,
    Nx: int = 256,
    T: float = 4.0,
    Nt: int = 8000,
    methods: Iterable[str] | None = None,
    initial_conditions: Iterable[str] | None = None,
):

    meta_resp = requests.get(META_URL, timeout=5)
    meta_resp.raise_for_status()
    meta = meta_resp.json()
    default_methods = meta.get("methods", [])
    default_initial_conditions = meta.get("initial_conditions", [])

    methods_to_use = methods or default_methods
    initial_conditions_to_use = initial_conditions or default_initial_conditions

    default_method_val = methods_to_use[0] if methods_to_use else None
    default_ic_val = initial_conditions_to_use[0] if initial_conditions_to_use else None

    method_dd = w.Dropdown(
        options=list(methods_to_use), value=default_method_val, description="Method:"
    )
    ic_dd = w.Dropdown(
        options=list(initial_conditions_to_use), value=default_ic_val, description="IC:"
    )

    L_sl = w.FloatSlider(
        value=L,
        min=1.0,
        max=200.0,
        step=1.0,
        description="L:",
        continuous_update=False,
        readout_format=".1f",
    )
    Nx_sl = w.IntSlider(
        value=Nx, min=32, max=2048, step=16, description="Nx:", continuous_update=False
    )
    T_sl = w.FloatSlider(
        value=T,
        min=0.1,
        max=15.0,
        step=0.1,
        description="T:",
        continuous_update=False,
        readout_format=".1f",
    )
    Nt_sl = w.IntSlider(
        value=Nt,
        min=100,
        max=40000,
        step=100,
        description="Nt:",
        continuous_update=False,
    )

    ic_params_txt = w.Textarea(
        value="{}",
        description="IC params (JSON):",
        layout=w.Layout(width="auto", min_width="300px", height="60px"),
    )

    run_btn = w.Button(
        description="Run / Restart",
        button_style="primary",
        icon="play",
        tooltip="Fetch solution and redraw plot",
    )
    status_out = w.Output(layout=w.Layout(min_height="30px"))
    fig_out = w.Output()

    cache: Dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def _run(_=None):
        with status_out:
            clear_output(wait=True)
        ic_params_val = json.loads(ic_params_txt.value.strip() or "{}")
        if not isinstance(ic_params_val, dict):
            raise ValueError("ic_params must be a JSON object.")

        current_L = L_sl.value
        current_Nx = Nx_sl.value
        current_T = T_sl.value
        current_Nt = Nt_sl.value
        current_method = method_dd.value
        current_ic = ic_dd.value

        ic_params_str = json.dumps(ic_params_val, sort_keys=True)
        key_parts = (
            current_method,
            current_ic,
            current_L,
            current_Nx,
            current_T,
            current_Nt,
            ic_params_str,
        )
        cache_key = "|".join(map(str, key_parts))

        x, t, u = np.array([]), np.array([]), np.array([])

        if cache_key in cache:
            x, t, u = cache[cache_key]
            with status_out:
                clear_output(wait=True)
        else:
            with status_out:
                clear_output(wait=True)

            x, t, u = _fetch_solution(
                method=current_method,
                ic=current_ic,
                L=current_L,
                Nx=current_Nx,
                T=current_T,
                Nt=current_Nt,
                ic_params=ic_params_val,
            )
            cache[cache_key] = (x, t, u)
            with status_out:
                clear_output(wait=True)
                print(
                    f"Calculation complete. (x: {x.shape}, t: {t.shape}, u: {u.shape})"
                )

        fig_title = f"KdV – {current_method} – {current_ic}"
        if x.size == 0 and t.size == 0 and u.size == 0 and not (cache_key in cache):
            fig_title += " (Error or No Data)"

        fig = _build_fig(x, t, u, title=fig_title)
        with fig_out:
            clear_output(wait=True)
            if x.size > 0 or t.size > 0 or u.size > 0:
                display(fig)

    run_btn.on_click(_run)

    controls_selection = w.HBox(
        [method_dd, ic_dd], layout=w.Layout(justify_content="space-around")
    )

    grid_params_col1 = w.VBox([L_sl, T_sl])
    grid_params_col2 = w.VBox([Nx_sl, Nt_sl])
    controls_grid = w.HBox(
        [grid_params_col1, grid_params_col2],
        layout=w.Layout(justify_content="space-around"),
    )

    controls_ic = w.HBox(
        [ic_params_txt, run_btn], layout=w.Layout(align_items="flex-end")
    )  # Align button with bottom of textarea

    parameter_panel = w.VBox(
        [
            w.HTML("<b>Solver & Initial Condition:</b>"),
            controls_selection,
            w.HTML(
                "<hr style='margin: 5px 0;'><b>Simulation Parameters:</b>"
            ),  # Thinner margin for hr
            controls_grid,
            w.HTML("<hr style='margin: 5px 0;'><b>IC Specific Parameters:</b>"),
            controls_ic,
        ],
        layout=w.Layout(
            width="auto", padding="5px", border="1px solid #ccc", margin_bottom="10px"
        ),
    )

    ui = w.VBox([parameter_panel, status_out, fig_out])

    if method_dd.value and ic_dd.value:
        _run()  # Initial run
    else:
        with status_out:
            clear_output(wait=True)
        with fig_out:
            clear_output(wait=True)

    return ui
