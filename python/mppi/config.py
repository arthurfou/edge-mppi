"""Loader for config/mppi.yaml.

Same file is parsed by cuda/include/mppi/config.hpp. Any field added here must
be added there, otherwise the Python/CUDA parity check compares two different
problems.
"""
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

STATE_DIM = {"kinematic": 4, "dynamic": 6}
TIRE_MODELS = ("linear", "tanh", "pacejka")
INTEGRATORS = ("euler", "rk4")


@dataclass(frozen=True)
class Vehicle:
    wheelbase: float
    lf: float
    lr: float
    width: float
    mu: float
    mass: float      # kg
    izz: float       # kg·m², inertie de lacet
    cornering_stiffness_front: float   # C_S, 1/rad, normalisée par la charge (partie C)
    cornering_stiffness_rear: float
    tire_model: str                    # linear | tanh | pacejka
    pacejka_c: float
    pacejka_e: float
    blend_speed_low: float             # m/s, cinématique pur en dessous
    blend_speed_high: float            # m/s, dynamique pur au-dessus
    integrator: str                    # euler | rk4, pour la partie dynamique
    substeps: int                      # sous-pas par intervalle de commande


@dataclass(frozen=True)
class MppiParams:
    num_samples: int
    horizon: int
    dt: float
    lam: float
    gamma: float
    noise_std: np.ndarray   # (control_dim,)
    seed: int


@dataclass(frozen=True)
class Bounds:
    u_min: np.ndarray   # [a_min, delta_min]
    u_max: np.ndarray   # [a_max, delta_max]


@dataclass(frozen=True)
class CostParams:
    w_lateral: float
    w_progress: float
    w_offtrack: float
    w_control: float
    w_adhesion: float
    w_speed: float
    lateral_scale: float
    speed_scale: float
    v_ref: float
    track_half_width: float


@dataclass(frozen=True)
class TrackParams:
    grid_resolution: float
    resample_step: float
    margin: float
    npz_path: Path
    bin_path: Path


@dataclass(frozen=True)
class Config:
    model: str
    state_dim: int
    control_dim: int
    mppi: MppiParams
    bounds: Bounds
    vehicle: Vehicle
    cost: CostParams
    track: TrackParams
    max_steps: int


def _readonly(values):
    """float64 array that cannot be modified in place."""
    arr = np.array(values, dtype=np.float64)
    arr.setflags(write=False)
    return arr


def load_config(path) -> Config:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    root = path.resolve().parent.parent   # config/mppi.yaml -> repo root

    model = raw["model"]
    if model not in STATE_DIM:
        raise ValueError(f"model={model!r} unknown, expected one of {list(STATE_DIM)}")
    if raw["state_dim"] != STATE_DIM[model]:
        raise ValueError(
            f"state_dim={raw['state_dim']} but model {model!r} needs {STATE_DIM[model]}"
        )

    m = raw["mppi"]
    noise_std = _readonly(m["noise_std"])
    if noise_std.shape != (raw["control_dim"],):
        raise ValueError(
            f"noise_std has {noise_std.size} entries, control_dim={raw['control_dim']}"
        )

    v = raw["vehicle"]
    if not math.isclose(v["lf"] + v["lr"], v["wheelbase"]):
        raise ValueError(
            f"lf + lr = {v['lf'] + v['lr']} but wheelbase = {v['wheelbase']}"
        )
    if v["tire_model"] not in TIRE_MODELS:
        raise ValueError(f"tire_model={v['tire_model']!r}, expected one of {TIRE_MODELS}")
    if not 0.0 < v["blend_speed_low"] < v["blend_speed_high"]:
        raise ValueError("need 0 < blend_speed_low < blend_speed_high")
    if v["integrator"] not in INTEGRATORS:
        raise ValueError(f"integrator={v['integrator']!r}, expected one of {INTEGRATORS}")
    if not (isinstance(v["substeps"], int) and v["substeps"] >= 1):
        raise ValueError(f"substeps must be an integer >= 1, got {v['substeps']!r}")

    b = raw["control_bounds"]
    u_min = _readonly([b["a_min"], b["delta_min"]])
    u_max = _readonly([b["a_max"], b["delta_max"]])
    if not (u_min < u_max).all():
        raise ValueError(
            f"control_bounds: min must be < max, got u_min={u_min}, u_max={u_max}"
        )

    t = raw["track"]

    # --- construction ---
    return Config(
        model=model,
        state_dim=raw["state_dim"],
        control_dim=raw["control_dim"],
        mppi=MppiParams(
            num_samples=m["num_samples"],
            horizon=m["horizon"],
            dt=m["dt"],
            lam=m["lambda"],   # "lambda" is a Python keyword
            gamma=m["gamma"],
            noise_std=noise_std,
            seed=m["seed"],
        ),
        bounds=Bounds(u_min=u_min, u_max=u_max),
        vehicle=Vehicle(**v),            # YAML keys match the field names exactly
        cost=CostParams(**raw["cost"]),   # YAML keys match the field names exactly
        track=TrackParams(
            grid_resolution=t["grid_resolution"],
            resample_step=t["resample_step"],
            margin=t["margin"],
            npz_path=root / t["npz_path"],
            bin_path=root / t["bin_path"],
        ),
        max_steps=raw["sim"]["max_steps"],
    )
