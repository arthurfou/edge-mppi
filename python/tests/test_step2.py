"""Step 2 tests: dynamic single-track model, tires, blend, and the 6-state controller."""
import dataclasses
from pathlib import Path

import numpy as np
import pytest

from mppi import dynamics as dyn
from mppi.config import load_config

CFG = load_config(Path(__file__).resolve().parents[2] / "config" / "mppi.yaml")
VEH = CFG.vehicle
DT = CFG.mppi.dt
TIRES = ("linear", "tanh", "pacejka")


def with_vehicle(**kw):
    return dataclasses.replace(VEH, **kw)


def on_manifold(vx, delta, veh=VEH):
    """Dynamic state at the origin, heading +x, with (vy, r) on the kinematic manifold."""
    tan_d = np.tan(delta)
    return np.array([0.0, 0.0, 0.0, vx, veh.lr / veh.wheelbase * vx * tan_d, vx * tan_d / veh.wheelbase])


def run(state, control, n, veh=VEH, f=dyn.step_dynamic):
    for _ in range(n):
        state = f(state, control, DT, veh)
    return state


# --- Configuration -----------------------------------------------------------

def test_cornering_stiffness_convention():
    """C_S is normalized by the load: C_alpha = mu * C_S * F_z, ~75 / 66 N/rad (theory 6.3)."""
    fzf, fzr = dyn.axle_loads(VEH)
    assert fzf + fzr == pytest.approx(VEH.mass * dyn.G)
    assert VEH.mu * VEH.cornering_stiffness_front * fzf == pytest.approx(74.9, abs=0.1)
    assert VEH.mu * VEH.cornering_stiffness_rear * fzr == pytest.approx(65.6, abs=0.1)


# --- Pneus --------------------------------------------------------------------

@pytest.mark.parametrize("tire", TIRES)
def test_tire_same_slope_at_origin(tire):
    fz, c_s = 18.7, VEH.cornering_stiffness_front
    alpha = 1e-4
    f = dyn.tire_force(np.array(alpha), fz, c_s, with_vehicle(tire_model=tire))
    assert f == pytest.approx(VEH.mu * c_s * fz * alpha, rel=1e-6)


@pytest.mark.parametrize("tire", ("tanh", "pacejka"))
def test_tire_saturates_at_mu_fz(tire):
    fz = 18.7
    alpha = np.linspace(-np.pi / 2, np.pi / 2, 2001)
    f = dyn.tire_force(alpha, fz, VEH.cornering_stiffness_front, with_vehicle(tire_model=tire))
    assert np.abs(f).max() <= VEH.mu * fz * (1 + 1e-12)
    assert np.abs(f).max() >= 0.95 * VEH.mu * fz
    assert np.allclose(f, -f[::-1])          # force impaire


# --- Modèle dynamique ---------------------------------------------------------

@pytest.mark.parametrize("tire", TIRES)
def test_positive_steer_pushes_left(tire):
    """Straight at 3 m/s, delta > 0: alpha_f > 0, the car accelerates to the left
    and starts turning left. Derivative only; the full-step version comes with part E."""
    veh = with_vehicle(tire_model=tire)
    state = np.array([0.0, 0.0, 0.0, 3.0, 0.0, 0.0])
    alpha_f, alpha_r = dyn.slip_angles(state, 0.1, veh)
    assert alpha_f == pytest.approx(0.1) and alpha_r == 0.0
    d = dyn.dynamic_derivative(state, 0.0, 0.1, veh)
    assert d[4] > 0 and d[5] > 0


# --- Basse vitesse : mélange cinématique/dynamique ----------------------------

def test_blend_is_kinematic_below_low_speed():
    """Below blend_speed_low the step is exactly the kinematic one in 6 states."""
    x = on_manifold(0.9 * VEH.blend_speed_low, 0.2)
    u = np.array([0.5, 0.2])
    assert np.array_equal(dyn.step_dynamic(x, u, DT, VEH), dyn.step_kinematic6(x, u, DT, VEH))


def test_blend_is_continuous():
    u = np.array([1.0, 0.3])
    for v in (VEH.blend_speed_low, VEH.blend_speed_high):
        lo = dyn.step_dynamic(on_manifold(v - 1e-7, 0.3), u, DT, VEH)
        hi = dyn.step_dynamic(on_manifold(v + 1e-7, 0.3), u, DT, VEH)
        assert np.allclose(lo, hi, atol=1e-5)


@pytest.mark.parametrize("tire", TIRES)
def test_finite_at_standstill_and_in_band(tire):
    """vx from 0 to 2 m/s with arbitrary vy, r: finite, no warning (0 * NaN trap).
    Part E adds the integrator as a second parametrize."""
    veh = with_vehicle(tire_model=tire)
    rng = np.random.default_rng(1)
    n = 401
    states = np.zeros((n, 6))
    states[:, 3] = np.linspace(0.0, 2.0, n)
    states[:, 4] = rng.normal(0, 0.5, n)
    states[:, 5] = rng.normal(0, 2.0, n)
    controls = rng.uniform(CFG.bounds.u_min, CFG.bounds.u_max, (n, 2))
    with np.errstate(all="raise"):
        out = run(states, controls, 50, veh)
    assert np.all(np.isfinite(out))


def test_never_reverses():
    x = dyn.step_dynamic(on_manifold(0.01, 0.0), np.array([-4.0, 0.0]), DT, VEH)
    assert x[3] == 0.0 and x[4] == 0.0 and x[5] == 0.0
