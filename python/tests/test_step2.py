"""Step 2 tests: dynamic single-track model, tires, blend, and the 6-state controller."""
import dataclasses
from pathlib import Path

import numpy as np
import pytest

from mppi import dynamics as dyn
from mppi import track as trk
from mppi.config import load_config
from mppi.controller import MPPI, rollout_costs

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

def test_config_is_dynamic():
    assert CFG.model == "dynamic" and CFG.state_dim == 6


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


def test_straight_line_stays_straight():
    x = run(on_manifold(3.0, 0.0), np.zeros(2), 50)
    assert np.allclose(x[[1, 2, 4, 5]], 0.0)
    assert x[0] == pytest.approx(3.0) and x[3] == pytest.approx(3.0)


def test_turns_left_with_positive_steer():
    """Same sign convention on the full step: 0.5 s at delta = 0.1 from a straight line."""
    x = run(on_manifold(3.0, 0.0), np.array([0.0, 0.1]), 25)
    assert x[1] > 0 and x[2] > 0 and x[5] > 0


@pytest.mark.parametrize("tire", TIRES)
def test_left_right_symmetry(tire):
    """Mirroring y, psi, vy, r and delta mirrors the result exactly."""
    veh = with_vehicle(tire_model=tire)
    rng = np.random.default_rng(0)
    states = rng.normal(size=(64, 6)) * [1, 1, 0.5, 0, 0.5, 1]
    states[:, 3] = rng.uniform(0, 8, 64)
    controls = rng.uniform(CFG.bounds.u_min, CFG.bounds.u_max, (64, 2))
    flip_x, flip_u = np.array([1, -1, -1, 1, -1, -1]), np.array([1, -1])
    a = dyn.step(states * flip_x, controls * flip_u, DT, veh)
    b = dyn.step(states, controls, DT, veh) * flip_x
    assert np.allclose(a, b, atol=1e-12)


def test_steady_state_cornering():
    """Steady turn: (F_yf cos(delta) + F_yr) / m = vx * r, zero yaw moment, and
    r matches the understeer formula r = vx delta / (L + K_us vx^2 / g)."""
    veh = with_vehicle(tire_model="linear")
    vx, delta = 3.0, 0.05
    fzf, fzr = dyn.axle_loads(veh)
    x = on_manifold(vx, 0.0)
    for _ in range(250):   # 5 s, vx maintenu constant par la commande a
        alpha_f, _ = dyn.slip_angles(x, delta, veh)
        fyf = dyn.tire_force(alpha_f, fzf, veh.cornering_stiffness_front, veh)
        a = fyf * np.sin(delta) / veh.mass - x[4] * x[5]
        x = dyn.step_dynamic(x, np.array([a, delta]), DT, veh)

    alpha_f, alpha_r = dyn.slip_angles(x, delta, veh)
    fyf = dyn.tire_force(alpha_f, fzf, veh.cornering_stiffness_front, veh)
    fyr = dyn.tire_force(alpha_r, fzr, veh.cornering_stiffness_rear, veh)
    assert x[3] == pytest.approx(vx, rel=1e-3)
    assert (fyf * np.cos(delta) + fyr) / veh.mass == pytest.approx(x[3] * x[5], rel=1e-6)
    assert veh.lf * fyf * np.cos(delta) - veh.lr * fyr == pytest.approx(0.0, abs=1e-6)
    k_us = 1 / veh.cornering_stiffness_front - 1 / veh.cornering_stiffness_rear
    assert x[5] == pytest.approx(vx * delta / (veh.wheelbase + k_us * vx**2 / dyn.G), rel=0.01)


def test_cross_check_with_kinematic_model():
    """Permanent non-regression: at low speed and small steer, the dynamic model
    (here 2 m/s, above the blend band, so purely dynamic) follows the kinematic one.

    Expected gap: understeer K_us vx^2 / (g L) = 1.5 % of the heading (0.009 rad
    out of 0.61), plus the slip build-up lag and the steering drag on vx:
    6.8 cm and 0.011 rad measured over 4 m. Tolerances x1.5 to x2.
    """
    vx, delta, n = 2.0, 0.05, 100
    beta = np.arctan(VEH.lr / VEH.wheelbase * np.tan(delta))
    k = run(np.array([0.0, 0.0, 0.0, vx / np.cos(beta)]), np.array([0.0, delta]), n, f=dyn.step)
    d = run(on_manifold(vx, delta), np.array([0.0, delta]), n, f=dyn.step)
    assert k.shape == (4,) and d.shape == (6,)   # step a bien choisi chaque modèle
    assert np.hypot(k[0] - d[0], k[1] - d[1]) < 0.10
    assert abs(k[2] - d[2]) < 0.02


def test_batch_matches_single():
    rng = np.random.default_rng(0)
    states = rng.normal(size=(8, 6))
    states[:, 3] = rng.uniform(0, 6, 8)
    controls = rng.normal(size=(8, 2)) * 0.1
    batch = dyn.step(states, controls, DT, VEH)
    loop = np.stack([dyn.step(states[i], controls[i], DT, VEH) for i in range(8)])
    assert batch.shape == (8, 6)
    assert np.allclose(batch, loop)


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
@pytest.mark.parametrize("integrator", ("euler", "rk4"))
def test_finite_at_standstill_and_in_band(tire, integrator):
    """vx from 0 to 2 m/s with arbitrary vy, r: finite, no warning (0 * NaN trap)."""
    veh = with_vehicle(tire_model=tire, integrator=integrator)
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


# --- Intégration --------------------------------------------------------------

def test_integrator_stable_at_blend_low():
    """Config guard: the pure dynamic step must be stable where the blend starts using it."""
    x = on_manifold(VEH.blend_speed_low, 0.0)
    x[5] = 0.1
    for _ in range(100):
        x = dyn.step_dynamic_only(x, np.zeros(2), DT, VEH)
    assert abs(x[5]) < 0.1


# --- Contrôleur avec l'état à 6 composantes ----------------------------------

@pytest.fixture(scope="module")
def track():
    return trk.load(CFG.track.npz_path)


def start_state(track, vx=2.0):
    return np.array([*track.centerline[0], track.heading[0], vx, 0.0, 0.0])


def test_rollout_costs_6_states(track):
    T = CFG.mppi.horizon
    eps = np.random.default_rng(0).normal(size=(64, T, 2)) * CFG.mppi.noise_std
    S1, _, X1 = rollout_costs(start_state(track), np.zeros((T, 2)), eps, track, CFG)
    S2, _, X2 = rollout_costs(start_state(track), np.zeros((T, 2)), eps, track, CFG)
    assert X1.shape == (64, T + 1, 6)
    assert np.array_equal(S1, S2) and np.all(np.isfinite(S1))


def test_closed_loop_short(track):
    """One second from a standstill with the dynamic model: stays on track, speeds up."""
    ctrl, x = MPPI(CFG, track), start_state(track, vx=0.0)
    d_max = CFG.cost.track_half_width - 0.5 * VEH.width
    for _ in range(50):
        u, _ = ctrl.command(x)
        x = dyn.step(x, u, DT, VEH)
        d, _ = trk.lookup(track, x[0], x[1])
        assert abs(d) < d_max
    assert x[3] > 2.0
