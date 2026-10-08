import dataclasses
from pathlib import Path

import numpy as np
import pytest

from mppi import track as trk
from mppi.config import STATE_DIM, load_config
from mppi.controller import MPPI, S_MAX, rollout_costs
from mppi.cost import G, stage_cost, terminal_cost
from mppi.dynamics import step

# Le YAML est passé en dynamique à l'étape 2 : ces tests gardent le cinématique
CFG = dataclasses.replace(load_config(Path(__file__).resolve().parents[2] / "config" / "mppi.yaml"),
                          model="kinematic", state_dim=STATE_DIM["kinematic"])
VEH = CFG.vehicle


def run(state, control, dt, n):
    """Applique step n fois avec une commande constante."""
    for _ in range(n):
        state = step(state, control, dt, VEH)
    return state


def test_straight_line():
    s = run(np.array([0.0, 0.0, 0.0, 2.0]), np.array([0.0, 0.0]), 0.02, 50)
    assert np.allclose(s, [2.0, 0.0, 0.0, 2.0])


def test_constant_steer_circle():
    v, delta, dt = 2.0, 0.2, 0.001
    tan_delta = np.tan(delta)
    beta = np.arctan(VEH.lr / VEH.wheelbase * tan_delta)
    r = VEH.wheelbase / (np.cos(beta) * tan_delta)
    t_lap = 2 * np.pi * r / v
    n = round(t_lap / dt)

    s = run(np.array([0.0, 0.0, 0.0, v]), np.array([0.0, delta]), dt, n)

    assert np.allclose(s[:2], [0.0, 0.0], atol=0.02)
    assert s[2] == pytest.approx(2 * np.pi, abs=0.01)
    assert s[3] == pytest.approx(v)


def test_turns_left_with_positive_steer():
    s = run(np.array([0.0, 0.0, 0.0, 2.0]), np.array([0.0, 0.2]), 0.02, 10)
    assert s[1] > 0.0
    assert s[2] > 0.0


def test_batch_matches_single():
    rng = np.random.default_rng(0)
    states = rng.standard_normal((8, 4))
    states[:, 3] = np.abs(states[:, 3])
    controls = rng.standard_normal((8, 2)) * 0.1

    batch = step(states, controls, 0.02, VEH)
    loop = np.stack([step(states[i], controls[i], 0.02, VEH) for i in range(8)])

    assert batch.shape == (8, 4)
    assert np.allclose(batch, loop)


def test_speed_never_negative():
    v, a, dt = 0.01, -4.0, 0.02
    s = step(np.array([0.0, 0.0, 0.0, v]), np.array([a, 0.0]), dt, VEH)
    assert s[3] == 0.0

# Grid tests: they need `pixi run track` to have been run first.

@pytest.fixture(scope="module")
def track():
    return trk.load(CFG.track.npz_path)


def test_grid_on_centerline(track):
    i = np.arange(0, len(track.centerline), 50)
    d, s = trk.lookup(track, track.centerline[i, 0], track.centerline[i, 1])
    assert np.all(np.abs(d) < track.res)
    ds = trk.wrap(s - i * track.length / len(track.centerline), track.length)
    assert np.all(np.abs(ds) < 2 * track.res)


def test_grid_sign_left_positive(track):
    i = 100
    n = np.array([-np.sin(track.heading[i]), np.cos(track.heading[i])])
    p = track.centerline[i] + 0.5 * n
    d, _ = trk.lookup(track, p[0], p[1])
    assert d == pytest.approx(0.5, abs=track.res)


def test_wrap():
    assert trk.wrap(-47.0, 48.0) == pytest.approx(1.0)
    assert trk.wrap(47.0, 48.0) == pytest.approx(-1.0)
    assert trk.wrap(0.3, 48.0) == pytest.approx(0.3)


def test_bin_matches_npz(track):
    grid, x_min, y_min, res, length = trk.load_bin(CFG.track.bin_path)
    assert np.array_equal(grid, track.grid)
    assert (x_min, y_min, res) == pytest.approx((track.x_min, track.y_min, track.res))


# Cost tests: they also need the grid.

def on_track(track, i, offset, v=CFG.cost.v_ref):
    """State at `offset` metres left of centerline point i, aligned with the track."""
    h = track.heading[i]
    p = track.centerline[i] + offset * np.array([-np.sin(h), np.cos(h)])
    return np.array([p[0], p[1], h, v])


ZERO_U = np.zeros(2)


def test_cost_zero_when_perfect(track):
    c = stage_cost(on_track(track, 100, 0.0), ZERO_U, track, CFG)
    assert c == pytest.approx(0.0, abs=CFG.cost.w_lateral * (track.res / CFG.cost.lateral_scale) ** 2)


def test_cost_lateral(track):
    x = on_track(track, 100, 0.3)
    d, _ = trk.lookup(track, x[0], x[1])
    expected = CFG.cost.w_lateral * (d / CFG.cost.lateral_scale) ** 2
    assert stage_cost(x, ZERO_U, track, CFG) == pytest.approx(expected)


def test_cost_offtrack_is_a_jump_then_graduated(track):
    d_max = CFG.cost.track_half_width - 0.5 * CFG.vehicle.width
    inside, out, further = (stage_cost(on_track(track, 100, d_max + e), ZERO_U, track, CFG)
                            for e in (-0.1, 0.1, 0.3))
    assert out - inside > CFG.cost.w_offtrack   # the indicator switches on
    assert further > out                         # and it keeps growing outside


def test_cost_offtrack_symmetric(track):
    left = stage_cost(on_track(track, 100, 0.75), ZERO_U, track, CFG)
    right = stage_cost(on_track(track, 100, -0.75), ZERO_U, track, CFG)
    assert left == pytest.approx(right, rel=0.1)   # nearest cell: not exactly symmetric


def test_cost_speed(track):
    x0 = on_track(track, 100, 0.0)
    x1 = on_track(track, 100, 0.0, v=CFG.cost.v_ref + CFG.cost.speed_scale)
    diff = stage_cost(x1, ZERO_U, track, CFG) - stage_cost(x0, ZERO_U, track, CFG)
    assert diff == pytest.approx(CFG.cost.w_speed)


def test_cost_effort(track):
    x = on_track(track, 100, 0.0, v=0.0)   # v = 0: no adhesion term
    u = CFG.bounds.u_max
    diff = stage_cost(x, u, track, CFG) - stage_cost(x, ZERO_U, track, CFG)
    assert diff == pytest.approx(2 * CFG.cost.w_control)   # (a/a_max)² + (d/d_max)² = 2


def test_cost_adhesion(track):
    veh, c = CFG.vehicle, CFG.cost
    u = np.array([0.0, 0.4])
    effort = c.w_control * (u[1] / CFG.bounds.u_max[1]) ** 2

    def extra(v):   # cost added by steering, minus the effort term
        x = on_track(track, 100, 0.0, v=v)
        return stage_cost(x, u, track, CFG) - stage_cost(x, ZERO_U, track, CFG) - effort

    assert extra(1.0) == pytest.approx(0.0, abs=1e-12)   # a_lat = 1.3 m/s² < mu g
    a_lat = 5.0**2 * np.tan(0.4) / veh.wheelbase
    assert extra(5.0) == pytest.approx(c.w_adhesion * (a_lat / (veh.mu * G) - 1) ** 2)


def test_cost_batch_matches_single(track):
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(track.centerline), 16)
    states = np.stack([on_track(track, i, o, v) for i, o, v in
                       zip(idx, rng.uniform(-1, 1, 16), rng.uniform(0, 6, 16))])
    controls = rng.uniform(CFG.bounds.u_min, CFG.bounds.u_max, (16, 2))

    batch = stage_cost(states, controls, track, CFG)
    loop = np.array([stage_cost(states[k], controls[k], track, CFG) for k in range(16)])
    assert batch.shape == (16,)
    assert np.allclose(batch, loop)


def test_terminal_at_v_ref(track):
    m, n = CFG.mppi, len(track.centerline)
    i = 100 + round(CFG.cost.v_ref * m.horizon * m.dt / (track.length / n))
    _, s0 = trk.lookup(track, *track.centerline[100])
    c = terminal_cost(on_track(track, i, 0.0)[None], s0, track, CFG)
    assert c.shape == (1,)
    assert c[0] == pytest.approx(-CFG.cost.w_progress, rel=0.05)


def test_terminal_across_start_line(track):
    n = len(track.centerline)
    _, s0 = trk.lookup(track, *track.centerline[n - 5])
    c = terminal_cost(on_track(track, 5, 0.0), s0, track, CFG)
    assert c < 0   # 10 points forward, not 47 m backward (Q-C4)
    assert terminal_cost(on_track(track, n - 10, 0.0), s0, track, CFG) > 0


# Controller tests.

def start_state(track, v=2.0):
    return np.array([*track.centerline[0], track.heading[0], v])


def fixed_noise(k=64, seed=0):
    return np.random.default_rng(seed).normal(size=(k, CFG.mppi.horizon, 2)) * CFG.mppi.noise_std


def test_rollout_costs_deterministic(track):
    x0, U, eps = start_state(track), np.zeros((CFG.mppi.horizon, 2)), fixed_noise()
    S1, e1, X1 = rollout_costs(x0, U, eps, track, CFG)
    S2, e2, X2 = rollout_costs(x0, U, eps, track, CFG)
    assert np.array_equal(S1, S2) and np.array_equal(X1, X2)
    assert np.all(np.isfinite(S1))
    V = U + e1
    assert np.all(V >= CFG.bounds.u_min) and np.all(V <= CFG.bounds.u_max)


def test_rollout_costs_shapes(track):
    T = CFG.mppi.horizon
    S, e, X = rollout_costs(start_state(track), np.zeros((T, 2)), fixed_noise(), track, CFG)
    assert S.shape == (64,) and e.shape == (64, T, 2) and X.shape == (64, T + 1, 4)
    assert np.array_equal(X[:, 0], np.broadcast_to(start_state(track), (64, 4)))


def test_rollout_costs_matches_manual_loop(track):
    """Rollout k must equal step + stage_cost applied by hand on its own controls."""
    x0, U, eps = start_state(track), np.zeros((CFG.mppi.horizon, 2)), fixed_noise()
    S, e, X = rollout_costs(x0, U, eps, track, CFG)
    k, x, cost = 7, x0, 0.0
    for t in range(CFG.mppi.horizon):
        x = step(x, U[t] + e[k, t], CFG.mppi.dt, VEH)
        cost += stage_cost(x, U[t] + e[k, t], track, CFG)
    _, s0 = trk.lookup(track, x0[0], x0[1])
    cost += terminal_cost(x, s0, track, CFG)
    assert np.allclose(X[k, -1], x)
    assert S[k] == pytest.approx(cost)


def test_rollout_costs_clips_noise(track):
    U = np.zeros((CFG.mppi.horizon, 2))
    eps = np.full((2, CFG.mppi.horizon, 2), 100.0)
    _, e, _ = rollout_costs(start_state(track), U, eps, track, CFG)
    assert np.allclose(e, CFG.bounds.u_max)


@pytest.mark.filterwarnings("ignore:invalid value:RuntimeWarning")
def test_rollout_costs_nan_becomes_s_max(track):
    x0 = start_state(track)
    x0[3] = np.nan
    S, _, _ = rollout_costs(x0, np.zeros((CFG.mppi.horizon, 2)), fixed_noise(4), track, CFG)
    assert np.all(S == S_MAX)


def test_command_outputs(track):
    ctrl = MPPI(CFG, track)
    u0, info = ctrl.command(start_state(track))
    K = CFG.mppi.num_samples
    assert u0.shape == (2,)
    assert np.all(u0 >= CFG.bounds.u_min) and np.all(u0 <= CFG.bounds.u_max)
    assert info["w"].sum() == pytest.approx(1.0)
    assert info["rho"] == info["S"].min()
    assert 1.0 <= info["ess"] <= K


def test_command_shift(track):
    ctrl = MPPI(CFG, track)
    u0, _ = ctrl.command(start_state(track))
    assert np.array_equal(ctrl.U[-1], ctrl.U[-2])   # last control repeated
    assert not np.array_equal(u0, ctrl.U[0])        # u0 was copied before the shift


def test_command_reproducible(track):
    u_a, _ = MPPI(CFG, track).command(start_state(track))
    u_b, _ = MPPI(CFG, track).command(start_state(track))
    assert np.array_equal(u_a, u_b)   # same seed, same result


def test_closed_loop_short(track):
    """One second of driving from the start line: stays on track and speeds up."""
    ctrl, x = MPPI(CFG, track), start_state(track, v=0.0)
    d_max = CFG.cost.track_half_width - 0.5 * VEH.width
    for _ in range(50):
        u, _ = ctrl.command(x)
        x = step(x, u, CFG.mppi.dt, VEH)
        d, _ = trk.lookup(track, x[0], x[1])
        assert abs(d) < d_max
    assert x[3] > 1.0
