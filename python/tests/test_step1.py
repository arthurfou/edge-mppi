from pathlib import Path

import numpy as np
import pytest

from mppi.config import load_config
from mppi.dynamics import step

CFG = load_config(Path(__file__).resolve().parents[2] / "config" / "mppi.yaml")
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