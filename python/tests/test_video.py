"""Tests of the MPPI video helpers (video_mppi.py): geometry and timing, no rendering."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mppi import track as trk
from mppi.config import load_config
from video_mppi import along, compute_cycles, lap_timeline, pose_at, scores, view_window, zoom

CFG = load_config(Path(__file__).resolve().parents[2] / "config" / "mppi.yaml")


def video_args(**kw):
    """The command-line arguments compute_cycles and the timelines read."""
    base = dict(seed=0, show=64, steps=3, stride=2, warmup=0, color="rank", margin=0.4, window=None,
                lap=True, speed=1.0, approach=1.0, durations=(1.0, 1.0, 1.0, 1.0), later=0.5,
                hold=0.5, fps=30)
    return SimpleNamespace(**{**base, **kw})


@pytest.fixture(scope="module")
def track():
    return trk.load(CFG.track.npz_path)


@pytest.fixture(scope="module")
def run(track):
    """A fake saved run: 200 steps on the centerline at 3 m/s, controls at zero."""
    n = 200
    idx = np.arange(n) * 3     # pas de rééchantillonnage 5 cm, 3 points = 15 cm ≈ 3 m/s × 50 ms
    states = np.column_stack([track.centerline[idx], track.heading[idx],
                              np.full(n, 3.0), np.zeros(n), np.zeros(n)])
    return states, np.zeros((n, 2))


# --- Interpolation -------------------------------------------------------------

def test_along_ends():
    P = np.arange(12.0).reshape(1, 6, 2)
    np.testing.assert_allclose(along(P, 1.0)[0, -1], P[0, -1])
    np.testing.assert_allclose(along(P, 0.0)[0], P[0, [0, 0]])   # le point de départ, doublé
    np.testing.assert_allclose(along(P, 0.5)[0, -1], 0.5 * (P[0, 2] + P[0, 3]))


def test_pose_at_wraps_heading():
    states = np.array([[0, 0, np.pi - 0.1, 1, 0, 0], [1, 0, -np.pi + 0.1, 1, 0, 0]], dtype=float)
    psi = pose_at(states, 0.5)[2]
    assert abs(abs(psi) - np.pi) < 1e-9   # le court chemin par ±pi, pas par 0


# --- Scores et cadrage ---------------------------------------------------------

@pytest.mark.parametrize("mode", ["rank", "weight", "cost"])
def test_scores_best_is_one(mode):
    S = np.array([5.0, 1.0, 3.0, 9.0])
    w = np.exp(-(S - S.min()))
    w /= w.sum()
    sc = scores(S, w, mode)
    assert sc.min() >= 0 and sc.max() == pytest.approx(1.0) and np.argmax(sc) == 1


def test_view_window_aspect_and_keep():
    beam = np.random.default_rng(0).normal(size=(1000, 2))
    keep = np.array([[5.0, 5.0]])
    win = view_window(beam, keep, 0.1, None, aspect=2.0)
    assert (win[1] - win[0]) / (win[3] - win[2]) == pytest.approx(2.0)
    assert win[0] <= 5 <= win[1] and win[2] <= 5 <= win[3]


def test_zoom_ends_on_both_views():
    wide, close = np.array([0.0, 40.0, 0.0, 20.0]), np.array([9.0, 11.0, 4.0, 5.0])
    car = np.array([10.0, 4.5])
    np.testing.assert_allclose(zoom(("in", wide, close, 0.0), car, car), wide)
    np.testing.assert_allclose(zoom(("in", wide, close, 1.0), car, car), close)
    np.testing.assert_allclose(zoom(("out", wide, close, 0.0), car, car), close)


# --- Cycles et mode --lap -------------------------------------------------------

def test_lap_cycles_follow_saved_run(track, run):
    states, controls = run
    args = video_args()
    k = 50
    cycles = compute_cycles(CFG, track, states, controls, k, args, aspect=1.5)
    assert len(cycles) == args.steps
    for i, c in enumerate(cycles):
        np.testing.assert_array_equal(c.path, states[k + i * args.stride:k + (i + 1) * args.stride + 1])
        assert c.X.shape == (args.show, CFG.mppi.horizon + 1, CFG.state_dim)
        assert np.all(np.diff(c.score) >= 0)   # les meilleurs dessinés en dernier


def test_lap_timeline_is_continuous(track, run):
    states, controls = run
    args = video_args()
    k = 50
    cycles = compute_cycles(CFG, track, states, controls, k, args, aspect=1.5)
    frames = lap_timeline(cycles, k, len(states), np.zeros(4), args, CFG.mppi.dt)
    steps = [f[1] for f in frames if f[0] == "drive"]
    before = [s for s in steps if s <= k]
    after = [s for s in steps if s > k]
    assert np.all(np.diff(before) >= 0) and np.all(np.diff(after) >= 0)
    assert before[-1] == pytest.approx(k)                            # le ralenti finit pile sur x_k
    assert after[0] == pytest.approx(k + args.steps * args.stride, abs=0.1)   # et repart d'où les cycles s'arrêtent
    assert steps[-1] == len(states) - 1
    real_time = args.speed / (CFG.mppi.dt * args.fps)
    assert np.max(np.diff(before)) <= real_time + 1e-9               # jamais plus vite que le temps réel
