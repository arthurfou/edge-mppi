"""Video of MPPI at work, replayed from a trajectory saved by run_sim.py.

Picks a state x_k of the saved run, then for each of --steps cycles shows:
  1. sample  the K rollouts growing together over the horizon, all in gray
  2. score   the rollouts colored by their score
  3. mean    the weighted mean U* = U + sum_i w_i eps_i, rolled out, thick line
  4. move    the plant applies the first --stride controls of U*, the horizon shifts

The rollouts are recomputed, not read from the run (run_sim.py keeps none on
disk). U starts from the controls actually applied after x_k, control[k:k+T],
a close stand-in for the nominal sequence the controller held then: the
picture is faithful in shape, not bit for bit. The cycles then run in closed
loop, plant included, from x_k.

With --lap, the whole lap is played around it: real time on the whole track,
slow down and zoom in up to x_k, the MPPI cycles, then zoom out and back to
real time until the end of the lap. The car then follows the saved run, also
during the cycles, so that the lap resumes without a jump.

Slowed down, not real time. Writes results/videos/<name>_mppi_k<k>.mp4, or
<name>_lap_k<k>.mp4 with --lap (a .gif if ffmpeg is missing).
"""
import argparse
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.animation import FuncAnimation, writers
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, to_rgba

from mppi import track as trk
from mppi.config import Config, load_config
from mppi.controller import rollout_costs
from mppi.dynamics import step, step_dynamic
from run_sim import BEAM_PROGRESS, draw_track, observe, with_model
from video import car_outline

PHASES = ("sample", "score", "mean", "move")
CAPTIONS = {
    "sample": "1 · sample {K} noisy control sequences, roll them out over {T} steps",
    "score": "2 · score each rollout: cost S, weight w ∝ exp(-S/λ)",
    "mean": "3 · weighted mean of the noise: U* = U + Σ w ε",
    "move": "4 · apply {applied} of U*, shift the horizon by {stride}",
}
COLOR_LABELS = {
    "rank": "cost rank (1 = cheapest)",
    "weight": "weight w / max w",
    "cost": "1 - (S - S_min) / spread",
}
GRAY = to_rgba("0.55")


@dataclass
class Cycle:
    X: np.ndarray       # (n_show, T+1, n) rollouts shown, worst first so the best are drawn on top
    score: np.ndarray   # (n_show,) in [0, 1], 1 = best
    best: np.ndarray    # (T+1, n) U* rolled out with the controller model
    path: np.ndarray    # (stride+1, 6) plant states while U*[:stride] is applied
    ess: float
    window: np.ndarray  # (4,) x_min, x_max, y_min, y_max of the view


def rollout(x0: np.ndarray, U: np.ndarray, cfg: Config) -> np.ndarray:
    X = [x0]
    for u in U:
        X.append(step(X[-1], u, cfg.mppi.dt, cfg.vehicle))
    return np.array(X)


def scores(S: np.ndarray, w: np.ndarray, mode: str) -> np.ndarray:
    """Score in [0, 1], 1 = best. Raw weights are very peaked (ESS a few % of K): rank reads best."""
    if mode == "rank":
        return np.argsort(np.argsort(-S)) / max(len(S) - 1, 1)
    if mode == "weight":
        return w / w.max()
    spread = max(np.percentile(S, 90) - S.min(), 1e-9)   # cost : les 10 % les pires saturent à 0
    return 1.0 - np.clip((S - S.min()) / spread, 0.0, 1.0)


def view_window(beam: np.ndarray, keep: np.ndarray, margin: float, width: float | None,
                aspect: float) -> np.ndarray:
    """Box around the beam (1-99 percentiles, a lost rollout does not blow the zoom) and
    every keep point (car, U*), at the axes aspect."""
    lo, hi = np.percentile(beam, [1, 99], axis=0)
    lo, hi = np.minimum(lo, keep.min(axis=0)), np.maximum(hi, keep.max(axis=0))
    center = 0.5 * (lo + hi)
    w, h = (width, width / aspect) if width else hi - lo + 2 * margin
    w, h = max(w, h * aspect), max(h, w / aspect)   # même échelle en x et en y
    return np.array([center[0] - w / 2, center[0] + w / 2, center[1] - h / 2, center[1] + h / 2])


def compute_cycles(cfg: Config, track: trk.Track, states: np.ndarray, controls: np.ndarray,
                   k: int, args, aspect: float) -> list[Cycle]:
    """All MPPI iterations of the video, computed before drawing anything."""
    m = cfg.mppi
    K, T, dt = m.num_samples, m.horizon, m.dt
    rng = np.random.default_rng(args.seed)
    U = controls[np.minimum(np.arange(k, k + T), len(controls) - 1)].copy()
    x = states[k].copy()
    shown = np.sort(rng.choice(K, args.show, replace=False)) if args.show < K else np.arange(K)

    def iterate(x, U):
        eps = rng.standard_normal((K, T, cfg.control_dim)) * m.noise_std
        S, eps, X = rollout_costs(observe(x, cfg), U, eps, track, cfg)
        w = np.exp(-(S - S.min()) / m.lam)
        w /= w.sum()
        return S, w, X, U + np.tensordot(w, eps, axes=1)

    for _ in range(args.warmup):   # on affine U sur place, la voiture ne bouge pas
        U = iterate(x, U)[3]

    cycles = []
    for i in range(args.steps):
        S, w, X, U_star = iterate(x, U)
        if args.lap:   # la voiture reste sur la course sauvegardée
            j = k + i * args.stride
            path = states[j:j + args.stride + 1]
        else:
            path = [x]
            for u in U_star[:args.stride]:
                path.append(step_dynamic(path[-1], u, dt, cfg.vehicle))   # le véhicule simulé est toujours dynamique
            path = np.array(path)
        best = rollout(observe(x, cfg), U_star, cfg)

        sc = scores(S, w, args.color)[shown]
        order = np.argsort(sc)
        Xs = X[shown][order]
        v = cfg.vehicle
        keep = np.vstack([best[:, :2], *(car_outline(*s[:3], v.wheelbase, v.width) for s in path)])
        cycles.append(Cycle(Xs, sc[order], best, path, 1.0 / np.sum(w**2),
                            view_window(Xs[..., :2].reshape(-1, 2), keep, args.margin, args.window, aspect)))

        x = path[-1]
        U = np.concatenate([U_star[args.stride:], np.repeat(U_star[-1:], args.stride, axis=0)])
    return cycles


def timeline(n_cycles: int, args) -> list[tuple]:
    """("mppi", cycle, phase, progress in ]0, 1]) for each frame of the cycles."""
    frames = []
    for c in range(n_cycles):
        scale = 1.0 if c == 0 else args.later
        for phase, dur in zip(PHASES, args.durations):
            n = max(1, round(dur * scale * args.fps))
            frames += [("mppi", c, phase, (f + 1) / n) for f in range(n)]
    return frames


def lap_timeline(cycles: list[Cycle], k: int, n_states: int, track_win: np.ndarray, args, dt: float) -> list[tuple]:
    """Whole lap around the cycles: ("drive", step s (float), view, caption, speed factor, U* opacity).

    Real time is r steps per frame. The slowdown is s(u) = k - L (1-u)^2 over
    n frames: its slope at u = 0 is r when L = r n / 2, so the speed falls
    smoothly from real time to zero, and rises back the same way after.
    """
    r = args.speed / (dt * args.fps)
    n = max(1, round(args.approach * args.fps))
    us = (np.arange(n) + 1) / n
    k1 = k + len(cycles) * args.stride
    L0, L1 = min(r * n / 2, k), min(r * n / 2, n_states - 1 - k1)
    full = lambda s: (track_win, "lap · real time" if args.speed == 1 else f"lap · ×{args.speed:g}", args.speed, 0.0)  # noqa: E731

    # Pas de r qui tombe pile sur k - L0, où le ralenti prend le relais à la même vitesse
    frames = [("drive", s, *full(s)) for s in (k - L0) - r * np.arange(int((k - L0) // r), -1, -1)]
    frames += [("drive", k - L0 * (1 - u) ** 2, ("in", track_win, cycles[0].window, ease(u)),
                "slowing down, zooming in on one MPPI iteration", args.speed * (1 - u), 0.0) for u in us]
    frames += timeline(len(cycles), args)
    frames += [("drive", k1 + L1 * u ** 2, ("out", track_win, cycles[-1].window, ease(u)),
                "back to the lap", args.speed * u, 1 - ease(min(2 * u, 1))) for u in us]
    frames += [("drive", s, *full(s)) for s in np.arange(k1 + L1 + r, n_states - 1, r)]
    frames += [("drive", n_states - 1.0, track_win, "lap done", 0.0, 0.0)] * round(args.hold * args.fps)
    return frames


def zoom(view, pos: np.ndarray, car0: np.ndarray) -> np.ndarray:
    """View while zooming between the whole track and a cycle view.

    The size changes geometrically (constant zoom speed); the center goes from
    the track center to the car, shifted the way the cycle view is, so the car
    never leaves the frame.
    """
    way, wide, close, e = view
    e = e if way == "in" else 1 - e
    c_wide, c_close = wide.reshape(2, 2).mean(axis=1), close.reshape(2, 2).mean(axis=1)
    center = (1 - e) * c_wide + e * (pos + c_close - car0)
    size = (wide[1::2] - wide[::2]) * ((close[1::2] - close[::2]) / (wide[1::2] - wide[::2])) ** e
    return np.array([center[0] - size[0] / 2, center[0] + size[0] / 2,
                     center[1] - size[1] / 2, center[1] + size[1] / 2])


def pose_at(states: np.ndarray, s: float) -> np.ndarray:
    """State at a fractional step, linear between two saved states (psi unwrapped)."""
    i = min(int(s), len(states) - 2)
    a, b = states[i], states[i + 1]
    d = b - a
    d[2] = (d[2] + np.pi) % (2 * np.pi) - np.pi
    return a + (s - i) * d


def along(P: np.ndarray, p: float) -> np.ndarray:
    """P (..., N+1, d) cut at fraction p of its length, the last point interpolated: (..., i+2, d)."""
    s = p * (P.shape[-2] - 1)
    i = min(int(s), P.shape[-2] - 2)
    tip = P[..., i, :] + (s - i) * (P[..., i + 1, :] - P[..., i, :])
    return np.concatenate([P[..., :i + 1, :], tip[..., None, :]], axis=-2)


def ease(p: float) -> float:
    return p * p * (3 - 2 * p)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("trajectory", nargs="?", default="results/trajectories/step2_dynamic.npz")
    ap.add_argument("--config", default="config/mppi.yaml")
    g = ap.add_argument_group("where and how long")
    g.add_argument("--start", default=None,
                   help=f"first step k: an index, 'random', or by default the step at {BEAM_PROGRESS:g} m "
                        "of progress (the beam of step2_*_trajectory.png)")
    g.add_argument("--steps", type=int, default=5, help="number of MPPI cycles shown")
    g.add_argument("--lap", action="store_true",
                   help="play the whole lap: real time, slow down and zoom in on x_k, the cycles, "
                        "zoom out, real time to the end")
    g.add_argument("--speed", type=float, default=1.0, help="--lap: playback speed of the lap (1 = real time)")
    g.add_argument("--approach", type=float, default=2.0,
                   help="--lap: duration of the slowdown + zoom in, and of the zoom out (s)")
    g.add_argument("--stride", type=int, default=1,
                   help="controls of U* applied per cycle before replanning (1 = real MPPI; "
                        "more makes the progress easier to see)")
    g = ap.add_argument_group("controller (YAML values by default)")
    g.add_argument("--model", choices=("kinematic", "dynamic"), default=None,
                   help="rollout model; by default read from the file name, else the YAML one")
    g.add_argument("--samples", "-K", type=int, default=None, help="number of rollouts K")
    g.add_argument("--horizon", "-T", type=int, default=None, help="horizon T, in steps")
    g.add_argument("--lam", type=float, default=None, help="temperature lambda")
    g.add_argument("--noise", type=float, nargs=2, default=None, metavar=("A", "DELTA"),
                   help="noise standard deviations on [a, delta]")
    g.add_argument("--seed", type=int, default=None, help="noise seed (YAML seed by default)")
    g.add_argument("--warmup", type=int, default=0,
                   help="silent MPPI iterations at x_k to refine U before the first cycle")
    g = ap.add_argument_group("drawing")
    g.add_argument("--show", type=int, default=None, help="rollouts drawn, a random subset of K (all by default)")
    g.add_argument("--color", choices=tuple(COLOR_LABELS), default="rank", help="score used to color the rollouts")
    g.add_argument("--cmap", default="viridis")
    g.add_argument("--alpha", type=float, default=0.35, help="opacity of one rollout")
    g.add_argument("--margin", type=float, default=0.4, help="margin around the rollouts in the view (m)")
    g.add_argument("--window", type=float, default=None, help="fixed view width (m) instead of the automatic zoom")
    g.add_argument("--no-minimap", action="store_true", help="hide the whole-track inset")
    g = ap.add_argument_group("timing and output")
    g.add_argument("--durations", type=float, nargs=4, default=(1.5, 1.0, 1.0, 1.0),
                   metavar=("SAMPLE", "SCORE", "MEAN", "MOVE"), help="phase durations of the first cycle (s)")
    g.add_argument("--later", type=float, default=0.5, help="duration factor for the cycles after the first")
    g.add_argument("--hold", type=float, default=1.0, help="still frames at the end (s)")
    g.add_argument("--fps", type=int, default=30)
    g.add_argument("--dpi", type=int, default=100)
    g.add_argument("--out", type=Path, default=None, help="output file (.mp4 or .gif)")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    track = trk.load(cfg.track.npz_path)
    data = np.load(args.trajectory)
    states, controls = data["state"], data["control"]
    stem = Path(args.trajectory).stem

    model = args.model or ("kinematic" if "kinematic" in stem else "dynamic" if "dynamic" in stem else cfg.model)
    cfg = with_model(cfg, model)
    m = cfg.mppi
    cfg = replace(cfg, mppi=replace(
        m,
        num_samples=args.samples or m.num_samples,
        horizon=args.horizon or m.horizon,
        lam=args.lam if args.lam is not None else m.lam,
        noise_std=np.array(args.noise) if args.noise else m.noise_std,
        seed=args.seed if args.seed is not None else m.seed,
    ))
    m = cfg.mppi
    args.seed = m.seed
    args.show = min(args.show or m.num_samples, m.num_samples)
    if not 1 <= args.stride < m.horizon:
        raise SystemExit(f"--stride must be in [1, T-1 = {m.horizon - 1}]")

    if args.start is None:
        _, s = trk.lookup(track, states[:, 0], states[:, 1])
        progress = np.concatenate([[0.0], np.cumsum(trk.wrap(np.diff(s), track.length))])
        k = int(np.argmax(progress >= BEAM_PROGRESS))
    elif args.start == "random":
        k = int(np.random.default_rng().integers(len(states)))
    else:
        k = int(args.start)
    if not 0 <= k < len(states):
        raise SystemExit(f"--start must be in [0, {len(states) - 1}]")
    if args.lap and k + args.steps * args.stride >= len(states):
        raise SystemExit(f"--lap: the cycles end after the saved run ({len(states)} steps), "
                         "take an earlier --start or fewer --steps")

    # --- figure fixe ; l'aspect des axes sert à cadrer chaque vue à la même échelle en x et y ---
    v = cfg.vehicle
    cmap = plt.get_cmap(args.cmap)
    fig = plt.figure(figsize=(11, 6.5))
    gs = fig.add_gridspec(2, 2, width_ratios=(4.5, 1), height_ratios=(1, 1.3))
    ax = fig.add_subplot(gs[:, 0])
    side = fig.add_subplot(gs[1, 1])   # porte la barre de couleur, sans cadre
    side.axis("off")
    draw_track(ax, track, cfg.cost.track_half_width)
    ax.set_aspect("auto")   # l'échelle est gérée par view_window
    ax.plot(*states[:, :2].T, color="0.6", lw=1, ls=":", zorder=1, label="saved run")
    fig.colorbar(ScalarMappable(Normalize(0, 1), cmap), cax=side.inset_axes([0.4, 0.05, 0.12, 0.9]),
                 label=COLOR_LABELS[args.color])
    if not args.no_minimap:
        mini = fig.add_subplot(gs[0, 1])
        draw_track(mini, track, cfg.cost.track_half_width)
        mini.set_xticks([]); mini.set_yticks([])
        mini.set_title("whole track", fontsize=9)
    ax.set_title(f"MPPI, {model} rollouts · K {m.num_samples} · T {m.horizon} · "
                 f"λ {m.lam:g} · from step {k} of {stem}", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    bbox = ax.get_window_extent()
    aspect = bbox.width / bbox.height

    cycles = compute_cycles(cfg, track, states, controls, k, args, aspect)
    if args.lap:
        normal = np.column_stack([-np.sin(track.heading), np.cos(track.heading)])
        edges = np.vstack([track.centerline + side * cfg.cost.track_half_width * normal for side in (-1, 1)])
        track_win = view_window(edges, edges, 0.5, None, aspect)
        frames = lap_timeline(cycles, k, len(states), track_win, args, m.dt)
    else:
        frames = timeline(len(cycles), args)
        frames += [frames[-1]] * round(args.hold * args.fps)

    lc = LineCollection([], linewidths=0.6, zorder=2)
    ax.add_collection(lc)
    best, = ax.plot([], [], color="tab:red", lw=2.5, zorder=4, label="U* rolled out",
                    path_effects=[patheffects.withStroke(linewidth=4.5, foreground="white")])
    trail, = ax.plot([], [], color="tab:blue", lw=2, zorder=3, label="driven")
    ghost = plt.Polygon(car_outline(*states[k, :3], v.wheelbase, v.width), color="0.2", alpha=0.25, zorder=5)
    car = plt.Polygon(car_outline(*states[k, :3], v.wheelbase, v.width), color="0.15", zorder=6)
    ax.add_patch(ghost); ax.add_patch(car)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    caption = ax.text(0.01, 0.98, "", transform=ax.transAxes, va="top", fontsize=11, weight="bold",
                      bbox=dict(boxstyle="round", fc="white", ec="0.8"), zorder=10)
    info = ax.text(0.01, 0.02, "", transform=ax.transAxes, va="bottom", family="monospace", fontsize=9,
                   bbox=dict(boxstyle="round", fc="white", ec="0.8"), zorder=10)

    if not args.no_minimap:
        frame_box = plt.Rectangle((0, 0), 1, 1, fill=False, ec="tab:red", lw=1.2)
        mini.add_patch(frame_box)
        mini_car, = mini.plot([], [], "o", color="tab:red", ms=3)

    # Chemin parcouru au début de chaque cycle ; avec --lap, depuis le départ
    starts = [states[:k + 1] if args.lap else cycles[0].path[:1]]
    for c in cycles:
        starts.append(np.vstack([starts[-1], c.path[1:]]))

    def drive(s, view, text, speed, best_alpha):
        """One frame of the lap, outside the cycles: the car on the saved run."""
        pose = pose_at(states, s)
        lc.set_segments([])
        ghost.set_visible(False)
        best.set_alpha(best_alpha)
        if best_alpha == 0:
            best.set_data([], [])
        car.set_xy(car_outline(*pose[:3], v.wheelbase, v.width))
        trail.set_data(*np.vstack([states[:int(s) + 1, :2], pose[:2]]).T)
        if isinstance(view, tuple):   # zoom : la voiture garde sa place dans la vue du cycle
            car0 = cycles[0].path[0] if view[0] == "in" else cycles[-1].path[-1]
            win = zoom(view, pose[:2], car0[:2])
        else:
            win = view
        ax.set_xlim(win[0], win[1]); ax.set_ylim(win[2], win[3])
        caption.set_text(text)
        info.set_text(f"t {s * m.dt:5.2f} s   vx {pose[3]:4.2f} m/s   playback ×{speed:4.2f}")
        if not args.no_minimap:
            frame_box.set_bounds(win[0], win[2], win[1] - win[0], win[3] - win[2])
            mini_car.set_data([pose[0]], [pose[1]])

    def update(f):
        if frames[f][0] == "drive":
            return drive(*frames[f][1:])
        _, ci, phase, p = frames[f]
        c = cycles[ci]
        best.set_alpha(1.0)
        e = ease(p)
        colored = cmap(c.score)
        colored[:, 3] = args.alpha
        gray = np.tile(GRAY, (len(c.X), 1))
        gray[:, 3] = args.alpha
        xy = c.X[..., :2]

        if phase == "sample":
            lc.set_segments(along(xy, e))
            lc.set_color(gray)
            best.set_data([], [])
        elif phase == "score":
            lc.set_segments(xy)
            lc.set_color((1 - e) * gray + e * colored)
        elif phase == "mean":
            lc.set_segments(xy)
            colored[:, 3] = args.alpha * (1 - 0.6 * e)   # le faisceau s'estompe sous U*
            lc.set_color(colored)
            best.set_data(*along(c.best[:, :2], e).T)
        else:
            colored[:, 3] = args.alpha * 0.4 * (1 - e)
            lc.set_color(colored)
            best.set_data(*c.best[:, :2].T)

        pose = along(c.path[:, :3], e)[-1] if phase == "move" else c.path[0, :3]
        car.set_xy(car_outline(*pose, v.wheelbase, v.width))
        ghost.set_xy(car_outline(*c.path[0, :3], v.wheelbase, v.width))
        ghost.set_visible(phase == "move")
        driven = np.vstack([starts[ci][:, :2], along(c.path[:, :2], e)[1:]]) if phase == "move" else starts[ci]
        trail.set_data(*driven[:, :2].T)

        win = c.window
        if phase == "move" and ci + 1 < len(cycles):
            win = (1 - e) * win + e * cycles[ci + 1].window   # la caméra suit vers la vue suivante
        ax.set_xlim(win[0], win[1]); ax.set_ylim(win[2], win[3])

        stride = len(c.path) - 1
        applied = "the first control" if stride == 1 else f"the first {stride} controls"
        caption.set_text(CAPTIONS[phase].format(K=m.num_samples, T=m.horizon, stride=stride, applied=applied))
        step_k = k + ci * stride + (stride * e if phase == "move" else 0)
        x_now = c.path[min(round(e * stride), stride)] if phase == "move" else c.path[0]
        info.set_text(f"cycle {ci + 1}/{len(cycles)}   t {step_k * m.dt:5.2f} s   "
                      f"vx {x_now[3]:4.2f} m/s   ESS {c.ess:4.0f}/{m.num_samples}")
        if not args.no_minimap:
            frame_box.set_bounds(win[0], win[2], win[1] - win[0], win[3] - win[2])
            mini_car.set_data([pose[0]], [pose[1]])

    out = args.out
    if out is None:
        ext = ".mp4" if writers.is_available("ffmpeg") else ".gif"
        out = cfg.track.npz_path.parent.parent / "videos" / f"{stem}_{'lap' if args.lap else 'mppi'}_k{k}{ext}"
    writer = "ffmpeg" if out.suffix == ".mp4" else "pillow"
    out.parent.mkdir(parents=True, exist_ok=True)
    anim = FuncAnimation(fig, update, frames=len(frames))   # pas de blit : les limites des axes bougent
    anim.save(out, writer=writer, fps=args.fps, dpi=args.dpi)
    plt.close(fig)
    print(f"video saved to {out} ({len(frames)} frames, {len(frames) / args.fps:.1f} s)")


if __name__ == "__main__":
    main()
