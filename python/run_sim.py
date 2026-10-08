"""Closed-loop simulation with the NumPy controller.

From step 2 on, the simulated vehicle (the "plant") is always the dynamic
model. The controller rolls out the model named in the config: dynamic
(perfect model) or kinematic (model mismatch, theory 7.9).

Writes the trajectory plot and the time series to results/figures/, and the
(state, control, next_state) triplets of the plant to results/trajectories/.
"""
import argparse
import time
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # pas de fenêtre : le script tourne aussi en SSH
import matplotlib.pyplot as plt
import numpy as np

from mppi import track as trk
from mppi.config import STATE_DIM, Config, Vehicle, load_config
from mppi.controller import MPPI
from mppi.dynamics import G, axle_loads, slip_angles, step_dynamic, tire_force

BEAM_PROGRESS = 12.0   # m : on photographie le faisceau à l'entrée du premier virage
BEAM_SIZE = 200        # nombre de rollouts tracés, les plus lourds


@dataclass
class SimLog:
    states: np.ndarray       # (N, 6)  x_k du véhicule simulé, toujours dynamique
    controls: np.ndarray     # (N, 2)  u_k
    next_states: np.ndarray  # (N, 6)  x_{k+1}
    ess: np.ndarray          # (N,)
    rho: np.ndarray          # (N,)    coût minimal de l'itération
    d: np.ndarray            # (N,)    écart latéral de x_{k+1}
    t_iter: np.ndarray       # (N,)    durée de command(), en secondes
    progress: float          # distance parcourue le long de la piste (m)
    beam: tuple[np.ndarray, np.ndarray] | None   # (X, w) d'une itération, pour la figure


def observe(x: np.ndarray, cfg: Config) -> np.ndarray:
    """Plant state (6,) -> the state the controller works with (state_dim,).

    A kinematic controller gets the speed norm: it has no notion of vy or r.
    """
    if cfg.state_dim == STATE_DIM["dynamic"]:
        return x
    return np.array([x[0], x[1], x[2], np.hypot(x[3], x[4])])


def simulate(cfg: Config, track: trk.Track, plant: Vehicle | None = None) -> SimLog:
    """Runs MPPI in closed loop from a standstill on the start line, until one lap or max_steps.

    plant: parameters of the simulated vehicle, cfg.vehicle by default (perfect
    model). Passing other ones simulates a model error (part H).
    """
    plant = cfg.vehicle if plant is None else plant
    x = np.array([*track.centerline[0], track.heading[0], 0.0, 0.0, 0.0])
    ctrl = MPPI(cfg, track)

    states, controls, nexts, ess, rho, d_log, t_iter = [], [], [], [], [], [], []
    beam = None
    progress = 0.0
    _, s_prev = trk.lookup(track, x[0], x[1])

    for _ in range(cfg.max_steps):
        tic = time.perf_counter()
        u, info = ctrl.command(observe(x, cfg))
        t_iter.append(time.perf_counter() - tic)   # on ne chronomètre que le contrôleur

        # Le "vrai" système est le modèle dynamique, quel que soit le modèle des rollouts
        x_next = step_dynamic(x, u, cfg.mppi.dt, plant)

        d, s = trk.lookup(track, x_next[0], x_next[1])
        progress += trk.wrap(s - s_prev, track.length)   # wrap : franchir la ligne ne fait pas -48 m
        s_prev = s

        states.append(x); controls.append(u); nexts.append(x_next)
        ess.append(info["ess"]); rho.append(info["rho"]); d_log.append(float(d))
        if beam is None and progress >= BEAM_PROGRESS:
            beam = (info["X"], info["w"])

        x = x_next
        if progress >= track.length or not np.all(np.isfinite(x)):
            break

    return SimLog(np.array(states), np.array(controls), np.array(nexts), np.array(ess),
                  np.array(rho), np.array(d_log), np.array(t_iter), float(progress), beam)


def chassis(log: SimLog, cfg: Config) -> dict[str, np.ndarray]:
    """Slip angles, body slip and lateral acceleration of the plant along the run."""
    veh = cfg.vehicle
    alpha_f, alpha_r = slip_angles(log.states, log.controls[:, 1], veh)
    fzf, fzr = axle_loads(veh)
    fyf = tire_force(alpha_f, fzf, veh.cornering_stiffness_front, veh)
    fyr = tire_force(alpha_r, fzr, veh.cornering_stiffness_rear, veh)
    vx, vy = log.states[:, 3], log.states[:, 4]
    return {
        "alpha_f": alpha_f, "alpha_r": alpha_r,
        "beta": np.arctan2(vy, np.maximum(vx, 1e-3)),            # dérive du véhicule
        "a_lat": (fyf * np.cos(log.controls[:, 1]) + fyr) / veh.mass,
        "speed": np.hypot(vx, vy),
    }


def print_summary(log: SimLog, cfg: Config, track: trk.Track) -> None:
    d_max = cfg.cost.track_half_width - 0.5 * cfg.vehicle.width
    duration = len(log.states) * cfg.mppi.dt
    lap = log.progress >= track.length
    off = np.abs(log.d) > d_max
    ch = chassis(log, cfg)
    p5, p50, p95 = np.percentile(log.ess, [5, 50, 95])

    print(f"controller model {cfg.model}, plant dynamic ({cfg.vehicle.tire_model} tires)")
    print(f"lap {'done' if lap else 'NOT done'} in {duration:.2f} s, "
          f"progress {log.progress:.1f}/{track.length:.1f} m")
    print(f"offtrack steps {off.sum()}, max |d| {np.abs(log.d).max():.3f} m (limit {d_max:.2f})")
    print(f"v mean {ch['speed'].mean():.2f} max {ch['speed'].max():.2f} m/s, "
          f"max a_lat {np.abs(ch['a_lat']).max():.1f} m/s2 (mu g = {cfg.vehicle.mu * G:.1f})")
    fast = log.states[:, 3] > cfg.vehicle.blend_speed_high   # dérive sans objet en dessous
    deg = lambda k: np.degrees(np.abs(ch[k][fast]).max()) if fast.any() else 0.0
    print(f"above {cfg.vehicle.blend_speed_high} m/s: max |beta| {deg('beta'):.1f} deg, "
          f"max |alpha_f| {deg('alpha_f'):.1f} deg, max |alpha_r| {deg('alpha_r'):.1f} deg")
    print(f"ESS median {p50:.0f} p5 {p5:.0f} p95 {p95:.0f} / K={cfg.mppi.num_samples}")
    print(f"iteration time median {1e3 * np.median(log.t_iter):.1f} ms")


def save_trajectories(log: SimLog, cfg: Config, path: Path) -> None:
    """(state, control, next_state) triplets of the plant, reused for the learned model."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, state=log.states, control=log.controls, next_state=log.next_states,
             ess=log.ess, rho=log.rho, dt=cfg.mppi.dt)


def draw_track(ax, track: trk.Track, hw: float) -> None:
    normal = np.column_stack([-np.sin(track.heading), np.cos(track.heading)])
    for side in (-1, 1):
        edge = track.centerline + side * hw * normal
        ax.plot(*np.vstack([edge, edge[:1]]).T, "k", lw=1)   # edge[:1] referme la boucle
    ax.plot(*track.centerline.T, "k--", lw=0.5)
    ax.set_aspect("equal")


def plot_trajectory(log: SimLog, cfg: Config, track: trk.Track, path: Path) -> None:
    """Track edges, driven path colored by body slip angle, and the rollout beam at one instant."""
    fig, ax = plt.subplots(figsize=(9, 6))
    zoom = ax.inset_axes([0.35, 0.3, 0.3, 0.4]) if log.beam is not None else None
    beta = np.degrees(chassis(log, cfg)["beta"])
    lim = max(np.abs(beta).max(), 1.0)

    for a in filter(None, (ax, zoom)):
        draw_track(a, track, cfg.cost.track_half_width)
        sc = a.scatter(log.states[:, 0], log.states[:, 1], c=beta, s=2,
                       cmap="coolwarm", vmin=-lim, vmax=lim)

    if zoom is not None:
        X, w = log.beam
        for i in np.argsort(w)[-BEAM_SIZE:]:
            zoom.plot(X[i, :, 0], X[i, :, 1], color=plt.cm.viridis(w[i] / w.max()), lw=0.5, alpha=0.6)
        lo, hi = X[..., :2].min(axis=(0, 1)) - 0.3, X[..., :2].max(axis=(0, 1)) + 0.3
        zoom.set_xlim(lo[0], hi[0]); zoom.set_ylim(lo[1], hi[1])
        zoom.set_xticks([]); zoom.set_yticks([])
        zoom.set_title("rollouts, colored by weight", fontsize=8)
        ax.indicate_inset_zoom(zoom, edgecolor="gray")

    fig.colorbar(sc, ax=ax, label="body slip beta (deg)")
    ax.set_title(f"Step 2, MPPI with {cfg.model} rollouts, dynamic plant")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_series(log: SimLog, cfg: Config, path: Path) -> None:
    """ESS, d, speed, yaw rate, slip angles and steering against time."""
    d_max = cfg.cost.track_half_width - 0.5 * cfg.vehicle.width
    t = np.arange(len(log.states)) * cfg.mppi.dt
    ch = chassis(log, cfg)
    alpha_sat = np.degrees(1.0 / cfg.vehicle.cornering_stiffness_front)   # fin de la zone linéaire

    fig, axs = plt.subplots(6, 1, figsize=(9, 13), sharex=True)
    axs[0].plot(t, log.ess); axs[0].set_ylabel("ESS")
    axs[0].axhline(cfg.mppi.num_samples, color="gray", ls=":")
    axs[1].plot(t, log.d)
    for sign in (-1, 1):
        axs[1].axhline(sign * d_max, color="r")
    axs[1].set_ylabel("d (m)")
    axs[2].plot(t, log.states[:, 3], label="vx")
    axs[2].plot(t, log.states[:, 4], label="vy")
    axs[2].axhline(cfg.cost.v_ref, color="gray", ls=":")
    axs[2].set_ylabel("m/s"); axs[2].legend(loc="upper right")
    axs[3].plot(t, log.states[:, 5]); axs[3].set_ylabel("r (rad/s)")
    axs[4].plot(t, np.degrees(ch["alpha_f"]), label="alpha_f")
    axs[4].plot(t, np.degrees(ch["alpha_r"]), label="alpha_r")
    axs[4].plot(t, np.degrees(ch["beta"]), label="beta", lw=0.8)
    for sign in (-1, 1):
        axs[4].axhline(sign * alpha_sat, color="gray", ls=":")
    axs[4].set_ylabel("deg"); axs[4].legend(loc="upper right")
    axs[5].plot(t, log.controls[:, 1]); axs[5].set_ylabel("delta (rad)")
    axs[5].set_xlabel("t (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def with_model(cfg: Config, model: str) -> Config:
    """Same config with another rollout model. Everything else is shared (protocol 0.3)."""
    return replace(cfg, model=model, state_dim=STATE_DIM[model])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/mppi.yaml")
    ap.add_argument("--model", choices=(*STATE_DIM, "both"), default=None,
                    help="rollout model, the YAML one by default; the plant is always dynamic")
    args = ap.parse_args()
    cfg = load_config(args.config)
    track = trk.load(cfg.track.npz_path)
    results = cfg.track.npz_path.parent.parent   # results/track/costmap.npz -> results/
    fig_dir = results / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    models = ("dynamic", "kinematic") if args.model == "both" else (args.model or cfg.model,)
    for model in models:
        c = with_model(cfg, model)
        log = simulate(c, track)
        print_summary(log, c, track)
        save_trajectories(log, c, results / "trajectories" / f"step2_{model}.npz")
        plot_trajectory(log, c, track, fig_dir / f"step2_{model}_trajectory.png")
        plot_series(log, c, fig_dir / f"step2_{model}_series.png")
        print()
    print(f"figures in {fig_dir}")


if __name__ == "__main__":
    main()
