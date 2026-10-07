"""Closed-loop simulation with the NumPy controller.

Writes the trajectory plot and the cost curve to results/figures/, and the
(state, control, next_state) triplets to results/trajectories/.
"""
import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # pas de fenêtre : le script tourne aussi en SSH
import matplotlib.pyplot as plt
import numpy as np

from mppi import track as trk
from mppi.config import Config, load_config
from mppi.controller import MPPI
from mppi.dynamics import step

BEAM_PROGRESS = 12.0   # m : on photographie le faisceau à l'entrée du premier virage
BEAM_SIZE = 200        # nombre de rollouts tracés, les plus lourds


@dataclass
class SimLog:
    states: np.ndarray       # (N, 4)  x_k
    controls: np.ndarray     # (N, 2)  u_k
    next_states: np.ndarray  # (N, 4)  x_{k+1} = step(x_k, u_k)
    ess: np.ndarray          # (N,)
    rho: np.ndarray          # (N,)    coût minimal de l'itération
    d: np.ndarray            # (N,)    écart latéral de x_{k+1}
    t_iter: np.ndarray       # (N,)    durée de command(), en secondes
    progress: float          # distance parcourue le long de la piste (m)
    beam: tuple[np.ndarray, np.ndarray] | None   # (X, w) d'une itération, pour la figure


def simulate(cfg: Config, track: trk.Track) -> SimLog:
    """Runs MPPI in closed loop from a standstill on the start line, until one lap or max_steps."""
    x = np.array([*track.centerline[0], track.heading[0], 0.0])
    ctrl = MPPI(cfg, track)

    states, controls, nexts, ess, rho, d_log, t_iter = [], [], [], [], [], [], []
    beam = None
    progress = 0.0
    _, s_prev = trk.lookup(track, x[0], x[1])

    for _ in range(cfg.max_steps):
        tic = time.perf_counter()
        u, info = ctrl.command(x)
        t_iter.append(time.perf_counter() - tic)   # on ne chronomètre que le contrôleur

        # Modèle parfait : le "vrai" système est le même modèle que celui des rollouts
        x_next = step(x, u, cfg.mppi.dt, cfg.vehicle)

        d, s = trk.lookup(track, x_next[0], x_next[1])
        progress += trk.wrap(s - s_prev, track.length)   # wrap : franchir la ligne ne fait pas -48 m
        s_prev = s

        states.append(x); controls.append(u); nexts.append(x_next)
        ess.append(info["ess"]); rho.append(info["rho"]); d_log.append(float(d))
        if beam is None and progress >= BEAM_PROGRESS:  # We take a picture for the plot once
            beam = (info["X"], info["w"])

        x = x_next
        if progress >= track.length:
            break

    return SimLog(np.array(states), np.array(controls), np.array(nexts), np.array(ess),
                  np.array(rho), np.array(d_log), np.array(t_iter), float(progress), beam)


def print_summary(log: SimLog, cfg: Config, track: trk.Track) -> None:
    d_max = cfg.cost.track_half_width - 0.5 * cfg.vehicle.width
    duration = len(log.states) * cfg.mppi.dt
    lap = log.progress >= track.length
    off = np.abs(log.d) > d_max
    v = log.states[:, 3]
    a_lat = v**2 * np.abs(np.tan(log.controls[:, 1])) / cfg.vehicle.wheelbase
    p5, p50, p95 = np.percentile(log.ess, [5, 50, 95])

    print(f"lap {'done' if lap else 'NOT done'} in {duration:.2f} s, "
          f"progress {log.progress:.1f}/{track.length:.1f} m")
    print(f"offtrack steps {off.sum()}, max |d| {np.abs(log.d).max():.3f} m (limit {d_max:.2f})")
    print(f"v mean {v.mean():.2f} max {v.max():.2f} m/s, max a_lat {a_lat.max():.1f} m/s2")
    print(f"ESS median {p50:.0f} p5 {p5:.0f} p95 {p95:.0f} / K={cfg.mppi.num_samples}")
    print(f"iteration time median {1e3 * np.median(log.t_iter):.1f} ms")


def save_trajectories(log: SimLog, cfg: Config, path: Path) -> None:
    """(state, control, next_state) triplets, reused in step 2 and for the learned model."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, state=log.states, control=log.controls, next_state=log.next_states,
             ess=log.ess, rho=log.rho, dt=cfg.mppi.dt)


def plot_trajectory(log: SimLog, cfg: Config, track: trk.Track, path: Path) -> None:
    """Track edges, driven path colored by speed, and the rollout beam at one instant."""
    hw = cfg.cost.track_half_width
    normal = np.column_stack([-np.sin(track.heading), np.cos(track.heading)])

    fig, ax = plt.subplots(figsize=(9, 6))
    # Encart : le faisceau ne fait que v_ref*T*dt = 1,8 m, illisible à l'échelle de la piste
    zoom = ax.inset_axes([0.35, 0.3, 0.3, 0.4]) if log.beam is not None else None

    for a in filter(None, (ax, zoom)):
        for side in (-1, 1):
            edge = track.centerline + side * hw * normal
            a.plot(*np.vstack([edge, edge[:1]]).T, "k", lw=1)   # edge[:1] referme la boucle
        a.plot(*track.centerline.T, "k--", lw=0.5)
        sc = a.scatter(log.states[:, 0], log.states[:, 1], c=log.states[:, 3], s=2, cmap="plasma")
        a.set_aspect("equal")

    if zoom is not None:
        X, w = log.beam
        for i in np.argsort(w)[-BEAM_SIZE:]:   # les plus lourds en dernier : tracés par-dessus
            zoom.plot(X[i, :, 0], X[i, :, 1], color=plt.cm.viridis(w[i] / w.max()), lw=0.5, alpha=0.6)
        lo, hi = X[..., :2].min(axis=(0, 1)) - 0.3, X[..., :2].max(axis=(0, 1)) + 0.3
        zoom.set_xlim(lo[0], hi[0])
        zoom.set_ylim(lo[1], hi[1])
        zoom.set_xticks([]); zoom.set_yticks([])
        zoom.set_title("rollouts, colored by weight", fontsize=8)
        ax.indicate_inset_zoom(zoom, edgecolor="gray")

    fig.colorbar(sc, ax=ax, label="v (m/s)")
    ax.set_title("Step 1, kinematic MPPI")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cost(log: SimLog, cfg: Config, path: Path) -> None:
    """rho, ESS, d and delta against time."""
    d_max = cfg.cost.track_half_width - 0.5 * cfg.vehicle.width
    t = np.arange(len(log.states)) * cfg.mppi.dt

    fig, axs = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
    axs[0].plot(t, log.rho)
    axs[0].set_ylabel("min cost")
    axs[1].plot(t, log.ess)
    axs[1].axhline(cfg.mppi.num_samples, color="gray", ls=":")
    axs[1].set_ylabel("ESS")
    axs[2].plot(t, log.d)
    for sign in (-1, 1):
        axs[2].axhline(sign * d_max, color="r")
    axs[2].set_ylabel("d (m)")
    axs[3].plot(t, log.controls[:, 1])
    axs[3].set_ylabel("delta (rad)")
    axs[3].set_xlabel("t (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/mppi.yaml")
    cfg = load_config(ap.parse_args().config)
    track = trk.load(cfg.track.npz_path)
    results = cfg.track.npz_path.parent.parent   # results/track/costmap.npz -> results/

    log = simulate(cfg, track)
    print_summary(log, cfg, track)

    fig_dir = results / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    save_trajectories(log, cfg, results / "trajectories" / "step1_kinematic.npz")
    plot_trajectory(log, cfg, track, fig_dir / "step1_trajectory.png")
    plot_cost(log, cfg, fig_dir / "step1_cost.png")
    print(f"figures in {fig_dir}")


if __name__ == "__main__":
    main()
