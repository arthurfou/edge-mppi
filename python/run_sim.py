"""Closed-loop simulation with the NumPy controller.

From step 2 on, the simulated vehicle (the "plant") is always the dynamic
model. The controller rolls out the model named in the config: dynamic
(perfect model) or kinematic (model mismatch, theory 7.9).

Writes the trajectory plot and the time series to results/figures/, the
(state, control, next_state) triplets of the plant to results/trajectories/,
and the summary to results/logs/ (Markdown + JSON).
"""
import argparse
import time
from collections.abc import Callable
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
import report
from report import cell

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
    plant: Vehicle           # paramètres du véhicule simulé (peuvent différer de cfg.vehicle)


def observe(x: np.ndarray, cfg: Config) -> np.ndarray:
    """Plant state (6,) -> the state the controller works with (state_dim,).

    A kinematic controller gets the speed norm: it has no notion of vy or r.
    """
    if cfg.state_dim == STATE_DIM["dynamic"]:
        return x
    return np.array([x[0], x[1], x[2], np.hypot(x[3], x[4])])


def simulate(cfg: Config, track: trk.Track, plant: Vehicle | None = None,
             on_step: Callable[[float], None] | None = None) -> SimLog:
    """Runs MPPI in closed loop from a standstill on the start line, until one lap or max_steps.

    plant: parameters of the simulated vehicle, cfg.vehicle by default (perfect
    model). Passing other ones simulates a model error (part H).
    on_step: called with the progress (m) after each step, for a progress bar.
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
        if on_step is not None:
            on_step(progress)
        if progress >= track.length or not np.all(np.isfinite(x)):
            break

    return SimLog(np.array(states), np.array(controls), np.array(nexts), np.array(ess),
                  np.array(rho), np.array(d_log), np.array(t_iter), float(progress), beam, plant)


def chassis(log: SimLog) -> dict[str, np.ndarray]:
    """Slip angles, body slip and lateral acceleration of the plant along the run.

    Uses the plant's parameters, not the controller's: with a model error
    (part H, plant mu = 0.8) the forces are the real ones.
    """
    veh = log.plant
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


def summarize(log: SimLog, cfg: Config, track: trk.Track) -> dict:
    """The numbers of one run, as plain floats (printed, logged, compared by sweep.py)."""
    plant = log.plant
    d_max = cfg.cost.track_half_width - 0.5 * cfg.vehicle.width
    ch = chassis(log)
    fast = log.states[:, 3] > plant.blend_speed_high   # dérive sans objet en dessous
    deg = lambda k: float(np.degrees(np.abs(ch[k][fast]).max())) if fast.any() else 0.0
    p5, p50, p95 = np.percentile(log.ess, [5, 50, 95])
    return {
        "lap": bool(log.progress >= track.length),
        "t": len(log.states) * cfg.mppi.dt,
        "progress": log.progress,
        "length": track.length,
        "off": int((np.abs(log.d) > d_max).sum()),
        "dmax": float(np.abs(log.d).max()),
        "d_limit": d_max,
        "v_mean": float(ch["speed"].mean()),
        "v_max": float(ch["speed"].max()),
        "a_lat_max": float(np.abs(ch["a_lat"]).max()),
        "mu_g": plant.mu * G,
        "beta_max": deg("beta"),
        "alpha_f_max": deg("alpha_f"),
        "alpha_r_max": deg("alpha_r"),
        "alpha_lin": float(np.degrees(1.0 / plant.cornering_stiffness_front)),   # fin de la zone linéaire
        "ess_med": float(p50),
        "ess_p5": float(p5),
        "ess_p95": float(p95),
        "K": cfg.mppi.num_samples,
        "jitter": float(np.abs(np.diff(log.controls[:, 1])).mean()),   # rad/pas, théorie §7.5
        "t_iter_ms": float(1e3 * np.median(log.t_iter)),
    }


def summary_rows(runs: dict[str, dict]) -> list[report.Row]:
    """One column per controller model, metrics grouped by theme."""
    rs = list(runs.values())
    line = lambda label, f: [cell(label)] + [f(r) for r in rs]   # noqa: E731
    return [
        line("lap", report.lap_cell),
        line("off-track steps", lambda r: report.off_cell(r["off"])),
        line("max |d| (m)", lambda r: report.dmax_cell(r["dmax"], r["d_limit"])),
        None,
        line("speed mean / max (m/s)", lambda r: cell(f"{r['v_mean']:.2f} / {r['v_max']:.2f}")),
        line("max a_lat (m/s²)", lambda r: cell(f"{r['a_lat_max']:.1f} ({r['a_lat_max'] / r['mu_g']:.0%} of μg)",
                                                 report.BAD if r["a_lat_max"] > r["mu_g"] else "")),
        None,
        line("max |β| (deg)", lambda r: cell(f"{r['beta_max']:.1f}")),
        line("max |α_f| (deg)", lambda r: report.alpha_cell(r["alpha_f_max"], r["alpha_lin"])),
        line("max |α_r| (deg)", lambda r: report.alpha_cell(r["alpha_r_max"], r["alpha_lin"])),
        None,
        line("ESS median / p5", lambda r: report.ess_cell(r["ess_med"], r["ess_p5"], r["K"])),
        line("steering jitter (rad/step)", lambda r: cell(f"{r['jitter']:.3f}")),
        line("iteration time (ms)", lambda r: cell(f"{r['t_iter_ms']:.1f}")),
    ]


def describe(cfg: Config) -> str:
    """The settings shared by every run of a script, on one line."""
    m, v = cfg.mppi, cfg.vehicle
    return (f"v_ref {cfg.cost.v_ref} · λ {m.lam} · K {m.num_samples} · T {m.horizon} · "
            f"plant {v.tire_model} μ={v.mu} {v.integrator}×{v.substeps}")


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
    beta = np.degrees(chassis(log)["beta"])
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
    ch = chassis(log)
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
    report.header("run_sim", describe(cfg))

    runs, files = {}, []
    with report.progress() as bar:
        for model in models:
            c = with_model(cfg, model)
            label = f"[bold]{model}[/] rollouts"
            task = bar.add_task(label, total=track.length, info="")
            on_step = lambda p: bar.update(task, completed=min(p, track.length),   # noqa: E731
                                           info=f"{p:5.1f}/{track.length:.1f} m")
            log = simulate(c, track, on_step=on_step)
            bar.stop_task(task)   # le chrono ne compte que la simulation, pas les figures
            runs[model] = summarize(log, c, track)
            ok = runs[model]["lap"] and runs[model]["off"] == 0
            bar.update(task, description=f"{'[green]✔' if ok else '[red]✘'}[/] {label}")

            paths = (results / "trajectories" / f"step2_{model}.npz",
                     fig_dir / f"step2_{model}_trajectory.png", fig_dir / f"step2_{model}_series.png")
            save_trajectories(log, c, paths[0])
            plot_trajectory(log, c, track, paths[1])
            plot_series(log, c, paths[2])
            files += paths

    columns = ["", *(f"{m} rollouts" for m in runs)]
    rows = summary_rows(runs)
    report.print_table(columns, rows)
    md = f"{describe(cfg)}\n\n{report.markdown_table(columns, rows)}\n\n" + \
         "\n".join(f"- `{report.relative(f)}`" for f in files)
    log_md = report.write_log(results, "sim", "-".join(runs), md, {"config": cfg, "runs": runs})
    for f in files:
        report.console.print(f"  [dim]{report.relative(f)}[/]")
    report.done(f"log [bold]{report.relative(log_md)}[/] (+ .json)")


if __name__ == "__main__":
    main()
