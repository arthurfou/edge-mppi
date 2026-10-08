"""Step 2 exit figure: MPPI with kinematic rollouts vs MPPI with dynamic rollouts.

Same plant (the dynamic model), same track, same v_ref, same lambda, K, T,
Sigma and seeds. Only the model inside the rollouts changes (theory 7.9).

    pixi run compare                 # 5 seeds, figure + table

With --save, the table is also written to results/reports/compare_<date>_seeds5.md
(or _<NAME> with --save=NAME), with the raw metrics and the base config in the .json.
"""
import argparse
import dataclasses
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # pas de fenêtre : le script tourne aussi en SSH
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from mppi import track as trk
from mppi.config import Config, load_config
import report
from run_sim import SimLog, chassis, describe, draw_track, simulate, summarize, with_model

SEEDS = range(5)


def variants(base: Config) -> dict[str, Config]:
    """The three controllers compared. Only the rollout model and the adhesion patch differ."""
    kin = with_model(base, "kinematic")
    return {
        "dynamic": with_model(base, "dynamic"),
        "kinematic": dataclasses.replace(kin, cost=dataclasses.replace(base.cost, w_adhesion=0.0)),
        "kinematic + adhesion term": kin,   # le contrôleur de l'étape 1, rustine comprise
    }


def with_seed(cfg: Config, seed: int) -> Config:
    return dataclasses.replace(cfg, mppi=dataclasses.replace(cfg.mppi, seed=seed))


def run(cfg: Config) -> SimLog:
    """One closed-loop lap. Top-level function: it must be picklable for the process pool."""
    return simulate(cfg, trk.load(cfg.track.npz_path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config" / "mppi.yaml"))
    report.add_save_flag(ap)
    args = ap.parse_args()
    base = load_config(args.config)
    track = trk.load(base.track.npz_path)
    d_max = base.cost.track_half_width - 0.5 * base.vehicle.width
    cfgs = variants(base)

    seeds_text = f" · seeds 0 to {len(SEEDS) - 1}"
    report.header("compare", describe(base) + seeds_text)

    # 15 simulations indépendantes, la barre avance à chaque run terminé
    jobs = {(name, s): with_seed(cfg, s) for name, cfg in cfgs.items() for s in SEEDS}
    logs: dict[tuple[str, int], SimLog] = {}
    with report.job_progress() as bar, ProcessPoolExecutor() as pool:
        task = bar.add_task(f"[bold]{len(cfgs)}[/] controllers, [bold]{len(jobs)}[/] closed-loop laps",
                            total=len(jobs))
        futures = {pool.submit(run, cfg): key for key, cfg in jobs.items()}
        for fut in as_completed(futures):
            logs[futures[fut]] = fut.result()
            bar.advance(task)

    # Propre = tour fini ET aucun pas hors piste : run_sim dit "lap" même après une sortie
    runs = {name: [summarize(logs[name, s], jobs[name, s], track) for s in SEEDS] for name in cfgs}
    columns = ["controller", *report.SEEDS_COLUMNS[1:]]
    rows = [report.seeds_row(f"{name} rollouts", rs) for name, rs in runs.items()]
    report.print_table(columns, rows, report.SEEDS_CAPTION)

    # Figure : graine 0, une piste par contrôleur, points hors piste en rouge
    status = report.console.status("Drawing the comparison figure", spinner_style="cyan")
    status.start()
    fig, axs = plt.subplots(1, len(cfgs), figsize=(5.5 * len(cfgs), 4.2), layout="constrained")
    for ax, name in zip(axs, cfgs):
        log = logs[name, 0]
        draw_track(ax, track, base.cost.track_half_width)
        speed = chassis(log)["speed"]
        sc = ax.scatter(log.states[:, 0], log.states[:, 1], c=speed, s=2, cmap="plasma",
                        vmin=0, vmax=base.cost.v_ref + 1)
        off = np.abs(log.d) > d_max   # d est celui de next_states (SimLog)
        ax.scatter(log.next_states[off, 0], log.next_states[off, 1], s=8, c="red", label="off track")
        ax.set_title(f"{name} rollouts\n"
                     f"{off.sum()} off-track steps, lap {len(log.states) * base.mppi.dt:.2f} s (seed 0)",
                     fontsize=10)
        ax.legend(loc="lower right", fontsize=8)
    fig.colorbar(sc, ax=axs, label="speed (m/s)", shrink=0.8)
    fig.suptitle(f"Same dynamic plant, v_ref = {base.cost.v_ref} m/s, lambda = {base.mppi.lam}")
    out = base.track.npz_path.parent.parent / "figures" / "step2_comparison.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    status.stop()

    md = (f"{describe(base)}{seeds_text}\n\n"
          f"{report.markdown_table(columns, rows, report.SEEDS_CAPTION)}\n\n"
          f"- `{report.relative(out)}`")
    data = {"config": base, "seeds": list(SEEDS),
            "controllers": {name: {"model": cfgs[name].model, "w_adhesion": cfgs[name].cost.w_adhesion,
                                   "runs": rs} for name, rs in runs.items()}}
    report.console.print(f"  [dim]{report.relative(out)}[/]")
    report.save_report(args.save, ROOT / "results", "compare", f"seeds{len(SEEDS)}", md, data)


if __name__ == "__main__":
    main()
