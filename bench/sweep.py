"""Parameter experiments: part G of step 1, parts G and H of step 2.

    pixi run sweep H1                # one experiment, seed of the YAML
    pixi run sweep H1 H3 H5          # several
    pixi run sweep H3 H4 --seeds 5   # seeds 0 to 4, clean laps per variant
    pixi run sweep                   # all of them

Write down your prediction before running each experiment.

Each variant changes the controller (its config, rollout model included) and,
separately, the simulated vehicle (the plant). This tells "my model is wrong"
apart from "the world changed".

The table is also written to results/logs/sweep_<date>_<experiments>.md, with
the raw metrics and the base config in the .json next to it.
"""
import argparse
import dataclasses
import sys
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from mppi import track as trk
from mppi.config import Config, Vehicle, load_config
import report
from report import cell
from run_sim import describe, simulate, summarize, with_model

ConfigFn = Callable[[Config], Config]
# (nom, modification de la config du contrôleur, modification des paramètres du véhicule simulé)
Variant = tuple[str, ConfigFn, dict]


def with_mppi(**kw) -> ConfigFn:
    """Config -> copy of Config with some cfg.mppi fields replaced."""
    return lambda c: dataclasses.replace(c, mppi=dataclasses.replace(c.mppi, **kw))


def with_cost(**kw) -> ConfigFn:
    return lambda c: dataclasses.replace(c, cost=dataclasses.replace(c.cost, **kw))


def with_vehicle(**kw) -> ConfigFn:
    """Vehicle parameters used by the rollouts (the controller's model), not by the plant."""
    return lambda c: dataclasses.replace(c, vehicle=dataclasses.replace(c.vehicle, **kw))


def kinematic(c: Config) -> Config:
    return with_model(c, "kinematic")


def chain(*fs: ConfigFn) -> ConfigFn:
    """Applies several modifications in a row."""
    def apply(c: Config) -> Config:
        for f in fs:
            c = f(c)
        return c
    return apply


SAME = lambda c: c   # noqa: E731


def both(**kw) -> tuple[ConfigFn, dict]:
    """Same change in the rollouts and in the simulated vehicle: perfect model."""
    return with_vehicle(**kw), kw


EXPERIMENTS: dict[str, list[Variant]] = {
    # Étape 1. Avec le YAML de l'étape 2 (dynamique, v_ref = 7, λ = 3), ces lignes
    # ne reproduisent plus les chiffres de step_1.md.
    "G1": [(f"lambda={lam}", with_mppi(lam=lam), {}) for lam in (1.0, 0.3, 0.1, 0.03)],
    "G2": [("w_speed=0", with_cost(w_speed=0.0), {})],
    "G3": [(f"v_ref={v}", with_cost(v_ref=v), {}) for v in (5.0, 7.0)],
    "G4": [(f"sigma_delta={s}", with_mppi(noise_std=np.array([0.5, s])), {}) for s in (0.02, 0.3)],
    "G5": [("gamma=lambda", lambda c: with_mppi(gamma=c.mppi.lam)(c), {})],
    "G6": [("T=10, v_ref=5", chain(with_mppi(horizon=10), with_cost(v_ref=5.0)), {})],
    "G7": [("seed=1", with_mppi(seed=1), {})],
    # Étape 2, partie G : 8.1 premier tour à 3 m/s, 8.2 re-réglage de λ à 7 m/s
    "first-lap": [("v_ref=3, lambda=0.3", chain(with_cost(v_ref=3.0), with_mppi(lam=0.3)), {})],
    "lambda-scan": [(f"lambda={lam}", with_mppi(lam=lam), {}) for lam in (0.3, 1.0, 3.0, 10.0)],
    # Étape 2, partie H
    "H1": [(f"lambda={lam}", with_mppi(lam=lam), {}) for lam in (1.0, 10.0)],
    "H2": [(f"tires {t}, both", *both(tire_model=t)) for t in ("linear", "tanh")],
    "H3": [("rollouts linear, plant pacejka", with_vehicle(tire_model="linear"), {}),
           ("rollouts tanh, plant pacejka", with_vehicle(tire_model="tanh"), {})],
    "H4": [("plant mu=0.8", SAME, {"mu": 0.8}), ("mu=0.8, both", *both(mu=0.8))],
    "H5": [("euler, both", *both(integrator="euler")),
           ("rollouts euler, plant rk4", with_vehicle(integrator="euler"), {})],
    "H6": [("T=15", with_mppi(horizon=15), {}), ("T=50", with_mppi(horizon=50), {}),
           ("T=50, lambda=6", with_mppi(horizon=50, lam=6.0), {})],
    "H7": [(f"sigma_delta={s}", with_mppi(noise_std=np.array([0.5, s])), {}) for s in (0.05, 0.2)],
    "H8": [("v_ref=9", with_cost(v_ref=9.0), {}),
           ("v_ref=9, lambda=10", chain(with_cost(v_ref=9.0), with_mppi(lam=10.0)), {}),
           ("v_ref=9, lambda=10, T=50", chain(with_cost(v_ref=9.0), with_mppi(lam=10.0, horizon=50)), {})],
    "H9": [("kinematic rollouts", chain(kinematic, with_cost(w_adhesion=0.0)), {}),
           ("kinematic + adhesion", kinematic, {}),
           ("kinematic + adhesion, lambda=10", chain(kinematic, with_mppi(lam=10.0)), {})],
}


def run(job: tuple[Config, Vehicle]) -> dict:
    """One closed-loop lap of controller cfg on the simulated vehicle plant."""
    cfg, plant = job
    track = trk.load(cfg.track.npz_path)
    return summarize(simulate(cfg, track, plant), cfg, track)


COLUMNS = ["variant", "lap", "off", "|d|", "ESS", "jitter", "v", "β", "α f / r"]
CAPTION = ("lap: time, or progress if not done · off: off-track steps · |d|: max lateral offset (m) · "
           "ESS: median / p5 · jitter: mean |Δδ| (rad/step) · v: mean speed (m/s) · "
           "β, α: max above blend_speed_high (deg)")


def table_rows(variants: list[Variant], results: list[dict]) -> list[report.Row]:
    """One seed: every metric of every variant, one group of rows per experiment."""
    rows, group = [], None
    for (name, *_), r in zip(variants, results):
        if name.split()[0] != group:   # nouvelle expérience : séparateur
            if group is not None:
                rows.append(None)
            group = name.split()[0]
        rows.append([
            cell(name, "bold" if name == "reference" else ""),
            report.lap_cell(r),
            report.off_cell(r["off"]),
            report.dmax_cell(r["dmax"], r["d_limit"]),
            report.ess_cell(r["ess_med"], r["ess_p5"], r["K"]),
            cell(f"{r['jitter']:.3f}"),
            cell(f"{r['v_mean']:.2f}"),
            cell(f"{r['beta_max']:.1f}"),
            cell(f"{r['alpha_f_max']:.1f} / {r['alpha_r_max']:.1f}",
                 report.WARN if max(r["alpha_f_max"], r["alpha_r_max"]) > r["alpha_lin"] else ""),
        ])
    return rows


def seeds_rows(variants: list[Variant], results: list[dict], n_seeds: int) -> list[report.Row]:
    """Several seeds: does the variant stay on track? One group of rows per experiment."""
    rows, group = [], None
    for i, (name, *_) in enumerate(variants):
        if name.split()[0] != group:
            if group is not None:
                rows.append(None)
            group = name.split()[0]
        rows.append(report.seeds_row(name, results[i * n_seeds:(i + 1) * n_seeds],
                                     "bold" if name == "reference" else ""))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiments", nargs="*", help=f"among {', '.join(EXPERIMENTS)}, default: all")
    ap.add_argument("--config", default=str(ROOT / "config" / "mppi.yaml"))
    ap.add_argument("--seeds", type=int, default=1,
                    help="1: the YAML seed, one row per variant; N > 1: seeds 0 to N-1, clean laps")
    args = ap.parse_args()
    unknown = set(args.experiments) - set(EXPERIMENTS)
    if unknown:
        ap.error(f"unknown experiments {sorted(unknown)}, expected among {list(EXPERIMENTS)}")
    if args.seeds < 1:
        ap.error("--seeds must be >= 1")
    base = load_config(args.config)

    variants: list[Variant] = [("reference", SAME, {})]
    for name in args.experiments or EXPERIMENTS:
        variants += [(f"{name} {label}", f, p) for label, f, p in EXPERIMENTS[name]]

    # Un job par (variante, graine). La graine est appliquée après la variante (G7 la fixe déjà).
    seeds = [None] if args.seeds == 1 else range(args.seeds)
    jobs = []
    for _, f, p in variants:
        plant = dataclasses.replace(base.vehicle, **p)
        for s in seeds:
            cfg = f(base) if s is None else with_mppi(seed=s)(f(base))
            jobs.append((cfg, plant))

    names = args.experiments or list(EXPERIMENTS)
    seeds_text = "" if args.seeds == 1 else f" · seeds 0 to {args.seeds - 1}"
    report.header(f"sweep {' '.join(names)}", describe(base) + seeds_text)

    # Les runs sont indépendants : un processus par job, la barre avance à chaque run terminé
    results: list[dict] = [{}] * len(jobs)
    with report.job_progress() as bar, ProcessPoolExecutor() as pool:
        task = bar.add_task(f"[bold]{len(variants)}[/] variants, [bold]{len(jobs)}[/] closed-loop laps",
                            total=len(jobs))
        futures = {pool.submit(run, job): i for i, job in enumerate(jobs)}
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()
            bar.advance(task)

    if args.seeds == 1:
        columns, caption, rows = COLUMNS, CAPTION, table_rows(variants, results)
    else:
        columns, caption, rows = report.SEEDS_COLUMNS, report.SEEDS_CAPTION, seeds_rows(variants, results, args.seeds)
    report.print_table(columns, rows, caption)

    md = f"{describe(base)}{seeds_text}\n\n{report.markdown_table(columns, rows, caption)}"
    data = {"config": base, "seeds": args.seeds,
            "variants": [{"name": name, "plant": p, "runs": results[i * len(seeds):(i + 1) * len(seeds)]}
                         for i, (name, _, p) in enumerate(variants)]}
    log_md = report.write_log(ROOT / "results", "sweep", "-".join(names) if args.experiments else "all", md, data)
    report.done(f"log [bold]{report.relative(log_md)}[/] (+ .json)")


if __name__ == "__main__":
    main()
