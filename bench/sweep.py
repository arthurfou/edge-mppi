"""Part G of step 1: one parameter changed at a time, closed-loop metrics side by side.

    pixi run sweep G1          # one experiment
    pixi run sweep G1 G3 G5    # several
    pixi run sweep             # all of them

Write down your prediction before running each experiment.
"""
import argparse
import dataclasses
import sys
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from mppi import track as trk
from mppi.config import Config, load_config
from run_sim import simulate

Variant = tuple[str, Callable[[Config], Config]]


def with_mppi(**kw) -> Callable[[Config], Config]:
    """Config -> copy of Config with some cfg.mppi fields replaced."""
    return lambda c: dataclasses.replace(c, mppi=dataclasses.replace(c.mppi, **kw))


def with_cost(**kw) -> Callable[[Config], Config]:
    return lambda c: dataclasses.replace(c, cost=dataclasses.replace(c.cost, **kw))


def chain(*fs: Callable[[Config], Config]) -> Callable[[Config], Config]:
    """Applies several modifications in a row (G6 changes T and v_ref)."""
    def apply(c: Config) -> Config:
        for f in fs:
            c = f(c)
        return c
    return apply


EXPERIMENTS: dict[str, list[Variant]] = {
    "G1": [(f"lambda={lam}", with_mppi(lam=lam)) for lam in (1.0, 0.3, 0.1, 0.03)],
    "G2": [("w_speed=0", with_cost(w_speed=0.0))],
    "G3": [(f"v_ref={v}", with_cost(v_ref=v)) for v in (5.0, 7.0)],
    "G4": [(f"sigma_delta={s}", with_mppi(noise_std=np.array([0.5, s]))) for s in (0.02, 0.3)],
    "G5": [("gamma=lambda", lambda c: with_mppi(gamma=c.mppi.lam)(c))],
    "G6": [("T=10, v_ref=5", chain(with_mppi(horizon=10), with_cost(v_ref=5.0)))],
    "G7": [("seed=1", with_mppi(seed=1))],
}


def run(cfg: Config) -> dict[str, float | bool]:
    """One closed-loop lap, summarized in a few numbers."""
    track = trk.load(cfg.track.npz_path)
    log = simulate(cfg, track)
    d_max = cfg.cost.track_half_width - 0.5 * cfg.vehicle.width
    return {
        "lap": log.progress >= track.length,
        "t": len(log.states) * cfg.mppi.dt,
        "progress": log.progress,
        "ess_med": float(np.median(log.ess)),
        "ess_p5": float(np.percentile(log.ess, 5)),
        "jitter": float(np.abs(np.diff(log.controls[:, 1])).mean()),   # rad/pas, théorie §7.5
        "dmax": float(np.abs(log.d).max()),
        "off": int((np.abs(log.d) > d_max).sum()),
        "v_mean": float(log.states[:, 3].mean()),
    }


def print_row(name: str, r: dict[str, float | bool]) -> None:
    lap = "yes" if r["lap"] else f"NO ({r['progress']:.1f} m)"
    print(f"| {name} | {lap} | {r['t']:.2f} | {r['ess_med']:.0f} | {r['ess_p5']:.0f} "
          f"| {r['jitter']:.3f} | {r['dmax']:.2f} | {r['off']} | {r['v_mean']:.2f} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiments", nargs="*", help=f"among {', '.join(EXPERIMENTS)}, default: all")
    ap.add_argument("--config", default=str(ROOT / "config" / "mppi.yaml"))
    args = ap.parse_args()
    unknown = set(args.experiments) - set(EXPERIMENTS)
    if unknown:
        ap.error(f"unknown experiments {sorted(unknown)}, expected among {list(EXPERIMENTS)}")
    base = load_config(args.config)

    variants: list[Variant] = [("reference", lambda c: c)]
    for name in args.experiments or EXPERIMENTS:
        variants += [(f"{name} {label}", f) for label, f in EXPERIMENTS[name]]

    # Les runs sont indépendants : un processus par variante
    with ProcessPoolExecutor() as pool:
        results = pool.map(run, [f(base) for _, f in variants])
        print("| variant | lap | t (s) | ESS med | ESS p5 | jitter | max abs(d) | off | v mean |")
        print("|---|---|---|---|---|---|---|---|---|")
        for (name, _), r in zip(variants, results):
            print_row(name, r)


if __name__ == "__main__":
    main()
