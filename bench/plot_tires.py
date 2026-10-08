"""Checkpoint C of step 2: lateral force of the front tire against its slip angle.

    pixi run tires

The three models must share the slope at the origin, and tanh / Pacejka must
peak at mu * F_z. Compare with the table of section 4.4.
"""
import dataclasses
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # pas de fenêtre, on écrit seulement le PNG
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from mppi import dynamics as dyn
from mppi.config import load_config

COLORS = {"linear": "#2a78d6", "tanh": "#eb6834", "pacejka": "#1baf7a"}


def main():
    veh = load_config(ROOT / "config" / "mppi.yaml").vehicle
    fzf, _ = dyn.axle_loads(veh)
    c_s = veh.cornering_stiffness_front
    f_max = veh.mu * fzf

    alpha = np.linspace(-np.pi / 2, np.pi / 2, 721)
    fig, ax = plt.subplots(figsize=(8, 5))
    for tire, color in COLORS.items():
        f = dyn.tire_force(alpha, fzf, c_s, dataclasses.replace(veh, tire_model=tire))
        ax.plot(np.degrees(alpha), f, color=color, lw=2, label=tire)

    for sign in (-1, 1):
        ax.axhline(sign * f_max, color="gray", ls=":", lw=1)                   # limite d'adhérence
        ax.axvline(sign * np.degrees(1 / c_s), color="gray", ls="--", lw=1)   # linéaire atteint mu*F_z
    ax.text(-88, f_max + 0.8, r"$\mu F_z$ = %.2f N" % f_max, color="dimgray", fontsize=9)

    ax.set_ylim(-1.6 * f_max, 1.6 * f_max)   # le linéaire sort du cadre : il ne sature jamais
    ax.set_xlim(-90, 90)
    ax.set_xlabel(r"slip angle $\alpha$ (deg)")
    ax.set_ylabel(r"lateral force $F_y$ (N)")
    ax.set_title(f"Front tire, F_z = {fzf:.2f} N, C_S = {c_s}, Pacejka C = {veh.pacejka_c}")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")

    path = ROOT / "results" / "figures" / "step2_tire_curves.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"figure in {path}")


if __name__ == "__main__":
    main()
