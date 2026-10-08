"""Step 2, part E: stiffness, stability, accuracy and cost of the integrators.

    pixi run python bench/integrators.py

Expected output: the tables of sections 5.2, 6.3, 6.4 and 6.5 of docs/step_2.md.
"""
import dataclasses
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from mppi import dynamics as dyn
from mppi.config import load_config

VEH = load_config(ROOT / "config" / "mppi.yaml").vehicle
DT = 0.02
VARIANTS = [("euler", 1), ("euler", 2), ("euler", 4), ("rk4", 1), ("rk4", 2)]


def eigenvalues():
    """Linearized (vy, r) system at constant vx, linear tires (theory 6.3). Table of 5.2."""
    fzf, fzr = dyn.axle_loads(VEH)
    cf = VEH.mu * VEH.cornering_stiffness_front * fzf
    cr = VEH.mu * VEH.cornering_stiffness_rear * fzr
    m, iz, lf, lr = VEH.mass, VEH.izz, VEH.lf, VEH.lr
    print(f"C_alpha front {cf:.1f} N/rad, rear {cr:.1f} N/rad, "
          f"yaw coefficient {(lf**2 * cf + lr**2 * cr) / iz:.1f}")
    for vx in (0.5, 1.0, 1.5, 3.0, 5.0, 8.0):
        a = np.array([[-(cf + cr) / (m * vx), -(lf * cf - lr * cr) / (m * vx) - vx],
                      [-(lf * cf - lr * cr) / (iz * vx), -(lf**2 * cf + lr**2 * cr) / (iz * vx)]])
        ev = np.linalg.eigvals(a)
        print(f"  vx {vx}: eigenvalues {np.round(ev, 1)}, "
              f"shortest time constant {1e3 / np.abs(ev.real).max():.0f} ms")


def unstable_below(integrator, substeps, tire):
    """Largest vx at which a yaw-rate perturbation does not decay, pure dynamic model. Table of 6.4."""
    # Bande de mélange abaissée : sinon le plancher vx_safe = max(vx, 1.0) masque l'instabilité
    veh = dataclasses.replace(VEH, integrator=integrator, substeps=substeps, tire_model=tire,
                              blend_speed_low=0.01, blend_speed_high=0.02)
    vxs = np.linspace(0.05, 3.0, 2951)
    s = np.zeros((len(vxs), 6))
    s[:, 3], s[:, 5] = vxs, 0.1
    with np.errstate(all="ignore"):
        for _ in range(100):
            s = dyn.step_dynamic_only(s, np.zeros((len(vxs), 2)), DT, veh)
        bad = ~(np.abs(s[:, 5]) < 0.1)        # NaN compte comme instable
    return vxs[bad].max() if bad.any() else 0.0


def accuracy():
    """Error after one horizon (30 steps) against RK4 with 200 sub-steps. Table of 6.3."""
    ref = dataclasses.replace(VEH, integrator="rk4", substeps=200)
    print("case (vx, delta, a) | " + " | ".join(f"{i} x{n}" for i, n in VARIANTS)
          + "   (position cm / max error on r rad/s)")
    for vx, delta, a in [(1.2, 0.3, 0.0), (2.0, 0.3, 0.0), (5.0, 0.1, 0.0), (5.0, 0.3, 0.0), (7.0, 0.4, -2.0)]:
        u = np.array([a, delta])

        def run(veh):
            s, rs = np.array([0, 0, 0, vx, 0, 0.0]), []
            for _ in range(30):
                s = dyn.step_dynamic(s, u, DT, veh)
                rs.append(s[5])
            return s, np.array(rs)

        s_ref, r_ref = run(ref)
        cells = []
        for integ, n in VARIANTS:
            s, r = run(dataclasses.replace(VEH, integrator=integ, substeps=n))
            cells.append(f"{100 * np.hypot(*(s[:2] - s_ref[:2])):.2f} / {np.abs(r - r_ref).max():.3f}")
        print(f"  ({vx}, {delta}, {a}) | " + " | ".join(cells))


def timing():
    """Cost of one blended step on a batch K = 1024. Table of 6.5."""
    rng = np.random.default_rng(0)
    x = np.zeros((1024, 6)); x[:, 3] = 5.0
    u = rng.normal(size=(1024, 2)) * [0.5, 0.1]
    for integ, n in VARIANTS:
        veh = dataclasses.replace(VEH, integrator=integ, substeps=n)
        tic = time.perf_counter()
        for _ in range(300):
            dyn.step_dynamic(x, u, DT, veh)
        print(f"  {integ} x{n}: {(time.perf_counter() - tic) / 300 * 1e3:.3f} ms per step, K=1024")


if __name__ == "__main__":
    print("--- 5.2 stiffness ---")
    eigenvalues()

    print("--- 6.4 stability ---")
    for integ, n in VARIANTS:
        print(f"  {integ} x{n}: unstable below {unstable_below(integ, n, 'linear'):.2f} m/s (linear), "
              f"{unstable_below(integ, n, 'pacejka'):.2f} m/s (pacejka)")
    z = np.linspace(-4.0, 0.0, 400001)
    g = 1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24   # facteur d'amplification de RK4 sur r' = -k r
    print(f"  RK4 real-axis limit k dt = {-z[np.abs(g) <= 1].min():.3f}")

    print("--- 6.3 accuracy ---")
    accuracy()

    print("--- 6.5 cost ---")
    timing()
