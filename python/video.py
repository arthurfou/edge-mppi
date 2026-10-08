"""Video of a trajectory saved by run_sim.py (results/trajectories/*.npz).

One frame per control step, played in real time (fps = 1/dt). The track is
drawn once; each frame only moves the car outline and extends its trail.
Writes results/videos/<name>.mp4 (a .gif if ffmpeg is missing).
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, writers

from mppi import track as trk
from mppi.config import load_config
from run_sim import draw_track


def car_outline(x: float, y: float, psi: float, length: float, width: float) -> np.ndarray:
    """The 4 corners of the car body, centered on (x, y), heading psi."""
    corners = 0.5 * np.array([[length, width], [-length, width], [-length, -width], [length, -width]])
    c, s = np.cos(psi), np.sin(psi)
    return corners @ np.array([[c, s], [-s, c]]) + (x, y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trajectory", nargs="?", default="results/trajectories/step2_dynamic.npz")
    ap.add_argument("--config", default="config/mppi.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    track = trk.load(cfg.track.npz_path)
    data = np.load(args.trajectory)
    states, dt = data["state"], float(data["dt"])
    v = cfg.vehicle

    fig, ax = plt.subplots(figsize=(8, 5.5))
    draw_track(ax, track, cfg.cost.track_half_width)
    trail, = ax.plot([], [], "tab:blue", lw=1.5)
    car = plt.Polygon(car_outline(*states[0, :3], v.wheelbase, v.width), color="tab:red")
    ax.add_patch(car)
    label = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top", family="monospace")
    ax.set_title(Path(args.trajectory).stem)
    fig.tight_layout()

    def update(k):
        x, y, psi, vx = states[k, :4]
        car.set_xy(car_outline(x, y, psi, v.wheelbase, v.width))
        trail.set_data(states[:k + 1, 0], states[:k + 1, 1])
        label.set_text(f"t {k * dt:5.2f} s   vx {vx:4.2f} m/s")
        return car, trail, label

    anim = FuncAnimation(fig, update, frames=len(states), blit=True)
    ext, writer = (".mp4", "ffmpeg") if writers.is_available("ffmpeg") else (".gif", "pillow")
    out = cfg.track.npz_path.parent.parent / "videos" / (Path(args.trajectory).stem + ext)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out, writer=writer, fps=round(1 / dt), dpi=100)
    plt.close(fig)
    print(f"video saved to {out}")


if __name__ == "__main__":
    main()
