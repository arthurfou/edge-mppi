"""Regenerates results/track/costmap.{npz,bin} from the control points.

The outputs are gitignored. This script is the thing that is versioned.
"""
import argparse

import numpy as np

from mppi.config import load_config
from mppi import track as trk

CONTROL_POINTS = np.array([
    [0, 0], [8, 0], [12, 1], [14, 4], [12, 7], [8, 7],
    [5, 9], [1, 10], [-3, 9], [-5, 6], [-4, 2],
], dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/mppi.yaml")
    ap.add_argument("--plot", action="store_true", help="save the d and s channels as a png")
    args = ap.parse_args()
    cfg = load_config(args.config)
    tp, hw = cfg.track, cfg.cost.track_half_width

    cl, heading, s, length, curvature = trk.centerline_from_points(CONTROL_POINTS, tp.resample_step)
    r_min = 1.0 / np.abs(curvature).max()
    assert r_min > 2 * hw, f"min radius {r_min:.2f} m too tight for half width {hw} m"

    grid, x_min, y_min = trk.bake_grid(cl, heading, s, tp.grid_resolution, hw + tp.margin)
    track = trk.Track(grid, x_min, y_min, tp.grid_resolution, length, cl, heading)
    trk.save(track, tp.npz_path, tp.bin_path)

    g2, *_ = trk.load_bin(tp.bin_path)
    assert np.array_equal(g2, grid)
    print(f"length {length:.2f} m, R_min {r_min:.2f} m, grid {grid.shape[1]}x{grid.shape[0]}")

    if args.plot:
        png = tp.npz_path.with_suffix(".png")
        trk.plot_grid(track, png)
        print(f"figure saved to {png}")


if __name__ == "__main__":
    main()
