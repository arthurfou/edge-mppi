// Cost map: load the binary written by python/mppi/track.py, look up a cell.
//
// TrackHost owns the grid in CPU memory. TrackView is what a rollout reads:
// a raw pointer plus the grid geometry, trivially copyable, passed by value to
// the kernel. The pointer is a host pointer on the CPU path and a device
// pointer on the GPU path; lookup() does not care.
#pragma once

#include <cmath>
#include <string>
#include <vector>

#include "mppi/hd.hpp"

namespace mppi {

struct TrackView {
    const float* grid;   // (ny, nx, 2) row-major: grid[(iy * nx + ix) * 2 + c], c = 0 d, 1 s
    int nx, ny;
    float x_min, y_min, res, length;
};

struct TrackHost {
    std::vector<float> grid;
    int nx = 0, ny = 0;
    float x_min = 0, y_min = 0, res = 0, length = 0;

    TrackView view() const { return {grid.data(), nx, ny, x_min, y_min, res, length}; }
};

// Reads results/track/costmap.bin. Throws std::runtime_error if the file is
// missing, has a bad header or is truncated.
TrackHost load_track(const std::string& bin_path);

// Nearest cell, indices clamped to the grid. Same as track.lookup in Python.
// floorf, not a cast: (int) truncates toward zero, wrong for x < x_min.
HD inline void lookup(const TrackView& tr, float x, float y, float& d, float& s) {
    int ix = static_cast<int>(floorf((x - tr.x_min) / tr.res));
    int iy = static_cast<int>(floorf((y - tr.y_min) / tr.res));
    ix = min(max(ix, 0), tr.nx - 1);
    iy = min(max(iy, 0), tr.ny - 1);
    const float* cell = tr.grid + (static_cast<long>(iy) * tr.nx + ix) * 2;
    d = cell[0];
    s = cell[1];
}

// Progress difference brought back to [-L/2, L/2). Python's % is a floored
// modulo (result has the sign of the divisor); fmodf truncates (sign of the
// dividend), hence the correction (Q-B2).
HD inline float wrap(float ds, float length) {
    float r = fmodf(ds + 0.5f * length, length);
    if (r < 0.0f) r += length;
    return r - 0.5f * length;
}

}  // namespace mppi
