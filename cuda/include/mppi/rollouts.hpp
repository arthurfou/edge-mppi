// The K rollout costs, on the CPU or on the GPU. Same function underneath
// (rollout_cost in cost.cuh), compiled twice.
//
// Plain C++ declarations: callable from a .cpp file compiled by g++.
// Layout of every array: row-major, as NumPy.
//   x0 (state_dim,), U (T, 2), eps (K, T, 2) raw noise, S (K,)
#pragma once

#include "mppi/config.hpp"
#include "mppi/track.cuh"

namespace mppi {

// All pointers are host pointers.
void rollouts_cpu(const float* x0, const float* U, const float* eps,
                  const TrackView& track, const Params& p, float* S);

// All pointers are DEVICE pointers (track.grid included). Asynchronous: returns
// as soon as the kernel is queued. The next cudaMemcpy waits for it.
void rollouts_gpu(const float* d_x0, const float* d_U, const float* d_eps,
                  const TrackView& d_track, const Params& p, float* d_S);

}  // namespace mppi
