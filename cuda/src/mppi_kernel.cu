// Step 3: one thread per trajectory, horizon loop inside the kernel, cost
// accumulated on the fly, one float written per thread. Correctness first.
// Step 4 fuses rollout and cost and keeps the state in registers.
#include "mppi/cost.cuh"
#include "mppi/cuda_check.hpp"
#include "mppi/rollouts.hpp"

namespace mppi {
namespace {

constexpr int BLOCK = 256;   // threads per block, a multiple of the warp size (32)

// TrackView and Params are passed BY VALUE: the launch copies them into the
// kernel's parameter space, so the GPU never dereferences a host address.
template <class Model>
__global__ void rollout_kernel(const float* x0, const float* U, const float* eps,
                               TrackView track, Params p, float* S) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;   // global thread index = rollout
    if (k >= p.K) return;                                   // last block may overhang
    S[k] = rollout_cost<Model>(k, x0, U, eps, track, p);
}

// The CPU twin: the same rollout_cost, in a for loop instead of K threads
template <class Model>
void cpu_loop(const float* x0, const float* U, const float* eps,
              const TrackView& track, const Params& p, float* S) {
    for (int k = 0; k < p.K; ++k) S[k] = rollout_cost<Model>(k, x0, U, eps, track, p);
}

}  // namespace

void rollouts_cpu(const float* x0, const float* U, const float* eps,
                  const TrackView& track, const Params& p, float* S) {
    if (p.model == ModelKind::Dynamic) cpu_loop<Dynamic>(x0, U, eps, track, p, S);
    else cpu_loop<Kinematic>(x0, U, eps, track, p, S);
}

void rollouts_gpu(const float* d_x0, const float* d_U, const float* d_eps,
                  const TrackView& d_track, const Params& p, float* d_S) {
    const int grid = (p.K + BLOCK - 1) / BLOCK;   // ceil(K / BLOCK)
    if (p.model == ModelKind::Dynamic)
        rollout_kernel<Dynamic><<<grid, BLOCK>>>(d_x0, d_U, d_eps, d_track, p, d_S);
    else
        rollout_kernel<Kinematic><<<grid, BLOCK>>>(d_x0, d_U, d_eps, d_track, p, d_S);
    CUDA_CHECK(cudaGetLastError());   // launch errors (bad grid, no kernel image for this GPU)
}

}  // namespace mppi
