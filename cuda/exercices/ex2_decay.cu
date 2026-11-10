// Exercise 2: "one thread per rollout" in miniature.
// Thread k integrates dr/dt = -k_k r with explicit Euler for T steps (the yaw
// mode of step 2, section 5.2), k_k = 95.2 / vx_k with vx_k = 0.5 + 0.01 k.
// Writes one float per thread. The host checks against the closed form
// r_T = r0 (1 - k dt)^T, counts the unstable threads (|1 - k dt| > 1), and
// (Q-A4) runs the very same function on the CPU to compare bit for bit.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "mppi/cuda_check.hpp"

// __host__ __device__: compiled twice, once for the CPU and once for the GPU.
__host__ __device__ float decay_rate(int k) { return 95.2f / (0.5f + 0.01f * k); }

__host__ __device__ float decay_one(float rate, int T, float dt, float r0) {
    float r = r0;                       // a local variable: lives in a register
    for (int t = 0; t < T; ++t) r += dt * (-rate * r);
    return r;
}

__global__ void decay(int K, int T, float dt, float r0, float* r_final) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    r_final[k] = decay_one(decay_rate(k), T, dt, r0);   // the only global write
}

int main() {
    const int K = 1024, T = 30;
    const float dt = 0.02f, r0 = 0.1f;

    mppi::DeviceBuffer<float> d_r(K);
    decay<<<(K + 255) / 256, 256>>>(K, T, dt, r0, d_r.get());
    CUDA_CHECK(cudaGetLastError());
    std::vector<float> r_gpu(K);
    d_r.download(r_gpu.data());         // synchronizes: waits for the kernel

    std::vector<float> r_cpu(K);
    for (int k = 0; k < K; ++k) r_cpu[k] = decay_one(decay_rate(k), T, dt, r0);

    int unstable = 0, bit_diff = 0;
    double worst = 0.0, worst_cpu_gpu = 0.0;
    for (int k = 0; k < K; ++k) {
        const double rate = 95.2 / (0.5 + 0.01 * k);
        const double exact = r0 * std::pow(1.0 - rate * dt, T);
        worst = std::fmax(worst, std::fabs(r_gpu[k] - exact) / std::fmax(1e-12, std::fabs(exact)));
        unstable += std::fabs(1.0 - rate * dt) > 1.0;

        bit_diff += std::memcmp(&r_gpu[k], &r_cpu[k], sizeof(float)) != 0;
        const double d = std::fabs((double)r_gpu[k] - r_cpu[k]);
        worst_cpu_gpu = std::fmax(worst_cpu_gpu, d / std::fmax(1e-30, std::fabs((double)r_cpu[k])));
    }
    std::printf("GPU vs exact : max relative error %.1e\n", worst);
    std::printf("unstable     : %d threads (vx < 0.95 m/s)\n", unstable);
    std::printf("GPU vs CPU   : %d / %d differ bit-wise, max relative diff %.1e\n",
                bit_diff, K, worst_cpu_gpu);
    std::printf("examples     : k=0 r_T=%.3e | k=45 r_T=%.3e | k=%d r_T=%.3e\n",
                r_gpu[0], r_gpu[45], K - 1, r_gpu[K - 1]);
    return 0;
}
