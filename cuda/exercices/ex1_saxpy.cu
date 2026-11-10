// Exercise 1: y = a * x + y on N floats, one thread per element.
//
// Error experiments (step_3.md 2.1, point 2), one at a time:
//   -DBIG_BLOCK    block = 2048 (> 1024, the hardware limit)
//   -DNO_GUARD     grid = n, and no `if (i < n)` in the kernel
//   -DBYTES_BUG    cudaMemcpy back with n instead of n * sizeof(float)
// e.g. pixi run nvcc -std=c++17 -arch=sm_89 -Icuda/include -DNO_GUARD \
//        -o build/ex1 cuda/exercices/ex1_saxpy.cu
#include <cstdio>
#include <vector>

#include "mppi/cuda_check.hpp"

__global__ void saxpy(int n, float a, const float* x, float* y) {
    // Global index: which block I am in, times the block size, plus my rank in it.
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
#ifdef NO_GUARD
    y[i] = a * x[i] + y[i];          // threads i >= n write past the end of y
#else
    if (i < n) y[i] = a * x[i] + y[i];
#endif
}

int main() {
    const int n = 1000000;
    // A different value in every cell, so a thread writing to the wrong index
    // is caught. Every value stays below 2^24, where floats are exact integers.
    std::vector<float> x(n), y(n, 2.0f);
    for (int i = 0; i < n; ++i) x[i] = (float)i;

    float *d_x = nullptr, *d_y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, y.data(), n * sizeof(float), cudaMemcpyHostToDevice));

#ifdef BIG_BLOCK
    const int block = 2048;
#else
    const int block = 256;
#endif
#ifdef NO_GUARD
    const int grid = n;                          // n blocks of 256 threads: far too many
#else
    const int grid = (n + block - 1) / block;    // round up so every element is covered
#endif
    std::printf("grid = %d blocks x %d threads = %lld threads for n = %d\n",
                grid, block, (long long)grid * block, n);

    saxpy<<<grid, block>>>(n, 3.0f, d_x, d_y);
    CUDA_CHECK(cudaGetLastError());              // launch errors (bad configuration)

    // cudaMemcpy waits for the kernel, so errors raised while it ran show up here.
#ifdef BYTES_BUG
    CUDA_CHECK(cudaMemcpy(y.data(), d_y, n, cudaMemcpyDeviceToHost));   // n bytes = n/4 floats
#else
    CUDA_CHECK(cudaMemcpy(y.data(), d_y, n * sizeof(float), cudaMemcpyDeviceToHost));
#endif

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));

    // Expected y[i] = 3 i + 2, exact in float (at most ~3e6 < 2^24 = 16.7e6).
    int wrong = 0;
    for (int i = 0; i < n; ++i) wrong += (y[i] != 3.0f * i + 2.0f);
    std::printf("y[0] = %.1f, y[n-1] = %.1f, %d wrong values\n", y[0], y[n - 1], wrong);
    return wrong == 0 ? 0 : 1;
}
