// Step 0 sanity check: the toolchain compiles, links and reaches the device.

#include <cstdio>
#include <cuda_runtime.h>

__global__ void probe(int* out) {
    out[threadIdx.x] = threadIdx.x;
}

int main() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count == 0) {
        std::printf("no CUDA device: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    std::printf("device 0: %s (sm_%d%d, %d SMs)\n",
                prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    int* d_out = nullptr;
    cudaMalloc(&d_out, 32 * sizeof(int));
    probe<<<1, 32>>>(d_out);
    cudaDeviceSynchronize();

    int h_out[32] = {};
    cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost);
    cudaFree(d_out);

    std::printf("kernel ok: last lane = %d\n", h_out[31]);
    return h_out[31] == 31 ? 0 : 1;
}
