// Error checking and a small RAII owner for device memory.
//
// Every CUDA runtime call returns a cudaError_t that is silently ignored unless
// you check it. Kernel launches return nothing: check cudaGetLastError() right
// after the launch (bad configuration), and the result of the next
// synchronizing call (errors raised while the kernel ran).
#pragma once

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <utility>

#include <cuda_runtime.h>

// do { ... } while (0) makes the macro a single statement, safe after an `if`
// without braces. #call turns the expression into a string for the message.
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n  %s\n",                \
                         cudaGetErrorString(err_), __FILE__, __LINE__, #call);    \
            std::exit(EXIT_FAILURE);                                              \
        }                                                                         \
    } while (0)

namespace mppi {

// Owns n elements of T in GPU memory, freed when it goes out of scope (RAII).
// Movable, not copyable: two owners of the same pointer would free it twice.
template <class T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t n) : n_(n) { CUDA_CHECK(cudaMalloc(&ptr_, n * sizeof(T))); }
    ~DeviceBuffer() { if (ptr_) cudaFree(ptr_); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& o) noexcept { *this = std::move(o); }
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        std::swap(ptr_, o.ptr_);
        std::swap(n_, o.n_);
        return *this;
    }

    T* get() const { return ptr_; }
    std::size_t size() const { return n_; }

    void upload(const T* host) { CUDA_CHECK(cudaMemcpy(ptr_, host, n_ * sizeof(T), cudaMemcpyHostToDevice)); }
    void download(T* host) const { CUDA_CHECK(cudaMemcpy(host, ptr_, n_ * sizeof(T), cudaMemcpyDeviceToHost)); }

private:
    T* ptr_ = nullptr;
    std::size_t n_ = 0;
};

}  // namespace mppi
