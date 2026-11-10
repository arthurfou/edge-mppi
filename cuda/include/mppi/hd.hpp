// HD marks a function compiled for both the CPU and the GPU.
//
// nvcc understands __host__ __device__. A plain C++ compiler (g++ on a .cpp
// file) does not, so the macro expands to nothing there and the header stays
// includable from host-only code. On the host side, min/max/isfinite come from
// the standard library; in device code CUDA provides them as builtins.
#pragma once

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#include <algorithm>
#include <cmath>
using std::isfinite;
using std::max;
using std::min;
#endif
