// Parses config/mppi.yaml, the same file the Python side reads.
//
// Two levels:
//   - Config: everything, host only (std::string, paths).
//   - Params: the plain numbers the kernel needs, passed BY VALUE to the kernel.
//     Only float, int and enums: no pointer, no std::string, no std::vector,
//     so it can be copied to the GPU as is.
//
// Same validation as python/mppi/config.py: a typo must fail at load time.
#pragma once

#include <string>
#include <vector>

namespace mppi {

constexpr int CONTROL_DIM = 2;   // [a, delta]

// The YAML strings become enums: an enum is a plain int, usable on the GPU.
enum class ModelKind : int { Kinematic = 0, Dynamic = 1 };
enum class TireModel : int { Linear = 0, Tanh = 1, Pacejka = 2 };
enum class Integrator : int { Euler = 0, Rk4 = 1 };

struct VehicleParams {
    float wheelbase, lf, lr, width, mu, mass, izz;
    float cs_front, cs_rear;           // cornering_stiffness_front / _rear
    TireModel tire_model;
    float pacejka_c, pacejka_e;
    float blend_speed_low, blend_speed_high;
    Integrator integrator;
    int substeps;
};

struct CostParams {
    float w_lateral, w_progress, w_offtrack, w_control, w_adhesion, w_speed;
    float lateral_scale, speed_scale, v_ref, track_half_width;
};

// Everything a rollout needs, in one trivially copyable struct.
struct Params {
    ModelKind model;
    int state_dim;
    int K, T;
    float dt, lambda, gamma;
    float noise_std[CONTROL_DIM];
    float u_min[CONTROL_DIM], u_max[CONTROL_DIM];
    VehicleParams vehicle;
    CostParams cost;
};

struct Config {
    Params p;
    unsigned long long seed;
    std::string track_bin_path;   // absolute
    int max_steps;
};

// path: config/mppi.yaml. Relative paths of the YAML are resolved from the repo
// root (parent of the config/ directory), like config.py.
// overrides: "section.key=value" strings applied to the YAML before parsing,
// e.g. {"vehicle.tire_model=tanh", "mppi.horizon=50"}. Used by the parity
// check and the sweeps, so that the C++ and Python sides see the same change.
// Throws std::runtime_error on a missing key, an unknown value or an
// inconsistent configuration.
Config load_config(const std::string& path, const std::vector<std::string>& overrides = {});

// Readable names, for printing.
const char* to_string(ModelKind m);
const char* to_string(TireModel t);
const char* to_string(Integrator i);

}  // namespace mppi
