// step(state, control, dt) -> state, compiled for the CPU and the GPU (HD).
// Mirrors python/mppi/dynamics.py line by line, same frame conventions:
//   kinematic state [x, y, psi, v], dynamic state [x, y, psi, vx, vy, r],
//   control [a, delta], SI units, radians, psi from the x axis, CCW positive.
//
// FP32 everywhere: float literals (0.5f) and float functions (sinf, atanf).
// A bare 0.5 or sin() would silently promote to double, which is slow on a
// GeForce-class GPU and changes the numbers.
//
// The model is a type (Kinematic or Dynamic) chosen at compile time by a
// template parameter: a static choice, never a branch on the state. DIM is a
// compile-time constant, so the local arrays float s[DIM] stay in registers.
#pragma once

#include "mppi/config.hpp"
#include "mppi/hd.hpp"

namespace mppi {

constexpr float G = 9.81f;   // m/s²

// ---------------------------------------------------------------------------
// Kinematic bicycle at the CG, state [x, y, psi, v] (step 1)
// ---------------------------------------------------------------------------
struct Kinematic {
    static constexpr int DIM = 4;

    // = step_kinematic
    HD static void step(const float* s, float a, float delta, float dt,
                        const VehicleParams& v, float* out) {
        const float tan_d = tanf(delta);
        const float beta = atanf(v.lr / v.wheelbase * tan_d);
        out[0] = s[0] + s[3] * cosf(s[2] + beta) * dt;
        out[1] = s[1] + s[3] * sinf(s[2] + beta) * dt;
        out[2] = s[2] + s[3] * cosf(beta) / v.wheelbase * tan_d * dt;
        out[3] = fmaxf(s[3] + a * dt, 0.0f);   // never reverses
    }
};

// ---------------------------------------------------------------------------
// Dynamic single-track model blended with the kinematic one (step 2)
// ---------------------------------------------------------------------------

// = tire_force. Lateral force (N) of one axle for a slip angle alpha (rad).
HD inline float tire_force(float alpha, float fz, float c_s, const VehicleParams& v) {
    // Static choice read from the YAML: the same for every thread, no divergence.
    switch (v.tire_model) {
        case TireModel::Linear:
            return v.mu * c_s * fz * alpha;
        case TireModel::Tanh:
            return v.mu * fz * tanhf(c_s * alpha);
        default: {   // Pacejka, B = c_s / C so that the slope at 0 is mu * c_s * fz
            const float b_alpha = c_s / v.pacejka_c * alpha;
            return v.mu * fz * sinf(v.pacejka_c *
                                    atanf(b_alpha - v.pacejka_e * (b_alpha - atanf(b_alpha))));
        }
    }
}

struct Dynamic {
    static constexpr int DIM = 6;

    // = dynamic_derivative (with axle_loads and slip_angles inlined), theory 6.3
    HD static void derivative(const float* s, float a, float delta,
                              const VehicleParams& v, float* ds) {
        const float psi = s[2], vx = s[3], vy = s[4], r = s[5];
        const float fzf = v.mass * G * v.lr / v.wheelbase;   // axle_loads
        const float fzr = v.mass * G * v.lf / v.wheelbase;

        // slip_angles: vx floored so the result stays finite at vx = 0, where the
        // blend gives this model a zero weight (0 * NaN = NaN, step 2 part D)
        const float vx_safe = fmaxf(vx, v.blend_speed_low);
        const float alpha_f = delta - atanf((vy + v.lf * r) / vx_safe);
        const float alpha_r = -atanf((vy - v.lr * r) / vx_safe);

        const float fyf = tire_force(alpha_f, fzf, v.cs_front, v);
        const float fyr = tire_force(alpha_r, fzr, v.cs_rear, v);
        const float cos_d = cosf(delta), sin_d = sinf(delta);
        const float cos_p = cosf(psi), sin_p = sinf(psi);

        ds[0] = vx * cos_p - vy * sin_p;                 // CG velocity in the world frame
        ds[1] = vx * sin_p + vy * cos_p;
        ds[2] = r;
        ds[3] = a - fyf * sin_d / v.mass + vy * r;       // + vy*r, - vx*r: rotating frame
        ds[4] = (fyf * cos_d + fyr) / v.mass - vx * r;
        ds[5] = (v.lf * fyf * cos_d - v.lr * fyr) / v.izz;
    }

    // = step_dynamic_only: `substeps` Euler or RK4 sub-steps of the pure dynamic model
    HD static void step_dynamic_only(const float* s0, float a, float delta, float dt,
                                     const VehicleParams& v, float* out) {
        const float h = dt / v.substeps;
        float s[DIM], k1[DIM], k2[DIM], k3[DIM], k4[DIM], tmp[DIM];
        for (int i = 0; i < DIM; ++i) s[i] = s0[i];
        for (int n = 0; n < v.substeps; ++n) {
            if (v.integrator == Integrator::Euler) {   // from the config: uniform across threads
                derivative(s, a, delta, v, k1);
                for (int i = 0; i < DIM; ++i) s[i] += h * k1[i];
            } else {                                   // rk4
                derivative(s, a, delta, v, k1);
                for (int i = 0; i < DIM; ++i) tmp[i] = s[i] + 0.5f * h * k1[i];
                derivative(tmp, a, delta, v, k2);
                for (int i = 0; i < DIM; ++i) tmp[i] = s[i] + 0.5f * h * k2[i];
                derivative(tmp, a, delta, v, k3);
                for (int i = 0; i < DIM; ++i) tmp[i] = s[i] + h * k3[i];
                derivative(tmp, a, delta, v, k4);
                for (int i = 0; i < DIM; ++i)
                    s[i] += h / 6.0f * (k1[i] + 2.0f * k2[i] + 2.0f * k3[i] + k4[i]);
            }
        }
        for (int i = 0; i < DIM; ++i) out[i] = s[i];
    }

    // = step_kinematic6: kinematic bicycle in the 6-state, explicit Euler,
    // (vy, r) put back on the kinematic manifold for the new vx (theory 6.5)
    HD static void step_kinematic6(const float* s, float a, float delta, float dt,
                                   const VehicleParams& v, float* out) {
        const float tan_d = tanf(delta);
        const float k_vy = v.lr / v.wheelbase * tan_d;   // vy = k_vy * vx
        const float k_r = tan_d / v.wheelbase;           // r  = k_r  * vx
        const float vx = s[3], vy = k_vy * vx;
        const float cos_p = cosf(s[2]), sin_p = sinf(s[2]);
        out[0] = s[0] + (vx * cos_p - vy * sin_p) * dt;
        out[1] = s[1] + (vx * sin_p + vy * cos_p) * dt;
        out[2] = s[2] + k_r * vx * dt;
        const float vx_next = fmaxf(vx + a * dt, 0.0f);
        out[3] = vx_next;
        out[4] = k_vy * vx_next;
        out[5] = k_r * vx_next;
    }

    // = step_dynamic: dynamic model blended with the kinematic one by kappa
    HD static void step(const float* s, float a, float delta, float dt,
                        const VehicleParams& v, float* out) {
        const float lo = v.blend_speed_low, hi = v.blend_speed_high;
        const float kappa = fminf(fmaxf((s[3] - lo) / (hi - lo), 0.0f), 1.0f);   // blend_weight
        float dyn[DIM], kin[DIM];
        step_dynamic_only(s, a, delta, dt, v, dyn);   // both always evaluated
        step_kinematic6(s, a, delta, dt, v, kin);
        for (int i = 0; i < DIM; ++i) out[i] = kappa * dyn[i] + (1.0f - kappa) * kin[i];
        out[3] = fmaxf(out[3], 0.0f);                 // never reverses
    }
};

}  // namespace mppi
