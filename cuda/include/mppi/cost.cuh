// Cost of one rollout, compiled for the CPU and the GPU (HD).
// Mirrors python/mppi/cost.py (stage_cost, terminal_cost) and
// controller.rollout_costs for a single rollout k.
//
// Same rules as dynamics.cuh: FP32 only, no branch on the state except short
// selects, model chosen at compile time by the template parameter.
#pragma once

#include "mppi/config.hpp"
#include "mppi/dynamics.cuh"
#include "mppi/hd.hpp"
#include "mppi/track.cuh"

namespace mppi {

constexpr float S_MAX = 1e6f;   // cost of a rollout that produced a NaN or an inf (= controller.S_MAX)

// Running cost of x_t and of the control u_{t-1} that led to it
template <class Model>
HD float stage_cost(const float* x, const float* u, const TrackView& tr, const Params& p) {
    const CostParams& c = p.cost;
    float d, s;
    lookup(tr, x[0], x[1], d, s);   // s unused here
    const float v = x[3];           // v (kinematic) or vx (dynamic): same index

    // Lateral offset, normalized by d0: 1 at 0.4 m from the centerline
    const float d0 = c.lateral_scale;
    const float lateral = c.w_lateral * (d / d0) * (d / d0);

    // Off-track: 0 inside, then 1 + excess/d0 outside. d is at the CG, the
    // wheels leave the track width/2 earlier.
    const float d_max = c.track_half_width - 0.5f * p.vehicle.width;
    const float excess = fabsf(d) - d_max;
    const float offtrack = excess > 0.0f ? c.w_offtrack * (1.0f + excess / d0) : 0.0f;   // Q-C2

    const float dv = (v - c.v_ref) / c.speed_scale;
    const float speed = c.w_speed * dv * dv;

    // Symmetric bounds: u_max is the scale for a and for delta
    float effort = 0.0f;
    for (int j = 0; j < CONTROL_DIM; ++j) {
        const float uj = u[j] / p.u_max[j];
        effort += uj * uj;
    }
    float cost = lateral + offtrack + speed + c.w_control * effort; // We don't do directly effort = effort * c.w_control

    // Adhesion penalty, kinematic model only. if constexpr is resolved at
    // compile time: the Dynamic instantiation does not even contain this code.
    if constexpr (Model::DIM == Kinematic::DIM) {
        const float a_lat = v * v * fabsf(tanf(u[1])) / p.vehicle.wheelbase;
        const float over = fmaxf(0.0f, a_lat / (p.vehicle.mu * G) - 1.0f);
        cost += c.w_adhesion * over * over;
    }
    return cost;
}

// Progress reward on x_T, normalized by the distance covered at v_ref over the horizon
HD inline float terminal_cost(const float* xT, float s0, const TrackView& tr, const Params& p) {
    float d, sT;
    lookup(tr, xT[0], xT[1], d, sT);
    const float progress = wrap(sT - s0, tr.length);   // otherwise the start line breaks everything
    return -p.cost.w_progress * progress / (p.cost.v_ref * p.T * p.dt);
}

// Total cost S^k of rollout k. Layout (row-major, as NumPy):
//   x0 (DIM,), U (T, 2), eps (K, T, 2) raw noise, before clipping.
// The CPU path calls it in a for loop, the GPU kernel with k = thread index.
template <class Model>
HD float rollout_cost(int k, const float* x0, const float* U, const float* eps,
                      const TrackView& tr, const Params& p) {
    float x[Model::DIM], x_next[Model::DIM];   // the state, in registers
    for (int i = 0; i < Model::DIM; ++i) x[i] = x0[i];

    float d, s0;
    lookup(tr, x0[0], x0[1], d, s0);           // progress of the start, shared by all rollouts

    float S = 0.0f, correction = 0.0f;
    for (int t = 0; t < p.T; ++t) {
        float vt[CONTROL_DIM];
        for (int j = 0; j < CONTROL_DIM; ++j) {
            const float u = U[t * CONTROL_DIM + j];
            const float e = eps[(static_cast<long>(k) * p.T + t) * CONTROL_DIM + j];
            vt[j] = fminf(fmaxf(u + e, p.u_min[j]), p.u_max[j]);   // V = clip(U + eps)
            const float sigma = p.noise_std[j];
            correction += u / (sigma * sigma) * (vt[j] - u);      // U Sigma^-1 eps_effective
        }
        Model::step(x, vt[0], vt[1], p.dt, p.vehicle, x_next);
        for (int i = 0; i < Model::DIM; ++i) x[i] = x_next[i];
        S += stage_cost<Model>(x, vt, tr, p);
    }
    S += terminal_cost(x, s0, tr, p);
    S += p.gamma * correction;
    return isfinite(S) ? S : S_MAX;
}

}  // namespace mppi
