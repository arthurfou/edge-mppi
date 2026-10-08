// Parses config/mppi.yaml, the same file the Python side reads.
//
// Fields added in step 2 (to parse in step 3):
//   vehicle.mass, vehicle.izz, vehicle.cornering_stiffness_front,
//   vehicle.cornering_stiffness_rear, vehicle.tire_model (linear|tanh|pacejka),
//   vehicle.pacejka_c, vehicle.pacejka_e, vehicle.blend_speed_low,
//   vehicle.blend_speed_high, vehicle.integrator (euler|rk4), vehicle.substeps
// Removed in step 2: vehicle.kinematic_blend_speed
// Changed in step 2: model (dynamic), state_dim (6), mppi.lambda, cost.v_ref
#pragma once
