"""Vehicle dynamics.

Frame conventions, fixed for the whole project:

- Right-handed world frame, x forward, y left, psi measured from the x axis,
  counter-clockwise positive.
- SI units and radians everywhere. No degrees, ever.
- Kinematic state order: [x, y, psi, v]
- Dynamic state order:   [x, y, psi, vx, vy, r]
  vx, vy are body-frame velocities, r is the yaw rate.
- Control order: [a, delta], longitudinal acceleration and steering angle.

Public interface, mirrored by dynamics.cuh as a __device__ function:

    step(state, control, dt) -> state

state_dim comes from the config. It is never written as a literal.
"""
