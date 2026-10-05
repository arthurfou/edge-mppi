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

    step(state, control, dt, vehicle) -> state

state_dim comes from the config. It is never written as a literal.

Project choice: the vehicle never reverses, v_next = max(v + a*dt, 0).
The CUDA kernel must do the same. Controls are not clipped here, the
controller applies the bounds.
"""
import numpy as np

from .config import Vehicle


def step(state: np.ndarray, control: np.ndarray, dt: float, vehicle: Vehicle) -> np.ndarray:
    """state (..., 4), control (..., 2) -> (..., 4). Vectorized over leading axes."""
    x, y, psi, v = state[..., 0], state[..., 1], state[..., 2], state[..., 3]
    a, delta = control[..., 0], control[..., 1]
    tan_delta = np.tan(delta)

    beta = np.arctan(vehicle.lr / vehicle.wheelbase * tan_delta)

    x_next   = x   + v * np.cos(psi + beta) * dt
    y_next   = y   + v * np.sin(psi + beta) * dt
    psi_next = psi + v * np.cos(beta) / vehicle.wheelbase * tan_delta * dt
    v_next   = np.maximum(v + a * dt, 0.0)   # la voiture ne recule jamais

    return np.stack([x_next, y_next, psi_next, v_next], axis=-1)
