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

The model is chosen by the state size (kinematic 4, dynamic 6), taken from
STATE_DIM in config.py. It is never written as a literal.

Project choice: the vehicle never reverses, v_next = max(v + a*dt, 0).
The CUDA kernel must do the same. Controls are not clipped here, the
controller applies the bounds.
"""
import numpy as np

from .config import STATE_DIM, Vehicle


def step(state: np.ndarray, control: np.ndarray, dt: float, vehicle: Vehicle) -> np.ndarray:
    """Model chosen by the state size: kinematic (4) or dynamic (6).

    The size is fixed by the config for a whole run, not by the state values:
    a static choice, not a branch on the state.
    """
    if state.shape[-1] == STATE_DIM["dynamic"]:
        return step_dynamic(state, control, dt, vehicle)
    return step_kinematic(state, control, dt, vehicle)


# ---------------------------------------------------------------------------
# Modèle cinématique, état [x, y, psi, v] (étape 1, inchangé)
# ---------------------------------------------------------------------------

def step_kinematic(state: np.ndarray, control: np.ndarray, dt: float, vehicle: Vehicle) -> np.ndarray:
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


# ---------------------------------------------------------------------------
# Modèle dynamique, état [x, y, psi, vx, vy, r]
# ---------------------------------------------------------------------------

G = 9.81   # m/s²


def axle_loads(vehicle: Vehicle) -> tuple[float, float]:
    """Static vertical loads (N) on the front and rear axles, no load transfer."""
    m, L = vehicle.mass, vehicle.wheelbase
    return m * G * vehicle.lr / L, m * G * vehicle.lf / L


def tire_force(alpha: np.ndarray, fz: float, c_s: float, vehicle: Vehicle) -> np.ndarray:
    """Lateral force (N) of one axle for a slip angle alpha (rad).

    All three models have the same slope at the origin, mu * c_s * fz (N/rad),
    and the two saturating ones peak at mu * fz.
    """
    mu = vehicle.mu
    if vehicle.tire_model == "linear":       # choix statique, pas une branche sur l'état
        return mu * c_s * fz * alpha
    if vehicle.tire_model == "tanh":
        return mu * fz * np.tanh(c_s * alpha)
    # Pacejka simplifié, B choisi pour que la pente à l'origine soit B*C*D = mu*c_s*fz
    c, e = vehicle.pacejka_c, vehicle.pacejka_e
    b_alpha = c_s / c * alpha
    return mu * fz * np.sin(c * np.arctan(b_alpha - e * (b_alpha - np.arctan(b_alpha))))


def slip_angles(state: np.ndarray, delta: np.ndarray, vehicle: Vehicle) -> tuple[np.ndarray, np.ndarray]:
    """Front and rear slip angles (rad). > 0 produces a force towards +y (left).

    vx is floored at blend_speed_low in the denominator: the result stays finite
    at vx = 0, where the blend gives this model a zero weight (0 * NaN = NaN).
    """
    vx, vy, r = state[..., 3], state[..., 4], state[..., 5]
    vx_safe = np.maximum(vx, vehicle.blend_speed_low)
    alpha_f = delta - np.arctan((vy + vehicle.lf * r) / vx_safe)
    alpha_r = -np.arctan((vy - vehicle.lr * r) / vx_safe)
    return alpha_f, alpha_r


def dynamic_derivative(state: np.ndarray, a: np.ndarray, delta: np.ndarray, vehicle: Vehicle) -> np.ndarray:
    """d(state)/dt of the dynamic single-track model, (..., 6) -> (..., 6). Theory 6.3."""
    psi, vx, vy, r = state[..., 2], state[..., 3], state[..., 4], state[..., 5]
    m, iz, lf, lr = vehicle.mass, vehicle.izz, vehicle.lf, vehicle.lr
    fzf, fzr = axle_loads(vehicle)

    alpha_f, alpha_r = slip_angles(state, delta, vehicle)
    fyf = tire_force(alpha_f, fzf, vehicle.cornering_stiffness_front, vehicle)
    fyr = tire_force(alpha_r, fzr, vehicle.cornering_stiffness_rear, vehicle)
    cos_d, sin_d = np.cos(delta), np.sin(delta)
    cos_p, sin_p = np.cos(psi), np.sin(psi)

    return np.stack([
        vx * cos_p - vy * sin_p,                    # vitesse du CdG dans le repère monde
        vx * sin_p + vy * cos_p,
        r,
        a - fyf * sin_d / m + vy * r,               # + vy*r, -vx*r : repère tournant
        (fyf * cos_d + fyr) / m - vx * r,
        (lf * fyf * cos_d - lr * fyr) / iz,
    ], axis=-1)


def step_dynamic_only(state: np.ndarray, control: np.ndarray, dt: float, vehicle: Vehicle) -> np.ndarray:
    """One control interval of the pure dynamic model: `substeps` RK4 or Euler sub-steps."""
    a, delta = control[..., 0], control[..., 1]
    h = dt / vehicle.substeps
    f = lambda s: dynamic_derivative(s, a, delta, vehicle)
    for _ in range(vehicle.substeps):              # nombre fixe : boucle déroulable en CUDA
        if vehicle.integrator == "euler":
            state = state + h * f(state)
        else:                                      # rk4
            k1 = f(state)
            k2 = f(state + 0.5 * h * k1)
            k3 = f(state + 0.5 * h * k2)
            k4 = f(state + h * k3)
            state = state + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return state


def step_kinematic6(state: np.ndarray, control: np.ndarray, dt: float, vehicle: Vehicle) -> np.ndarray:
    """Kinematic bicycle in the 6-component state, explicit Euler.

    Positions and heading move with the no-slip velocities, then (vy, r) are
    put back on the kinematic manifold for the new vx (theory 6.5).
    """
    x, y, psi, vx = state[..., 0], state[..., 1], state[..., 2], state[..., 3]
    a, delta = control[..., 0], control[..., 1]
    k_vy = vehicle.lr / vehicle.wheelbase * np.tan(delta)   # vy = k_vy * vx
    k_r = np.tan(delta) / vehicle.wheelbase                 # r  = k_r  * vx

    vy = k_vy * vx
    x_next = x + (vx * np.cos(psi) - vy * np.sin(psi)) * dt
    y_next = y + (vx * np.sin(psi) + vy * np.cos(psi)) * dt
    psi_next = psi + k_r * vx * dt
    vx_next = np.maximum(vx + a * dt, 0.0)
    return np.stack([x_next, y_next, psi_next, vx_next, k_vy * vx_next, k_r * vx_next], axis=-1)


def blend_weight(vx: np.ndarray, vehicle: Vehicle) -> np.ndarray:
    """kappa = 0 below blend_speed_low (kinematic), 1 above blend_speed_high (dynamic)."""
    lo, hi = vehicle.blend_speed_low, vehicle.blend_speed_high
    return np.clip((vx - lo) / (hi - lo), 0.0, 1.0)


def step_dynamic(state: np.ndarray, control: np.ndarray, dt: float, vehicle: Vehicle) -> np.ndarray:
    """state (..., 6), control (..., 2) -> (..., 6). Dynamic model blended with the kinematic one."""
    kappa = blend_weight(state[..., 3], vehicle)[..., None]
    dyn = step_dynamic_only(state, control, dt, vehicle)
    kin = step_kinematic6(state, control, dt, vehicle)
    nxt = kappa * dyn + (1.0 - kappa) * kin           # les deux sont toujours évalués
    vx_next = np.maximum(nxt[..., 3], 0.0)            # pas de marche arrière
    return np.concatenate([nxt[..., :3], vx_next[..., None], nxt[..., 4:]], axis=-1)
