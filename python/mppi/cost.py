"""Per-trajectory cost.

Terms: lateral offset to the centerline, curvilinear progress, off-track
penalty, control regularization.

Each term is normalized to the same order of magnitude. Without that, tuning
lambda against the cost scale is guesswork.
"""
import numpy as np

from .config import Config
from .track import Track, lookup, wrap

G = 9.81   # m/s², pour la limite d'adhérence mu*g


def stage_cost(state: np.ndarray, control: np.ndarray, track: Track, cfg: Config) -> np.ndarray:
    """Running cost of x_t and of the control u_{t-1} that led to it.

    state (..., 4), control (..., 2) -> (...,). Vectorized over leading axes,
    typically K.
    """
    c, veh, u_max = cfg.cost, cfg.vehicle, cfg.bounds.u_max
    d, _ = lookup(track, state[..., 0], state[..., 1])   # s inutile ici
    v = state[..., 3]

    # Écart latéral, normalisé par d0 : vaut 1 à 0,4 m de la ligne centrale
    d0 = c.lateral_scale
    lateral = c.w_lateral * (d / d0) ** 2

    # Sortie de piste : 0 dedans, puis 1 + (dépassement / d0) dehors.
    # d est mesuré au CdG, les roues sortent à width/2 avant le bord (Q-D2).
    d_max = c.track_half_width - 0.5 * veh.width
    excess = np.abs(d) - d_max
    offtrack = c.w_offtrack * (excess > 0) * (1.0 + excess / d0)

    speed = c.w_speed * ((v - c.v_ref) / c.speed_scale) ** 2

    # Bornes symétriques : u_max sert d'échelle pour a et pour delta
    effort = c.w_control * ((control / u_max) ** 2).sum(axis=-1)

    # Accélération latérale du modèle cinématique, pénalisée au-delà de mu*g (Q-D4)
    a_lat = v**2 * np.abs(np.tan(control[..., 1])) / veh.wheelbase
    adhesion = c.w_adhesion * np.maximum(0.0, a_lat / (veh.mu * G) - 1.0) ** 2

    return lateral + offtrack + speed + effort + adhesion


def terminal_cost(state_T: np.ndarray, s0: float, track: Track, cfg: Config) -> np.ndarray:
    """Progress reward on the last state x_T. state_T (..., 4) -> (...,).

    s0 is the progress of the rollout start x0, shared by all K rollouts.
    Normalized by the distance covered at v_ref over the horizon: ~ -w_progress
    for a rollout driven at v_ref.
    """
    c, m = cfg.cost, cfg.mppi
    _, s_T = lookup(track, state_T[..., 0], state_T[..., 1])
    progress = wrap(s_T - s0, track.length)   # sinon la ligne de départ casse tout (Q-C4)
    return -c.w_progress * progress / (c.v_ref * m.horizon * m.dt)
