"""Vectorized NumPy MPPI, the reference implementation.

Per control step:
  1. sample noise of shape (K, T, control_dim)
  2. roll out the dynamics
  3. accumulate a cost per trajectory
  4. w = exp(-(S - S_min) / lambda), normalized
  5. weighted mean over the noise, applied to the nominal sequence
  6. shift the nominal sequence by one step

This is the ground truth the CUDA kernel is checked against.
"""
from typing import Any

import numpy as np

from .config import Config
from .cost import stage_cost, terminal_cost
from .dynamics import step
from .track import Track, lookup

S_MAX = 1e6   # coût d'un rollout qui a produit un NaN ou un inf


def rollout_costs(
    x0: np.ndarray, U: np.ndarray, eps: np.ndarray, track: Track, cfg: Config
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic part of one iteration. Draws nothing at random.

    Args:
        x0:  (n,)      current state, shared by all rollouts
        U:   (T, m)    nominal control sequence
        eps: (K, T, m) raw noise

    Returns:
        S:   (K,)        total cost of each rollout
        eps: (K, T, m)   effective noise, V - U, after clipping (Q-E1)
        X:   (K, T+1, n) rolled-out states, X[:, 0] = x0
    """
    m, b = cfg.mppi, cfg.bounds

    # Commandes réellement simulées, dans les bornes. Le bruit effectif est
    # recalculé pour que U + eps soit exactement ce qui a été simulé.
    V = np.clip(U[None] + eps, b.u_min, b.u_max)
    eps = V - U[None]
    K, T, _ = V.shape

    # Boucle séquentielle sur t, vectorisée sur les K rollouts (Q-E4)
    X = np.empty((K, T + 1, cfg.state_dim))
    X[:, 0] = x0
    S = np.zeros(K)
    for t in range(T):
        X[:, t + 1] = step(X[:, t], V[:, t], m.dt, cfg.vehicle)
        S += stage_cost(X[:, t + 1], V[:, t], track, cfg)

    _, s0 = lookup(track, x0[0], x0[1])
    S += terminal_cost(X[:, T], s0, track, cfg)

    # Terme de correction gamma * sum_t U_t^T Sigma^-1 eps_t, Sigma diagonale (théorie §4.5)
    sigma_inv = 1.0 / m.noise_std**2    #m.noise_std est la matrice de covar diagonale.
    S += m.gamma * np.einsum("tj,ktj->k", U * sigma_inv, eps)

    S[~np.isfinite(S)] = S_MAX
    return S, eps, X


class MPPI:
    """Receding-horizon controller. Keeps the nominal sequence U between calls."""

    def __init__(self, cfg: Config, track: Track) -> None:
        self.cfg, self.track = cfg, track
        self.U = np.zeros((cfg.mppi.horizon, cfg.control_dim))
        self.rng = np.random.default_rng(cfg.mppi.seed)

    def command(self, x0: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """One MPPI iteration from state x0 (n,).

        Returns the control to apply now (m,) and diagnostics:
        ess, rho (min cost), S (K,), w (K,), X (K, T+1, n).
        """
        m = self.cfg.mppi
        shape = (m.num_samples, m.horizon, self.cfg.control_dim)
        eps = self.rng.standard_normal(shape) * m.noise_std   # (K, T, m), un sigma par commande
        S, eps, X = rollout_costs(x0, self.U, eps, self.track, self.cfg)

        # Poids softmin. Soustraire rho évite exp(-S/lambda) = 0 partout (Q-E2)
        rho = S.min()
        w = np.exp(-(S - rho) / m.lam)
        w /= w.sum()

        # Moyenne pondérée du bruit : (K,) . (K, T, m) -> (T, m)
        self.U = self.U + np.tensordot(w, eps, axes=1)

        u0 = self.U[0].copy()   # U[0] est une vue, le décalage l'écraserait
        self.U[:-1] = self.U[1:]
        self.U[-1] = self.U[-2]   # garder la dernière commande plutôt que 0 (Q-E3)

        info = {"ess": 1.0 / np.sum(w**2), "rho": rho, "S": S, "w": w, "X": X}
        return u0, info
