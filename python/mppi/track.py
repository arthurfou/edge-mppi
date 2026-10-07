"""Track generation and cost map baking.

Control points -> periodic spline (scipy.interpolate.splprep) -> resampling at
constant arc length -> 2D grid filled by KD-tree lookup.

Each cell stores the signed lateral offset to the centerline and the
curvilinear progress. Exported as .npz for Python and as a raw binary for C++:
header with the grid dimensions and the domain bounds, then row-major float32.
"""

import struct
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree

MAGIC = b"MPPG"
VERSION = 1
HEADER = struct.Struct("<4sIiiffff")   # magic, version, nx, ny, x_min, y_min, res, length


@dataclass
class Track:
    grid: np.ndarray         # (ny, nx, 2) float32: [d, s]
    x_min: float
    y_min: float
    res: float
    length: float
    centerline: np.ndarray   # (n, 2)
    heading: np.ndarray      # (n,)


def centerline_from_points(
    points: np.ndarray, ds: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """Smooth closed centerline resampled every ds metres.

    Args:
        points: (k, 2) control points, NOT closed (last != first).
        ds: target spacing in metres (0.05).

    Returns:
        centerline: (n, 2) positions x, y
        heading:    (n,)   track direction theta (rad)
        s:          (n,)   arc length from the start: 0, ~ds, ~2ds, ...
        length:     float  lap length (~48.31 m)
        curvature:  (n,)   kappa (1/m), > 0 when turning left
    """
    # ------------------------------------------------------------------
    # ÉTAPE 1 — Construire la spline fermée
    #   np.vstack([A, B])  : empile A puis B verticalement -> (k+1, 2)
    #   points[:1]         : le 1er point, gardé en 2D, forme (1, 2)
    #   splprep([xs, ys], s=0, per=1) -> (tck, u)
    #       xs, ys : deux tableaux 1D (les colonnes x et y)
    #       tck    : la spline, à repasser tel quel à splev
    #       u      : on ne s'en sert pas -> le nommer _
    # ------------------------------------------------------------------
    closed = np.vstack([points, points[:1]])                     # (12, 2)
    tck, _ = splprep([closed[:, 0], closed[:, 1]], s=0, per=1)

    # ------------------------------------------------------------------
    # ÉTAPE 2 — Échantillonner u très finement
    #   np.linspace(a, b, N, endpoint=False) : N valeurs de a à b, b exclu
    #       -> ici 20000 valeurs dans [0, 1)
    #   splev(u, tck) -> [x, y] : deux tableaux 1D de même taille que u
    # ------------------------------------------------------------------
    u_dense = np.linspace(0.0, 1.0, 20000, endpoint=False)
    x, y = splev(u_dense, tck)

    # ------------------------------------------------------------------
    # ÉTAPE 3 — Distance parcourue le long de la courbe
    #   np.diff(x, append=x[0]) : x[i+1] - x[i], et le dernier vaut
    #       x[0] - x[-1] (segment qui referme la boucle). Taille 20000.
    #   np.hypot(a, b) : sqrt(a² + b²) élément par élément
    #   np.cumsum(a)   : sommes cumulées [a0, a0+a1, a0+a1+a2, ...]
    #
    #   seg[i]     = longueur du segment entre le point i et le point i+1
    #   s_dense[i] = distance parcourue pour ARRIVER au point i
    #                -> s_dense[0] doit valoir 0, taille 20000
    #                -> indice : [0] suivi de cumsum(seg) sans son dernier
    #                   élément (np.concatenate([[0.0], ...]))
    #   length     = somme de tous les segments (convertir en float)
    # ------------------------------------------------------------------
    seg = np.hypot(np.diff(x, append=x[0]), np.diff(y, append=y[0]))
    s_dense = np.concatenate([[0.0], np.cumsum(seg)[:-1]])
    length = float(seg.sum())   

    # ------------------------------------------------------------------
    # ÉTAPE 4 — Les distances cibles (déjà écrit)
    #   n points, pas = length / n ≈ ds, qui tombe pile sur un tour
    # ------------------------------------------------------------------
    n = int(round(length / ds))
    s = np.arange(n) * length / n

    # ------------------------------------------------------------------
    # ÉTAPE 5 — Pour chaque s cible, retrouver le u correspondant
    #   np.interp(x_voulus, xp, fp) : interpolation linéaire de la
    #       fonction xp -> fp aux abscisses x_voulus
    #       (xp doit être croissant : c'est le cas de s_dense)
    #   Ici la « fonction » est distance -> u :
    #       xp = s_dense, fp = u_dense, x_voulus = s
    # ------------------------------------------------------------------
    u_s = np.interp(s, s_dense, u_dense)     # 966 valeurs de u

    # ------------------------------------------------------------------
    # ÉTAPE 6 — Évaluer la spline aux u_s
    #   splev(u_s, tck)        -> [x, y]       positions
    #   splev(u_s, tck, der=1) -> [x', y']     dérivées premières
    #   splev(u_s, tck, der=2) -> [x'', y'']   dérivées secondes
    # ------------------------------------------------------------------
    x, y = splev(u_s, tck)
    dx, dy = splev(u_s, tck, der=1)
    ddx, ddy = splev(u_s, tck, der=2)

    # ------------------------------------------------------------------
    # ÉTAPE 7 — Cap et courbure
    #   np.arctan2(Y, X) : angle du vecteur (X, Y). ATTENTION à l'ordre :
    #       Y en premier -> np.arctan2(dy, dx)
    #   kappa = (x' y'' - y' x'') / (x'² + y'²)^1.5   (** pour la puissance)
    #   np.column_stack([x, y]) : deux tableaux (n,) -> un tableau (n, 2)
    # ------------------------------------------------------------------
    heading = np.arctan2(dy, dx)
    curvature = (dx * ddy - dy * ddx) / (dx**2 + dy**2) ** 1.5
    centerline = np.column_stack([x, y])

    return centerline, heading, s, length, curvature

def bake_grid(
    centerline: np.ndarray, heading: np.ndarray, s: np.ndarray, res: float, margin: float
) -> tuple[np.ndarray, float, float]:
    """Fills a (ny, nx, 2) float32 grid with [d, s] of the nearest centerline point.

    margin: distance kept around the centerline (track half width + extra margin).
    Returns the grid and the lower-left corner (x_min, y_min) of the domain.
    """
    x_min, y_min = centerline.min(axis=0) - margin
    x_max, y_max = centerline.max(axis=0) + margin
    nx = int(np.ceil((x_max - x_min) / res))
    ny = int(np.ceil((y_max - y_min) / res))

    xs = x_min + (np.arange(nx) + 0.5) * res      # cell centres
    ys = y_min + (np.arange(ny) + 0.5) * res
    X, Y = np.meshgrid(xs, ys)                    # (ny, nx), row = y
    P = np.column_stack([X.ravel(), Y.ravel()])

    _, idx = cKDTree(centerline).query(P)
    normal = np.column_stack([-np.sin(heading), np.cos(heading)])   # left of the track
    # Row-wise dot product as a batch of (1, 2) @ (2, 1) products
    d = ((P - centerline[idx])[:, None, :] @ normal[idx][:, :, None])[:, 0, 0]
    grid = np.stack([d, s[idx]], axis=-1).reshape(ny, nx, 2).astype(np.float32)
    return grid, float(x_min), float(y_min)


def save(track: Track, npz_path, bin_path) -> None:
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, grid=track.grid, x_min=track.x_min, y_min=track.y_min,
             res=track.res, length=track.length,
             centerline=track.centerline, heading=track.heading)
    ny, nx, _ = track.grid.shape
    with open(bin_path, "wb") as f:
        f.write(HEADER.pack(MAGIC, VERSION, nx, ny, track.x_min, track.y_min,
                            track.res, track.length))
        f.write(np.ascontiguousarray(track.grid, dtype="<f4").tobytes())


def load(npz_path) -> Track:
    z = np.load(npz_path)
    return Track(z["grid"], float(z["x_min"]), float(z["y_min"]), float(z["res"]),
                 float(z["length"]), z["centerline"], z["heading"])


def load_bin(bin_path):
    """Returns (grid, x_min, y_min, res, length) read from the C++ binary."""
    raw = bin_path.read_bytes()
    magic, version, nx, ny, x_min, y_min, res, length = HEADER.unpack_from(raw)
    assert magic == MAGIC and version == VERSION, f"bad header in {bin_path}"
    grid = np.frombuffer(raw, dtype="<f4", offset=HEADER.size).reshape(ny, nx, 2)
    return grid, x_min, y_min, res, length


def lookup(track: Track, x, y):
    """Nearest cell, indices clamped to the grid (same as cudaAddressModeClamp)."""
    ny, nx, _ = track.grid.shape
    ix = np.clip(np.floor((x - track.x_min) / track.res).astype(np.int64), 0, nx - 1)
    iy = np.clip(np.floor((y - track.y_min) / track.res).astype(np.int64), 0, ny - 1)
    cell = track.grid[iy, ix]
    return cell[..., 0], cell[..., 1]


def wrap(ds, length):
    """Progress difference brought back to [-L/2, L/2)."""
    return (ds + 0.5 * length) % length - 0.5 * length


def plot_grid(track: Track, path=None):
    """Shows the d and s channels side by side, centerline in black.

    Saves the figure to path if given, otherwise opens a window.
    """
    import matplotlib.pyplot as plt   # local import: plotting is optional

    ny, nx, _ = track.grid.shape
    extent = [track.x_min, track.x_min + nx * track.res,
              track.y_min, track.y_min + ny * track.res]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    channels = [("d (m)", "RdBu", dict(vmin=-2, vmax=2)), ("s (m)", "viridis", {})]
    for c, (ax, (title, cmap, scale)) in enumerate(zip(axes, channels)):
        im = ax.imshow(track.grid[..., c], origin="lower", extent=extent, cmap=cmap, **scale)
        ax.plot(track.centerline[:, 0], track.centerline[:, 1], "k", lw=0.8)
        ax.set_title(title)
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax)

    if path is None:
        plt.show()
    else:
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
