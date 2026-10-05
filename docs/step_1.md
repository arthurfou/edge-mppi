# Étape 1 : MPPI cinématique en Python/NumPy

TP autonome. Tout ce qu'il faut pour réussir l'étape est dans ce fichier : les objectifs, les notions, les rappels d'API, les pièges, les vérifications et, à la fin, les solutions complètes. La théorie de fond est dans `docs/mppi-theory.md`, référencée par numéro de section (par exemple « théorie §4.1 »).

**Règle du jeu.** Les questions marquées **Q** demandent de réfléchir avant de coder. Écris ta réponse (dans ta tête, sur papier ou dans le journal), puis compare-la à la section [Solutions](#solutions). Les blocs de code de la partie énoncé sont des squelettes et des rappels d'API, pas la solution.

**Durée estimée.** 2 à 3 jours, comme prévu dans `edge-mppi.md`.

## Sommaire

- [0. Objectif et critère de sortie](#0-objectif-et-critère-de-sortie)
- [1. Préparer le terrain](#1-préparer-le-terrain)
- [2. Partie A : charger la configuration](#2-partie-a--charger-la-configuration)
- [3. Partie B : le modèle cinématique](#3-partie-b--le-modèle-cinématique)
- [4. Partie C : générer la piste et la grille](#4-partie-c--générer-la-piste-et-la-grille)
- [5. Partie D : la fonction de coût](#5-partie-d--la-fonction-de-coût)
- [6. Partie E : le contrôleur MPPI](#6-partie-e--le-contrôleur-mppi)
- [7. Partie F : la simulation en boucle fermée](#7-partie-f--la-simulation-en-boucle-fermée)
- [8. Partie G : réglage et compréhension des paramètres](#8-partie-g--réglage-et-compréhension-des-paramètres)
- [9. Dépannage](#9-dépannage)
- [10. Clôturer l'étape](#10-clôturer-létape)
- [Solutions](#solutions)

---

## 0. Objectif et critère de sortie

Tu vas écrire la **référence** NumPy de tout le projet. Le kernel CUDA de l'étape 3 sera jugé contre elle, coût par coût, à 1e-4 près. Chaque choix fait ici (ordre des opérations, saturation, lecture de grille) devra être reproduit à l'identique en CUDA. Code simple, explicite et déterministe : c'est plus important que rapide.

**Critère de sortie** (repris de `edge-mppi.md`) :

- [ ] un tour complet sans sortie de piste à `K = 1024`, `T = 30` ;
- [ ] un plot de la trajectoire et une courbe de coût dans `results/figures/` ;
- [ ] `λ` et la variance du bruit réglés à la main, et tu sais expliquer qualitativement leur effet (partie G) ;
- [ ] les tests unitaires passent ;
- [ ] une entrée dans `docs/journal.md` avec des chiffres mesurés ;
- [ ] le tag `git tag step-1`.

**Vue d'ensemble des fichiers.**

| Fichier | Rôle | Partie |
|---|---|---|
| `config/mppi.yaml` | nouveaux champs | A |
| `python/mppi/config.py` | YAML → dataclasses figées, avec validation | A |
| `python/mppi/dynamics.py` | `step(state, control, dt, vehicle)` vectorisé | B |
| `python/mppi/track.py` | spline, grille, export, lecture | C |
| `python/generate_track.py` | script qui produit `results/track/*` | C |
| `python/mppi/cost.py` | coût d'étape et coût terminal | D |
| `python/mppi/controller.py` | `rollout_costs` (pur) + classe `MPPI` | E |
| `python/run_sim.py` | boucle fermée, figures, journal | F |
| `python/tests/test_step1.py` | tests unitaires | toutes |

Ordre conseillé : A → B (avec ses tests) → C (avec ses vérifications visuelles) → D → E → F → G. Ne passe à la partie suivante que lorsque la précédente est vérifiée. Un bug de dynamique découvert pendant le réglage de λ coûte une demi-journée.

---

## 1. Préparer le terrain

### 1.1 Ajouter pytest et deux tâches pixi

```bash
pixi add pytest
```

Puis dans `pixi.toml`, section `[tasks]` :

```toml
track = "python python/generate_track.py --config config/mppi.yaml"
test = { cmd = "pytest -q python/tests", env = { PYTHONPATH = "python" } }
```

Et fais dépendre `sim` de `track` :

```toml
sim = { cmd = "python python/run_sim.py --config config/mppi.yaml", depends-on = ["track"] }
```

**Pourquoi `PYTHONPATH=python`.** Les scripts sont lancés par `python python/run_sim.py`. Python ajoute alors le dossier du script (`python/`) à `sys.path`, donc `import mppi` fonctionne. pytest, lui, ne le fait pas pour `python/`. Il faut donc le lui dire.

### 1.2 Rappels NumPy utiles pour tout le TP

**Vectoriser sur K avec `...`.** Toutes les fonctions prennent des tableaux dont **le dernier axe** est l'axe de l'état ou de la commande. `state[..., 0]` est `x`, que `state` soit de forme `(4,)`, `(K, 4)` ou `(K, T, 4)`. Une même fonction `step` sert alors pour la simulation (un seul état) et pour les rollouts (K états).

```python
x, y, psi, v = state[..., 0], state[..., 1], state[..., 2], state[..., 3]
new = np.stack([x2, y2, psi2, v2], axis=-1)   # recolle sur le dernier axe
```

**Broadcasting.** `U` de forme `(T, m)` plus `eps` de forme `(K, T, m)` : écris `U[None] + eps`, ce qui donne `(K, T, m)`.

**Moyenne pondérée sur K.** `np.tensordot(w, eps, axes=1)` avec `w` de forme `(K,)` et `eps` de forme `(K, T, m)` donne `(T, m)`. C'est `sum_k w[k] * eps[k]`. Équivalent : `np.einsum("k,ktm->tm", w, eps)`.

**Générateur aléatoire.** Utilise `rng = np.random.default_rng(seed)` puis `rng.standard_normal(shape)`. N'utilise jamais `np.random.seed` ni `np.random.randn`, qui passent par un état global.

**Tests.** Un test pytest est une fonction `test_*` dans un fichier `test_*.py` qui fait des `assert`. Pour comparer des flottants : `np.allclose(a, b)` ou `x == pytest.approx(y, abs=1e-3)`.

---

## 2. Partie A : charger la configuration

### 2.1 Ce qu'il faut ajouter au YAML

La configuration actuelle ne suffit pas pour un coût normalisé (théorie §5.3). Ajoute les champs suivants. Les valeurs sont des **valeurs de départ vérifiées** : avec elles, la solution de référence boucle le tour.

```yaml
mppi:
  lambda: 0.3             # était 1.0, voir partie G
  gamma: 0.0              # poids du terme de correction, théorie §4.5

vehicle:
  width: 0.30             # largeur hors tout, pour la marge de sortie de piste
  mu: 1.0                 # coefficient d'adhérence, pour le terme d'adhérence

cost:
  w_lateral: 1.0
  w_progress: 10.0        # était 1.0
  w_offtrack: 100.0
  w_control: 0.01
  w_adhesion: 10.0
  w_speed: 1.0
  lateral_scale: 0.4      # d0 (m)
  speed_scale: 1.0        # v0 (m/s)
  v_ref: 3.0              # vitesse de référence (m/s)
  track_half_width: 0.8

track:
  grid_resolution: 0.05
  resample_step: 0.05     # pas de rééchantillonnage de la ligne centrale (m)
  margin: 1.0             # marge de la grille au-delà du bord de piste (m)
  npz_path: results/track/costmap.npz   # était costmap.npy, voir Q-A2
  bin_path: results/track/costmap.bin

sim:
  max_steps: 3000         # 60 s à 50 Hz
```

> `config.hpp` dit que tout champ ajouté côté Python devra l'être côté C++. Garde une liste des champs ajoutés dans le journal : tu en auras besoin à l'étape 3.

### 2.2 Ce que doit faire `config.py`

- Une fonction `load_config(path) -> Config`.
- Des `@dataclass(frozen=True)` par section (`MppiParams`, `Bounds`, `Vehicle`, `CostParams`, `TrackParams`), regroupées dans `Config`. `frozen=True` interdit de modifier un paramètre en cours de simulation par accident. Pour une variante (partie G), on crée une copie avec `dataclasses.replace(cfg, mppi=dataclasses.replace(cfg.mppi, lam=0.1))`.
- Les tableaux NumPy sont convertis au chargement : `noise_std` en `np.array` de forme `(control_dim,)`, et les bornes en deux vecteurs `u_min = [a_min, delta_min]`, `u_max = [a_max, delta_max]`.
- Les chemins sont résolus **par rapport à la racine du repo** (le dossier parent de `config/`), pas par rapport au dossier courant.
- Une validation qui lève `ValueError` si :
  - `model` n'est ni `kinematic` ni `dynamic` ;
  - `state_dim` ne correspond pas à `model` (4 ou 6) ;
  - `noise_std` n'a pas `control_dim` composantes ;
  - `lf + lr != wheelbase`.

Attention, `lambda` est un mot-clé Python : le champ de la dataclass s'appellera `lam`.

```python
import yaml
raw = yaml.safe_load(Path(path).read_text())    # dict imbriqué
```

**Q-A1.** Pourquoi vérifier la cohérence `state_dim`/`model` au chargement plutôt que laisser le code planter plus tard ?

**Q-A2.** Le README prévoit un `.npy` pour Python. Pourquoi passer à un `.npz` ?

**Vérification.** `python -c "import sys; sys.path.insert(0,'python'); from mppi.config import load_config; print(load_config('config/mppi.yaml'))"` affiche la config. Mets ensuite `state_dim: 6` dans une copie du YAML : le chargement doit échouer avec un message clair.

---

## 3. Partie B : le modèle cinématique

### 3.1 Les équations

État `[x, y, ψ, v]`, commande `[a, δ]`. Bicycle cinématique **écrit au centre de gravité** (théorie §6.2) :

$$
\beta = \arctan\!\Big(\frac{l_r}{L}\tan\delta\Big),\quad
\dot x = v\cos(\psi+\beta),\quad
\dot y = v\sin(\psi+\beta),\quad
\dot\psi = \frac{v\cos\beta}{L}\tan\delta,\quad
\dot v = a.
$$

Intégration d'Euler explicite : `état_suivant = état + dérivée(état, commande) * dt`, toutes les dérivées étant évaluées sur l'état **courant**. Dans la mise à jour de `x`, utilise donc le `psi` d'avant le pas, pas le nouveau.

**Choix de projet : `v ≥ 0`.** Après intégration, `v_next = max(v + a·dt, 0)`. La voiture ne recule pas. Écris ce choix dans l'en-tête de `dynamics.py`, car le kernel devra le reproduire.

**La saturation des commandes n'est pas dans `step`.** `step` est une fonction physique pure. Les bornes sont appliquées par le contrôleur (partie E), qui doit de toute façon recalculer le bruit effectif après saturation.

### 3.2 Interface

```python
def step(state, control, dt, vehicle):
    """state (..., 4), control (..., 2) -> (..., 4). Vectorized over leading axes."""
```

`vehicle` est la dataclass `Vehicle` de la partie A. Aucun `4` littéral dans le corps de la fonction (décision d'architecture du README).

**Q-B1.** Pourquoi la forme au centre de gravité plutôt que la forme à l'essieu arrière, plus simple ?

**Q-B2.** À braquage constant δ et vitesse constante v, la voiture décrit un cercle. Quel rayon ? (Tu en as besoin pour un test.)

**Q-B3.** Euler explicite est-il un problème pour ce modèle à dt = 20 ms ?

### 3.3 Tests à écrire

Dans `python/tests/test_step1.py` :

1. **Ligne droite.** δ = 0, a = 0, v = 2 m/s, 50 pas de 20 ms : la voiture est en `(2, 0)`, cap et vitesse inchangés.
2. **Cercle.** δ constant, v constante, petit dt (1 ms) pendant la durée d'un tour : retour au point de départ à 2 cm près, ψ ≈ 2π.
3. **Convention de signe.** δ > 0 fait tourner à gauche : y > 0 et ψ > 0. C'est le test le plus important de la partie. Une erreur de signe ici donne un contrôleur qui « marche presque ».
4. **Batch = boucle.** `step` sur un batch `(8, 4)` donne la même chose que 8 appels sur `(4,)`.
5. **Pas de marche arrière.** v = 0,01 et a = −4 donnent v = 0.

Lance-les avec `pixi run test`.

---

## 4. Partie C : générer la piste et la grille

C'est la plus grosse partie. Prends le temps de **regarder** chaque résultat intermédiaire.

### 4.1 La chaîne de traitement

```
points de contrôle ──splprep(per=1)──▶ spline périodique
      ──échantillonnage dense──▶ abscisse curviligne s(u)
      ──interpolation inverse──▶ points tous les 5 cm (ligne centrale, cap, courbure)
      ──cKDTree sur la ligne centrale──▶ pour chaque cellule : point le plus proche
      ──projection sur la normale──▶ d signé ; s du point le plus proche
      ──▶ grille (ny, nx, 2) float32 ──▶ .npz + .bin
```

### 4.2 Les points de contrôle (fournis)

Ils sont dans le script, pas dans le YAML, conformément à la docstring de `generate_track.py` : c'est le script qui est versionné.

```python
CONTROL_POINTS = np.array([
    [0, 0], [8, 0], [12, 1], [14, 4], [12, 7], [8, 7],
    [5, 9], [1, 10], [-3, 9], [-5, 6], [-4, 2],
], dtype=float)
```

Résultat attendu : un tour de 48,3 m, un rayon de courbure minimal de 2,19 m (vers (12, 7)), parcouru dans le sens trigonométrique. Tu pourras dessiner ta propre piste plus tard. Garde d'abord celle-ci pour pouvoir comparer tes chiffres à ceux des solutions.

### 4.3 Tutoriel `splprep` / `splev`

```python
from scipy.interpolate import splprep, splev

closed = np.vstack([pts, pts[:1]])          # per=1 : le dernier point doit répéter le premier
tck, u = splprep([closed[:, 0], closed[:, 1]], s=0, per=1)
#  s=0   : la spline passe exactement par les points
#  per=1 : spline périodique, raccord C2 au point de départ
x, y   = splev(u_query, tck)                # positions, u_query dans [0, 1)
dx, dy = splev(u_query, tck, der=1)         # dérivées premières par rapport à u
ddx, ddy = splev(u_query, tck, der=2)       # dérivées secondes
```

**Le piège : `u` n'est pas l'abscisse curviligne.** Le paramètre `u` ne progresse pas à vitesse constante le long de la courbe. Échantillonner `u` uniformément donne des points serrés dans certaines zones et espacés dans d'autres. Il faut donc :

1. échantillonner `u` très finement (20 000 points sur [0, 1), `endpoint=False`) ;
2. calculer les longueurs des segments, y compris celui qui referme la boucle (`np.diff(x, append=x[0])`), puis leur somme cumulée `s_dense` (commençant à 0) et la longueur totale `L` ;
3. choisir `n = round(L / ds)` et `s = arange(n) * L / n`, pour que le pas réel divise exactement le tour ;
4. inverser `s(u)` par interpolation : `u_s = np.interp(s, s_dense, u_dense)` ;
5. évaluer la spline en `u_s`.

**Cap et courbure** (à partir des dérivées en `u`, valables quel que soit le paramétrage) :

$$
\theta = \operatorname{atan2}(y', x'), \qquad
\kappa = \frac{x' y'' - y' x''}{(x'^2 + y'^2)^{3/2}}.
$$

**Q-C1.** Le script doit refuser une piste dont le rayon de courbure minimal est trop petit. Trop petit par rapport à quoi ? Donne deux bornes.

### 4.4 Tutoriel `cKDTree` et remplissage de la grille

**Géométrie de la grille.**

- Boîte englobante de la ligne centrale, élargie de `track_half_width + margin` de chaque côté, ce qui donne `x_min, y_min, x_max, y_max`.
- `nx = ceil((x_max − x_min) / res)`, `ny` de même.
- La cellule `(iy, ix)` couvre `[x_min + ix·res, x_min + (ix+1)·res[` en x, et de même en y. Son centre est en `x_min + (ix + 0.5)·res`.
- Disposition mémoire : `grid[iy, ix, c]`, avec la **ligne = y** et le **canal c = 0 pour d, 1 pour s**. Les deux valeurs d'une cellule sont adjacentes en mémoire, et cela correspondra à une texture `float2` en CUDA.

```python
from scipy.spatial import cKDTree

xs = x_min + (np.arange(nx) + 0.5) * res
ys = y_min + (np.arange(ny) + 0.5) * res
X, Y = np.meshgrid(xs, ys)                 # formes (ny, nx) : indexation "xy", ligne = y
P = np.column_stack([X.ravel(), Y.ravel()])
dist, idx = cKDTree(centerline).query(P)   # idx[i] = indice du point de ligne centrale le plus proche
```

**Écart latéral signé.** Normale **à gauche** de la direction de la piste : `n = (−sin θ, cos θ)`. Alors `d = (P − c[idx]) · n[idx]`, positif à gauche de la ligne centrale, conformément au repère du projet (y à gauche). Pour un produit scalaire ligne à ligne : `np.einsum("ij,ij->i", A, B)`.

**Progression.** `s = s_centerline[idx]`.

**Ordre de grandeur.** Avec la piste fournie : 454 × 279 cellules, soit environ 1 Mo en float32. Le KD-tree fait tout en moins d'une seconde.

**Q-C2.** Dans une épingle, ou quand deux portions de piste se rapprochent, que se passe-t-il pour `d` et `s` dans la grille ? Pourquoi la piste fournie n'a-t-elle pas ce problème ?

### 4.5 Lecture de la grille (lookup)

```python
def lookup(track, x, y):
    ix = floor((x - x_min) / res), ramené dans [0, nx-1]
    iy = floor((y - y_min) / res), ramené dans [0, ny-1]
    return grid[iy, ix, 0], grid[iy, ix, 1]     # d, s
```

- **Plus proche cellule, pas d'interpolation bilinéaire** (voir Q-C3).
- **Hors grille : on ramène l'indice dans la grille (clamp).** La bordure de la grille est à plus de `margin` du bord de piste, donc la cellule de bordure est déjà franchement hors piste. C'est aussi exactement ce que fait le mode d'adressage `cudaAddressModeClamp` d'une texture CUDA : pas de branche, parité gratuite à l'étape 3.
- `np.floor(...).astype(np.int64)` puis `np.clip`. N'utilise pas `astype(int)` seul, qui tronque vers zéro : −0,3 donnerait 0 au lieu de −1.

**Q-C3.** Pourquoi ne pas interpoler bilinéairement la grille ? Distingue `d` et `s`.

### 4.6 Le `wrap` de la progression

$$
\operatorname{wrap}(\Delta s) = \big((\Delta s + L/2) \bmod L\big) - L/2 \in [-L/2, L/2[.
$$

En NumPy, `%` sur des flottants donne un résultat du signe du diviseur. La formule fonctionne donc telle quelle pour des Δs négatifs.

**Q-C4.** Que se passe-t-il sans `wrap` quand la voiture approche de la ligne de départ ?

### 4.7 Export binaire pour le C++

Format, **little-endian** :

| Offset | Type | Champ |
|---|---|---|
| 0 | `char[4]` | magic `MPPG` |
| 4 | `uint32` | version = 1 |
| 8 | `int32` | nx |
| 12 | `int32` | ny |
| 16 | `float32` | x_min |
| 20 | `float32` | y_min |
| 24 | `float32` | res |
| 28 | `float32` | longueur du tour L |
| 32 | `float32[ny][nx][2]` | grille, row-major |

```python
import struct
HEADER = struct.Struct("<4sIiiffff")      # "<" = little-endian, sans padding ; taille 32
f.write(HEADER.pack(b"MPPG", 1, nx, ny, x_min, y_min, res, length))
f.write(np.ascontiguousarray(grid, dtype="<f4").tobytes())
# relecture
magic, ver, nx, ny, x_min, y_min, res, L = HEADER.unpack_from(raw)
grid = np.frombuffer(raw, dtype="<f4", offset=HEADER.size).reshape(ny, nx, 2)
```

Le `.npz` contient la grille, les métadonnées, la ligne centrale et le cap (utiles pour les figures) : `np.savez(path, grid=..., x_min=..., ...)` puis `z = np.load(path); z["grid"]`.

**Q-C5.** Pourquoi un champ magic et un champ version, pour un fichier que toi seul utilises ?

### 4.8 Vérifications obligatoires

1. **Dans le script** : `assert` sur le rayon minimal (Q-C1), et relecture du `.bin` comparée bit à bit à la grille (`np.array_equal`).
2. **Visuellement** : affiche les deux canaux avec `plt.imshow(grid[..., 0], origin="lower", extent=[x_min, x_max, y_min, y_max])`, puis la même chose pour `grid[..., 1]`. `origin="lower"` est indispensable, sinon l'image est retournée verticalement. Tu dois voir :
   - pour `d` : un dégradé qui change de signe sur la ligne centrale, positif à l'intérieur de la boucle (piste parcourue dans le sens trigonométrique, donc l'intérieur est à gauche) ;
   - pour `s` : un dégradé qui fait le tour, avec **une seule** discontinuité, sur la ligne de départ. Une autre discontinuité près de la piste signale le problème de Q-C2.
3. **Tests unitaires** :
   - un point de la ligne centrale donne `|d| < res`, et un `s` égal à son abscisse à `2·res` près (après `wrap`) ;
   - un point décalé de +0,5 m selon la normale gauche donne `d ≈ +0,5` ;
   - `wrap(−47, 48) = 1` ;
   - le `.bin` relu est égal au `.npz`.

---

## 5. Partie D : la fonction de coût

### 5.1 Les termes

Pour un état `x_t` (après le pas) et la commande `u_{t-1}` qui y a mené. Avec `d0 = lateral_scale`, `v0 = speed_scale` et `d_max = track_half_width − width/2` (= 0,65 m) :

| Terme | Formule | Rôle |
|---|---|---|
| latéral | `w_lateral · (d/d0)²` | rester près de la ligne centrale |
| sortie de piste | `w_offtrack · 1[abs(d) > d_max] · (1 + (abs(d) − d_max)/d0)` | interdit, mais gradué (théorie §5.4) |
| vitesse | `w_speed · ((v − v_ref)/v0)²` | rouler à `v_ref` |
| effort | `w_control · Σ_j (u_j / u_max,j)²` | commandes modérées |
| adhérence | `w_adhesion · max(0, a_lat/(μg) − 1)²`, avec `a_lat = v²·abs(tan δ)/L` | interdire les virages que les pneus ne tiendraient pas (théorie §6.2) |

Coût **terminal** (une seule fois, sur `x_T`) :

$$
\phi = -w_\text{progress} \cdot \frac{\operatorname{wrap}(s_T - s_0)}{v_\text{ref}\, T\, \Delta t}
$$

où `s_0` est la progression de l'état initial `x0` du rollout (le même pour les K trajectoires).

Signature conseillée :

```python
def stage_cost(state, control, track, cfg):      # state (K, 4), control (K, 2) -> (K,)
def terminal_cost(state_T, s0, track, cfg):      # state_T (K, 4) -> (K,)
```

**Q-D1.** Le plan initial (`edge-mppi.md`) ne prévoyait que latéral + progression + sortie + effort. Pourquoi ajouter un terme de vitesse ? Essaie de prédire ce que fait la voiture sans lui. Tu vérifieras en partie G.

**Q-D2.** Pourquoi `d_max = demi-largeur − width/2`, et pas la demi-largeur ?

**Q-D3.** Calcule l'ordre de grandeur du coût d'une **bonne** trajectoire sur l'horizon (T = 30), terme par terme. C'est l'échelle à laquelle λ devra se comparer.

**Q-D4.** Pourquoi le terme d'adhérence est-il propre au modèle cinématique ?

---

## 6. Partie E : le contrôleur MPPI

### 6.1 Découpage imposé par l'étape 3

Sépare la partie **déterministe** de la partie aléatoire :

```python
def rollout_costs(x0, U, eps, track, cfg):
    """x0 (4,), U (T, m), eps (K, T, m) bruit brut
    -> S (K,), eps_effectif (K, T, m), X (K, T+1, n)"""

class MPPI:
    def __init__(self, cfg, track): ...      # U = zeros((T, m)), rng = default_rng(seed)
    def command(self, x0): ...               # -> u0, info
```

`rollout_costs` est exactement ce que `bench/check_parity.py` appellera : même `x0`, même `U`, même `eps`, et K coûts à comparer au kernel. Elle ne doit **rien** tirer au hasard.

### 6.2 L'algorithme

Il est décrit pas à pas en théorie §2.2 et §2.3. Résumé de ce que tu dois coder :

**Dans `rollout_costs`** :

1. `V = clip(U[None] + eps, u_min, u_max)`, puis `eps = V − U[None]` ;
2. `X[:, 0] = x0`, puis pour `t` de 0 à T−1 (boucle Python sur t, **vectorisée sur K**) : `X[:, t+1] = step(X[:, t], V[:, t])` et `S += stage_cost(X[:, t+1], V[:, t])` ;
3. `S += terminal_cost(X[:, T], s0)` ;
4. `S += gamma · Σ_t U_t^T Σ^{-1} eps_t` : avec Σ diagonale, c'est `np.einsum("tj,ktj->k", U / sigma**2, eps)` ;
5. remplace les coûts non finis par `S_MAX = 1e6`.

**Dans `command`** :

1. `eps = rng.standard_normal((K, T, m)) * noise_std` ;
2. `S, eps, X = rollout_costs(...)` ;
3. `ρ = min S`, `w = exp(−(S − ρ)/λ)`, puis `w /= w.sum()` ;
4. `ESS = 1 / Σ w²` ;
5. `U ← U + tensordot(w, eps, axes=1)` ;
6. `u0 = U[0].copy()` ;
7. décalage : `U[:-1] = U[1:]`, puis `U[-1] = U[-2]` ;
8. renvoie `u0` et un dict `info` (ESS, ρ, S, w, X) pour les diagnostics.

Attention au `.copy()` à l'étape 6 : `U[0]` est une **vue**, et le décalage de l'étape 7 l'écrase.

**Q-E1.** Pourquoi recalculer `eps` après la saturation ?

**Q-E2.** Pourquoi soustraire ρ avant l'exponentielle, alors que la normalisation le compense de toute façon ?

**Q-E3.** Au décalage, pourquoi recopier l'avant-dernière commande plutôt que mettre zéro ?

**Q-E4.** Pourquoi la boucle sur t ne peut-elle pas être vectorisée, alors que celle sur K l'est ? Quel lien avec le kernel CUDA ?

### 6.3 Test

Pour un `x0`, un `U = 0` et un `eps` fixés (64 échantillons) : deux appels à `rollout_costs` donnent des `S` identiques (`np.array_equal`), tous finis, et `U + eps_effectif` reste dans les bornes.

---

## 7. Partie F : la simulation en boucle fermée

### 7.1 Ce que fait `run_sim.py`

```
charger cfg et la piste (.npz)
x = [x_c[0], y_c[0], θ[0], 0]        # départ arrêté, sur la ligne centrale, dans l'axe
pour k < max_steps :
    u, info = mppi.command(x)
    x_next  = step(x, u, dt)         # le "vrai" système est le même modèle (modèle parfait)
    journaliser (x, u, x_next, ESS, ρ, d, temps de calcul)
    progression cumulée += wrap(s(x_next) − s(x))
    x = x_next
    si progression ≥ L : tour fini
afficher le bilan ; sauver figures et trajectoires
```

**Mesurer le temps** : `time.perf_counter()` autour de `command` uniquement.

### 7.2 Sorties

- **Bilan en console** : tour fini ou non, temps au tour, nombre de pas hors piste, `max abs(d)`, vitesse moyenne et maximale, accélération latérale maximale, ESS (médiane, 5e et 95e centiles), temps médian d'une itération.
- **`results/figures/step1_trajectory.png`** : bords de piste (ligne centrale ± `track_half_width · n`), trajectoire colorée par la vitesse et, à un instant choisi (par exemple à l'entrée du premier virage), le **faisceau** des 200 rollouts les plus lourds, colorés par leur poids. C'est le diagnostic le plus utile du projet (théorie §4.2).
- **`results/figures/step1_cost.png`** : quatre courbes en fonction du temps, ρ (coût minimal), ESS, d (avec les droites ±d_max) et δ.
- **`results/trajectories/step1_kinematic.npz`** : les triplets `(state, control, next_state)` exigés par la docstring de `run_sim.py`. Ils serviront à l'étape 2 et au modèle appris plus tard.

Dans un script lancé sans affichage (SSH), mets `matplotlib.use("Agg")` avant d'importer `pyplot`.

### 7.3 Résultats de référence

Avec la config de la partie A, la solution de référence donne (machine de dev, WSL2, NumPy) :

```
lap done in 16.10 s, progress 48.4/48.3 m
offtrack steps 0, max |d| 0.057 m (limit 0.65)
v mean 2.99 max 3.24 m/s, max a_lat 6.3 m/s2
ESS median 444 p5 104 p95 519 / K=1024
iteration time median 8.2 ms
```

Tes chiffres exacts différeront (ordre de tirage du bruit, détails d'implémentation), mais les ordres de grandeur doivent être les mêmes. Garde le temps d'itération : c'est le premier point de comparaison avec CUDA.

---

## 8. Partie G : réglage et compréhension des paramètres

Le critère de sortie demande une compréhension **qualitative** des paramètres. Pour chaque expérience : **écris ta prédiction avant de lancer**, puis compare. Les résultats mesurés sont dans les solutions (S-G).

Pour enchaîner les variantes sans toucher au YAML, écris un petit `sweep.py` qui charge la config, la modifie avec `dataclasses.replace` et appelle une fonction `run(cfg)` renvoyant un dict de métriques. Ajoute à ces métriques la **gigue de braquage** `mean(abs(diff(δ)))`, qui mesure le bruit haute fréquence de la commande (théorie §7.5).

| # | Expérience | Question |
|---|---|---|
| G1 | λ ∈ {1, 0,3, 0,1, 0,03} | Comment évoluent l'ESS et la gigue ? Quel λ choisir ? |
| G2 | `w_speed = 0` | Prédiction de Q-D1 ? |
| G3 | `v_ref = 5`, puis 7, à λ = 0,3 | Que devient l'ESS ? Pourquoi ? |
| G4 | `noise_std = [0.5, 0.02]` puis `[0.5, 0.3]` | Effet sur l'ESS et la gigue ? |
| G5 | `gamma = λ` | Effet sur la gigue et le temps au tour ? |
| G6 | `T = 10`, `v_ref = 5` | Que fait un horizon court ? |
| G7 | `seed = 1` | Le comportement doit-il changer ? |

**Méthode de réglage** (théorie §4.6). Normaliser le coût (fait en partie D), choisir Σ en regardant le faisceau, régler λ pour une ESS entre 1 % et 10 % de K au minimum, puis vérifier la gigue. À K = 1024, une ESS médiane de quelques centaines est confortable.

---

## 9. Dépannage

| Symptôme | Cause probable | Vérification |
|---|---|---|
| La voiture tourne du mauvais côté, ou part en spirale | signe de δ ou de la normale | test « δ > 0 tourne à gauche », image de `d` |
| La voiture ralentit puis s'arrête, ESS ≈ K | coût plat (pas de terme de vitesse, λ trop grand) | G2, théorie §7.2 |
| Le braquage vibre de pas en pas, ESS ≈ 1 | λ trop petit pour l'échelle du coût | histogramme de `(S − ρ)/λ`, théorie §7.1 |
| La voiture freine devant la ligne de départ | `wrap` manquant | Q-C4 |
| La progression fait un saut près de la piste | grille ambiguë (Q-C2) | image de `s` |
| `RuntimeWarning: overflow in exp` ou `w` en NaN | ρ non soustrait, ou coût NaN | Q-E2, `np.isfinite(S).all()` |
| La commande sort des bornes | `eps` non recalculé après clip | Q-E1 |
| Les résultats changent d'une exécution à l'autre | RNG global ou non initialisé | `default_rng(seed)` |
| `ModuleNotFoundError: mppi` dans pytest | `PYTHONPATH` | §1.1 |
| L'image de la grille est retournée verticalement | `imshow` sans `origin="lower"` | §4.8 |
| Une itération prend plus de 50 ms | boucle Python sur K | seule la boucle sur t est en Python |

---

## 10. Clôturer l'étape

1. `pixi run track && pixi run test && pixi run sim` passent.
2. Coche le critère de sortie de la [section 0](#0-objectif-et-critère-de-sortie).
3. Écris l'entrée du journal (`docs/journal.md`) en suivant son format. Dans **Measured**, mets le bilan console de la simulation et le tableau de la partie G **avec la config utilisée**. Dans **No effect / reverted**, mets ce que tu as essayé sans succès. Ajoute la liste des champs YAML créés, à reporter dans `config.hpp`.
4. Coche `Step 1` dans le README.
5. Commit, puis `git tag step-1`.

---
---

# Solutions

Ne lis une solution qu'après avoir écrit ta propre réponse.

## Réponses aux questions

**S-A1.** Une incohérence `state_dim`/`model` ne plante pas forcément. Un tableau `(K, 6)` passé à un modèle qui lit les colonnes 0 à 3 tourne sans erreur et produit des résultats faux. Échouer tôt, avec un message explicite, coûte une ligne. Le bug silencieux coûte une demi-journée. Le même raisonnement vaut pour `lf + lr = wheelbase`.

**S-A2.** Un `.npy` ne contient qu'un tableau. Or Python a besoin de la grille **et** de ses métadonnées (`x_min`, `y_min`, `res`, `L`), plus de la ligne centrale pour les figures. Un `.npz` (archive de plusieurs `.npy`) regroupe tout dans un seul fichier, sans format maison. Côté C++, le `.bin` porte les métadonnées dans son en-tête.

**S-B1.** Le modèle dynamique de l'étape 2 est écrit au centre de gravité, et la bascule à basse vitesse mélange les deux modèles état par état (théorie §6.5). Ce mélange n'a de sens que si les deux modèles décrivent le même point. Écrire le cinématique au CdG dès maintenant évite de le réécrire, et garde le test croisé cinématique/dynamique de l'étape 2 direct.

**S-B2.** Avec ψ̇ = v cos β tan δ / L et une vitesse de norme v, le rayon vaut R = v/ψ̇ = L / (cos β · tan δ). Pour δ = 0,2 rad : β = 0,111 rad et R ≈ 1,63 m. Le test utilise un petit dt, parce qu'Euler sur un cercle accumule une erreur en O(dt).

**S-B3.** Non. Le modèle cinématique n'a aucune dynamique raide : pas de terme qui ramène une grandeur vers zéro avec une constante de temps courte. Le cap avance de ψ̇·dt, au plus 0,34 rad par pas à 8 m/s et δ = 0,4. Euler introduit une petite erreur de position, mais le système simulé et le modèle des rollouts sont **le même code**, donc cette erreur ne crée aucun écart entre prédiction et réalité. Le problème d'Euler apparaîtra à l'étape 2 (théorie §6.5).

**S-C1.** Deux bornes :
- **R_min > demi-largeur de piste.** Le bord intérieur est la courbe décalée de `hw` vers l'intérieur. Si le rayon de courbure de la ligne centrale descend sous `hw`, ce bord se replie sur lui-même et forme un rebroussement, ce qui rend la grille absurde dans cette zone.
- **R_min > rayon de braquage minimal**, soit L / tan(0,4) = 0,78 m au point de l'essieu arrière, avec de la marge. La voiture doit pouvoir suivre la ligne centrale sans être en butée.

La solution prend `R_min > 2·hw` = 1,6 m, qui couvre les deux avec de la marge. La piste fournie est à 2,19 m.

**S-C2.** Une cellule prend le `d` et le `s` du point de ligne centrale **le plus proche**, qui peut appartenir à l'autre portion de piste. Sur la frontière entre les deux zones d'influence, `s` saute de plusieurs mètres et `d` change brutalement. Un rollout qui traverse cette frontière reçoit une progression fictive énorme et MPPI coupe à travers l'herbe (théorie §5.4). La frontière se trouve à mi-distance entre les deux lignes centrales. Elle est inoffensive tant que ces lignes sont séparées de plus de `2·(hw + margin)`, car elle tombe alors loin hors piste, dans une zone où la pénalité de sortie domine de toute façon. Sur la piste fournie, les portions les plus proches (en bas et en haut) sont à plus de 7 m. Pour une piste plus serrée, il faudrait vérifier cette distance dans le script.

**S-C3.**
- **`s`** : l'interpolation bilinéaire est fausse sur la ligne de départ, où `s` saute de L à 0. La moyenne de 48,2 et 0,05 donne 24, au milieu du tour. Il faudrait interpoler les écarts avec `wrap`, ce qui complique tout.
- **`d`** : l'interpolation serait correcte et plus précise (l'erreur au plus proche voisin est au plus `res·√2/2` ≈ 3,5 cm, négligeable devant `d0 = 0,4 m`). Mais elle coûte quatre lectures au lieu d'une, et la parité NumPy/CUDA demande que les deux côtés interpolent exactement de la même façon. Les textures CUDA interpolent en virgule fixe 9 bits, ce qui donnerait des écarts supérieurs à 1e-4.

Plus proche cellule partout : simple, identique des deux côtés. On pourra y revenir à l'étape 4 si la précision manque.

**S-C4.** Juste avant la ligne, `s_0 ≈ 48`. Un rollout qui la franchit finit à `s_T ≈ 0,5`, soit `s_T − s_0 ≈ −47,5` m : une énorme pénalité de progression. Tous les rollouts qui franchissent la ligne sont éliminés, et la voiture freine devant la ligne et s'y arrête. Le même bug touche le comptage des tours dans `run_sim.py`.

**S-C5.** Si l'en-tête change à l'étape 4 (par exemple en ajoutant un canal pour le cap de la piste), un vieux `.bin` lu par du nouveau code doit produire une erreur claire, pas des cellules décalées d'un octet qui donneraient une piste aberrante sans aucun message. Le magic attrape aussi un mauvais fichier, et le test d'endianness est gratuit.

**S-D1.** Sans terme de vitesse, seule la progression terminale pousse à avancer. Deux échecs :
1. La progression ne limite pas la vitesse. La voiture accélère dans les lignes droites. Avec un horizon de 0,6 s, elle ne voit pas assez loin pour freiner à temps (à 4 m/s², l'arrêt depuis 5 m/s prend 1,25 s), elle freine brutalement dans le virage et peut finir arrêtée.
2. À l'arrêt, le coût est **plat**. Avec σ_a = 0,5 m/s² sur 0,6 s, les rollouts avancent tous de quelques centimètres, la progression les distingue à peine, l'ESS tend vers K et la commande devient inerte (théorie §7.2). La voiture reste arrêtée.

C'est exactement ce qu'a fait la première version de la solution de référence : 5,6 m/s en pointe, puis un arrêt définitif au bout de 18 s, à 24 m. Le terme de vitesse rend le coût informatif partout, y compris à l'arrêt.

**S-D2.** `d` est mesuré au centre de gravité, mais ce sont les roues qui sortent. La voiture fait 30 cm de large, donc ses roues touchent le bord quand le CdG est à `0,8 − 0,15 = 0,65` m de la ligne centrale (théorie §5.2 et §7.6). On pourrait ajouter une marge de sécurité.

**S-D3.** Pour une trajectoire correcte, avec T = 30 :
- latéral : `d ≈ 0,05 à 0,1 m`, donc `(d/0,4)² ≈ 0,02 à 0,06` par pas, soit environ 1 sur l'horizon ;
- vitesse : `abs(v − v_ref) ≈ 0,1 à 0,3`, donc 0,01 à 0,1 par pas, soit 0,3 à 3 ;
- effort : `0,01 · ((a/4)² + (δ/0,4)²)` ≈ 0,01 par pas, soit 0,3 ;
- adhérence : 0 si tout va bien ;
- progression : environ −10 (le véhicule parcourt à peu près `v_ref·T·dt`).

Le coût total est d'environ −10 à −8 (dans la simulation de référence, ρ a une médiane de −10,1) et, surtout, la **dispersion** entre échantillons voisins est de l'ordre de l'unité. C'est cette dispersion qui fixe λ : avec s ≈ 0,5 à 1 et λ = 0,3, on est dans la zone visée. En revanche, un seul pas hors piste coûte 100. Les sorties de piste reçoivent donc un poids nul, ce qui est voulu.

**S-D4.** Dans le modèle cinématique, les pneus n'ont pas de limite : à δ = 0,4 rad, la voiture tourne sur 0,78 m de rayon à n'importe quelle vitesse, ce qui représente 82 m/s² latéraux à 8 m/s (théorie §6.2). Le contrôleur planifierait des virages physiquement impossibles. Ce terme les interdit par le coût. Avec le modèle dynamique de l'étape 2, la saturation des pneus est dans le modèle et le terme devient inutile, voire nuisible.

**S-E1.** Si une commande perturbée est écrêtée, le rollout a simulé `V`, pas `U + eps`. Moyenner les `eps` bruts pousserait `U` vers des commandes jamais simulées, et hors des bornes. Avec `eps = V − U`, la mise à jour `U + Σ w·eps = Σ w·V` est une combinaison convexe de commandes admissibles, donc admissible (théorie §2.2, étape 2).

**S-E2.** Les poids normalisés sont mathématiquement inchangés, mais numériquement, `exp(−S/λ)` vaut 0 en float64 dès que `S/λ > 745` (dès 103 en float32). Avec des coûts de 150 à 400 et λ = 0,3, tous les poids valent 0, ce qui donne `0/0 = NaN`. Après soustraction, le meilleur échantillon a un poids brut exactement égal à 1, donc la somme vaut au moins 1.

**S-E3.** La commande de fin d'horizon est la meilleure estimation pour le pas suivant. En plein virage, la meilleure hypothèse est de garder le braquage. Mettre zéro imposerait à chaque itération un « redressement » en fin d'horizon que MPPI devrait ensuite corriger (théorie §2.2, étape 8).

**S-E4.** `x_{t+1}` dépend de `x_t` : la boucle en temps est intrinsèquement séquentielle. Les K rollouts, eux, n'échangent rien. En NumPy, on vectorise donc sur K et on boucle sur T : T = 30 appels de fonctions qui traitent chacun 1024 états. Le kernel CUDA fait l'inverse : un thread par rollout (parallèle sur K) et une boucle sur T dans le thread, avec l'état gardé dans les registres (théorie §2.4 et §2.5). C'est la raison d'être du projet.

## S-G : résultats mesurés de la partie G

Mesurés avec la solution de référence, la config de la partie A et un seul paramètre modifié à la fois. La gigue est `mean(abs(Δδ))` en rad par pas.

| Variante | Tour | Temps (s) | ESS médiane | ESS p5 | Gigue δ | `max abs(d)` (m) |
|---|---|---|---|---|---|---|
| λ = 1 | oui | 16,44 | 776 | 336 | 0,027 | 0,14 |
| **λ = 0,3 (réf.)** | **oui** | **16,12** | **444** | **104** | **0,037** | **0,06** |
| λ = 0,1 | oui | 16,24 | 140 | 53 | 0,049 | 0,06 |
| λ = 0,03 | oui | 16,32 | 9 | 1 | 0,110 | 0,05 |
| `w_speed = 0` | **non**, 24,2 m en 60 s | n/a | 1023 | 421 | 0,018 | 0,20 |
| `v_ref = 5` | oui | 9,56 | **1** | 1 | 0,116 | 0,25 |
| `v_ref = 7` | oui | 7,56 | **1** | 1 | 0,125 | 0,56 |
| `σ_δ = 0,02` | oui | 16,14 | 790 | 22 | 0,009 | 0,13 |
| `σ_δ = 0,3` | oui | 16,80 | 65 | 6 | 0,057 | 0,05 |
| `γ = λ = 0,3` | oui | 17,04 | 423 | 340 | 0,009 | 0,10 |
| `T = 10`, `v_ref = 5` | oui | 6,92 | 1 | 1 | 0,055 | **0,65** |
| `seed = 1` | oui | 16,10 | 438 | 217 | 0,036 | 0,09 |

**Ce qu'il faut en retenir.**

- **G1, λ.** C'est un compromis entre suivi et bruit de commande. Quand λ baisse, l'ESS chute et la gigue monte : à λ = 0,03, la commande recopie le bruit du meilleur échantillon (ESS ≈ 1 à 9), avec une gigue 4 fois supérieure. À λ = 1, la commande est plus douce mais le suivi se dégrade (`max abs(d)` est multiplié par 2,3). λ = 0,3 est un bon point, avec une ESS ≈ 40 % de K. La tâche est facile à 3 m/s : ici, même λ = 0,03 boucle le tour.
- **G2.** Confirme S-D1 : sans terme de vitesse, la voiture finit arrêtée avec une ESS ≈ K. C'est le coût qui est plat, pas λ qui est mal réglé.
- **G3, la leçon la plus importante.** En passant `v_ref` à 5, la voiture boucle toujours, mais l'ESS tombe à 1. Le coût n'a pas changé de forme, mais son **échelle** a changé : les écarts de vitesse, l'adhérence et la progression augmentent avec la vitesse. λ = 0,3 est maintenant beaucoup trop petit. **λ se règle relativement à l'échelle du coût, et doit être revérifié à chaque changement de poids ou de consigne** (théorie §4.1). Exercice : trouve le λ qui remet l'ESS vers 100 à `v_ref = 5`.
- **G4, Σ.** Un σ_δ petit donne une commande douce (gigue 0,009), mais une exploration étroite. L'ESS p5 tombe à 22 dans les virages : il arrive qu'aucun échantillon ne braque assez. Un σ_δ grand élargit l'exploration, mais les échantillons sortent de piste et l'ESS chute. Modifier Σ modifie la dispersion des coûts, donc il faut revérifier λ.
- **G5, γ.** Avec `γ = λ`, le terme de correction tire `U` vers zéro : la commande est nettement plus douce (gigue 0,009), mais la voiture est un peu plus lente (17,0 s contre 16,1 s), car le terme pénalise aussi l'accélération nécessaire pour tenir 3 m/s. Les implémentations réelles utilisent `γ ≪ λ` (théorie §4.5). Garder `γ = 0` à l'étape 1 est un choix défendable.
- **G6, horizon court.** Avec 0,2 s d'horizon, la normalisation de la progression par `v_ref·T·dt` donne un poids énorme à quelques centimètres gagnés. La voiture roule à 7,4 m/s de moyenne, bien au-dessus de `v_ref`, et frôle la limite de sortie (`max abs(d) = d_max` = 0,65 m). Le contrôleur est myope (théorie §4.3).
- **G7.** Changer la graine ne change presque rien : temps au tour identique à 0,02 s près, ESS et gigue identiques. Le résultat ne tient pas à un tirage chanceux. Même à λ = 0,03, la graine 1 donne 16,32 s et une gigue de 0,108, comme la graine 0 : sur cette piste facile, la sensibilité à la graine décrite en théorie §7.1 ne se voit pas sur le temps au tour, seulement sur la gigue, qui est haute quelle que soit la graine.

## Code de référence

Le code est testé : `pixi run track && pixi run test && pixi run sim` donnent les résultats des sections 7.3 et S-G.

### `config/mppi.yaml`

```yaml
# Single source of truth for both the Python reference and the C++/CUDA path.
# SI units and radians everywhere. psi is measured from the x axis.
# State order is fixed: [x, y, psi, v] kinematic, [x, y, psi, vx, vy, r] dynamic.

model: kinematic          # kinematic | dynamic
state_dim: 4              # 4 for kinematic, 6 for dynamic
control_dim: 2            # [a, delta]

mppi:
  num_samples: 1024       # K, raised to 8192 once the CUDA path is validated
  horizon: 30             # T, raised to 50
  dt: 0.02
  lambda: 0.3
  gamma: 0.0              # importance-sampling correction weight, <= lambda
  noise_std: [0.5, 0.1]   # [a, delta]
  seed: 0                 # fixed noise for the parity check

control_bounds:
  a_min: -4.0
  a_max: 4.0
  delta_min: -0.4
  delta_max: 0.4

vehicle:                  # F1TENTH parameters
  wheelbase: 0.33
  lf: 0.15
  lr: 0.18
  width: 0.30
  mu: 1.0
  mass: 3.5
  izz: 0.04
  cornering_stiffness_front: 4.0
  cornering_stiffness_rear: 4.2
  kinematic_blend_speed: 1.5

cost:                     # every term normalized by its scale, see docs/mppi-theory.md 5.3
  w_lateral: 1.0
  w_progress: 10.0
  w_offtrack: 100.0
  w_control: 0.01
  w_adhesion: 10.0
  w_speed: 1.0
  lateral_scale: 0.4
  speed_scale: 1.0
  v_ref: 3.0
  track_half_width: 0.8

track:
  grid_resolution: 0.05
  resample_step: 0.05
  margin: 1.0
  npz_path: results/track/costmap.npz
  bin_path: results/track/costmap.bin

sim:
  max_steps: 3000
```

### `python/mppi/config.py`

```python
"""Loader for config/mppi.yaml.

Same file is parsed by cuda/include/mppi/config.hpp. Any field added here must
be added there, otherwise the Python/CUDA parity check compares two different
problems.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

STATE_DIM = {"kinematic": 4, "dynamic": 6}


@dataclass(frozen=True)
class MppiParams:
    num_samples: int
    horizon: int
    dt: float
    lam: float
    gamma: float
    noise_std: np.ndarray
    seed: int


@dataclass(frozen=True)
class Bounds:
    u_min: np.ndarray
    u_max: np.ndarray


@dataclass(frozen=True)
class Vehicle:
    wheelbase: float
    lf: float
    lr: float
    width: float
    mu: float


@dataclass(frozen=True)
class CostParams:
    w_lateral: float
    w_progress: float
    w_offtrack: float
    w_control: float
    w_adhesion: float
    w_speed: float
    lateral_scale: float
    speed_scale: float
    v_ref: float
    track_half_width: float


@dataclass(frozen=True)
class TrackParams:
    grid_resolution: float
    resample_step: float
    margin: float
    npz_path: Path
    bin_path: Path


@dataclass(frozen=True)
class Config:
    model: str
    state_dim: int
    control_dim: int
    mppi: MppiParams
    bounds: Bounds
    vehicle: Vehicle
    cost: CostParams
    track: TrackParams
    max_steps: int


def load_config(path):
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    root = path.resolve().parent.parent

    model = raw["model"]
    if model not in STATE_DIM:
        raise ValueError(f"unknown model {model!r}")
    if raw["state_dim"] != STATE_DIM[model]:
        raise ValueError(f"state_dim={raw['state_dim']} but model {model} needs {STATE_DIM[model]}")

    m = raw["mppi"]
    noise_std = np.asarray(m["noise_std"], dtype=np.float64)
    if noise_std.shape != (raw["control_dim"],):
        raise ValueError("noise_std must have control_dim entries")

    b = raw["control_bounds"]
    v = raw["vehicle"]
    if abs(v["lf"] + v["lr"] - v["wheelbase"]) > 1e-9:
        raise ValueError("lf + lr must equal wheelbase")
    t = raw["track"]

    return Config(
        model=model,
        state_dim=raw["state_dim"],
        control_dim=raw["control_dim"],
        mppi=MppiParams(m["num_samples"], m["horizon"], m["dt"], m["lambda"],
                        m["gamma"], noise_std, m["seed"]),
        bounds=Bounds(np.array([b["a_min"], b["delta_min"]]),
                      np.array([b["a_max"], b["delta_max"]])),
        vehicle=Vehicle(v["wheelbase"], v["lf"], v["lr"], v["width"], v["mu"]),
        cost=CostParams(**raw["cost"]),
        track=TrackParams(t["grid_resolution"], t["resample_step"], t["margin"],
                          root / t["npz_path"], root / t["bin_path"]),
        max_steps=raw["sim"]["max_steps"],
    )
```

### `python/mppi/dynamics.py`

```python
"""Vehicle dynamics.

(garder l'en-tête de conventions existant, puis ajouter :)

Kinematic bicycle written at the center of gravity, explicit Euler. Speed is
clamped at zero: the vehicle never reverses. Control bounds are not applied
here, the controller saturates before calling step.
"""
import numpy as np


def step(state, control, dt, vehicle):
    x, y, psi, v = state[..., 0], state[..., 1], state[..., 2], state[..., 3]
    a, delta = control[..., 0], control[..., 1]
    tan_d = np.tan(delta)
    beta = np.arctan(vehicle.lr / vehicle.wheelbase * tan_d)
    x_next = x + v * np.cos(psi + beta) * dt
    y_next = y + v * np.sin(psi + beta) * dt
    psi_next = psi + v * np.cos(beta) / vehicle.wheelbase * tan_d * dt
    v_next = np.maximum(v + a * dt, 0.0)
    return np.stack([x_next, y_next, psi_next, v_next], axis=-1)
```

### `python/mppi/track.py`

```python
"""Track generation and cost map baking.

(garder la docstring existante, en remplaçant .npy par .npz)
"""
import struct
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import splev, splprep
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


def centerline_from_points(points, ds):
    closed = np.vstack([points, points[:1]])
    tck, _ = splprep([closed[:, 0], closed[:, 1]], s=0, per=1)

    u = np.linspace(0.0, 1.0, 20000, endpoint=False)
    x, y = splev(u, tck)
    seg = np.hypot(np.diff(x, append=x[0]), np.diff(y, append=y[0]))
    s_dense = np.concatenate([[0.0], np.cumsum(seg)[:-1]])
    length = seg.sum()

    n = int(round(length / ds))
    s = np.arange(n) * length / n
    u_s = np.interp(s, s_dense, u)
    x, y = splev(u_s, tck)
    dx, dy = splev(u_s, tck, der=1)
    ddx, ddy = splev(u_s, tck, der=2)
    curvature = (dx * ddy - dy * ddx) / (dx**2 + dy**2) ** 1.5
    return np.column_stack([x, y]), np.arctan2(dy, dx), s, length, curvature


def bake_grid(centerline, heading, s, res, margin):
    x_min, y_min = centerline.min(axis=0) - margin
    x_max, y_max = centerline.max(axis=0) + margin
    nx = int(np.ceil((x_max - x_min) / res))
    ny = int(np.ceil((y_max - y_min) / res))

    xs = x_min + (np.arange(nx) + 0.5) * res
    ys = y_min + (np.arange(ny) + 0.5) * res
    X, Y = np.meshgrid(xs, ys)                    # (ny, nx), row = y
    P = np.column_stack([X.ravel(), Y.ravel()])

    _, idx = cKDTree(centerline).query(P)
    normal = np.column_stack([-np.sin(heading), np.cos(heading)])
    d = np.einsum("ij,ij->i", P - centerline[idx], normal[idx])
    grid = np.stack([d, s[idx]], axis=-1).reshape(ny, nx, 2).astype(np.float32)
    return grid, float(x_min), float(y_min)


def save(track, npz_path, bin_path):
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, grid=track.grid, x_min=track.x_min, y_min=track.y_min,
             res=track.res, length=track.length,
             centerline=track.centerline, heading=track.heading)
    ny, nx, _ = track.grid.shape
    with open(bin_path, "wb") as f:
        f.write(HEADER.pack(MAGIC, VERSION, nx, ny, track.x_min, track.y_min,
                            track.res, track.length))
        f.write(np.ascontiguousarray(track.grid, dtype="<f4").tobytes())


def load(npz_path):
    z = np.load(npz_path)
    return Track(z["grid"], float(z["x_min"]), float(z["y_min"]), float(z["res"]),
                 float(z["length"]), z["centerline"], z["heading"])


def load_bin(bin_path):
    raw = bin_path.read_bytes()
    magic, version, nx, ny, x_min, y_min, res, length = HEADER.unpack_from(raw)
    assert magic == MAGIC and version == VERSION
    grid = np.frombuffer(raw, dtype="<f4", offset=HEADER.size).reshape(ny, nx, 2)
    return grid, x_min, y_min, res, length


def lookup(track, x, y):
    """Nearest cell, indices clamped to the grid (same as cudaAddressModeClamp)."""
    ny, nx, _ = track.grid.shape
    ix = np.clip(np.floor((x - track.x_min) / track.res).astype(np.int64), 0, nx - 1)
    iy = np.clip(np.floor((y - track.y_min) / track.res).astype(np.int64), 0, ny - 1)
    cell = track.grid[iy, ix]
    return cell[..., 0], cell[..., 1]


def wrap(ds, length):
    return (ds + 0.5 * length) % length - 0.5 * length
```

Remarque : `x_min` et `y_min` sont stockés en float64 dans le `.npz` et en float32 dans le `.bin`. L'écart (environ 1e-7 m) est sans effet ici. À l'étape 3, pour une parité stricte, le plus sûr est que Python relise le `.bin`, ou arrondisse `x_min` et `y_min` en float32 avant de les sauver.

### `python/generate_track.py`

```python
"""Regenerates results/track/costmap.{npz,bin} from the control points.

The outputs are gitignored. This script is the thing that is versioned.
"""
import argparse

import numpy as np

from mppi.config import load_config
from mppi import track as trk

CONTROL_POINTS = np.array([
    [0, 0], [8, 0], [12, 1], [14, 4], [12, 7], [8, 7],
    [5, 9], [1, 10], [-3, 9], [-5, 6], [-4, 2],
], dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/mppi.yaml")
    cfg = load_config(ap.parse_args().config)
    tp, hw = cfg.track, cfg.cost.track_half_width

    cl, heading, s, length, curvature = trk.centerline_from_points(CONTROL_POINTS, tp.resample_step)
    r_min = 1.0 / np.abs(curvature).max()
    assert r_min > 2 * hw, f"min radius {r_min:.2f} m too tight for half width {hw} m"

    grid, x_min, y_min = trk.bake_grid(cl, heading, s, tp.grid_resolution, hw + tp.margin)
    track = trk.Track(grid, x_min, y_min, tp.grid_resolution, length, cl, heading)
    trk.save(track, tp.npz_path, tp.bin_path)

    g2, *_ = trk.load_bin(tp.bin_path)
    assert np.array_equal(g2, grid)
    print(f"length {length:.2f} m, R_min {r_min:.2f} m, grid {grid.shape[1]}x{grid.shape[0]}")


if __name__ == "__main__":
    main()
```

Sortie attendue : `length 48.31 m, R_min 2.19 m, grid 454x279`.

### `python/mppi/cost.py`

```python
"""Per-trajectory cost.

(garder la docstring existante)
"""
import numpy as np

from mppi.track import lookup, wrap

G = 9.81


def stage_cost(state, control, track, cfg):
    c, veh, ub = cfg.cost, cfg.vehicle, cfg.bounds.u_max
    d, _ = lookup(track, state[..., 0], state[..., 1])
    v = state[..., 3]

    d0 = c.lateral_scale
    d_max = c.track_half_width - 0.5 * veh.width
    lateral = c.w_lateral * (d / d0) ** 2
    excess = np.abs(d) - d_max
    offtrack = c.w_offtrack * (excess > 0) * (1.0 + excess / d0)
    speed = c.w_speed * ((v - c.v_ref) / c.speed_scale) ** 2
    effort = c.w_control * ((control / ub) ** 2).sum(axis=-1)
    a_lat = v**2 * np.abs(np.tan(control[..., 1])) / veh.wheelbase
    adhesion = c.w_adhesion * np.maximum(0.0, a_lat / (veh.mu * G) - 1.0) ** 2
    return lateral + offtrack + speed + effort + adhesion


def terminal_cost(state_T, s0, track, cfg):
    c, m = cfg.cost, cfg.mppi
    _, s_T = lookup(track, state_T[..., 0], state_T[..., 1])
    progress = wrap(s_T - s0, track.length)
    return -c.w_progress * progress / (c.v_ref * m.horizon * m.dt)
```

`control / ub` normalise par `u_max`. Les bornes sont symétriques ici. Si elles ne l'étaient pas, il faudrait choisir explicitement l'échelle.

### `python/mppi/controller.py`

```python
"""Vectorized NumPy MPPI, the reference implementation.

(garder la docstring existante)
"""
import numpy as np

from mppi.cost import stage_cost, terminal_cost
from mppi.dynamics import step
from mppi.track import lookup

S_MAX = 1e6


def rollout_costs(x0, U, eps, track, cfg):
    """Deterministic part of one iteration: given U and raw noise eps (K, T, m),
    returns the costs S (K,), the effective noise and the states (K, T+1, n)."""
    m, b = cfg.mppi, cfg.bounds
    V = np.clip(U[None] + eps, b.u_min, b.u_max)
    eps = V - U[None]
    K, T, _ = V.shape

    X = np.empty((K, T + 1, cfg.state_dim))
    X[:, 0] = x0
    S = np.zeros(K)
    for t in range(T):
        X[:, t + 1] = step(X[:, t], V[:, t], m.dt, cfg.vehicle)
        S += stage_cost(X[:, t + 1], V[:, t], track, cfg)
    _, s0 = lookup(track, x0[0], x0[1])
    S += terminal_cost(X[:, T], s0, track, cfg)

    sigma_inv = 1.0 / m.noise_std**2
    S += m.gamma * np.einsum("tj,ktj->k", U * sigma_inv, eps)
    S[~np.isfinite(S)] = S_MAX
    return S, eps, X


class MPPI:
    def __init__(self, cfg, track):
        self.cfg, self.track = cfg, track
        self.U = np.zeros((cfg.mppi.horizon, cfg.control_dim))
        self.rng = np.random.default_rng(cfg.mppi.seed)

    def command(self, x0):
        m = self.cfg.mppi
        eps = self.rng.standard_normal((m.num_samples, m.horizon, self.cfg.control_dim)) * m.noise_std
        S, eps, X = rollout_costs(x0, self.U, eps, self.track, self.cfg)

        rho = S.min()
        w = np.exp(-(S - rho) / m.lam)
        w /= w.sum()
        self.U = self.U + np.tensordot(w, eps, axes=1)

        u0 = self.U[0].copy()
        self.U[:-1] = self.U[1:]
        self.U[-1] = self.U[-2]
        info = {"ess": 1.0 / np.sum(w**2), "rho": rho, "S": S, "w": w, "X": X}
        return u0, info
```

### `python/run_sim.py`

```python
"""Closed-loop simulation with the NumPy controller.

Writes the trajectory plot and the cost curve to results/figures/, and the
(state, control, next_state) triplets to results/trajectories/.
"""
import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mppi import track as trk
from mppi.config import load_config
from mppi.controller import MPPI
from mppi.dynamics import step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/mppi.yaml")
    cfg = load_config(ap.parse_args().config)
    track = trk.load(cfg.track.npz_path)
    out = cfg.track.npz_path.parent.parent

    x = np.array([*track.centerline[0], track.heading[0], 0.0])
    ctrl = MPPI(cfg, track)
    d_max = cfg.cost.track_half_width - 0.5 * cfg.vehicle.width

    states, controls, nexts, ess, rho, d_log, t_iter = [], [], [], [], [], [], []
    beam = None
    progress = 0.0
    _, s_prev = trk.lookup(track, x[0], x[1])
    for k in range(cfg.max_steps):
        tic = time.perf_counter()
        u, info = ctrl.command(x)
        t_iter.append(time.perf_counter() - tic)
        x_next = step(x, u, cfg.mppi.dt, cfg.vehicle)

        states.append(x); controls.append(u); nexts.append(x_next)
        ess.append(info["ess"]); rho.append(info["rho"])
        d, s = trk.lookup(track, x_next[0], x_next[1])
        d_log.append(float(d))
        progress += trk.wrap(s - s_prev, track.length)
        s_prev = s
        if beam is None and abs(track.centerline[:, 0].max() - x[0]) < 3 and x[1] < 3:
            beam = (info["X"], info["w"])          # first corner
        x = x_next
        if progress >= track.length:
            break

    states, controls, d_log = np.array(states), np.array(controls), np.array(d_log)
    t = np.arange(len(states)) * cfg.mppi.dt
    lap = progress >= track.length
    off = np.abs(d_log) > d_max
    a_lat = states[:, 3] ** 2 * np.abs(np.tan(controls[:, 1])) / cfg.vehicle.wheelbase
    print(f"lap {'done' if lap else 'NOT done'} in {t[-1]:.2f} s, progress {progress:.1f}/{track.length:.1f} m")
    print(f"offtrack steps {off.sum()}, max |d| {np.abs(d_log).max():.3f} m (limit {d_max:.2f})")
    print(f"v mean {states[:, 3].mean():.2f} max {states[:, 3].max():.2f} m/s, max a_lat {a_lat.max():.1f} m/s2")
    print(f"ESS median {np.median(ess):.0f} p5 {np.percentile(ess, 5):.0f} p95 {np.percentile(ess, 95):.0f} / K={cfg.mppi.num_samples}")
    print(f"iteration time median {1e3 * np.median(t_iter):.1f} ms")

    traj_dir, fig_dir = out / "trajectories", out / "figures"
    traj_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    np.savez(traj_dir / "step1_kinematic.npz", state=states, control=controls,
             next_state=np.array(nexts), ess=ess, rho=rho, dt=cfg.mppi.dt)

    hw = cfg.cost.track_half_width
    n = np.column_stack([-np.sin(track.heading), np.cos(track.heading)])
    fig, ax = plt.subplots(figsize=(9, 6))
    for side in (-1, 1):
        edge = track.centerline + side * hw * n
        ax.plot(*np.vstack([edge, edge[:1]]).T, "k", lw=1)
    ax.plot(*track.centerline.T, "k--", lw=0.5)
    if beam is not None:
        X, w = beam
        for i in np.argsort(w)[-200:]:
            ax.plot(X[i, :, 0], X[i, :, 1], color=plt.cm.viridis(w[i] / w.max()), lw=0.5, alpha=0.6)
    sc = ax.scatter(states[:, 0], states[:, 1], c=states[:, 3], s=2, cmap="plasma")
    fig.colorbar(sc, label="v (m/s)")
    ax.set_aspect("equal")
    ax.set_title("Step 1, kinematic MPPI")
    fig.savefig(fig_dir / "step1_trajectory.png", dpi=150)

    fig, axs = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
    axs[0].plot(t, rho); axs[0].set_ylabel("min cost")
    axs[1].plot(t, ess); axs[1].set_ylabel("ESS")
    axs[2].plot(t, d_log)
    axs[2].axhline(d_max, color="r"); axs[2].axhline(-d_max, color="r")
    axs[2].set_ylabel("d (m)")
    axs[3].plot(t, controls[:, 1]); axs[3].set_ylabel("delta (rad)"); axs[3].set_xlabel("t (s)")
    fig.tight_layout()
    fig.savefig(fig_dir / "step1_cost.png", dpi=150)


if __name__ == "__main__":
    main()
```

### `python/tests/test_step1.py`

```python
from pathlib import Path

import numpy as np
import pytest

from mppi import track as trk
from mppi.config import load_config
from mppi.controller import rollout_costs
from mppi.dynamics import step

CFG = load_config(Path(__file__).resolve().parents[2] / "config" / "mppi.yaml")
VEH = CFG.vehicle


def test_straight_line():
    x = np.array([0.0, 0.0, 0.0, 2.0])
    for _ in range(50):
        x = step(x, np.array([0.0, 0.0]), 0.02, VEH)
    assert np.allclose(x, [2.0, 0.0, 0.0, 2.0])


def test_constant_steer_circle():
    delta = 0.2
    beta = np.arctan(VEH.lr / VEH.wheelbase * np.tan(delta))
    radius = VEH.wheelbase / (np.cos(beta) * np.tan(delta))
    v, dt = 1.0, 0.001
    n = int(round(2 * np.pi * radius / v / dt))
    x = np.array([0.0, 0.0, 0.0, v])
    for _ in range(n):
        x = step(x, np.array([0.0, delta]), dt, VEH)
    assert np.hypot(x[0], x[1]) < 0.02
    assert x[2] == pytest.approx(2 * np.pi, abs=0.02)


def test_turns_left_with_positive_steer():
    x = np.array([0.0, 0.0, 0.0, 1.0])
    for _ in range(20):
        x = step(x, np.array([0.0, 0.3]), 0.02, VEH)
    assert x[1] > 0 and x[2] > 0


def test_batch_matches_single():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(8, 4))
    U = rng.normal(size=(8, 2)) * 0.1
    batch = step(X, U, 0.02, VEH)
    for i in range(8):
        assert np.allclose(batch[i], step(X[i], U[i], 0.02, VEH))


def test_speed_never_negative():
    x = step(np.array([0.0, 0.0, 0.0, 0.01]), np.array([-4.0, 0.0]), 0.02, VEH)
    assert x[3] == 0.0


@pytest.fixture(scope="module")
def track():
    return trk.load(CFG.track.npz_path)


def test_grid_on_centerline(track):
    i = np.arange(0, len(track.centerline), 50)
    d, s = trk.lookup(track, track.centerline[i, 0], track.centerline[i, 1])
    assert np.all(np.abs(d) < track.res)
    ds = trk.wrap(s - i * track.length / len(track.centerline), track.length)
    assert np.all(np.abs(ds) < 2 * track.res)


def test_grid_sign_left_positive(track):
    i = 100
    n = np.array([-np.sin(track.heading[i]), np.cos(track.heading[i])])
    p = track.centerline[i] + 0.5 * n
    d, _ = trk.lookup(track, p[0], p[1])
    assert d == pytest.approx(0.5, abs=track.res)


def test_wrap():
    assert trk.wrap(-47.0, 48.0) == pytest.approx(1.0)
    assert trk.wrap(1.0, 48.0) == pytest.approx(1.0)


def test_bin_matches_npz(track):
    grid, x_min, y_min, res, length = trk.load_bin(CFG.track.bin_path)
    assert np.array_equal(grid, track.grid)
    assert (x_min, y_min, res) == pytest.approx((track.x_min, track.y_min, track.res))


def test_rollout_costs_deterministic(track):
    x0 = np.array([*track.centerline[0], track.heading[0], 2.0])
    U = np.zeros((CFG.mppi.horizon, 2))
    eps = np.random.default_rng(0).normal(size=(64, CFG.mppi.horizon, 2)) * CFG.mppi.noise_std
    S1, e1, _ = rollout_costs(x0, U, eps, track, CFG)
    S2, e2, _ = rollout_costs(x0, U, eps, track, CFG)
    assert np.array_equal(S1, S2) and np.all(np.isfinite(S1))
    V = U + e1
    assert np.all(V >= CFG.bounds.u_min) and np.all(V <= CFG.bounds.u_max)
```

Les tests de grille supposent que `pixi run track` a été lancé.

### `sweep.py` (partie G, dans le scratch ou `bench/`)

```python
import dataclasses
import sys

import numpy as np

sys.path.insert(0, "python")
from mppi import track as trk
from mppi.config import load_config
from mppi.controller import MPPI
from mppi.dynamics import step


def run(cfg):
    track = trk.load(cfg.track.npz_path)
    x = np.array([*track.centerline[0], track.heading[0], 0.0])
    ctrl = MPPI(cfg, track)
    d_max = cfg.cost.track_half_width - 0.5 * cfg.vehicle.width
    progress, (_, s_prev) = 0.0, trk.lookup(track, x[0], x[1])
    ess, dl, us = [], [], []
    for k in range(cfg.max_steps):
        u, info = ctrl.command(x)
        x = step(x, u, cfg.mppi.dt, cfg.vehicle)
        d, s = trk.lookup(track, x[0], x[1])
        progress += trk.wrap(s - s_prev, track.length)
        s_prev = s
        ess.append(info["ess"]); dl.append(abs(d)); us.append(u)
        if progress >= track.length:
            break
    us = np.array(us)
    return dict(lap=bool(progress >= track.length), t=round((k + 1) * cfg.mppi.dt, 2),
                off=int((np.array(dl) > d_max).sum()), dmax=round(float(max(dl)), 2),
                ess_med=int(np.median(ess)), ess_p5=int(np.percentile(ess, 5)),
                jitter=round(float(np.abs(np.diff(us[:, 1])).mean()), 4))


if __name__ == "__main__":
    base = load_config("config/mppi.yaml")
    for lam in (1.0, 0.3, 0.1, 0.03):
        cfg = dataclasses.replace(base, mppi=dataclasses.replace(base.mppi, lam=lam))
        print(f"lambda={lam}", run(cfg))
```

Chaque run dure environ 15 s. Pour aller plus vite, lance les variantes en parallèle dans plusieurs terminaux.
