# Work journal

One entry per session. Record what was measured, not what was expected.
Optimizations that gave nothing go in here too: that is the record showing
measurement rather than belief.

Format:

## YYYY-MM-DD - step N - short title

**Goal.**

**Done.**

**Measured.** Numbers, with the config they came from (K, T, model, hardware,
power mode).

**No effect / reverted.** What was tried, why a gain was expected, what
actually happened.

**Next.**

---

## 2026-09-09 - step 0 - repo foundations

**Goal.** Repo structure, CMake, pixi environment, shared YAML config.

**Done.** Directory layout, `pixi.toml` with linux-64 and linux-aarch64,
CMake with CUDA as a first-class language targeting sm_86 and sm_87,
`config/mppi.yaml` as the single shared config.

**Measured.**

**No effect / reverted.**

**Next.** Step 1, kinematic MPPI in NumPy.

---

## 2026-10-07 - step 1 - kinematic MPPI in NumPy

**Goal.** NumPy reference MPPI on the kinematic bicycle model: one clean lap
at K = 1024, T = 30, and a qualitative understanding of lambda, Sigma, gamma
and T. This code is the ground truth the CUDA kernel will be checked against.

**Done.**
- `config.py`: YAML -> frozen dataclasses, with validation (state_dim vs
  model, lf + lr = wheelbase, noise_std size, u_min < u_max).
- `dynamics.py`: kinematic bicycle at the CG, explicit Euler, v clamped >= 0.
- `track.py` + `generate_track.py`: periodic spline, resampling at 5 cm,
  KD-tree baked grid [d, s] at 5 cm, `.npz` for Python, `.bin` (MPPG v1) for C++.
- `cost.py`: lateral, off-track (graduated), speed, effort, adhesion;
  terminal progress normalized by v_ref*T*dt.
- `controller.py`: deterministic `rollout_costs` (what `check_parity.py` will
  call) + `MPPI` class. Effective noise recomputed after clipping, rho
  subtracted before exp, non-finite costs -> S_MAX.
- `run_sim.py`: closed loop, summary, figures, (state, control, next_state)
  triplets in `results/trajectories/step1_kinematic.npz`.
- `bench/sweep.py`: part G experiments, one parameter at a time, in parallel.
- 28 unit tests. `pixi run test` now depends on `track`, so CI regenerates
  the gitignored grid instead of failing.

**Measured.** Config: K = 1024, T = 30, dt = 0.02, lambda = 0.3, gamma = 0,
noise_std = [0.5, 0.1], v_ref = 3, kinematic model, NumPy, dev PC (28 cores).

```
lap done in 16.12 s, progress 48.4/48.3 m
offtrack steps 0, max |d| 0.057 m (limit 0.65)
v mean 2.99 max 3.24 m/s, max a_lat 6.3 m/s2
ESS median 444 p5 104 p95 519 / K=1024
iteration time median 3.8 ms
```

Iteration time 3.8 ms is the NumPy baseline for the CUDA comparison.

Part G (one parameter changed at a time from the config above):

| variant | lap | t (s) | ESS med | ESS p5 | jitter | max abs(d) | v mean |
|---|---|---|---|---|---|---|---|
| reference | yes | 16.12 | 444 | 104 | 0.037 | 0.06 | 2.99 |
| lambda = 1 | yes | 16.44 | 777 | 337 | 0.027 | 0.14 | 2.93 |
| lambda = 0.1 | yes | 16.24 | 140 | 53 | 0.049 | 0.06 | 2.97 |
| lambda = 0.03 | yes | 16.32 | 9 | 1 | 0.109 | 0.05 | 2.96 |
| w_speed = 0 | NO, 24.2 m | 60.00 | 1023 | 422 | 0.018 | 0.20 | 0.40 |
| v_ref = 5 | yes | 9.56 | 1 | 1 | 0.116 | 0.25 | 5.01 |
| v_ref = 7 | yes | 7.56 | 1 | 1 | 0.125 | 0.56 | 6.18 |
| sigma_delta = 0.02 | yes | 16.14 | 790 | 22 | 0.009 | 0.13 | 2.99 |
| sigma_delta = 0.3 | yes | 16.80 | 66 | 6 | 0.057 | 0.05 | 2.87 |
| gamma = lambda | yes | 17.04 | 423 | 341 | 0.008 | 0.10 | 2.82 |
| T = 10, v_ref = 5 | yes | 6.92 | 1 | 1 | 0.055 | 0.65 | 7.40 |
| seed = 1 | yes | 16.10 | 439 | 217 | 0.036 | 0.09 | 3.00 |

v_ref = 5, lambda re-tuned: lambda = 1 -> ESS med 19; lambda = 3 -> ESS med
238, p5 126, jitter 0.018, 11.18 s; lambda = 10 -> ESS med 562.

Findings (details in `docs/step1_predictions.md`):
- lambda must be set relative to the cost spread. At v_ref = 5 the adhesion
  term spreads the costs (median - min: 0.7 at v_ref = 3, 57 at v_ref = 5),
  so lambda = 0.3 collapses ESS to 1. lambda ~ 2-3 restores it.
- T = 10 at v_ref = 5: the car never stops accelerating (12.4 m/s at the end
  for v_ref = 5). Over 0.2 s, acceleration barely changes the cost while
  steering dominates the spread, so with ESS = 1 the acceleration of U drifts
  as a random walk from the +3.8 m/s2 taken at start-up. Measured d(cost)/d(a)
  is positive for every term (all prefer braking); progress has the smallest
  effect. This differs from the explanation in step_1.md S-G.
- seed 0 -> 1 changes nothing measurable except ESS p5: the tuning is robust.

**No effect / reverted.**
- Beam snapshot for the trajectory figure triggered on |delta| > 0.15, then on
  |delta| > 0.1 with v > 0.9 v_ref: both fired near the start, because the
  steering noise alone crosses these thresholds. Replaced by a trigger on
  track progress (12 m, first corner entry).

**YAML fields added in step 1** (to mirror in `cuda/include/mppi/config.hpp`):
- `mppi.gamma`; `mppi.lambda` changed 1.0 -> 0.3
- `vehicle.width`, `vehicle.mu`
- `cost.w_adhesion`, `cost.w_speed`, `cost.lateral_scale`, `cost.speed_scale`,
  `cost.v_ref`; `cost.w_progress` changed 1.0 -> 10.0
- `track.resample_step`, `track.margin`, `track.npz_path` (replaces `npy_path`)
- `sim.max_steps`

**Questions posées pendant l'étape (et réponses).** Trace de ce que je ne
comprenais pas, pour pouvoir y revenir.

*Partie D, coût*

- **Pourquoi normaliser chaque terme (`d/d0`, `(v − v_ref)/v0`, `u/u_max`) ?**
  Pour que chaque terme vaille environ 1 « quand ça va mal ». Les poids `w_*`
  se lisent alors directement, et l'échelle globale du coût est connue, ce qui
  est indispensable pour régler λ.
- **Pourquoi `d_max = demi-largeur − width/2` ?** `d` est mesuré au CdG, mais
  ce sont les roues qui sortent, 15 cm plus loin. Les bords dessinés sur la
  figure sont à ±0,8 m (bords physiques), la limite du coût est à ±0,65 m.
- **Pourquoi `(excess > 0) * (...)` et pas un `if` ?** Un `if` sur un tableau
  NumPy lève une erreur. Et en CUDA, l'écriture sans branche évite la
  divergence des threads d'un warp, et garde le calcul identique des deux
  côtés.
- **`np.maximum` vs `np.max` ?** `np.maximum(0, x)` est un max élément par
  élément entre deux tableaux. `np.max(x)` est le maximum de tout le tableau.
- **Dans `-c.w_progress * progress / (...)`, le `-` porte sur quoi ?**
  Seulement sur `c.w_progress`. Ordre de priorité : attribut/appel/indexation,
  puis `**`, puis `-` unaire, puis `* / // %` (de gauche à droite), puis
  `+ -` binaires. Piège : `-2 ** 2 == -4`, car `**` est plus prioritaire que
  le `-` unaire. Ici, le résultat est identique au bit près : changer le signe
  d'un flottant est exact. On peut vérifier l'arbre avec
  `ast.dump(ast.parse("-a * b / c", mode="eval").body)`.
- **C'est quoi `w_speed` ?** Le poids du terme de vitesse
  `w_speed · ((v − v_ref)/v0)²`, compté à chaque pas de l'horizon. Il vaut 9
  par pas à l'arrêt, ce qui rend le coût informatif même quand la voiture ne
  bouge pas.

*Partie E, contrôleur*

- **Pourquoi les tests « déclenchent des NaN » ?** C'est le test
  `test_rollout_costs_nan_becomes_s_max` qui met volontairement `v = NaN`
  dans `x0`, pour vérifier le garde-fou `S[~np.isfinite(S)] = S_MAX`. Le NaN
  se propage (`step` → x, y NaN → coût de vitesse NaN). Les
  `RuntimeWarning: invalid value encountered in cast` viennent de `lookup` :
  `np.floor(NaN).astype(np.int64)` n'a pas de sens, car un entier ne peut pas
  représenter NaN. Ils sont masqués pour ce seul test avec
  `@pytest.mark.filterwarnings`.
- **Forme de `X = np.empty((K, T + 1, state_dim))` ?** `(1024, 31, 4)`, soit
  (rollout, instant, composante). On a T + 1 états pour T commandes :
  `x_0 --u_0--> x_1 ... --u_29--> x_30`. `np.empty` n'initialise pas, ce qui
  est sans danger car tout est écrit avant d'être lu.
- **`X[:, 0]` alors que X a 3 dimensions ?** Les axes non précisés sont
  complétés **à droite** par `:`, donc `X[:, 0] == X[:, 0, :]`, de forme
  `(1024, 4)` : l'état complet à l'instant 0. `X[:, :, 0]` est autre chose :
  la coordonnée x à tous les instants, de forme `(1024, 31)`.
- **Compléter par des `:` à gauche ?** Avec l'Ellipsis `...` :
  `X[..., 0] == X[:, :, 0]`. Elle peut aussi être au milieu
  (`X[0, ..., 0]`), mais une seule par expression. Intérêt : `state[..., 3]`
  désigne la vitesse pour un état `(4,)`, un lot `(K, 4)` ou `X` entier.
  `state[:, 3]` sur `X` renverrait l'état à l'instant 3, sans erreur.
- **`X[..., 0, :, 1]` ?** `...` remplit ce qui reste avant les 3 indices
  explicites (`:` compte comme indice). Sur un tableau 3D, ça donne
  `X[0, :, 1]`, de forme `(31,)`. Sur un tableau 4D, `A[:, 0, :, 1]`. Sur un
  tableau 2D, `IndexError`. Un indice entier supprime son axe, `:` et `...`
  les gardent.
- **La mise à jour `U += tensordot(w, eps, axes=1)` ?** C'est
  `U + Σ_k w_k eps_k`, une moyenne pondérée des perturbations après
  saturation, avec des poids qui ne dépendent que des coûts. Comme
  `Σ w = 1`, c'est aussi `Σ_k w_k V_k` : une moyenne des commandes réellement
  simulées, donc dans les bornes.
- **Pourquoi soustraire ρ = min S ?** Les poids normalisés ne changent pas
  (le facteur `e^{ρ/λ}` se simplifie), mais `exp(x)` vaut 0 en float64 pour
  `x < −745`. Sans ρ, tous les poids peuvent valoir 0, ce qui donne
  `0/0 = NaN`. Avec ρ, le meilleur rollout a un poids brut de 1, donc la somme
  vaut au moins 1. Même astuce que `x − x.max()` dans un softmax.
- **Étape 4, le terme `γ Σ_t U_tᵀ Σ⁻¹ ε_t` ?** Il pénalise les perturbations
  qui éloignent encore la commande de 0 (produit `U·ε > 0`). Il vient de la
  théorie (§3.5) : on échantillonne autour de U alors que la référence est
  centrée en 0. Σ⁻¹ mesure la perturbation en unités de son écart-type. γ = 0
  dans la config.
- **Σ en général ?** C'est la matrice de covariance `(m, m)` du bruit. Si elle
  est pleine (bruits corrélés), il faut tirer `ε = z @ L.T` avec
  `L = cholesky(Σ)`, calculer `Σ⁻¹ = inv(Σ)` une fois, et utiliser
  `einsum("ti,ij,ktj->k", U, Sigma_inv, eps)`. Ici, Σ est diagonale,
  `Σ = diag(noise_std**2)`, donc `Σ⁻¹` revient à multiplier par
  `1 / noise_std**2`. `noise_std` est un **vecteur d'écarts-types**, pas la
  matrice de covariance.
- **Étape 5 : un coût grand donne déjà `exp → 0`, pourquoi un garde-fou ?**
  Pour un coût grand mais **fini**, c'est vrai. Le garde-fou vise NaN et inf,
  qui ne sont pas de « grands nombres » : `exp(NaN) = NaN`,
  `min([…, NaN]) = NaN`, `0 · NaN = NaN`. Un seul NaN donne ρ = NaN, donc tous
  les poids NaN, donc U NaN pour toujours. Avec +inf partout,
  `inf − inf = NaN`. Remplacé par 1e6, le rollout a juste un poids nul.
  Ce n'est pas une question de vitesse.

*Partie F, simulation*

- **À quoi servent `d` et `s` dans `simulate` ?** C'est la mesure faite par
  le simulateur, pas par le contrôleur. `d` est seulement journalisé, pour
  compter les pas hors piste, calculer `max abs(d)` et tracer la courbe.
  `s` sert à cumuler la progression pas à pas, car `s` revient à 0 à chaque
  tour. `wrap` évite le saut de −48 m au passage de la ligne.
- **C'est quoi `beam` (le faisceau) ?** Une photo, à un seul instant (12 m de
  progression, entrée du premier virage), des 1024 rollouts `info["X"]` et de
  leurs poids `info["w"]`. Elle sert uniquement à la figure, mais doit être
  prise **pendant** la simulation : `info` est écrasé à chaque itération, et
  tout garder ferait environ 820 Mo. Lecture : un faisceau fin et centré est
  sain ; trop large → σ trop grand ; un seul fil coloré → λ trop petit.
- **Le format de `edge` ?** C'est le bord de la piste, de forme `(966, 2)` :
  `centerline + side · hw · normal`, avec `normal = (−sin θ, cos θ)` et
  `side = ±1`. `np.vstack([edge, edge[:1]])` rajoute le premier point à la fin
  pour refermer la boucle. `.T` transpose en `(2, 967)`, et `*` déplie en
  `ax.plot(xs, ys)`.
- **`ax.inset_axes([0.35, 0.3, 0.3, 0.4])` ?** Un encart, c'est-à-dire un
  graphique complet dans le grand. Les nombres sont
  `[gauche, bas, largeur, hauteur]` en fraction du grand graphique.
  `indicate_inset_zoom` dessine le rectangle et les traits de liaison.

*Partie G, réglage*

- **C'est quoi la gigue (jitter) ?** `mean(abs(diff(δ)))`, en rad/pas : la
  variation moyenne du braquage d'un pas à l'autre. Deux séries de même
  moyenne peuvent avoir des gigues très différentes. Elle ne se voit presque
  pas en simulation, mais sur une vraie voiture, le servomoteur ne suit pas et
  la voiture vibre. `max abs(d)` dit **si** on suit bien la piste, la gigue
  dit **à quel prix** pour les actionneurs.
- **L'ESS, sa formule, la médiane et le p5 ?** `ESS = 1 / Σ w_k²`, avec des
  poids normalisés. Elle vaut K si tous les poids sont égaux, 1 si un seul
  porte tout, m si m rollouts sont à égalité. Elle se lit comme « combien de
  rollouts participent vraiment ». Il y a une ESS par itération (environ 806
  sur un tour). La **médiane** décrit le régime normal (moins sensible que la
  moyenne au démarrage, où ESS ≈ 1). Le **p5** décrit les pires moments
  récurrents (virages), en ignorant les cas isolés. Règle : ESS p5 entre 1 et
  10 % de K. Exemple G4 : médiane 790 (tout va bien en apparence), mais p5 22
  (problème dans les virages).
- **Pourquoi l'ESS tombe à 1 à `v_ref = 5` (G3) ?** Le terme d'adhérence :
  à 5 m/s, la limite μg est atteinte dès δ ≈ 0,13 rad, donc dans le bruit
  σ_δ = 0,1. La dispersion des coûts passe de 0,7 à 57, soit 190 fois λ.
  **λ se règle par rapport à la dispersion des coûts**, pas dans l'absolu.
- **Pourquoi la voiture va si vite avec `T = 10` (G6) ?** Voir **Measured** :
  l'accélération de U dérive comme une marche au hasard, car sur 0,2 s elle
  change à peine le coût.
- **G7, `seed = 1` :** la graine **change** (0 → 1), donc tout le bruit est
  différent. Un résultat quasi identique veut dire que le réglage est robuste
  au hasard, ce qui valide les autres expériences. Ce n'est pas « la même
  graine ».

*Partie 10, clôture*

- **Pourquoi la CI échouait ?** `results/` est dans `.gitignore`, donc la
  grille n'existe pas sur GitHub. Corrigé par `depends-on = ["track"]` sur la
  tâche `test`.

**Next.** Step 2, dynamic bicycle model with tire model. Re-check lambda
after the model change (cost spread will move).
