# Work journal

One entry per session. Record what was measured, not what was expected.
Optimizations that gave nothing go in here too: that is the record showing
measurement rather than belief.

Format:

## YYYY-MM-DD - step N - short title

**Goal.**

**Done.**

**Measured.** Numbers, with the config they came from (K, T, model, hardware,
power limit, other jobs on the node).

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

---

## 2026-10-08 - step 2 - parts 1 and A (setup, why kinematic is not enough)

**Goal.** Prepare step 2 (part 1) and get the orders of magnitude that justify
a dynamic model (part A, paper only).

**Done.**
- Part 1: `pixi.toml`, added the `# STEP 2` header above the `compare` task
  (the task itself was already there, under `# STEP 1`). `pixi run compare`
  will fail until `bench/compare_models.py` exists (part G).
- Part 1.1 checks, all true on the current code: `controller.py` only uses
  `cfg.state_dim` and `step(...)` (no change needed); the kinematic state is
  at the CG (`cos(psi + beta)` in `dynamics.py`, choice S-B1 of step 1);
  `mass`, `izz`, `cornering_stiffness_*`, `kinematic_blend_speed`, `mu` are in
  the YAML but not loaded by `config.py` yet (part F).
- Part A: answers to Q-A1 to Q-A3 and checkpoint A written in
  `docs/step2_predictions.md` (new file, also holds the part H table).

**Measured.** `pixi run test`: 28 passed (step 1 suite, unchanged).

Part A, with L = 0.33, lr = 0.18, delta_max = 0.4, mu = 1:

```
geometric beta at delta_max   13.0 deg
R at delta_max                0.801 m at the CG (0.78 m at the rear axle)
v_max = sqrt(mu g R)          4.64 m/s on R_min = 2.19 m, 2.80 m/s on R = 0.80 m
tight corner at 7 m/s         a_lat = 22.4 m/s2 = 2.3 g (kinematic accepts it)
step 1 at v_ref = 3           max a_lat 6.3 m/s2 = 0.64 g, kinematic still valid
```

Note: 3 m/s on the centerline of R_min only gives 4.1 m/s2. The 6.3 measured
in step 1 comes from the driven line, locally tighter than the centerline.

**No effect / reverted.**

**Questions posées pendant l'étape (et réponses).**

*Partie 1, préparation*

- **Le modèle cinématique est au CdG ou à l'essieu arrière ?** Au **CdG**,
  depuis l'étape 1 (`step_1.md:170`, Q-B1/S-B1). Le code le montre :
  `x += v cos(ψ + β) dt`. À l'essieu arrière, il n'y aurait pas de β. Raison :
  le modèle dynamique s'écrit au CdG (Newton s'applique au CdG), et le
  mélange de la partie D fait une moyenne état par état. Si les deux modèles
  décrivaient deux points différents (séparés de `lr`), la position
  « sauterait » pendant la transition. Le 0,78 m de `step_1.md:577` est
  seulement le rayon calculé à l'essieu arrière (`L / tan δ`).

*Partie A, repères et physique*

- **Quel repère pour chaque modèle ?** Les deux en utilisent deux. `x, y, ψ`
  sont dans le **repère monde**, pour les deux modèles (indispensable pour
  les mélanger). La vitesse est dans le **repère véhicule** : explicitement
  dans le dynamique (`vx, vy`), implicitement dans le cinématique
  (`vx = v cos β`, `vy = v sin β`, β imposé par la géométrie). Preuve :
  `vx cos ψ − vy sin ψ = v cos(ψ + β)`.
- **β est dans quel repère ?** C'est un angle entre deux directions (le nez
  et la vitesse du CdG), donc il ne dépend pas du repère. Repère véhicule :
  `β = atan(vy/vx)`. Repère monde : `β = χ − ψ`, avec χ la direction de la
  vitesse, d'où `cos(ψ + β)` dans le code. Cinématique : β fixé par δ
  (≤ 13°). Dynamique : β libre (20 à 30° en dérapage).
- **Le repère véhicule n'est pas galiléen ?** Exact, mais on n'y applique pas
  Newton. Il faut distinguer **référentiel** (le sol, galiléen, où l'on
  dérive) et **base de projection** (`e_x, e_y`, qui tournent). `vx, vy` sont
  la vitesse par rapport au sol, projetée sur les axes de la voiture. Les
  termes `+vy·r` et `−vx·r` viennent de `ė_x = r e_y`, `ė_y = −r e_x` : ce ne
  sont pas des forces fictives. Dans le référentiel véhicule, on les
  retrouverait comme force d'entraînement (Coriolis nulle, le CdG y est
  immobile). `Izz ṙ = Mz` au CdG reste valable même si la voiture accélère.
- **Le « plan de la roue » ?** Le plan vertical qui contient le disque de la
  roue (vu de dessus : la direction où elle pointe). « Rouler sans glisser » =
  la vitesse du point de contact est dans ce plan (α = 0). Un vrai pneu dérive
  (α ≠ 0), et c'est cette dérive qui crée sa force latérale.
- **Bicycle ou 4 roues ?** Bicycle (single-track, théorie §6.1) : une roue
  par essieu, sur l'axe. Négligés : transfert de charge gauche/droite,
  Ackermann (une seule valeur de δ), roulis. `width` ne sert qu'au coût.
- **C'est quoi δ ?** Une **commande**, pas une variable calculée : l'angle
  entre le plan de la roue avant et l'axe x du véhicule. Radians, > 0 à
  gauche, borné à ±0,4 (par le contrôleur). `control[..., 1]`.
- **Pourquoi `a_lat = v²/R` ?** Base de Frenet (ou polaire pour un cercle),
  dans le référentiel du sol : `a = (dv/dt)·T + (v²/R)·N`. Sur un cercle :
  `OM = R e_r`, `a = R θ̈ e_θ − R θ̇² e_r`, et `R θ̇² = v²/R`.
- **μ, c'est quoi ?** Le coefficient de la loi de Coulomb `|T| ≤ μ N`.
  Empirique (≈ 1 caoutchouc sur sec, ≈ 0,5 mouillé, ≈ 0,1 glace).
  `mu: 1.0` est **supposé**, pas mesuré sur la voiture.
- **D'où vient `v_max = √(μgR)` ?** `N = mg`, frottement latéral
  `T = m v²/R`, pas de glissement si `T ≤ μN`. La masse se simplifie.
  Le 2,8 m/s de la théorie est **au braquage maximal** (R ≈ 0,8 m, géométrie
  de la voiture). Sur la piste, la limite est 4,6 m/s (R_min = 2,19 m,
  `generate_track.py:27`). Et le cinématique ne devient pas faux d'un coup :
  l'erreur grandit avec la dérive des pneus.
- **Pourquoi séparer l'avant et l'arrière, Coulomb ne parle que du tout ?**
  Coulomb s'applique **à chaque contact**. Charges :
  `Fz_f = m g lr/L = 18,7 N`, `Fz_r = m g lf/L = 15,6 N`. En virage
  stabilisé, les forces demandées sont exactement proportionnelles aux
  charges, donc les deux essieux saturent ensemble à μg (c'est pour ça que le
  raisonnement global marche). En transitoire, en accélération ou avec des
  pneus différents, un essieu lâche avant l'autre : avant = sous-virage
  (stable), arrière = survirage (tête-à-queue). Ramener au CdG : oui, mais
  **force + moment** (torseur). Le moment `lf F_yf cos δ − lr F_yr` est
  ce qui fait tourner ou déraper ; un point matériel ne peut pas déraper.
- **Lacet, vitesse de lacet ?** Lacet = rotation autour de l'axe vertical
  (roulis : axe x, tangage : axe y, ignorés). Angle de lacet = ψ (le cap).
  Vitesse de lacet `r = ψ̇` (rad/s), 6e composante de l'état dynamique.
- **Fx, Fy ?** Composantes des forces dans le repère véhicule : Fx
  longitudinale (accélérer, freiner), Fy latérale (tourner). `F_yf`, `F_yr` :
  forces latérales des pneus avant (*front*) et arrière (*rear*), à ne pas
  confondre avec `r` la vitesse de lacet. La roue avant étant braquée,
  `F_yf` se projette en `(−F_yf sin δ, F_yf cos δ)`.
- **Les seules forces sont a, F_yf, F_yr ?** Dans le plan, oui : `Fx = m·a`
  (a est une accélération, pas une force), `F_yf`, `F_yr`. `Fx` passe par le
  CdG, donc pas de moment. Verticalement : poids et réactions `Fz`, qui
  s'annulent mais fixent les limites `μ Fz`. Négligés : traînée, résistance
  au roulement, pente.
- **La voiture peut perdre l'adhérence en x ?** Physiquement oui :
  patinage, blocage, taux de glissement `κ = (ωR − vx)/vx`, et le **cercle
  d'adhérence** `Fx² + Fy² ≤ (μFz)²` par pneu. **Notre modèle n'en
  représente rien** : `Fx = m·a` toujours obtenue, pas de `F_xf/F_xr`, pas de
  cercle, pas de transfert de charge (pas de `h`). Acceptable en ligne droite
  (|a| ≤ 4 → 14 N sur 34,3 N, 41 %). Optimiste en virage à la limite en
  accélérant : il resterait `√(1 − 0,41²) ≈ 91 %` de capacité latérale, le
  modèle en garde 100 %.

*Hors périmètre, pour plus tard*

- **Vraie voiture : modèle 4 roues ?** Non, le bicycle dynamique est le
  standard (Liniger et al., F1TENTH). Priorités : identifier μ, C_S, Izz sur
  la vraie voiture ; modéliser l'actionneur (servo du 1er ordre, moteur) ;
  compenser la latence.
- **Comment le robot voit la piste ?** Il ne la voit pas : état exact donné
  par le simulateur, costmap précalculée. Voulu (`edge-mppi.md:50`). En vrai :
  carte par SLAM lidar (une fois), localisation par filtre particulaire,
  `vx, vy, r` par EKF (IMU + odométrie) ; `vy` est la plus dure à estimer.
  Faisable : c'est l'architecture F1TENTH, et AutoRally (Georgia Tech) a fait
  MPPI sur GPU embarqué en dérapage.
- **Ajouter `δ + δ₀ ω` pour simuler un contrôle imparfait ?** Bonne idée,
  mais **seulement dans le véhicule simulé** (`run_sim.py`), jamais dans
  `step()` (parité CUDA). Le bruit blanc est le moins réaliste (s'annule à
  50 Hz). Mieux : retard du servo `δ_réel += (δ_cmd − δ_réel)·dt/τ`, vitesse
  limitée, biais de trim, gain `k δ_cmd`, bruit corrélé (Ornstein-Uhlenbeck).
  Expérience possible après la partie H : tours propres en fonction de τ.
- **Cercle d'adhérence** : extension la moins chère pour le réalisme
  longitudinal (répartir `m·a` entre essieux, limiter `F_y` à
  `√((μFz)² − Fx²)`), sans nouvel état.
- **Pourquoi `step_2.md` cite `f1tenth_gym` ?** Simulateur officiel F1TENTH,
  interface Gym (RL **et** contrôle classique). Utilisé seulement comme source
  pour vérifier les paramètres (§3.5) et l'interprétation de `C_S` normalisée
  par la charge (§4.2). Pas utilisé à la place de notre modèle : une seule
  voiture à la fois (MPPI fait 1024 × 30 pas par itération), pas maîtrisable
  ligne à ligne pour la parité CUDA, et c'est le but de l'étape. Pourrait
  servir plus tard de véhicule simulé indépendant (test de robustesse).
- **PyTorch passe par NumPy pour CUDA ?** Non. NumPy est CPU uniquement, et
  CUDA ne « comprend » pas NumPy : notre kernel est une réécriture C++ à la
  main, comparée par fichiers (`check_parity.py`). PyTorch a ses propres
  tenseurs et appelle directement ses kernels CUDA précompilés (ATen, cuBLAS,
  cuDNN). NumPy n'intervient que pour convertir (`from_numpy`,
  `.cpu().numpy()`). « NumPy sur GPU » : CuPy ; kernels en Python : Numba.
- **CUDA à la main ou bibliothèque ?** CUDA à la main pour ce projet :
  c'est l'objectif, MPPI y gagne (un seul lancement, un thread par rollout,
  état dans les registres, contre environ 1000 petits lancements par
  itération en PyTorch naïf ; `torch.compile` et les CUDA graphs réduisent
  l'écart), et latence prévisible. Idée : ajouter une version
  PyTorch/CuPy comme point de comparaison dans le benchmark.

**Next.** Part B: dynamic bicycle equations in `dynamics.py`
(Q-B1 to Q-B3), then part C (tire models, `C_S` normalized by load).

---

## 2026-10-08 - step 2 - parts B to E (dynamic model, tires, blend, integrator)

**Goal.** Write the dynamic single-track model in NumPy (part B), the three
tire models (part C), make it safe at low speed by blending with the
kinematic model (part D), then pick the integrator from measurements
(part E). Same rule as step 1: this code is the reference for the CUDA kernel.

**Done.**
- Part B: `dynamic_derivative(state, a, delta, vehicle)` in `dynamics.py`,
  the right-hand side `f(x, u)` of the 6 equations (world-frame position,
  rotating-frame `+vy r` / `-vx r` terms, steering drag `-F_yf sin(delta)/m`,
  yaw moment `lf F_yf cos(delta) - lr F_yr`). Written as a derivative, not a
  step, so that the part E integrator can call it several times per step.
  `Vehicle` gets `mass`, `izz`, `cornering_stiffness_front/rear`.
- Part C: `axle_loads` (static, no load transfer), `tire_force` (linear, tanh,
  simplified Pacejka with `B = C_S / C`, same slope `mu C_S F_z` at the origin
  for all three), `slip_angles` (`alpha > 0` pushes towards +y). YAML/config:
  `tire_model`, `pacejka_c`, `pacejka_e`, `TIRE_MODELS` + validation. The
  `if tire_model` is a static choice read from the YAML, not a state branch.
  `bench/plot_tires.py` (`pixi run tires`) draws `F(alpha)` on [-90, 90] deg
  to `results/figures/step2_tire_curves.png` (checkpoint C).
- Part D: `kinematic_blend_speed` replaced by `blend_speed_low: 1.0` /
  `blend_speed_high: 1.5` (validated `0 < low < high`). Floor
  `vx_safe = max(vx, blend_speed_low)` in `slip_angles`. New functions:
  `step_kinematic6` (kinematic model in the 6-state, `vy`, `r` put back on the
  manifold with `vx_next`), `blend_weight` (`kappa` by `clip`, no `if`),
  `step_dynamic` (`kappa * dyn + (1 - kappa) * kin`, then `vx >= 0`).
- Part E: `integrator: rk4`, `substeps: 1` in the YAML (`INTEGRATORS` +
  validation, `substeps` must be an int >= 1). `step_dynamic_only` runs
  `substeps` Euler or RK4 sub-steps (it was a provisional single Euler step
  during part D). `bench/integrators.py` (S-E script) measures stiffness,
  stability, accuracy and cost.
- Tests added in `test_step2.py` for B to E: stiffness convention, same slope
  (x3 tires), saturation + odd force (tanh, Pacejka), sign on the derivative
  (x3 tires), sign on the full step, steady-state cornering (official Q-B2),
  blend = kinematic below `v_low` (bit-exact), blend continuity at both band
  edges, finiteness from 0 to 2 m/s under `np.errstate(all="raise")`
  (3 tires x 2 integrators), never reverses, integrator stable at `v_low`.

**Measured.** Dev PC (WSL2, no GPU), NumPy, dt = 0.02.

Part B check (linear tire, steady turn at vx = 3, delta = 0.05, solved with
`fsolve`): `F_yf cos(delta) + F_yr - m vx r = -9e-16`, yaw moment 0,
r = 0.43972 rad/s (doc: 0.43974).

Part C, front tire, F_zf = 18.73 N (N):

```
alpha (deg)   linear    tanh   pacejka
14.3          18.70    14.25    14.44
22.9          29.94    17.26    17.63
37.2          48.64    18.52    18.73   (Pacejka peak)
57.3          74.92    18.72    18.16
85.9         112.31    18.73    17.12
```

Matches the table of section 4.4 (low-angle cells differ only because the doc
rounds the degree values). Rear C_alpha prints 65.5 N/rad (exact 65.548), doc
says 65.6: rounding, the test tolerance is 0.1.

Part D:
- checkpoint D: with `/ vx_safe` replaced by `/ vx`, the 3 finiteness cases
  fail (`FloatingPointError: divide by zero`, `slip_angles`). Restored after.
- from rest `[0, 0, 0, 0, 0, 0]`, a = 1, delta = 0.2: finite next state
  `[0, 0, 0, 0.02, 0.0022, 0.0123]` (would be all NaN without the floor).
- `step_kinematic6` vs step 1 `step` (4 states), 50 steps, delta = 0.3:
  4e-16 with a = 0; 3.4 mm with a = 0.5. Not a bug: in the 6-state model `a`
  accelerates `vx` (body axis, like the dynamic model), in the 4-state model it
  accelerates the total speed `v`. They differ by `cos(beta)` (<= 3 %).

Part E, `bench/integrators.py`:

```
stiffness (linear tires): yaw coefficient 95.2, shortest time constant
  5 ms @ 0.5, 11 @ 1.0, 16 @ 1.5, 32 @ 3.0, 57 @ 5.0, 118 ms @ 8.0 m/s
stability, unstable below (linear / pacejka):
  euler x1 0.95 / 0.95   euler x2 0.47 / 0.47   euler x4 0.24 / 0.23
  rk4 x1   0.68 / 0.67   rk4 x2   0.34 / 0.30   RK4 limit k dt = 2.785
accuracy after 30 steps vs RK4 x200 (cm / max error on r):
  (7.0, 0.4, -2.0): euler x1 22.09 / 0.584, euler x4 4.09 / 0.302, rk4 x1 0.00 / 0.000
  (5.0, 0.3, 0.0):  euler x1 10.91 / 0.350, euler x4 2.53 / 0.079, rk4 x1 0.00 / 0.000
cost, ms per blended step, K = 1024:
  euler x1 0.192, x2 0.276, x4 0.475, rk4 x1 0.488, x2 0.896
```

All stability and accuracy numbers match sections 5.2, 6.3, 6.4. Timings are
~1.5x the doc's (slower machine), same ratios. Decision: RK4 x1, same cost as
Euler x4, 100 to 1000x more accurate, stable down to 0.68 < `v_low` = 1.0.

`test_integrator_stable_at_blend_low` can fail (checked in memory): rk4 /
v_low 1.0 -> |r| 6e-39 OK; euler / 1.0 -> 3e-6 OK (threshold 0.95); euler /
0.9 -> 0.52 FAIL; rk4 / 0.6 -> 0.20 FAIL.

`pixi run test` at the end of part E: 49 passed.

**No effect / reverted.**
- `pixi run bench/plot_tires.py` -> `Permission denied (os error 13)`.
  `pixi run` expects a task name or a command, so it tried to execute the
  file itself (not executable, no shebang). Use `pixi run tires` or
  `pixi run python bench/plot_tires.py`.

**YAML fields added in parts B to E** (to mirror in `cuda/include/mppi/config.hpp`):
- `vehicle.mass`, `vehicle.izz`, `vehicle.cornering_stiffness_front`,
  `vehicle.cornering_stiffness_rear` (were in the YAML, now loaded)
- `vehicle.tire_model`, `vehicle.pacejka_c`, `vehicle.pacejka_e`
- `vehicle.blend_speed_low`, `vehicle.blend_speed_high`; removed
  `vehicle.kinematic_blend_speed`
- `vehicle.integrator`, `vehicle.substeps`

**Questions posées pendant l'étape (et réponses).**

*Partie B, modèle dynamique*

- **C'est quoi `check_b.py` ?** Un script jetable, hors du projet (dossier
  temporaire de la session), pour vérifier les équations avant la partie C :
  il remplace en mémoire les fonctions de pneu par un pneu linéaire, puis
  vérifie la rotation de repère, le broadcasting `(5, 7, 6)`, Q-B2 et le
  signe de `v̇x` (Q-B1). Rien à garder : `test_steady_state_cornering` fait
  la vérification officielle.
- **Que fait `dynamic_derivative`, et où est-elle ?** `dynamics.py`, section
  « Modèle dynamique ». Entrée : état + `(a, δ)`. Sortie : les 6 dérivées
  `[ẋ, ẏ, ψ̇, v̇x, v̇y, ṙ]`. Elle ne fait **pas** avancer le temps : c'est
  l'intégrateur qui fait `x + dt·f(x)` (Euler) ou 4 évaluations (RK4).
  Ordre interne : forces de pneu, Newton dans la base qui tourne, rotation de
  la vitesse vers le repère monde.
- **Pourquoi séparer dérivée et intégrateur pour le dynamique, et pas pour le
  cinématique ?** Ce n'est pas parce que `step` était déjà pris : `step` reste
  l'interface unique et choisit le modèle par la taille de l'état. Le
  dynamique est **raide** (constantes de temps de `vy`, `r` de 5 à 32 ms,
  du même ordre que `dt = 20` ms) : il faut pouvoir appeler `f` plusieurs
  fois par pas (RK4, sous-pas), comparer les intégrateurs sur la même
  physique, et tester la physique seule (`v̇y = ṙ = 0` en virage stabilisé).
  Le cinématique n'a pas d'évolution rapide (`r` y est imposé, pas intégré) :
  Euler en un pas suffit, et `max(v + a dt, 0)` porte sur l'état après le
  pas, pas sur la dérivée.

*Partie C, pneus*

- **F(α), c'est la force de frottement ?** Presque. C'est la **force
  latérale** du pneu. À petit angle, le pneu ne glisse pas : la bande de
  roulement se déforme comme un ressort (`F ≈ μ C_S F_z α`). À grand angle,
  le contact glisse vraiment et la force plafonne à `μ F_z` : là, c'est du
  frottement de Coulomb. Adhérence sous la limite, frottement à la limite.
- **Pourquoi `pixi run bench/plot_tires.py` donne `Permission denied` ?**
  Voir **No effect / reverted**.
- **Partie C finie ?** Oui, sauf Q-C1 à Q-C4. Le test 4 de §4.6 (signe sur le
  pas complet) attendait l'intégrateur : ajouté en partie E.

*Partie D, basse vitesse*

- **C'était prévu que le dynamique rate à basse vitesse ?** Oui, dès le
  départ : `edge-mppi.md:140` (bascule sous 1,5 m/s avec interpolation),
  théorie §6.5, `kinematic_blend_speed: 1.5` dans le YAML depuis l'étape 0,
  et le cinématique écrit au CdG à l'étape 1 (S-B1) exprès pour le mélange.
  C'est une limite de tous les modèles de pneu à angle de dérive (`α`
  indéfini à `vx = 0`, pneu arrêté = contrainte de non-glissement).
  Solution standard (AMZ, ETH Zurich, Kabzan et al. 2020).
- **Cette « disjonction de cas » pose problème en CUDA ?** Non : ce n'est pas
  un `if`, c'est un mélange. Tous les threads calculent `dyn` **et** `kin`,
  seule la valeur de `κ` change. `clip`/`maximum` deviennent `fminf`/`fmaxf`,
  des instructions machine sans saut. C'est pour ça qu'il faut le plancher
  `vx_safe` : on calcule toujours le dynamique, même à l'arrêt. Les `if`
  restants (`tire_model`, `integrator`) dépendent du YAML, pas de l'état :
  uniformes sur le warp, et futurs paramètres de template.
- **On calcule cinématique ET dynamique, alors qu'on croyait faire « que
  dynamique » ?** Le « modèle dynamique » du projet est le modèle **mélangé**.
  `model: kinematic` ne calcule toujours que le cinématique. On ne peut pas
  faire que du dynamique : chaque simulation part de `vx = 0` (un pas de
  dynamique pur donne un état tout NaN), et des rollouts qui freinent
  repassent sous 1 m/s. Surcoût estimé (non mesuré) : quelques pourcents en
  CUDA (3 fonctions transcendantes, déjà partagées avec le dynamique, contre
  ~50 pour RK4 + Pacejka). Si ça compte à l'étape 4 : vote de warp
  `__all_sync(κ == 1)` pour sauter le cinématique sans divergence, à mesurer.
- **Partie D finie ?** Oui, sauf Q-D1 à Q-D3. Restait pour la partie E :
  l'intégrateur réel et le second `parametrize`.

*Partie E, intégration*

- **L'idée de la partie E, c'est la divergence de l'approximation ?** En
  partie. Trois critères : **stabilité** (l'erreur explose-t-elle ?),
  **précision** (quelle erreur ?), **coût**. La stabilité est une condition,
  déjà satisfaite par tous les candidats au-dessus de `v_low = 1,0` grâce à la
  partie D ; elle ne départage pas. C'est la précision à coût égal qui
  choisit RK4 ×1 (22 cm d'erreur pour Euler ×1 à 7 m/s, 0,00 pour RK4).
- **Il n'y a pas un Euler implicite dans `step_kinematic6` ?** Non, les
  positions et `vx` sont en Euler **explicite**. Il ne diverge pas car aucune
  de ces dérivées ne rappelle l'état vers une valeur (pas de terme
  `−k·x`). `vy` et `r` ne sont pas intégrés du tout, ils sont **imposés**.
  Mais l'intuition est juste : imposer `r = r_cin`, c'est la limite `k → ∞`
  de l'Euler implicite sur `ṙ = −k (r − r_cin)`, qui donne
  `r_{n+1} = (r_n + k dt r_cin)/(1 + k dt)`, stable pour tout `dt`.
- **Deux cas d'école pour la différence ?** (1) Voiture à vitesse constante,
  `ẋ = v` : Euler exact pour tout `dt`, car `ẋ` ne dépend pas de `x`.
  (2) Café qui refroidit, `Ṫ = −k (T − 20)`, `k = 3`, `dt = 1` : explicite
  80 → −100 → 260 → −460 (écart ×(1 − k dt) = ×−2) ; implicite
  80 → 35 → 23,75 → 20,94 (écart ÷(1 + k dt) = ÷4) ; exact 80 → 23,0 → 20,15.
  `k → ∞` : on impose `T = 20`. Projet : `x, y, ψ` = cas 1 ; `vy, r`
  dynamiques à basse vitesse = cas 2 (`k = 95/vx`) ; cinématique = `T`
  imposé.
- **Mais si la vitesse change beaucoup pendant `dt`, un plus petit `dt` est
  meilleur ?** Oui : c'est la **précision**, pas la stabilité. `ẋ = 2t`,
  `x(3) = 9` : Euler donne 6 / 7,5 / 8,7 pour `dt` = 1 / 0,5 / 0,1 (erreur
  ∝ `dt`, ordre 1). L'erreur s'accumule mais ne s'amplifie pas. Stabilité =
  effet de seuil (`k dt < 2`), précision = continue. C'est le tableau de
  §6.3 : Euler ×1/×2/×4 = 22 / 7,4 / 4,1 cm. RK4 (erreur ∝ `dt⁴`) serait
  exact sur `ẋ = 2t` dès `dt = 1`.
- **Et `v(t) = v_inf (1 − e^{−kt})`, ce n'est pas comme le café ?** Ça dépend
  de **comment on l'écrit**, pas de la forme de la solution. Si `v(t)` est
  une fonction connue et qu'on n'intègre que `x` : imprécis mais stable
  (Euler `x(3)` = 1,95 contre 2,67, `k = 3`, `dt = 1`). Si `v` est un état
  avec `v̇ = −k (v − v_inf)` : c'est exactement le café, `v` = 0 → 3 → −1 → 5.
  Règle : Euler explicite peut exploser quand la dérivée d'un état dépend de
  cet état lui-même avec un rappel fort (`k dt > 2`). Cinématique : `r` lu
  (cas A). Dynamique : `r` intégré avec rappel `95/vx` (cas B).

**Next.** Answer Q-B1 to Q-E4 in `docs/step2_predictions.md` if not done.
Parts F to H are committed (`6ee97ef`, `a756e55`) and `pixi run test` gives
58 passed: they get their own entry.

---

## 2026-10-08 - step 2 - parts F to H (wiring, comparison figure, experiments tooling)

**Goal.**

**Done.**

**Measured.**

**No effect / reverted.**

**Questions posées pendant l'étape (et réponses).**

*Partie F, brancher le modèle*

- **La partie F est finie ?** Oui pour le code : le point de contrôle F
  (`pixi run test` vert, `pixi run sim` boucle un tour avec la config de 7.1)
  passe. 58 tests au lieu des 55 du doc : les 3 de plus sont
  `test_positive_steer_pushes_left` (×3 pneus), absent de la solution. Restent
  les questions Q-F1 à Q-F5.
- **Faire deux configs, `mppi-dynamic.yaml` et `mppi-kinematic.yaml` ?** Non.
  Le protocole 0.3 impose deux contrôleurs identiques sauf le modèle des
  rollouts : deux YAML complets seraient des copies à 99 %, à garder
  synchronisées à la main (λ re-réglé dans l'un, oublié dans l'autre = 
  comparaison faussée sans que rien ne le signale), et le C++ ne lira qu'un
  fichier. Choix : un seul YAML et `run_sim.py --model {kinematic, dynamic,
  both}`, qui fait `replace(cfg, model=m, state_dim=STATE_DIM[m])` (même idée
  que l'en-tête de `test_step1.py`).
- **Que fait exactement `pixi run sim --model kinematic` ?** Pixi colle les
  arguments en plus à la fin du `cmd` de la tâche : `python
  python/run_sim.py --config config/mppi.yaml --model kinematic`, lancé dans
  l'environnement pixi (pas un `pixi run python` imbriqué). Les arguments ne
  vont qu'à la tâche lancée, pas à ses `depends-on` (`track` tourne sans).
  Avec argparse, la dernière valeur gagne : `pixi run sim --config autre.yaml`
  surcharge celle de la tâche.
- **Une tâche pixi peut contenir plusieurs commandes ? Les arguments vont où ?**
  Oui, `cmd` est une ligne de shell (`&&`, `;`, `|`). Les arguments libres
  sont collés à la fin de toute la chaîne, donc ne vont qu'à la **dernière**
  commande (`echo A && echo B extra`). Pour les placer ailleurs : `args =
  [{ arg = "model", default = "dynamic" }]` et `{{ model }}` dans le `cmd`,
  passés par position (`pixi run sim kinematic`). Une tâche qui déclare des
  `args` refuse alors tout argument en plus (« received more arguments than
  expected »).
- **Sans `--config`, quelle config prend `run_sim.py` ?** `config/mppi.yaml`,
  la valeur par défaut d'argparse. Chemin **relatif au dossier courant** :
  sans souci avec pixi (les tâches tournent depuis la racine du projet),
  `FileNotFoundError` si on lance le script à la main depuis `python/`.
  Sans `--model`, c'est le `model:` du YAML.

*Partie G, figure comparative*

- **Pourquoi `pixi run compare` refusait de démarrer (« expected a table key,
  found a newline ») ?** La tâche `sim` avait été écrite sur trois lignes :
  en TOML, une table en ligne `{ … }` doit tenir sur **une seule ligne**. Pour
  l'écrire sur plusieurs lignes, utiliser une section `[tasks.sim]` avec
  `cmd = …`, `args = …`, `depends-on = …`.
- **Pourquoi 1/5 tours propres pour le cinématique + rustine, contre 2/5 dans
  le doc ?** Pas un bug : les lignes dynamique et cinématique pur sont
  identiques au chiffre près. Avec une ESS de 2 à 19, ce contrôleur ne garde
  qu'un ou deux rollouts par itération : il est chaotique, le moindre écart
  numérique change l'issue d'une graine. La conclusion ne change pas.

*Outillage (affichage, rapports)*

- **Comment `rich` fait les en-têtes, barres et tableaux ?** Cinq briques :
  `Console` + balisage `[bold magenta]…[/]` (`escape()` pour le texte venu
  d'ailleurs, `highlight=False` pour que seules nos couleurs aient un sens) ;
  `Progress` = suite de colonnes (spinner, texte, barre, champ libre, chrono)
  et des tâches qu'on fait avancer (`update`, `advance`, `stop_task`) ;
  `Table` (`box.SIMPLE_HEAD`, `no_wrap` sur les nombres, `overflow="fold"`
  sur les noms, `add_section()`) ; `console.status(...)` pour un spinner seul.
  `simulate(..., on_step=...)` fait avancer la barre sans que la simulation
  connaisse `rich`. Pour un pool de processus, `as_completed` plutôt que
  `pool.map`, sinon la barre attend la simulation la plus lente.
- **À quoi sert `report.py` ?** À l'affichage et aux rapports, rien d'autre :
  il ne calcule aucun résultat. Chaîne : `simulate()` → `SimLog`,
  `summarize()` → nombres bruts, `report.*` → écran et fichiers. Un seul
  endroit pour les règles de couleur (vert dans la cible, jaune proche d'une
  limite, rouge échec), partagé par `sim`, `sweep` et `compare`. Une cellule
  est `(texte, style)` : le terminal reçoit le style, le Markdown le texte
  seul, donc les deux montrent les mêmes chiffres. Hors de `python/mppi/` :
  ce n'est pas du code de référence pour CUDA.
- **Ne sauvegarder un rapport que quand je le demande ?** Flag `--save` sur
  les trois scripts : rien n'est écrit sans lui ; `--save` →
  `results/reports/<kind>_<date>_<tag>.md` + `.json` (config complète,
  métriques brutes) ; `--save=NOM` remplace le tag. Pièges : avec `sim`, le
  modèle d'abord (`pixi run sim dynamic --save`, arguments positionnels) ;
  avec `sweep`, `--save` après les expériences ou `--save=NOM`, sinon
  `--save H1` prend `H1` pour le nom.

*Bilan*

- **Hormis les questions H et le journal, l'étape 2 est finie ?** Le code oui,
  tous les points techniques du critère 0.2 sont cochés. Restent : les
  prédictions et mesures H1 à H9, les réponses Q-B à Q-G, la vérification de
  Q-A3 sur la figure, la case du README, le commit et `git tag step-2`.

**Next.**

---

## 2026-10-09 - plan - hardware target: RTX 4000 Ada instead of Jetson

**Goal.** Drop the Jetson Orin Nano: its price (RAM costs) is not justified by
what it adds to the project. Retarget the plan to the GPU node of the cluster.

**Done.** `README.md` and `docs/*` rewritten for an NVIDIA RTX 4000 Ada
Generation (AD104, compute capability 8.9, 48 SMs, 20 GB GDDR6, PCIe 4.0 x16,
130 W). Step 5 becomes a real-time closed loop with simulator and controller in
two processes on the node; the Jetson-specific work (unified memory,
`nvpmodel`) is replaced by discrete-GPU work: pinned / mapped / managed memory
for the tiny per-iteration transfers, CUDA Graphs, reduced compute budget
(power limit, clock lock or SM share), jitter on a shared node. The RTX 3060 /
Orin comparison of step 6 becomes NumPy vs C++ CPU vs CUDA, full and throttled.
Entries above this one are left as they were written.

**Measured.**

**No effect / reverted.**

Also: `CMAKE_CUDA_ARCHITECTURES 89` in `CMakeLists.txt`, `linux-aarch64`
dropped from `pixi.toml` (lockfile regenerated), Orin column removed from
`bench/results.md`. Clean rebuild compiles for sm_89.

**Next.** Run `hello_cuda` on the cluster node to confirm the sm_89 binary.

---

## 2026-10-09 - tooling - MPPI videos and README media

**Goal.** Show how MPPI works, not only the path it drives: a video of single
iterations (rollouts, scores, weighted mean, applied control), inside a lap.

**Done.** `python/video_mppi.py`. From a state x_k of a saved trajectory it
recomputes the iterations (`run_sim.py` keeps no rollout on disk), with U
warm-started from the controls actually applied, `control[k:k+T]`: faithful in
shape, not bit for bit. Each cycle: K rollouts growing in gray, colored by cost
rank, `U*` rolled out, first control applied. `--lap` plays the whole lap
around it: real time, slowdown and zoom in to x_k (s(u) = k - L(1-u)^2, same
speed at the junction), cycles on the saved run, zoom out, real time. Defaults
to the step at 12 m, the beam of `step2_*_trajectory.png`. Pixi tasks
`video-mppi`, `video-lap`, `media`. `docs/media/` (versioned, gitignore
exception) holds the README GIF and MP4. `python/tests/test_video.py` covers the
geometry and timing helpers.

**Measured.** Rendering about 1.5 min for the default lap video (759 frames at
30 fps, 25 s). GIF through ffmpeg palette, 800 px, 15 fps: 4.4 MB (pillow at
the same settings: 6.5 MB, coarser).

**No effect / reverted.** A first version jumped up to two steps where the real
time part hands over to the slowdown (`np.arange` stopped short of k - L);
caught by `test_lap_timeline_is_continuous`, fixed.

**Next.** Regenerate `docs/media/` with `pixi run media` whenever the
step-2 trajectory changes.
