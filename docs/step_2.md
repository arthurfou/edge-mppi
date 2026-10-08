# Étape 2 : modèle dynamique de véhicule en Python/NumPy

TP autonome, suite directe de `docs/step_1.md`. Ce fichier contient tout ce qu'il faut pour réussir l'étape : les notions physiques expliquées depuis zéro, les dérivations pas à pas, les ordres de grandeur calculés sur **tes** paramètres, les rappels d'API, les pièges, les vérifications intermédiaires et, à la fin, les solutions complètes. La théorie de fond est dans `docs/mppi-theory.md`, référencée par numéro de section (« théorie §6.5 »). Ce document l'explique plus lentement, sans la recopier : quand une notion y est déjà bien écrite, on y renvoie.

**Règle du jeu (inchangée).** Les questions marquées **Q** demandent de réfléchir avant de coder. Écris ta réponse, puis compare-la à la section [Solutions](#solutions). Les blocs de code de l'énoncé sont des squelettes et des rappels d'API, pas la solution. Les **points de contrôle** (✔) te disent quoi vérifier avant de passer à la suite : si un point de contrôle ne passe pas, ne continue pas, va voir le [dépannage](#10-dépannage).

**Durée estimée.** 3 à 4 jours, comme prévu dans `edge-mppi.md`. Les parties B, C et D (la physique) prennent la moitié du temps. Ne les survole pas : tout ce que tu écris ici sera réécrit en CUDA à l'étape 3, et une erreur de signe dans une force de pneu ne se voit pas à l'œil.

**D'où viennent les chiffres.** Tous les résultats « mesurés » de ce document (temps au tour, ESS, seuils de stabilité, figure comparative, expériences) ont été obtenus en exécutant la solution de référence donnée à la fin, sur la machine de dev (28 cœurs, NumPy 2.4, Python 3.11). Tes chiffres exacts différeront un peu, les ordres de grandeur doivent être les mêmes. Quand un résultat attendu par le plan ne s'est **pas** produit, c'est écrit.

## Sommaire

- [0. Objectif, critère de sortie et protocole de la figure](#0-objectif-critère-de-sortie-et-protocole-de-la-figure)
- [1. Préparer le terrain](#1-préparer-le-terrain)
- [2. Partie A : pourquoi le modèle cinématique ne suffit plus](#2-partie-a--pourquoi-le-modèle-cinématique-ne-suffit-plus)
- [3. Partie B : le bicycle dynamique, des repères aux équations](#3-partie-b--le-bicycle-dynamique-des-repères-aux-équations)
- [4. Partie C : angles de dérive et modèles de pneu](#4-partie-c--angles-de-dérive-et-modèles-de-pneu)
- [5. Partie D : la basse vitesse et le mélange cinématique/dynamique](#5-partie-d--la-basse-vitesse-et-le-mélange-cinématiquedynamique)
- [6. Partie E : l'intégration numérique](#6-partie-e--lintégration-numérique)
- [7. Partie F : brancher le modèle dans le projet](#7-partie-f--brancher-le-modèle-dans-le-projet)
- [8. Partie G : premier tour rapide et figure comparative](#8-partie-g--premier-tour-rapide-et-figure-comparative)
- [9. Partie H : réglage et expériences](#9-partie-h--réglage-et-expériences)
- [10. Dépannage](#10-dépannage)
- [11. Clôturer l'étape](#11-clôturer-létape)
- [Solutions](#solutions)

---

## 0. Objectif, critère de sortie et protocole de la figure

### 0.1 Ce que tu vas construire

À l'étape 1, le contrôleur et le « vrai » véhicule simulé utilisaient le même modèle cinématique : les roues ne glissent jamais, la voiture tourne sur un cercle de 0,78 m de rayon à n'importe quelle vitesse. Ce monde est trop gentil. À l'étape 2, tu écris un modèle **dynamique** : la voiture a une masse, une inertie de rotation, et ses pneus produisent des forces limitées par l'adhérence. Au-delà d'une certaine vitesse en virage, ils glissent, et la voiture dérape.

Tu vas :

1. écrire le modèle dynamique (état `[x, y, ψ, vx, vy, r]`) avec trois modèles de pneu (linéaire, tanh, Pacejka simplifié) ;
2. le rendre sûr à basse vitesse par un mélange continu avec le modèle cinématique ;
3. choisir un intégrateur numérique en le **mesurant** ;
4. faire du modèle dynamique le « vrai » véhicule de la simulation ;
5. faire rouler MPPI vite, avec dérapage, et montrer que le modèle cinématique échoue là où le dynamique passe.

Comme à l'étape 1, le code NumPy est la **référence** du kernel CUDA de l'étape 3 : chaque formule, chaque `clip`, chaque ordre d'opération devra y être reproduit et comparé coût par coût à 1e-4 près en FP32. Garde donc le même esprit : simple, explicite, sans branche qui dépend de l'état.

### 0.2 Critère de sortie

Repris de `edge-mppi.md`, précisé :

- [ ] un tour à vitesse élevée (`v_ref = 7` m/s ici) **sans sortie de piste**, avec un dérapage visible (dérive du véhicule |β| > 10° dans les virages rapides) tenu par le contrôleur ;
- [ ] la **figure comparative** `results/figures/step2_comparison.png` et son tableau sur 5 graines (protocole en 0.3) ;
- [ ] le **test croisé** cinématique/dynamique à basse vitesse, conservé comme test de non-régression permanent ;
- [ ] pneu linéaire, puis tanh et Pacejka simplifié, sélectionnables dans le YAML ;
- [ ] la bascule cinématique à basse vitesse par mélange continu sur la bande [1,0 ; 1,5] m/s ;
- [ ] un intégrateur choisi et justifié par des mesures (partie E) ;
- [ ] λ re-réglé en mesurant l'ESS, et le tableau de prédictions de la partie H rempli ;
- [ ] tous les tests passent (ceux de l'étape 1 compris) ;
- [ ] une entrée dans `docs/journal.md` avec les chiffres mesurés et la liste des champs YAML ajoutés ;
- [ ] le tag `git tag step-2`.

### 0.3 Le protocole de la figure comparative (à lire attentivement)

La phrase du plan, « le cinématique doit échouer là où le dynamique passe », peut se lire de deux façons :

| Lecture | Ce qu'on simule | Ce que ça montre |
|---|---|---|
| (a) chaque contrôleur avec « son » monde | MPPI cinématique sur un véhicule cinématique, MPPI dynamique sur un véhicule dynamique | rien d'utile : dans un monde cinématique, la voiture n'a pas de limite d'adhérence, elle ne peut **pas** échouer par excès de vitesse. On compare deux problèmes différents. |
| (b) un seul monde, deux modèles internes | le véhicule simulé est **toujours** le modèle dynamique ; seul change le modèle utilisé **dans les rollouts** de MPPI | l'effet du **désaccord de modèle** (théorie §7.9) : MPPI optimise contre son modèle, et un modèle qui ignore la saturation des pneus planifie des virages impossibles. |

La lecture (b) est la seule défendable, et c'est celle du projet. Le modèle dynamique joue le rôle de la réalité (comme le ferait la vraie voiture), et la question posée est : « quel modèle faut-il mettre **dans** le contrôleur pour piloter cette réalité ? ». Tout le reste est identique entre les deux contrôleurs : piste, `v_ref`, coût, λ, K, T, Σ, graines.

Deux subtilités, que la figure doit afficher honnêtement :

- **Le terme d'adhérence de l'étape 1.** C'était une rustine (S-D4 de l'étape 1) : une connaissance de la limite des pneus injectée dans le coût du contrôleur cinématique. On compare donc **trois** contrôleurs : dynamique, cinématique pur (sans rustine), et cinématique + rustine (le contrôleur de l'étape 1 tel quel).
- **Une seule graine ne prouve rien.** On lance 5 graines par contrôleur et on compte les tours propres.

Résultat mesuré (détaillé en partie G) : à `v_ref = 7` m/s et λ = 3, le contrôleur dynamique fait 5 tours propres sur 5 ; le cinématique pur 0 sur 5 ; le cinématique avec rustine 2 sur 5. Et un résultat moins confortable, que tu dois connaître avant de présenter la figure : en montant λ à 10, le cinématique avec rustine passe, mais 2 s plus lentement au tour (partie H, H9).

### 0.4 Vue d'ensemble des fichiers

| Fichier | Rôle | Partie |
|---|---|---|
| `config/mppi.yaml` | nouveaux champs véhicule, `model: dynamic`, λ et `v_ref` re-réglés | F, H |
| `python/mppi/config.py` | `Vehicle` étendue, validation | F |
| `python/mppi/dynamics.py` | pneus, modèle dynamique, mélange, intégrateur | B à E |
| `python/mppi/cost.py` | terme d'adhérence réservé au cinématique | F |
| `python/mppi/controller.py` | **aucun changement** (et c'est voulu) | F |
| `python/run_sim.py` | véhicule simulé toujours dynamique, nouvelles courbes | F |
| `python/tests/test_step1.py` | une ligne : forcer le cinématique | F |
| `python/tests/test_step2.py` | nouveaux tests | B à F |
| `bench/compare_models.py` | figure comparative | G |
| `bench/sweep.py` | expériences H1 à H9 | H |
| `cuda/include/mppi/config.hpp` | commentaire listant les champs à parser à l'étape 3 | F |

Ordre conseillé : 1 → A (lecture et calculs) → B → C (avec tests) → D (avec tests) → E (mesures) → F → G → H. Chaque partie a ses points de contrôle.

---

## 1. Préparer le terrain

### 1.1 Ce qui existe déjà et que tu réutilises

- `controller.py` lit `cfg.state_dim` et appelle `step(state, control, dt, vehicle)`. Il ne sait pas quel modèle il déroule. C'est la décision d'architecture §6.1 de `edge-mppi.md`, et c'est elle qui te fait gagner du temps maintenant.
- L'état cinématique est écrit **au centre de gravité** (CdG), pas à l'essieu arrière. Tu avais fait ce choix exprès (S-B1 de l'étape 1) : le mélange de la partie D n'a de sens que si les deux modèles décrivent le même point.
- `config/mppi.yaml` contient déjà `mass: 3.5`, `izz: 0.04`, `cornering_stiffness_front: 4.0`, `cornering_stiffness_rear: 4.2`, `kinematic_blend_speed: 1.5` et `mu: 1.0`. Ils ne sont pas encore chargés par `config.py`.

### 1.2 Deux tâches pixi de plus

Dans `pixi.toml`, section `[tasks]`, sous les tâches de l'étape 1 :

```toml
# STEP 2
compare = { cmd = "python bench/compare_models.py", depends-on = ["track"] }
```

`sim`, `test` et `sweep` restent les mêmes.

### 1.3 Rappels NumPy pour cette étape

**`np.clip` et `np.where` à la place des `if`.** Un `if` sur un tableau lève une erreur, et en CUDA un `if` qui dépend de l'état fait diverger les threads d'un warp (journal de l'étape 1). Deux outils :

```python
kappa = np.clip((vx - lo) / (hi - lo), 0.0, 1.0)      # rampe 0 → 1, bornée
vx_safe = np.maximum(vx, v_low)                         # plancher élément par élément
y = np.where(cond, a, b)                                # choisit élément par élément
```

**Attention, `np.where` évalue les deux branches.** `np.where(vx > 0, vy / vx, 0.0)` calcule quand même `vy / 0` pour les éléments où `vx = 0`, ce qui donne un `RuntimeWarning` et un `inf` ou un `NaN` dans le tableau intermédiaire. Ici ça passe (le `NaN` n'est pas sélectionné), mais si ce tableau est ensuite **multiplié** au lieu d'être sélectionné, `0 * NaN = NaN` contamine tout. C'est le piège central de la partie D.

**Faire échouer un test au premier `NaN`.** Par défaut, NumPy ne fait qu'avertir. Dans un test :

```python
with np.errstate(all="raise"):     # division par zéro, overflow, invalide -> exception
    out = step(states, controls, dt, veh)
```

**Recoller des composantes.** `np.stack([a, b, c], axis=-1)` crée un nouvel axe à la fin (formes `(...,)` → `(..., 3)`). `np.concatenate([X[..., :3], v[..., None], X[..., 4:]], axis=-1)` recolle des morceaux qui ont déjà l'axe d'état. `v[..., None]` ajoute un axe de taille 1 à la fin.

**Broadcasting d'un coefficient par état.** `kappa` a la forme `(K,)` et `dyn` la forme `(K, 6)`. `kappa * dyn` échoue (les formes se comparent par la droite : 6 contre K). Il faut `kappa[..., None] * dyn` : `(K, 1)` contre `(K, 6)`.

**Remplacer un champ d'une dataclass figée.** `dataclasses.replace(veh, tire_model="linear")` renvoie une copie. Tu t'en serviras beaucoup dans les tests et la partie H.

**Paramétrer un test.** Pour lancer le même test sur les trois pneus :

```python
@pytest.mark.parametrize("tire", ("linear", "tanh", "pacejka"))
def test_quelque_chose(tire):
    ...
```

Deux `parametrize` empilés donnent le produit cartésien (3 pneus × 2 intégrateurs = 6 tests).

### 1.4 Ce que tu ne dois PAS changer

- `controller.py` : rien. Si tu as envie d'y écrire `if cfg.model == ...`, c'est que l'interface est mal utilisée.
- `track.py`, `generate_track.py` : rien.
- L'ordre de l'état `[x, y, ψ, vx, vy, r]` : il est écrit dans l'en-tête de `dynamics.py` et dans `config/mppi.yaml` depuis l'étape 0.

---

## 2. Partie A : pourquoi le modèle cinématique ne suffit plus

Pas de code dans cette partie : des calculs à faire sur papier, qui te donnent les ordres de grandeur de toute l'étape.

### 2.1 Ce que suppose le modèle cinématique

Le bicycle cinématique (théorie §6.2) suppose que **chaque roue roule sans glisser** : la vitesse du point de contact est exactement dans le plan de la roue. Avec cette hypothèse, la géométrie seule fixe le mouvement : les perpendiculaires aux deux roues se coupent au centre instantané de rotation, et le rayon de virage ne dépend que du braquage.

```
   roue arrière        CdG         roue avant, braquée de δ
       ║══════════════●═══════════╱╱
       │                         ╱   perpendiculaire à la roue avant
       │                       ╱
       │ perpendiculaire     ╱
       │ à la roue arrière ╱
       │                 ╱
       └───────────────●   centre instantané de rotation : le point de rencontre
                           ne dépend que de δ et de L, pas de la vitesse
```

Conséquence : à δ = 0,4 rad, le CdG décrit un cercle de rayon `R = L / (cos β · tan δ)` ≈ 0,80 m (S-B2 de l'étape 1), **à 1 m/s comme à 8 m/s**. L'accélération latérale `v²/R` n'est limitée par rien.

### 2.2 Ce que fait un vrai pneu

Un pneu ne peut pousser la voiture sur le côté qu'avec une force limitée, environ `μ · F_z`, où `F_z` est le poids qu'il porte et `μ` le coefficient d'adhérence (≈ 1 ici). Pour toute la voiture, l'accélération latérale plafonne donc vers `μ g` = 9,81 m/s². Un virage de rayon `R` ne peut être pris qu'en dessous de

$$
v_{\max} = \sqrt{\mu\, g\, R}.
$$

**Q-A1.** Calcule `v_max` pour le virage le plus serré de la piste (R_min = 2,19 m, sortie de `generate_track.py`) et pour le cercle à braquage maximal (R ≈ 0,80 m). Quelle accélération latérale le modèle cinématique prédit-il si la voiture passe le virage serré à 7 m/s ? À l'étape 1, à `v_ref = 3`, la simulation affichait `max a_lat 6.3 m/s2` : le modèle cinématique y était-il encore valable ?

### 2.3 Ce qui se passe au-delà de la limite

Quand on demande plus que `μ g`, les pneus glissent : la voiture ne suit plus la direction de ses roues. Deux angles décrivent ce glissement :

- l'**angle de dérive du véhicule** `β = atan(vy / vx)` : l'écart entre la direction où pointe le nez et la direction où va réellement le CdG. Quelques degrés en conduite normale, 10 à 30° quand la voiture « dérape » ;
- les **angles de dérive des pneus** `α_f`, `α_r` : la même chose, roue par roue (partie C).

```
   conduite normale (β ≈ 0)            dérapage dans un virage à gauche (β < 0)

        ↑ nez = vitesse                    nez ↖   ↑ vitesse réelle du CdG
        │                                       ╲  │
        │                                        ╲ │   β = angle entre les deux
        ●                                          ●
```

Un modèle cinématique ne peut pas représenter `β` autrement que par sa valeur géométrique (`atan(l_r/L · tan δ)`), et il n'a ni `vy` ni `r` comme variables libres. Le dérapage n'existe pas dans son monde.

### 2.4 Ce que l'étape 1 faisait pour compenser

Le terme d'adhérence du coût, `w_adhesion · max(0, a_lat/(μg) − 1)²` avec `a_lat = v² tan δ / L`, pénalisait les virages que les pneus ne tiendraient pas. C'est une **rustine** : la physique manquante est mise dans le coût au lieu du modèle. Le journal de l'étape 1 a montré son prix : à `v_ref = 5`, ce terme étale les coûts (médiane − min passe de 0,7 à 57), et λ = 0,3 effondre l'ESS à 1.

**Q-A2.** Donne deux raisons pour lesquelles ce terme ne remplace pas un modèle dynamique, même bien réglé. Pense à ce qu'il mesure (quelle accélération latérale ?) et à ce que la voiture fait **après** avoir dépassé la limite.

**Q-A3.** Avec le protocole de la section 0.3, le véhicule simulé est dynamique, le contrôleur peut être cinématique. Que fait concrètement la voiture quand le contrôleur cinématique lui demande un virage à 2 g ? Fais une prédiction écrite : tu la vérifieras sur la figure de la partie G.

✔ **Point de contrôle A.** Tu sais dire, avec des chiffres, à partir de quelle vitesse le virage serré de la piste n'est plus faisable, et ce que veut dire `β = −15°`.

---

## 3. Partie B : le bicycle dynamique, des repères aux équations

### 3.1 Deux repères

```
          y monde                                     
          ↑                     y véhicule  x véhicule
          │                          ↖      ↗
          │                            ╲  ╱  ψ (cap) mesuré depuis x monde,
          │                             ● CdG   sens trigonométrique
          │                                 
          └──────────────→ x monde
```

- **Repère monde** : fixe. On y exprime la position `(x, y)` du CdG et le cap `ψ`.
- **Repère véhicule** : attaché au CdG, axe `x` vers l'avant de la voiture, axe `y` vers sa gauche. Il **tourne** avec la voiture, à la vitesse `r = ψ̇` (vitesse de lacet, en rad/s).

Les vitesses `vx`, `vy` de l'état sont les composantes de la vitesse du CdG **dans le repère véhicule**. En ligne droite sans glisser, `vy = 0` ; en dérapage, `vy ≠ 0`. C'est ce choix qui rend les forces de pneu simples à écrire (elles sont naturellement exprimées par rapport à la voiture) et qui rend les équations un peu plus subtiles (le repère tourne).

Les vecteurs unitaires du repère véhicule, vus du monde :

$$
\mathbf{e}_x = (\cos\psi,\ \sin\psi), \qquad \mathbf{e}_y = (-\sin\psi,\ \cos\psi).
$$

### 3.2 La position : changer de repère

La vitesse du CdG est `V = vx·e_x + vy·e_y`. En remplaçant `e_x` et `e_y` :

$$
\dot x = v_x\cos\psi - v_y\sin\psi, \qquad
\dot y = v_x\sin\psi + v_y\cos\psi, \qquad
\dot\psi = r.
$$

C'est une simple rotation d'angle ψ. Vérification : `ψ = 0`, `vy = 0` donne `ẋ = vx`, `ẏ = 0`.

### 3.3 Les vitesses : dériver dans un repère qui tourne

C'est le point qui pose problème à tout le monde la première fois, alors on le fait pas à pas.

Newton s'écrit dans le repère monde : `m · dV/dt = F`. On dérive `V = vx·e_x + vy·e_y` **en tenant compte du fait que `e_x` et `e_y` bougent** :

$$
\frac{d\mathbf V}{dt} = \dot v_x\,\mathbf e_x + \dot v_y\,\mathbf e_y + v_x\,\dot{\mathbf e}_x + v_y\,\dot{\mathbf e}_y.
$$

Dérivons les vecteurs unitaires (règle de dérivation en chaîne, `ψ̇ = r`) :

$$
\dot{\mathbf e}_x = r\,(-\sin\psi,\ \cos\psi) = r\,\mathbf e_y, \qquad
\dot{\mathbf e}_y = r\,(-\cos\psi,\ -\sin\psi) = -r\,\mathbf e_x.
$$

En regroupant sur `e_x` et `e_y` :

$$
\frac{d\mathbf V}{dt} = (\dot v_x - r\,v_y)\,\mathbf e_x + (\dot v_y + r\,v_x)\,\mathbf e_y.
$$

Avec `F = F_x·e_x + F_y·e_y` (forces exprimées dans le repère véhicule), Newton composante par composante donne

$$
\dot v_x = \frac{F_x}{m} + v_y\, r, \qquad \dot v_y = \frac{F_y}{m} - v_x\, r.
$$

**Lecture physique du terme `− vx r`.** En virage stabilisé, la voiture tourne à `r` constant et `vy` reste constant (`v̇y = 0`). Il faut donc `F_y = m·vx·r` : les pneus fournissent exactement l'accélération centripète `vx·r` (qui vaut `v²/R` sur un cercle). Sans force latérale, `v̇y = −vx·r` : la voiture tourne mais sa vitesse ne suit pas, elle part vers l'extérieur dans son propre repère. C'est le dérapage.

C'est la même chose que la formule `v̇ + ω × v` de la théorie §6.3, avec `ω × v = (−r vy, r vx)`.

### 3.4 Les forces et le moment

```
   vue de dessus, x véhicule vers la droite, y véhicule vers le haut

        F_yr ↑                                  ↖ F_yf : perpendiculaire à la roue avant,
             │                                   ╲  donc inclinée de δ vers l'arrière
   arrière   ●═══════════════●═══════════════════●  avant, braquée de δ
             ←──── l_r ─────→←─────── l_f ───────→
                            CdG
```

- La roue avant est braquée de δ : sa force latérale `F_yf` est perpendiculaire à la roue, donc dans le repère véhicule elle vaut `(−F_yf sin δ, F_yf cos δ)`.
- La roue arrière n'est pas braquée : `(0, F_yr)`.
- La commande `a` est traitée comme une force longitudinale totale divisée par la masse (choix du projet, théorie §6.3 ; le vrai variateur est plus compliqué, théorie §6.6).

Somme des forces et moment autour du CdG (bras de levier `+l_f` pour l'avant, `−l_r` pour l'arrière) :

$$
\dot v_x = a - \frac{F_{yf}\sin\delta}{m} + v_y r, \qquad
\dot v_y = \frac{F_{yf}\cos\delta + F_{yr}}{m} - v_x r, \qquad
\dot r = \frac{l_f F_{yf}\cos\delta - l_r F_{yr}}{I_z}.
$$

Avec les trois équations de position, ce sont les six équations du modèle (théorie §6.3). Tout ce qui reste à définir, ce sont `F_yf` et `F_yr` : partie C.

**Q-B1.** Le terme `− F_yf sin δ / m` dans `v̇x` n'existe pas dans le modèle cinématique. Que représente-t-il physiquement ? Prédis son signe et son effet sur la vitesse quand la voiture tourne avec un braquage constant et `a = 0`.

**Q-B2.** En virage stabilisé (`v̇y = 0`, `ṙ = 0`), que valent `F_yf cos δ + F_yr` et `l_f F_yf cos δ − l_r F_yr` ? Tu en feras un test (`test_steady_state_cornering`).

**Q-B3.** Où l'inertie `I_z` intervient-elle, et que se passe-t-il quand on braque brusquement, comparé au modèle cinématique ?

### 3.5 Les paramètres, en chiffres

| Symbole | YAML | Valeur | Commentaire |
|---|---|---|---|
| `m` | `mass` | 3,5 kg | voiture F1TENTH avec batterie |
| `I_z` | `izz` | 0,04 kg·m² | inertie de lacet |
| `l_f`, `l_r` | `lf`, `lr` | 0,15 / 0,18 m | CdG plus près de l'avant |
| `L` | `wheelbase` | 0,33 m | |
| `μ` | `mu` | 1,0 | |
| `C_Sf`, `C_Sr` | `cornering_stiffness_*` | 4,0 / 4,2 rad⁻¹ | partie C |

Remarque : le simulateur F1TENTH (`f1tenth_gym`) utilise des valeurs proches mais pas identiques (m = 3,74 kg, I_z = 0,04712, l_f = 0,15875, l_r = 0,17145, C_Sf = 4,718, C_Sr = 5,4562, μ = 1,0489). Les valeurs du projet sont « de type F1TENTH », pas une copie. Ce n'est pas grave (la plateforme est figée, `edge-mppi.md` §6.5), mais ne prétends pas dans le rapport qu'elles sont identifiées sur la vraie voiture.

### 3.6 Squelette

```python
def dynamic_derivative(state, a, delta, vehicle):
    """d(state)/dt, (..., 6) -> (..., 6). Theory 6.3."""
    psi, vx, vy, r = state[..., 2], state[..., 3], state[..., 4], state[..., 5]
    # 1. angles de dérive (partie C)
    # 2. forces de pneu (partie C)
    # 3. les six dérivées, recollées avec np.stack(..., axis=-1)
```

On écrit la **dérivée** (le second membre `f(x, u)` de `ẋ = f(x, u)`), pas directement le pas de temps : l'intégrateur de la partie E l'appellera une ou plusieurs fois par pas.

✔ **Point de contrôle B.** Tu sais refaire la dérivation de 3.3 sans regarder, et tu sais dire pourquoi `F_y = m·vx·r` en virage stabilisé.

---

## 4. Partie C : angles de dérive et modèles de pneu

### 4.1 L'angle de dérive d'un pneu, depuis zéro

Un pneu en caoutchouc ne transmet une force latérale que s'il se déforme, c'est-à-dire s'il « crabe » un peu : la vitesse de son point de contact fait un petit angle avec le plan de la roue. Cet angle est l'**angle de dérive du pneu** `α`. Pas d'angle, pas de force ; un peu d'angle, une force à peu près proportionnelle ; trop d'angle, le pneu glisse et la force plafonne.

```
       plan de la roue
            │
            │   α  ← angle de dérive
            │  ╱
            │ ╱   vitesse réelle du point de contact
            │╱
            ●
   force latérale ← poussée perpendiculaire à la roue, vers le côté où « pointe » la roue
```

**Vitesse du centre de chaque roue.** Un point d'un solide en rotation a la vitesse du CdG plus `ω × (position relative)`. Pour la roue avant, à `(+l_f, 0)` dans le repère véhicule, avec `ω = r` autour de l'axe vertical : `ω × (l_f, 0) = (0, r·l_f)`. Donc :

$$
\text{avant : } (v_x,\ v_y + l_f r), \qquad \text{arrière : } (v_x,\ v_y - l_r r).
$$

**Direction de ces vitesses** : `atan((vy + l_f r)/vx)` à l'avant, `atan((vy − l_r r)/vx)` à l'arrière. Le plan de la roue avant fait l'angle δ avec l'axe du véhicule, celui de la roue arrière l'angle 0. L'angle de dérive est l'écart entre le plan de la roue et la direction de la vitesse :

$$
\alpha_f = \delta - \arctan\!\frac{v_y + l_f r}{v_x}, \qquad
\alpha_r = -\arctan\!\frac{v_y - l_r r}{v_x}.
$$

**Convention de signe** (théorie §6.3) : `α > 0` produit une force vers `+y` (la gauche). Vérifie-la sur le cas le plus simple : la voiture roule tout droit (`vy = r = 0`) et tu braques à gauche (`δ > 0`) : `α_f = δ > 0`, force vers la gauche à l'avant, moment `l_f F_yf > 0`, donc `ṙ > 0` : la voiture tourne à gauche. C'est le test le plus important de la partie, comme le test de signe de l'étape 1.

**Q-C1.** La voiture tourne à gauche en régime établi, à vitesse modérée. Quel est le signe de `α_r` ? Et celui de `vy` à basse vitesse (pense au modèle cinématique : le CdG se déplace-t-il vers l'intérieur ou l'extérieur du virage par rapport au nez) ? Et à haute vitesse, quand l'arrière doit glisser pour fournir de la force ?

### 4.2 La rigidité de dérive : trancher l'interprétation de 4,0 et 4,2

Le modèle linéaire s'écrit `F_y = C_α · α`, où `C_α` est la **rigidité de dérive** en N/rad. Une vraie voiture a des `C_α` de dizaines de milliers de N/rad. Une voiture de 3,5 kg, de l'ordre de la centaine. Des valeurs de 4,0 et 4,2 sont donc trop petites pour être des N/rad : une force de 4 N pour un radian d'angle (57°) ne tiendrait pas un virage à 1 m/s.

Elles suivent la convention du modèle *single-track* de CommonRoad, reprise par le simulateur F1TENTH : la rigidité est **normalisée par la charge verticale** et multipliée par μ,

$$
F_y = \mu\, C_S\, F_z\, \alpha .
$$

**Vérification par la source.** Dans `f1tenth_gym` (`gym/f110_gym/envs/dynamic_models.py`, fonction `vehicle_dynamics_st`), la dérivée de la vitesse de lacet contient le terme `mu*m/(I*(lr+lf)) * lf*C_Sf*(g*lr - u[1]*h) * x[2]` (où `x[2]` est le braquage). En réarrangeant, c'est `(l_f / I_z) · μ C_Sf · (m g l_r / L) · δ`, c'est-à-dire un moment `l_f · F_yf` avec `F_yf = μ C_Sf F_zf α_f` et `F_zf = m g l_r / L`. Le `u[1]*h` est le transfert de charge longitudinal (hauteur du CdG `h`), que le projet néglige : il n'a pas de paramètre `h`.

**Charges statiques par essieu** (le CdG est plus près de l'avant, donc l'avant porte plus) :

$$
F_{zf} = m g \frac{l_r}{L} = 3{,}5 \times 9{,}81 \times \frac{0{,}18}{0{,}33} = 18{,}73\ \text{N}, \qquad
F_{zr} = m g \frac{l_f}{L} = 15{,}61\ \text{N}.
$$

**Rigidités en N/rad** : `C_αf = 1,0 × 4,0 × 18,73 = 74,9 N/rad`, `C_αr = 1,0 × 4,2 × 15,61 = 65,6 N/rad`. Ce sont les valeurs de la théorie §6.3, et `test_cornering_stiffness_convention` les vérifie.

**Q-C2.** Le pneu linéaire ne sature jamais. À quel angle `α` atteint-il la force `μ F_z` qu'un vrai pneu ne peut pas dépasser ? Donne la valeur en degrés. (Indice : elle ne dépend que de `C_S`.)

### 4.3 Sous-vireur ou survireur

Le **gradient de sous-virage** (théorie §6.3) compare la facilité avec laquelle l'avant et l'arrière fournissent leur force :

$$
K_{us} = \frac{F_{zf}}{C_{\alpha f}} - \frac{F_{zr}}{C_{\alpha r}} = \frac{1}{C_{Sf}} - \frac{1}{C_{Sr}} = 0{,}25 - 0{,}238 = 0{,}0119 \text{ rad/g}.
$$

Positif : légèrement sous-vireur (l'avant atteint sa limite le premier, la voiture « tire tout droit » à la limite). Très petit : presque neutre. En régime établi, à petit angle, la vitesse de lacet vaut

$$
r = \frac{v_x\, \delta}{L + K_{us}\, v_x^2 / g},
$$

un peu moins que la valeur cinématique `vx tan δ / L`. Mesuré avec le modèle linéaire, vx = 3 m/s, δ = 0,05 rad : `r = 0,43974` rad/s, formule `0,43998`, cinématique `0,45492`. L'écart au cinématique, 3,3 %, est l'effet du sous-virage ; il sert à calibrer la tolérance du test croisé (partie F).

### 4.4 Trois modèles de pneu

On veut trois modèles **qui ont la même pente à l'origine** `μ C_S F_z` : ainsi, à petit angle, ils sont interchangeables, et le test croisé à basse vitesse est le même pour les trois.

| Modèle | Formule | Sature à | Coût (fonctions transcendantes) |
|---|---|---|---|
| linéaire | `μ C_S F_z α` | jamais | 0 |
| tanh | `μ F_z tanh(C_S α)` | `μ F_z`, sans redescendre | 1 (`tanh`) |
| Pacejka simplifié | `μ F_z sin(C atan(Bα − E(Bα − atan Bα)))`, `B = C_S / C` | pic `μ F_z`, puis redescend vers `μ F_z sin(Cπ/2)` | 3 (`sin`, 2 × `atan`) |

**Pourquoi `B = C_S / C`.** La pente de Pacejka à l'origine vaut `B·C·D` (dérive la formule en α = 0 : `sin(u) ≈ u`, `atan(u) ≈ u`). Avec `D = μ F_z`, imposer `BCD = μ C_S F_z` donne `B = C_S / C`.

Remarque de cohérence avec la théorie : §6.4 écrit `tanh(C_S α / μ)` et `B = C_S/(μC)`, ce qui donne une pente `C_S F_z` sans le facteur μ. Le modèle linéaire CommonRoad, lui, a la pente `μ C_S F_z`. Ce document garde `μ C_S F_z` pour les trois modèles, pour qu'ils coïncident à petit angle. Avec `μ = 1`, les deux écritures sont identiques ; la différence n'apparaît que si tu changes μ (expérience H4).

**Les coefficients de Pacejka retenus : `C = 1,5`, `E = 0`.** Honnêtement : ils ne sont **pas identifiés** sur une vraie voiture. `C = 1,5` est au milieu de la plage typique 1,2 à 1,9 donnée par la théorie §6.4 ; `E = 0` est le choix neutre. Avec `C_S = 4`, cela donne `B = 2,67`, un pic à `tan(π/2C)/B = 0,650` rad (37,2°) et une force qui redescend à 0,907 `μ F_z` à 90°. Valeurs calculées pour le pneu avant (`F_z` = 18,73 N) :

| α (deg) | linéaire (N) | tanh (N) | Pacejka (N) |
|---|---|---|---|
| 1,1 | 1,50 | 1,50 | 1,50 |
| 5,7 | 7,49 | 7,12 | 7,14 |
| 14,3 | 18,73 | 14,26 | 14,46 |
| 22,9 | 29,97 | 17,26 | 17,63 |
| 37,2 | 48,69 | 18,52 | 18,73 (pic) |
| 57,3 | 74,91 | 18,72 | 18,16 |
| 85,9 | 112,37 | 18,73 | 17,12 |

Conséquence, à garder en tête pour la partie H : **avec ces coefficients, tanh et Pacejka diffèrent de moins de 3 % jusqu'à 57°.** La branche descendante de Pacejka, qui est censée rendre les glissades « punitives » (théorie §6.4), n'agit qu'au-delà de 37°, et la voiture n'y va presque jamais. Un `C` plus grand (1,9) avance le pic à 30° ; l'expérience correspondante est dans les solutions (S-H2).

**Q-C3.** Pourquoi est-ce important que le modèle de pneu soit une fonction **impaire** (`F(−α) = −F(α)`) ? Quel test le vérifie ?

**Q-C4.** En CUDA, quel modèle coûte le moins cher ? Le choix du modèle est lu dans le YAML : est-ce une « branche dépendant de l'état » au sens de l'étape 1 ?

### 4.5 Squelettes

```python
def axle_loads(vehicle) -> tuple[float, float]:
    """Static vertical loads (N) on the front and rear axles, no load transfer."""

def tire_force(alpha, fz, c_s, vehicle):
    """Lateral force (N) for a slip angle alpha (rad). Same slope mu*c_s*fz at 0 for all models."""
    if vehicle.tire_model == "linear":     # choix statique, lu dans la config
        ...

def slip_angles(state, delta, vehicle):
    """alpha_f, alpha_r (rad), > 0 pushes towards +y."""
```

### 4.6 Tests de la partie C

Dans `python/tests/test_step2.py` :

1. **Convention** : `μ C_S F_z` vaut 74,9 et 65,6 N/rad à 0,1 près, et `F_zf + F_zr = m g`.
2. **Même pente à l'origine** pour les trois modèles (`α = 1e-4`, tolérance relative 1e-6), avec `parametrize`.
3. **Saturation** (tanh, Pacejka) : sur `α ∈ [−π/2, π/2]`, `max |F| ≤ μ F_z` et `≥ 0,95 μ F_z`, et `F` impaire (`np.allclose(f, -f[::-1])` sur une grille symétrique).
4. **Signe** : en ligne droite à 3 m/s, δ = 0,1 pendant 0,5 s donne `y > 0`, `ψ > 0`, `r > 0` (ce test utilise le pas complet, il passera après la partie E).

✔ **Point de contrôle C.** Les tests 1 à 3 passent. Trace les trois courbes `F(α)` sur `[−90°, 90°]` avec matplotlib : tu dois retrouver le tableau ci-dessus.

---

## 5. Partie D : la basse vitesse et le mélange cinématique/dynamique

La voiture part **arrêtée** (`vx = 0`) à chaque simulation, et chaque rollout qui freine fort repasse près de zéro. Le modèle dynamique s'y comporte mal pour trois raisons de nature différente (théorie §6.5). Il faut les distinguer, parce que chacune a son remède.

### 5.1 Problème 1 : une singularité algébrique

Les angles de dérive divisent par `vx`. À `vx = 0` :

- si `vy = r = 0` (la voiture est arrêtée sur la ligne de départ), on calcule `atan(0/0) = atan(NaN) = NaN` ;
- si `vy ≠ 0`, on calcule `atan(±inf) = ±π/2` : fini, mais absurde (un pneu arrêté qui « glisse à 90° »).

Mesuré sur la solution de référence en retirant le plancher de vitesse : **un seul pas** depuis `[0, 0, 0, 0, 0, 0]` renvoie un état entièrement `NaN`, alors que le coefficient de mélange vaut 0 à cette vitesse. C'est exactement l'état de départ de la simulation.

`atan2(vy, vx)` éviterait la division, mais pas l'absurdité physique (théorie §6.5).

### 5.2 Problème 2 : une raideur numérique

Linéarisons le couple `(vy, r)` à `vx` constant avec le pneu linéaire. Le terme dominant de l'équation de lacet est

$$
\dot r \approx -\frac{l_f^2 C_{\alpha f} + l_r^2 C_{\alpha r}}{I_z\, v_x}\, r = -\frac{95{,}2}{v_x}\, r .
$$

(`l_f² C_αf + l_r² C_αr = 0,0225 × 74,9 + 0,0324 × 65,6 = 3,81`, divisé par `I_z = 0,04`.) C'est une équation `ṙ = −k r` : `r` revient vers 0 avec une constante de temps `1/k = vx / 95,2` secondes. **Plus la voiture est lente, plus ce retour est rapide**, ce qui est contre-intuitif mais logique : à basse vitesse, un petit `r` crée un grand angle de dérive (on divise par `vx`), donc une grande force de rappel.

Valeurs propres du système linéarisé complet `(vy, r)`, calculées avec `np.linalg.eigvals` (script de la solution S-E) :

| vx (m/s) | valeurs propres (s⁻¹) | constante de temps la plus courte |
|---|---|---|
| 0,5 | −80,3 ; −190,4 | 5 ms |
| 1,0 | −40,3 ; −95,0 | 11 ms |
| 1,5 | −27,1 ; −63,1 | 16 ms |
| 3,0 | −14,2 ; −31,0 | 32 ms |
| 5,0 | −9,5 ; −17,6 | 57 ms |
| 8,0 | −8,5 ± 1,5 j | 118 ms (oscillant) |

Un système est dit **raide** quand une de ses constantes de temps est plus courte que le pas d'intégration. À `dt = 20 ms`, c'est le cas sous environ 1,5 m/s. La partie E montre ce que fait Euler dans ce cas.

### 5.3 Problème 3 : une invalidité physique

Le modèle « force proportionnelle à l'angle de dérive » décrit un pneu **qui roule** en régime établi. Un pneu presque arrêté se comporte comme une contrainte : il ne glisse pas latéralement. Même avec un intégrateur parfait, le modèle dynamique est **faux** à très basse vitesse, alors que le modèle cinématique y est exact.

### 5.4 Le remède : mélanger les deux modèles

Puisque le cinématique est juste à basse vitesse et le dynamique à haute vitesse, on passe de l'un à l'autre progressivement sur une bande `[v_bas, v_haut] = [1,0 ; 1,5]` m/s :

$$
\kappa = \operatorname{clip}\!\Big(\frac{v_x - v_\text{bas}}{v_\text{haut} - v_\text{bas}},\ 0,\ 1\Big), \qquad
x_{t+1} = \kappa\, F_\text{dyn}(x_t, u_t) + (1-\kappa)\, F_\text{cin}(x_t, u_t).
$$

```
  κ
  1 ┤                    ┌──────────── dynamique pur
    │                   ╱
    │                  ╱   mélange linéaire
    │                 ╱
  0 ┼────────────────┘
    0               1,0  1,5              vx (m/s)
     cinématique pur
```

Le mélange se fait sur le **pas complet** (`F` est la fonction « état suivant »), pas sur les dérivées : le cinématique impose `vy` et `r` algébriquement (ci-dessous), ce qui n'est pas une dérivée.

`κ` est continu en `vx` : pas de saut quand un rollout traverse la bande, donc pas de discontinuité artificielle dans les coûts. Et il est calculé par `clip`, sans `if` : un thread CUDA à 0,8 m/s et un thread à 6 m/s exécutent les mêmes instructions.

**Pourquoi la bande commence à 1,0 m/s.** Il faut que le modèle dynamique ne soit jamais utilisé **seul** dans une zone où son intégration est instable. Le seuil d'Euler à 20 ms est 0,95 m/s (partie E), celui de RK4 0,68 m/s. Avec `v_bas = 1,0`, les deux sont sous la bande. Le YAML remplace donc le seuil unique `kinematic_blend_speed: 1.5` par deux champs, `blend_speed_low: 1.0` et `blend_speed_high: 1.5`.

### 5.5 Le modèle cinématique écrit dans l'état à 6 composantes

Pour mélanger, les deux modèles doivent produire le même type d'état. Il faut donc un modèle cinématique « à 6 composantes ». Le roulement sans glissement impose `vy` et `r` en fonction de `vx` et δ : c'est la **variété cinématique**.

Dérivation, en partant du modèle cinématique au CdG de l'étape 1 :

- la vitesse du CdG fait l'angle `β = atan(l_r tan δ / L)` avec l'axe du véhicule, donc `vy / vx = tan β`, soit **`vy = (l_r / L) · vx · tan δ`** ;
- `ψ̇ = v cos β · tan δ / L` avec `v cos β = vx` (projection sur l'axe du véhicule), soit **`r = vx · tan δ / L`**.

Le pas cinématique à 6 composantes :

1. positions et cap avancent avec les vitesses **de la variété** (calculées avec le `vx` courant) ;
2. `vx_next = max(vx + a·dt, 0)` ;
3. `vy` et `r` sont **replacés** sur la variété, calculés avec `vx_next` et δ.

L'étape 3 « oublie » volontairement les `vy` et `r` d'entrée : sous `v_bas`, l'état est ramené sur la variété à chaque pas, ce qui supprime toute dynamique raide.

**Q-D1.** Pourquoi replacer `vy` et `r` sur la variété plutôt que les garder tels quels dans le pas cinématique ? Que se passerait-il au moment où un rollout qui dérapait repasse sous 1 m/s ?

### 5.6 Le piège `0 × NaN` et le plancher de vitesse

Le mélange est sans branche : on évalue **toujours** `F_dyn`, même quand `κ = 0`. Si cette évaluation produit un `NaN` (§5.1), alors `0 × NaN = NaN`, et le `NaN` contamine l'état, puis le coût, puis toute la séquence nominale (théorie §7.7).

Remède : dans les dénominateurs du modèle dynamique, utiliser une vitesse bornée,

```python
vx_safe = np.maximum(vx, vehicle.blend_speed_low)
alpha_f = delta - np.arctan((vy + vehicle.lf * r) / vx_safe)
```

**Q-D2.** Pourquoi `max(vx, v_bas)` et pas `max(vx, 1e-6)` ? Ce plancher change-t-il le comportement du modèle là où `κ > 0` ?

### 5.7 La marche arrière

Choix du projet depuis l'étape 1 : la voiture ne recule pas. On le garde, et on l'applique **après** le mélange : `vx_next = max(vx_next, 0)`, ainsi que dans le pas cinématique (pour que `vy` et `r` de la variété soient calculés avec un `vx` positif).

**Q-D3.** Pourquoi le modèle dynamique tel qu'il est écrit ne peut-il pas servir en marche arrière, même avec le plancher ? Pense au signe de `α` quand `vx < 0`. Quel est le prix de ce choix pour un robot réel ?

### 5.8 Squelettes et tests

```python
def step_kinematic6(state, control, dt, vehicle):    # (..., 6) -> (..., 6)
def blend_weight(vx, vehicle):                       # clip, (...,) -> (...,)
def step_dynamic(state, control, dt, vehicle):       # mélange + plancher vx >= 0
    kappa = blend_weight(state[..., 3], vehicle)[..., None]   # (..., 1) pour le broadcasting
    ...
```

Tests (dans `test_step2.py`) :

1. **Sous `v_bas`, c'est exactement le cinématique** : `np.array_equal(step(x), step_kinematic6(x))` à 0,9 m/s.
2. **Continuité** aux deux bords de la bande : à `v ± 1e-7`, états suivants égaux à 1e-5.
3. **Finitude** : 401 états de `vx = 0` à 2 m/s avec `vy`, `r` aléatoires, commandes aléatoires dans les bornes, 50 pas, sous `np.errstate(all="raise")`, pour les 3 pneus × 2 intégrateurs.
4. **Pas de marche arrière** : `vx = 0,01`, `a = −4` donne `vx = vy = r = 0`.

✔ **Point de contrôle D.** Les 4 tests passent. Puis vérifie que le test 3 **échoue** si tu remplaces `vx_safe` par `vx` : un test qui ne peut pas échouer ne protège de rien.

---

## 6. Partie E : l'intégration numérique

### 6.1 Euler sur l'équation la plus simple possible

Prends `ṙ = −k r` (le mode de lacet de §5.2). La solution exacte décroît : `r(t) = r₀ e^{−kt}`. Euler explicite fait

$$
r_{n+1} = r_n + \Delta t\,(-k\, r_n) = (1 - k\Delta t)\, r_n .
$$

À chaque pas, `r` est multiplié par le **facteur d'amplification** `g = 1 − kΔt`. Trois régimes :

| `kΔt` | `g` | comportement |
|---|---|---|
| < 1 | entre 0 et 1 | décroît sans changer de signe (correct qualitativement) |
| entre 1 et 2 | entre −1 et 0 | décroît en **changeant de signe à chaque pas** (faux, mais borné) |
| > 2 | < −1 | **explose** en alternant de signe : ×\|g\| à chaque pas |

Stabilité : `kΔt < 2`, soit avec `k = 95,2 / vx` et `Δt = 0,02` :

$$
v_x > \frac{95{,}2 \times 0{,}02}{2} = 0{,}95 \text{ m/s}.
$$

Et l'alternance de signe commence dès `kΔt > 1`, soit sous 1,9 m/s.

**Mesuré** (modèle dynamique pur, sans mélange, `r₀ = 0,1`, δ = 0), suite des `r` sur les premiers pas :

- Euler à `vx = 1,0` : −0,090, +0,081, −0,073, +0,065… (facteur −0,90 : stable de justesse, oscillant) ;
- Euler à `vx = 1,5` : −0,027, +0,007, −0,002… (oscillant mais vite amorti) ;
- RK4 à `vx = 1,0` : +0,030, +0,009, +0,003… (décroissance monotone, comme la vraie solution).

**Q-E1.** Calcule le seuil de stabilité d'Euler pour `Δt = 40 ms`, puis pour `Δt = 20 ms` avec 2 sous-pas de 10 ms. Compare à la bande de mélange.

### 6.2 Les options

| Option | Principe | Évaluations de la dérivée par pas | Seuil de stabilité théorique |
|---|---|---|---|
| Euler ×1 | `x + Δt f(x)` | 1 | `kΔt < 2` |
| Euler ×n sous-pas | n pas de `Δt/n` | n | divisé par n |
| RK4 ×1 | 4 évaluations pondérées (ci-dessous) | 4 | `kΔt < 2,785` |
| Euler semi-implicite | termes raides évalués au pas suivant | 1 + une résolution | aucun pour le mode traité |

**RK4** (Runge-Kutta d'ordre 4), pour `ẋ = f(x)` à commande constante sur le pas :

```python
k1 = f(x)
k2 = f(x + 0.5 * h * k1)
k3 = f(x + 0.5 * h * k2)
k4 = f(x + h * k3)
x_next = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
```

L'erreur par pas est en `O(h⁵)` contre `O(h²)` pour Euler. Le seuil 2,785 vient de la fonction d'amplification de RK4 sur `ṙ = −kr`, `g(z) = 1 + z + z²/2 + z³/6 + z⁴/24` avec `z = −kΔt` : `|g| ≤ 1` tant que `z ≥ −2,785` (calculé numériquement dans la solution).

**Euler semi-implicite** (théorie §6.5) : pour le lacet, `r_{t+1} = (r_t + Δt τ_t)/(1 + Δt k)`, stable pour tout Δt. Mais `k` est la pente **locale** de la courbe de pneu : avec tanh ou Pacejka, il faudrait dériver la force de pneu à chaque pas, et traiter aussi `vy`, couplé à `r`. C'est faisable, mais c'est du code de plus à écrire deux fois (NumPy et CUDA) pour un gain que les mesures ci-dessous ne justifient pas. Non retenu.

### 6.3 Mesures : précision

Référence : RK4 avec 200 sous-pas par intervalle. Pour chaque cas, 30 pas de 20 ms (un horizon complet, T = 30) depuis la variété cinématique, commande constante. On mesure l'écart de position final (cm) et l'écart maximal sur `r` (rad/s) :

| cas (vx, δ, a) | Euler ×1 | Euler ×2 | Euler ×4 | RK4 ×1 | RK4 ×2 |
|---|---|---|---|---|---|
| (1,2 ; 0,3 ; 0) | 0,37 / 0,161 | 0,17 / 0,050 | 0,08 / 0,022 | 0,01 / 0,007 | 0,00 / 0,000 |
| (2,0 ; 0,3 ; 0) | 2,09 / 0,264 | 1,02 / 0,113 | 0,50 / 0,053 | 0,00 / 0,002 | 0,00 / 0,000 |
| (5,0 ; 0,1 ; 0) | 3,79 / 0,123 | 1,88 / 0,056 | 0,94 / 0,027 | 0,00 / 0,000 | 0,00 / 0,000 |
| (5,0 ; 0,3 ; 0) | 10,91 / 0,350 | 5,18 / 0,164 | 2,53 / 0,079 | 0,00 / 0,000 | 0,00 / 0,000 |
| (7,0 ; 0,4 ; −2) | 22,09 / 0,584 | 7,37 / 0,598 | 4,09 / 0,302 | 0,00 / 0,000 | 0,00 / 0,000 |

À coût égal (4 évaluations), RK4 ×1 est d'un à trois ordres de grandeur plus précis qu'Euler ×4. Euler ×1 se trompe de 22 cm sur 0,6 s dans le cas le plus dur, soit un tiers de la marge latérale `d_max = 0,65` m.

### 6.4 Mesures : stabilité

Modèle dynamique pur (mélange désactivé), `r₀ = 0,1`, 100 pas : plus grande vitesse à laquelle `|r|` ne décroît pas.

| Intégrateur | pneu linéaire | Pacejka |
|---|---|---|
| Euler ×1 | 0,95 m/s | 0,95 m/s |
| Euler ×2 | 0,48 m/s | 0,47 m/s |
| Euler ×4 | 0,24 m/s | 0,23 m/s |
| RK4 ×1 | 0,68 m/s | 0,67 m/s |
| RK4 ×2 | 0,34 m/s | 0,30 m/s |

Les seuils mesurés tombent sur la théorie (0,95 et 0,68 m/s). Tous sont sous `v_bas = 1,0` m/s.

**Piège de mesure.** Pour mesurer le modèle dynamique « pur » sous 1 m/s, il ne suffit pas d'appeler `step_dynamic_only` : le plancher `vx_safe = max(vx, blend_speed_low)` de la partie D fait que, sous 1 m/s, les angles de dérive sont calculés comme à 1 m/s, et tout paraît stable (première version du script de mesure : « instable sous 0,00 m/s » pour tous les intégrateurs). Il faut abaisser la bande dans une copie du véhicule (`blend_speed_low=0.01`), comme le fait `bench/integrators.py`.

**Un résultat que la théorie ne dit pas.** Le mélange lui-même stabilise. Dans la bande, le facteur d'amplification effectif du lacet est `κ · g` (la part cinématique remet `r` sur la variété). Avec une bande [0,5 ; 1,0] et Euler, `max κ|1 − kΔt|` vaut 0,91 : stable, alors qu'Euler seul est instable sous 0,95. Avec une bande [0,2 ; 0,5], il vaut 2,81 (Euler) et 15,0 (RK4) : un `r₀ = 0,01` grandit jusqu'à 1,33 rad/s en 1 s avec Euler (la saturation du pneu Pacejka l'empêche d'aller à l'infini). La règle « `v_bas` au-dessus du seuil de l'intégrateur » est donc suffisante, pas nécessaire ; garde-la quand même, c'est la plus simple à vérifier (`test_integrator_stable_at_blend_low`).

### 6.5 Mesures : coût

Temps d'un pas de `step_dynamic` sur un lot `K = 1024` (NumPy, moyenne sur 300 appels) :

| | cinématique 4 états | Euler ×1 | Euler ×2 | Euler ×4 | RK4 ×1 | RK4 ×2 |
|---|---|---|---|---|---|---|
| ms par pas | 0,029 | 0,122 | 0,189 | 0,325 | 0,335 | 0,615 |

En boucle fermée (médiane d'une itération MPPI, K = 1024, T = 30) : cinématique 3,8 ms, dynamique Euler 7,3 ms, dynamique RK4 14,9 ms.

### 6.6 La décision : RK4, un sous-pas

- **Précision** : RK4 ×1 est exact au centimètre près sur l'horizon, Euler ×1 se trompe de 4 à 22 cm à haute vitesse.
- **Stabilité** : 0,68 m/s, avec de la marge sous `v_bas`.
- **Boucle fermée** : l'expérience H5 montre qu'Euler ×1 fait **le même tour** (9,06 s contre 9,02 s, 0 sortie, et 5/5 tours propres avec des rollouts Euler sur un véhicule simulé RK4). La différence n'est donc pas visible sur le temps au tour : le contrôleur corrige à 50 Hz les erreurs de prédiction de quelques centimètres. Ce n'est pas un argument pour Euler, c'est un argument pour ne pas croire que la boucle fermée valide le modèle (Q-E3).
- **Coût CUDA** (étape 3) : 4 évaluations de la dérivée par pas au lieu d'une. Une évaluation coûte, avec Pacejka, une douzaine de fonctions transcendantes (2 `atan` pour les dérives, 3 par pneu, `sin`/`cos` de ψ et de δ). C'est **non mesuré** à ce stade. Deux remarques pour l'étape 4 : `sin δ`, `cos δ` et `tan δ` sont constants sur le pas et peuvent être calculés une fois hors des 4 étages ; et l'intégrateur reste un champ du YAML (`integrator: euler`), donc « RK4 → Euler » pourra être une ligne mesurée du tableau d'optimisation, avec son effet sur la qualité de contrôle.

On garde le **même intégrateur pour le véhicule simulé et pour les rollouts** (modèle parfait, comme à l'étape 1). Le véhicule simulé pourrait utiliser un intégrateur plus fin que les rollouts ; c'est une expérience (H5), pas la référence.

**Q-E2.** Pourquoi RK4 ×1 plutôt qu'Euler ×4, qui coûte autant ?

**Q-E3.** En H5, Euler et RK4 donnent le même tour. Pourquoi est-ce quand même important d'avoir une intégration précise dans la **référence** NumPy ? Donne deux raisons.

**Q-E4.** Dans la boucle de sous-pas, le nombre de sous-pas est lu dans la config. En CUDA, est-ce une boucle qui pose problème ?

### 6.7 Squelette

```python
def step_dynamic_only(state, control, dt, vehicle):
    """`substeps` RK4 or Euler sub-steps of the pure dynamic model."""
    a, delta = control[..., 0], control[..., 1]
    h = dt / vehicle.substeps
    f = lambda s: dynamic_derivative(s, a, delta, vehicle)
    for _ in range(vehicle.substeps):
        ...
```

✔ **Point de contrôle E.** Écris le petit script de mesure de stabilité (solution S-E) et retrouve 0,95 m/s pour Euler et environ 0,68 m/s pour RK4. Tous les tests de `test_step2.py` écrits jusqu'ici passent.

---

## 7. Partie F : brancher le modèle dans le projet

### 7.1 Le YAML

Dans `config/mppi.yaml` :

```yaml
model: dynamic            # était kinematic
state_dim: 6              # était 4

mppi:
  lambda: 3.0             # était 0.3, re-réglé en partie G

vehicle:
  mu: 1.0                 # sert maintenant aux pneus
  cornering_stiffness_front: 4.0   # C_S (1/rad) : F_y = mu * C_S * F_z * alpha, convention CommonRoad
  cornering_stiffness_rear: 4.2
  tire_model: pacejka     # linear | tanh | pacejka
  pacejka_c: 1.5          # forme ; B = C_S / C (pente à l'origine identique au linéaire)
  pacejka_e: 0.0
  blend_speed_low: 1.0    # m/s : cinématique pur en dessous (remplace kinematic_blend_speed)
  blend_speed_high: 1.5   # m/s : dynamique pur au-dessus
  integrator: rk4         # euler | rk4, partie dynamique seulement
  substeps: 1             # sous-pas par intervalle dt

cost:
  w_adhesion: 10.0        # utilisé seulement par le modèle cinématique
  v_ref: 7.0              # était 3.0, voir partie G
```

`kinematic_blend_speed` **disparaît** (remplacé par les deux bornes de la bande).

**Pour l'étape 3.** `cuda/include/mppi/config.hpp` n'est encore qu'un en-tête vide. Ajoutes-y en commentaire la liste des champs que le parseur C++ devra lire, et reporte-la dans le journal : `vehicle.mass`, `vehicle.izz`, `vehicle.cornering_stiffness_front`, `vehicle.cornering_stiffness_rear`, `vehicle.tire_model`, `vehicle.pacejka_c`, `vehicle.pacejka_e`, `vehicle.blend_speed_low`, `vehicle.blend_speed_high`, `vehicle.integrator`, `vehicle.substeps` ; supprimé : `vehicle.kinematic_blend_speed` ; modifiés : `model`, `state_dim`, `mppi.lambda`, `cost.v_ref`.

### 7.2 `config.py`

- `Vehicle` reçoit tous les champs de la section `vehicle` du YAML. Puisque les clés YAML sont exactement les noms des champs, `Vehicle(**v)` suffit (comme `CostParams(**raw["cost"])` à l'étape 1). Un champ YAML en trop ou manquant lève alors un `TypeError` explicite : c'est voulu.
- Ajoute deux constantes à côté de `STATE_DIM` : `TIRE_MODELS = ("linear", "tanh", "pacejka")` et `INTEGRATORS = ("euler", "rk4")`.
- Validation, `ValueError` si :
  - `tire_model` ou `integrator` est inconnu ;
  - on n'a pas `0 < blend_speed_low < blend_speed_high` (sinon division par zéro dans `κ`) ;
  - `substeps` n'est pas un entier ≥ 1.

**Q-F1.** Une faute de frappe `tire_model: pacjeka` : que se passerait-il sans la validation, compte tenu de la façon dont `tire_force` est écrite (`if linear … if tanh … sinon Pacejka`) ?

### 7.3 `dynamics.py` : l'interface

Le contrôleur appelle `step(state, control, dt, vehicle)`. On garde cette fonction et on choisit le modèle selon la taille de l'état :

```python
def step(state, control, dt, vehicle):
    """Model chosen by the state size: kinematic (4) or dynamic (6)."""
    if state.shape[-1] == STATE_DIM["dynamic"]:
        return step_dynamic(state, control, dt, vehicle)
    return step_kinematic(state, control, dt, vehicle)
```

`step_kinematic` est l'ancien `step` de l'étape 1, renommé, inchangé. Pas de `6` littéral : `STATE_DIM` vient de `config.py`.

**Q-F2.** Ce `if` est-il une « branche qui dépend de l'état » au sens de l'étape 1 ? Quel est son équivalent en CUDA ?

### 7.4 `cost.py`

- **Le terme de vitesse** lit `state[..., 3]`, qui est `v` en cinématique et `vx` en dynamique : même indice, rien à changer. On pourrait prendre la norme `hypot(vx, vy)` ; en dérapage à 5 m/s avec β = 16°, elle vaut 5,2 m/s au lieu de 5,0. L'écart est faible devant `speed_scale = 1` et `vx` garde le coût identique entre les deux modèles. On garde `vx`.
- **Le terme d'adhérence** n'est calculé que si `cfg.model == "kinematic"`. Le modèle dynamique sature ses pneus lui-même : garder ce terme pénaliserait des virages que le modèle sait déjà rendre impossibles, et avec une formule fausse (l'accélération latérale cinématique `v² tan δ / L`, pas celle des pneus). Le champ `w_adhesion` reste dans le YAML pour la comparaison de la partie G.
- **Une pénalité de dérive ?** On pourrait ajouter `w_slip · (β/β₀)²`. Elle limiterait la dérive, ce qui est un bon réflexe sur un vrai robot (c'est un remède de la théorie §7.9 : éviter les zones où le modèle est peu fiable), mais le critère de sortie demande justement un dérapage visible, et chaque terme ajouté est un champ YAML et une ligne CUDA de plus. **Non ajoutée, non mesurée.** Note-la dans le journal comme piste pour quand le modèle sera confronté à des données réelles.

```python
cost = lateral + offtrack + speed + effort
if cfg.model == "kinematic":    # choix statique (config), pas une branche sur l'état
    cost = cost + adhesion
```

**Q-F3.** Pourquoi `v_ref = 7` rend-il le terme d'adhérence encore plus gênant pour le contrôleur cinématique qu'à `v_ref = 5` (journal de l'étape 1) ?

### 7.5 `controller.py` : rien

Relis `rollout_costs` : `X = np.empty((K, T + 1, cfg.state_dim))`, `X[:, 0] = x0`, `step(X[:, t], V[:, t], m.dt, cfg.vehicle)`, `stage_cost`, `terminal_cost`. Aucune de ces lignes ne connaît le modèle. Le garde-fou `S[~np.isfinite(S)] = S_MAX` reste la dernière ligne de défense contre un `NaN` qui aurait échappé à la partie D.

Si tu as dû modifier `controller.py`, cherche pourquoi : c'est le signe qu'une hypothèse sur la dimension de l'état s'est glissée quelque part.

### 7.6 `run_sim.py` : le véhicule simulé est toujours dynamique

Les changements, dans l'ordre :

1. **L'état initial** a 6 composantes : `[x_c[0], y_c[0], θ[0], 0, 0, 0]`.
2. **Le véhicule simulé** est toujours `step_dynamic`, quel que soit `cfg.model` (protocole 0.3). `simulate(cfg, track, plant=None)` prend en option les paramètres du véhicule simulé (`Vehicle`) pour simuler une erreur de modèle en partie H ; par défaut `plant = cfg.vehicle`.
3. **L'observation.** Un contrôleur cinématique attend `[x, y, ψ, v]`. On lui donne `v = hypot(vx, vy)`, la norme de la vitesse :

   ```python
   def observe(x, cfg):
       if cfg.state_dim == STATE_DIM["dynamic"]:
           return x
       return np.array([x[0], x[1], x[2], np.hypot(x[3], x[4])])
   ```

4. **La boucle s'arrête** si l'état devient non fini (`np.all(np.isfinite(x))`), pour ne pas tracer 3000 pas de `NaN`.
5. **Les grandeurs de châssis** : une fonction `chassis(log, cfg)` recalcule, à partir des états et commandes journalisés, `α_f`, `α_r`, `β = atan2(vy, max(vx, 1e-3))`, l'accélération latérale réelle `(F_yf cos δ + F_yr)/m` et la norme de la vitesse.
6. **Le bilan console** ajoute la dérive maximale `|β|` et les `|α|` maximaux, **calculés seulement au-dessus de `blend_speed_high`** : en dessous, la voiture est sur la variété cinématique et `α` n'a pas de sens physique (au démarrage, `α_f ≈ δ` atteint 27° sans aucun glissement réel, ce qui pollue le maximum).
7. **Les figures** :
   - `step2_<model>_trajectory.png` : trajectoire colorée par `β` (palette divergente `coolwarm`, centrée sur 0) au lieu de la vitesse, et le faisceau ;
   - `step2_<model>_series.png` : six courbes, ESS, `d` (avec ±`d_max`), `vx` et `vy` (avec `v_ref`), `r`, `α_f`, `α_r` et `β` en degrés (avec les droites ±`1/C_S` = ±14,3°, fin de la zone linéaire), δ.
8. **Les triplets** `(state, control, next_state)` du véhicule simulé vont dans `results/trajectories/step2_<model>.npz`. Ce sont maintenant des états à 6 composantes : c'est le dataset du jalon 2.

**Q-F4.** Pourquoi observer `hypot(vx, vy)` plutôt que `vx` pour le contrôleur cinématique ?

### 7.7 Les tests

**`test_step1.py`.** Le YAML est passé en dynamique, donc les tests de l'étape 1 (états à 4 composantes, `X.shape == (64, T+1, 4)`, terme d'adhérence) doivent forcer le cinématique. Une seule ligne change :

```python
CFG = dataclasses.replace(load_config(...), model="kinematic", state_dim=STATE_DIM["kinematic"])
```

Ces 28 tests deviennent des tests de non-régression du modèle cinématique.

**`test_step2.py`**, en plus des tests des parties C et D :

| Test | Ce qu'il vérifie | Tolérance |
|---|---|---|
| `test_config_is_dynamic` | le YAML est cohérent | exact |
| `test_straight_line_stays_straight` | δ = 0 : `y`, `ψ`, `vy`, `r` restent nuls, `x = 3` m après 1 s | `allclose` |
| `test_turns_left_with_positive_steer` | convention de signe | signes |
| `test_left_right_symmetry` | miroir `(y, ψ, vy, r, δ) → −` : résultat miroir, 3 pneus | 1e-12 |
| `test_steady_state_cornering` | `F_y/m = vx·r`, moment nul, `r` = formule de sous-virage | 1e-6, 1e-6, 1 % |
| `test_cross_check_with_kinematic_model` | **test croisé permanent** (ci-dessous) | 10 cm, 0,02 rad |
| `test_integrator_stable_at_blend_low` | garde-fou de config : stable à `v_bas` | `r` décroît |
| `test_batch_matches_single` | lot = boucle | `allclose` |
| `test_rollout_costs_6_states` | `X` de forme `(64, T+1, 6)`, coûts finis et déterministes | exact |
| `test_closed_loop_short` | 1 s depuis l'arrêt : reste en piste, `vx > 2` m/s | — |

**Le test croisé, en détail.** À 2 m/s (au-dessus de la bande, donc dynamique pur) et δ = 0,05 rad, on fait rouler 2 s le modèle cinématique 4 états (`v = vx / cos β`, même vitesse longitudinale) et le modèle complet à 6 états (départ sur la variété). Écart mesuré : **6,8 cm de position et 0,011 rad de cap après 4 m**, identique pour les trois pneus (on reste dans la zone linéaire). D'où vient cet écart ?

- le sous-virage : `K_us vx² / (g L) = 0,0119 × 4 / (9,81 × 0,33) = 1,5 %` de moins en vitesse de lacet, soit 0,009 rad sur un cap de 0,61 rad ;
- le temps d'établissement de la dérive : le modèle dynamique part avec des forces nulles (sur la variété, `α_f = α_r = 0`) et met quelques dizaines de ms à établir ses angles ;
- la traînée de braquage (Q-B1) : le modèle dynamique perd 0,017 m/s en 2 s, le cinématique rien.

Tolérances : 10 cm et 0,02 rad, soit 1,5 à 2 fois les valeurs mesurées, et justifiées par le calcul de sous-virage. Pour mémoire, le même écart vaut 3,1 cm à 1,6 m/s, 27 cm à 3 m/s et 1,38 m à 5 m/s (δ = 0,05) ; 16 cm à 2 m/s avec δ = 0,1 et 42 cm avec δ = 0,2. Le test n'a de sens qu'à basse vitesse et petit braquage, comme le dit le plan.

**Q-F5.** Pourquoi ce test ne doit-il pas être fait sous 1 m/s ? (Indice : que vaut alors `step` ?)

✔ **Point de contrôle F.** `pixi run test` : 55 tests passent (28 de l'étape 1, 27 de l'étape 2, en comptant chaque cas paramétré). `pixi run sim` boucle un tour avec la config de la section 7.1.

---

## 8. Partie G : premier tour rapide et figure comparative

### 8.1 Premier tour : à 3 m/s d'abord

Ne passe pas tout de suite à 7 m/s. Lance d'abord `pixi run sim` avec `v_ref = 3` et λ = 0,3 (la config de l'étape 1 avec `model: dynamic`). Résultat de référence :

```
controller model dynamic, plant dynamic (pacejka tires)
lap done in 15.94 s, progress 48.3/48.3 m
offtrack steps 0, max |d| 0.115 m (limit 0.65)
ESS median 592 p5 149 p95 675 / K=1024
iteration time median 15.4 ms
```

C'est presque le tour de l'étape 1 (16,12 s) : à 3 m/s, les pneus restent dans leur zone linéaire (au-dessus de 1,5 m/s, |α_f| ≤ 7,5°, |α_r| ≤ 5,9°, |β| ≤ 1,7°) et le modèle dynamique se comporte comme le cinématique. C'est le test croisé, en boucle fermée. Si ce tour ne passe pas, le problème est dans le code, pas dans le réglage : retourne aux tests.

### 8.2 Monter en vitesse : re-régler λ

Passe `v_ref` à 7. L'échelle du coût change (écarts de vitesse plus grands, rollouts qui sortent dans les virages), donc **λ doit être re-réglé en mesurant l'ESS**, comme l'a appris la partie G de l'étape 1. Mesuré, à `v_ref = 7`, modèle dynamique, une graine :

| λ | tour | t (s) | sorties | max \|d\| | ESS méd | ESS p5 | gigue δ |
|---|---|---|---|---|---|---|---|
| 0,3 | oui | 8,54 | 0 | 0,63 | 31 | 1 | 0,075 |
| 1 | oui | 8,56 | 0 | 0,63 | 208 | 1 | 0,050 |
| **3** | **oui** | **9,02** | **0** | **0,54** | **484** | **13** | **0,023** |
| 10 | oui | 10,52 | 0 | 0,41 | 784 | 343 | 0,013 |

Règle de la théorie §4.6 : ESS minimale (p5) entre 1 % et 10 % de K, soit 10 à 100 à K = 1024. λ = 3 est le seul qui y tombe. λ = 0,3 et 1 sont plus rapides mais frôlent la limite (`max |d|` = 0,63 pour `d_max` = 0,65) avec une ESS p5 de 1 : ils passent par chance (vérifie avec d'autres graines en partie H). λ = 10 est sûr mais mou : 1,5 s de plus.

Note la cohérence avec le journal de l'étape 1 : à `v_ref = 5` en cinématique, λ ≈ 2 à 3 rétablissait l'ESS. Le même ordre de grandeur revient ici.

### 8.3 Le tour de référence

`pixi run sim` avec la config de 7.1 (`v_ref = 7`, λ = 3) :

```
controller model dynamic, plant dynamic (pacejka tires)
lap done in 9.02 s, progress 48.4/48.3 m
offtrack steps 0, max |d| 0.536 m (limit 0.65)
v mean 5.46 max 6.72 m/s, max a_lat 9.0 m/s2 (mu g = 9.8)
above 1.5 m/s: max |beta| 16.4 deg, max |alpha_f| 22.8 deg, max |alpha_r| 19.5 deg
ESS median 484 p5 13 p95 759 / K=1024
iteration time median 14.9 ms
```

Comment lire ce bilan :

- **9,02 s** contre 16,12 s à l'étape 1 : 44 % plus rapide, sans sortie.
- **`max a_lat 9,0`** : la voiture utilise 92 % de l'adhérence disponible (μg = 9,81). Elle ne la dépasse pas, et pour cause : le modèle ne le permet pas.
- **Le dérapage** : les pneus arrière dépassent la zone linéaire (|α_r| > 14,3°) pendant 2,6 s sur 9 s, et la dérive du véhicule dépasse 10° pendant 4,0 s. Le maximum, β = −16,4°, est atteint dans le grand virage à gauche situé sur le côté droit de la piste (vers x = 14,3, y = 4,2), à vx = 5,1 m/s et r = 1,62 rad/s, avec un braquage de seulement 0,082 rad : le modèle cinématique demanderait `atan(r L / vx)` = 0,105 rad pour la même rotation. La voiture tourne avec **moins** de braquage que la géométrie ne le dit, parce que l'arrière glisse et la fait pivoter. C'est le dérapage visible du critère de sortie, et le contrôleur le tient (0 sortie).
- **Le démarrage** : sous 1,5 m/s (atteint en 0,48 s), l'ESS médiane est de 15 et le braquage s'agite (visible sur la courbe δ). Comme à l'étape 1 : à l'arrêt, les coûts des rollouts sont dominés par le terme de vitesse et très dispersés relativement à leur faible progression. C'est sans conséquence sur la piste.
- **14,9 ms par itération**, 4 fois le cinématique (3,8 ms) : c'est le prix du RK4 en NumPy (partie E).

Regarde `step2_dynamic_series.png` : tu dois voir `vy` et `β` prendre de grandes valeurs dans les virages, `α_f` et `α_r` sortir des droites ±14,3°, et `r` jusqu'à 1,7 rad/s (−2 rad/s dans l'enchaînement à droite).

**Q-G1.** Sur la courbe, `α_f` et `α_r` restent très proches l'un de l'autre pendant tout le tour. Pourquoi ? (Indice : partie C, sous-virage.)

### 8.4 La figure comparative

`bench/compare_models.py` (code complet en solution) :

1. charge la config ;
2. construit trois variantes qui ne diffèrent **que** par le modèle des rollouts : `dynamic` ; `kinematic` (avec `w_adhesion = 0`) ; `kinematic + adhesion term` (le contrôleur de l'étape 1) ;
3. lance chaque variante sur 5 graines, en parallèle (`ProcessPoolExecutor`), avec le même véhicule simulé dynamique ;
4. imprime un tableau (tours propres, sorties par graine, temps, ESS) ;
5. trace trois pistes côte à côte (graine 0) : trajectoire colorée par la vitesse, pas hors piste en rouge.

Résultat mesuré, `v_ref = 7`, λ = 3, K = 1024, T = 30, graines 0 à 4 :

| contrôleur | tours propres | pas hors piste par graine | temps au tour (s) | ESS médiane |
|---|---|---|---|---|
| dynamique | **5/5** | [0, 0, 0, 0, 0] | 8,68 à 9,18 | 407 à 592 |
| cinématique | **0/5** | [29, 16, 54, 26, 41] | 10,24 à 10,84 | 250 à 326 |
| cinématique + terme d'adhérence | **2/5** | [50, 0, 0, 93, 15] | 10,16 à 11,12 | 6 à 21 |

Où et comment ils échouent (graine 0, sur la figure) :

- **cinématique pur** : sort dans l'enchaînement droite-gauche du haut de la piste (vers x = 5 à 6, y = 7 à 10). Il a planifié un changement de direction que ses pneus ne pouvaient pas fournir ;
- **cinématique + adhérence** : sort à l’**extérieur** du grand virage situé à droite de la piste (x ≈ 14,7, y ≈ 4,5 à 7) et de celui situé à gauche (x ≈ −5,6). La voiture « part tout droit », exactement le scénario de la théorie §6.2.

**Lecture honnête.** Le cinématique pur a une ESS correcte (250 à 326) : son λ n'est pas en cause, il échoue **parce que son modèle est faux**, c'est la démonstration la plus propre du désaccord de modèle. Le cinématique avec rustine, lui, a une ESS effondrée (6 à 21) : le terme d'adhérence étale les coûts à haute vitesse, comme à l'étape 1. Une partie de son échec vient donc du réglage de λ.

Et voici ce que la figure **ne** montre **pas**, mesuré en partie H (H9) : à λ = 10, le cinématique avec rustine passe 5 graines sur 5 à `v_ref = 7` (10,86 à 11,08 s), et le cinématique pur aussi (10,62 à 10,94 s). Ils sont alors **2 s plus lents** que le dynamique : un λ grand rend le contrôleur timide (ESS proche de K, commande moyennée), et la rustine bride le braquage avec une limite trop prudente. À `v_ref = 8` et λ = 10, ils échouent de nouveau (3 graines sur 5 pour le pur, 1 sur 5 pour la rustine), alors que le dynamique fait 5/5. La conclusion défendable n'est donc pas « le cinématique ne peut pas faire le tour », mais :

> **À réglage égal, MPPI avec rollouts cinématiques sort de la piste là où MPPI avec rollouts dynamiques passe ; on peut le sauver en le rendant timide, au prix d'environ 20 % de temps au tour.**

Écris la légende de la figure dans ce sens, avec le tableau. Une figure qui cache la variante λ = 10 serait attaquable en entretien.

**Q-G2.** Le cinématique pur et le dynamique ont des ESS comparables à λ = 3. Pourquoi est-ce important pour l'interprétation de la figure ?

**Q-G3.** Pourquoi un λ plus grand sauve-t-il le contrôleur cinématique ? Relis la théorie §4.1 (« trop grand »).

✔ **Point de contrôle G.** `pixi run compare` produit la figure et un tableau qualitativement identique (dynamique 5/5, cinématique pur 0 ou 1/5).

---

## 9. Partie H : réglage et expériences

Même format qu'à l'étape 1 (`docs/step1_predictions.md`), que tu as trouvé utile : **écris ta prédiction AVANT de lancer**, puis compare. Une prédiction se limite à un sens (↑, ↓, =, « sort », « ne sort pas ») et une ligne de « pourquoi ». C'est le raisonnement qui compte, pas le chiffre. Crée `docs/step2_predictions.md` en recopiant le tableau de 9.2, remplis les colonnes **Prédiction** et **Pourquoi**, et seulement ensuite lance `pixi run sweep Hx`. Les mesures de référence sont dans les solutions (S-H) : ne les lis qu'après.

### 9.1 L'outil

`bench/sweep.py` est étendu (code complet en solution) :

- chaque variante est maintenant un triplet `(nom, modification de la config du contrôleur, modification du véhicule simulé)`. Cela permet de distinguer « je change le modèle des rollouts » de « je change la réalité » ;
- `with_vehicle(**kw)` modifie les paramètres véhicule **du contrôleur** ; `both(**kw)` modifie les deux (modèle parfait) ;
- la fonction `run` reçoit `(cfg, plant)` et passe `plant` à `simulate` ;
- les métriques ajoutent le nombre de pas hors piste (`off`) et la dérive maximale `β` au-dessus de 1,5 m/s.

**Attention à la colonne « tour ».** Elle vaut « oui » dès que la progression atteint la longueur du tour, même si la voiture est sortie en route. Le critère qui compte est `off = 0`.

**Toujours plusieurs graines pour conclure.** Une seule graine suffit pour une tendance nette (G7 de l'étape 1 l'a montré à 3 m/s), mais à 7 m/s, près de la limite, le hasard du bruit décide parfois de la sortie. Pour toute expérience dont le résultat est « sort / ne sort pas », relance avec `with_mppi(seed=s)` pour s = 0 à 4.

### 9.2 Le tableau des expériences

Référence : config de 7.1 (`v_ref = 7`, λ = 3, σ = [0,5 ; 0,1], T = 30, K = 1024, Pacejka C = 1,5, RK4, graine 0). Mesure de référence : tour en 9,02 s, 0 sortie, `max |d|` 0,54 m, ESS 484 / 13, gigue 0,023, β max 16,4°.

| # | Expérience | Question | Prédiction | Pourquoi |
|---|---|---|---|---|
| H1 | λ ∈ {1, 10} | Temps au tour, ESS, gigue, `max \|d\|` ? Lequel choisirais-tu sur un vrai robot ? | | |
| H2 | pneus linéaires, puis tanh, **dans le modèle et dans la réalité** | La voiture va-t-elle plus vite avec des pneus linéaires ? Que devient `a_lat` max ? tanh ≠ Pacejka ? | | |
| H3 | rollouts avec pneus linéaires (puis tanh), réalité Pacejka | Le contrôleur sort-il de la piste ? (5 graines) | | |
| H4 | réalité à μ = 0,8, contrôleur à μ = 1 ; puis les deux à 0,8 | Effet d'une surestimation de 25 % de l'adhérence ? (5 graines) | | |
| H5 | Euler ×1 dans les deux ; puis rollouts Euler, réalité RK4 | Le tour change-t-il ? Le temps de calcul ? | | |
| H6 | T = 15 ; T = 50 ; T = 50 avec λ = 6 | Horizon court à 7 m/s ? Horizon long : ESS ? | | |
| H7 | σ_δ = 0,05 ; σ_δ = 0,2 | Gigue, ESS, temps au tour ? | | |
| H8 | `v_ref = 9` ; puis λ = 10 ; puis λ = 10 et T = 50 | Aller plus vite en demandant plus ? | | |
| H9 | rollouts cinématiques ; + adhérence ; + adhérence et λ = 10 | Retrouve et prolonge la figure de la partie G | | |

### 9.3 Les questions à se poser pendant les expériences

- **H1** : relie l'ESS et la gigue à la théorie §4.1 et §4.2 (`σ/√ESS`). Calcule la gigue attendue à ESS = 13 et à ESS = 343.
- **H2** : un modèle linéaire autorise-t-il des accélérations latérales supérieures à μg ? Si oui, la voiture simulée sort-elle pour autant ? (Rappel : ici, la réalité **et** le modèle sont linéaires.)
- **H3** : c'est le cas « modèle de pneu sans saturation dans le contrôleur ». Quel est le mode d'échec de la théorie §7.9 qui correspond ?
- **H4** : c'est la pluie. Qu'est-ce que la théorie §6.7 dit de l'erreur sur μ ?
- **H5** : si la boucle fermée ne voit pas la différence, est-ce que l'intégrateur n'a pas d'importance ? (Q-E3.)
- **H6** : calcule la distance vue à 7 m/s avec T = 15 et la distance de freinage de 7 à 4,6 m/s (vitesse max du virage serré) à 4 m/s² (théorie §4.3). Pour T = 50 : pourquoi l'ESS baisse-t-elle, et pourquoi remonter λ ?
- **H7** : un σ_δ plus grand explore plus. Pourquoi pourrait-il **ralentir** la voiture ?
- **H8** : pourquoi demander 9 m/s ne donne-t-il pas un tour plus rapide à λ = 10 ?
- **H9** : pourquoi la rustine d'adhérence est-elle **plus** nuisible à `v_ref = 7` qu'à `v_ref = 3` ?

✔ **Point de contrôle H.** Le tableau de prédictions est rempli avant les mesures, et chaque écart prédiction/mesure a une ligne d'explication.

---

## 10. Dépannage

| Symptôme | Cause probable | Vérification |
|---|---|---|
| L'état devient `NaN` dès le premier pas de la simulation | `0/0` dans les angles de dérive à `vx = 0` (piège `0 × NaN`) | `vx_safe = max(vx, v_bas)` ; `test_finite_at_standstill_and_in_band` sous `errstate(all="raise")` |
| La voiture tourne à droite quand δ > 0, ou part en tête-à-queue en ligne droite | signe d'un angle de dérive ou du moment de lacet | `test_turns_left_with_positive_steer`, §4.1 ; vérifie `α_f = δ − atan(...)` et `+l_f F_yf − l_r F_yr` |
| `r` alterne de signe d'un pas à l'autre à basse vitesse | Euler dans la zone `kΔt > 1` | §6.1 ; RK4, ou bande de mélange plus haute |
| `r` explose à basse vitesse, coûts `S_MAX` au démarrage | Euler instable sous la bande (`v_bas` < 0,95 m/s) | `test_integrator_stable_at_blend_low` |
| Saut du coût ou de l'état à 1,0 ou 1,5 m/s | `κ` mal borné, ou `vy`, `r` cinématiques calculés avec l'ancien `vx` | `test_blend_is_continuous` |
| `ValueError: operands could not be broadcast together ... (1024,) (1024,6)` | `κ` sans axe final | `kappa[..., None]` |
| Les tests de l'étape 1 échouent avec des formes `(…, 6)` | `test_step1.py` lit le YAML dynamique | forcer le cinématique (§7.7) |
| Le contrôleur cinématique plante dans `run_sim` | il reçoit un état à 6 composantes | `observe(x, cfg)` |
| La voiture va beaucoup plus vite que prévu et ne dérape jamais | `tire_model: linear`, ou `μ` trop grand | `a_lat` max > μg dans le bilan = pneu non saturé |
| ESS ≈ 1 en permanence à 7 m/s | λ réglé pour 3 m/s | §8.2, `S - ρ` des 10 % meilleurs |
| ESS ≈ K, la voiture roule à 4,5 m/s au lieu de 7 | λ trop grand (timide) | H1, λ = 10 |
| Sortie à l'extérieur des virages rapides | horizon trop court, ou modèle trop optimiste (μ, pneu linéaire) | H6, H3, H4 |
| `β` max énorme (> 40°) en début de tour | statistique calculée sur la phase de démarrage | ne calculer `β`, `α` qu'au-dessus de `blend_speed_high` |
| `TypeError: Vehicle.__init__() got an unexpected keyword argument` | champ YAML ajouté sans champ dans la dataclass (ou l'inverse) | §7.2 |
| Une itération prend plus de 30 ms | RK4 avec plusieurs sous-pas, ou boucle Python sur K | `substeps: 1` ; seule la boucle sur t et la boucle de sous-pas sont en Python |

---

## 11. Clôturer l'étape

1. `pixi run track && pixi run test && pixi run sim && pixi run compare` passent.
2. Coche le critère de sortie de la [section 0.2](#02-critère-de-sortie).
3. Remplis `docs/step2_predictions.md` (partie H), prédictions d'abord.
4. Écris l'entrée du journal (`docs/journal.md`), au format habituel :
   - **Measured** : le bilan console de 8.3 avec la config utilisée, le tableau de la figure comparative (5 graines), le tableau de la partie H, les seuils de stabilité et la table de précision de la partie E ;
   - **No effect / reverted** : par exemple Euler contre RK4 en boucle fermée (aucune différence mesurable), tanh contre Pacejka (aucune différence mesurable avec C = 1,5) ;
   - la liste des champs YAML ajoutés, supprimés et modifiés (§7.1), à reporter dans `config.hpp` ;
   - les questions que tu t'es posées pendant l'étape, avec leurs réponses, comme à l'étape 1.
5. Coche `Step 2` dans le README.
6. Commit, puis `git tag step-2`.

---
---

# Solutions

Ne lis une solution qu'après avoir écrit ta propre réponse.

## Réponses aux questions

**S-A1.** `v_max = √(μ g R)` : 4,64 m/s pour le virage serré (R = 2,19 m), 2,80 m/s pour le cercle à braquage maximal (R ≈ 0,80 m). Passé à 7 m/s, le virage serré demanderait `49 / 2,19 = 22,4` m/s², soit 2,3 g : le modèle cinématique le prédit sans broncher. À l'étape 1, `max a_lat = 6,3` m/s² restait sous μg = 9,81 : les pneus réels auraient tenu, et le modèle cinématique était une bonne approximation (le tour à 3 m/s avec le modèle dynamique, 15,94 s contre 16,12 s, le confirme en 8.1).

**S-A2.** Trois raisons :
1. **Il mesure la mauvaise accélération.** `v² tan δ / L` est l'accélération latérale que **le modèle cinématique** prédit, pas celle que les pneus fournissent. Près de la limite, les deux divergent (la voiture glisse, tourne moins, `vy` apparaît).
2. **Il ne dit rien de ce qui se passe après la limite.** Un rollout cinématique qui dépasse μg paie une pénalité, mais **sa trajectoire prédite reste sur le cercle**. Le contrôleur ne sait pas que la voiture part à l'extérieur, donc il ne peut pas planifier de rattrapage, ni distinguer une petite glissade récupérable d'une sortie.
3. **Il dégrade le réglage.** Le terme croît très vite avec la vitesse et étale les coûts : à λ = 3 et `v_ref = 7`, l'ESS du contrôleur avec rustine tombe à 6 à 21 (partie G), contre 250 à 326 sans rustine.

**S-A3.** Les pneus avant saturent, la voiture tourne moins que prévu et part vers l'extérieur du virage (sous-virage, « elle part tout droit »). C'est ce que montre la figure comparative : le contrôleur cinématique avec rustine sort à l’extérieur des deux grandes boucles (côtés droit et gauche de la piste), qui sont toutes deux des virages à gauche.

**S-B1.** C'est la **traînée de braquage** : la force latérale de la roue avant braquée a une composante dirigée vers l'arrière, `−F_yf sin δ`. En virage avec `a = 0`, `F_yf` et δ ont le même signe, donc le terme est négatif : la voiture ralentit. Mesuré avec le modèle dynamique à 2 m/s pendant 2 s : perte de 0,017 m/s à δ = 0,05, de 0,23 m/s à δ = 0,2. Le modèle cinématique garde `v` constant ; c'est une des sources d'écart du test croisé.

**S-B2.** `F_yf cos δ + F_yr = m vx r` (les pneus fournissent l'accélération centripète) et `l_f F_yf cos δ − l_r F_yr = 0` (pas de moment, sinon `r` changerait). Le test le vérifie à 1e-6 près, et la vitesse de lacet à 0,05 % près de la formule de sous-virage (0,43974 contre 0,43998 rad/s).

**S-B3.** Dans `ṙ`, l'inertie `I_z` fixe la vitesse à laquelle la rotation s'établit : le moment des pneus produit une **accélération** angulaire, pas une vitesse. Dans le modèle cinématique, `r = vx tan δ / L` saute instantanément quand δ saute. Dans le modèle dynamique, `r` rejoint sa valeur avec les constantes de temps du tableau de §5.2 (32 à 70 ms à 3 m/s). Pour MPPI, cela veut dire qu'un coup de volant n'a pas d'effet immédiat : il faut anticiper, ce qui est une raison de plus pour un horizon suffisant.

**S-C1.** En virage à gauche, l'arrière doit pousser vers la gauche : `F_yr > 0`, donc `α_r > 0`. Comme `α_r = −atan((vy − l_r r)/vx)`, il faut `vy < l_r r`.
- À basse vitesse, sur la variété cinématique, `vy = (l_r/L) vx tan δ > 0` : le CdG se déplace vers l'**intérieur** par rapport au nez (β géométrique positif, jusqu’à 13° à braquage maximal).
- À haute vitesse, l'arrière doit avoir un grand `α_r`, donc `vy = l_r r − vx tan α_r` devient **négatif** : le nez pointe vers l'intérieur, la vitesse vers l'extérieur. C'est l'attitude de dérapage, et c'est bien le signe mesuré (β = −16,4° dans un virage à gauche, 8.3).

**S-C2.** `μ C_S F_z α = μ F_z` donne `α = 1/C_S` : 0,25 rad = 14,3° à l'avant, 0,238 rad = 13,6° à l'arrière. Au-delà, le pneu linéaire produit une force qu'aucun pneu réel ne peut produire. Les droites pointillées à ±14,3° de `step2_dynamic_series.png` marquent cette limite.

**S-C3.** La voiture doit se comporter exactement de la même façon en virage à gauche et à droite. Un modèle de pneu non impair ferait tirer la voiture d'un côté, et le contrôleur apprendrait à compenser un défaut qui n'existe pas. `test_tire_saturates_at_mu_fz` le vérifie directement (`f == -f[::-1]` sur une grille symétrique) et `test_left_right_symmetry` le vérifie sur le pas complet, à 1e-12.

**S-C4.** Linéaire (aucune fonction transcendante), puis tanh (une), puis Pacejka (`sin` et deux `atan`). Le modèle est lu dans le YAML : il est **le même pour tous les threads**. Ce n'est donc pas une branche dépendant de l'état : tous les threads d'un warp prennent la même branche, il n'y a pas de divergence. En CUDA, le plus propre est d'en faire un paramètre de template (`template <TireModel M>`) ou un `if constexpr`, résolu à la compilation.

**S-D1.** La variété décrit l'unique mouvement possible sans glissement à `(vx, δ)` donnés. Garder `vy` et `r` hors de la variété dans le pas cinématique serait incohérent (les positions avancent avec les vitesses de la variété, pas avec ces `vy`, `r`). Et un rollout qui dérapait et repasse sous 1 m/s garderait en mémoire une glissade que le modèle cinématique ne sait pas faire évoluer : quand il remonte au-dessus de 1 m/s, le modèle dynamique recevrait des `vy`, `r` arbitraires. Avec la projection, sous `v_bas`, l'état est entièrement déterminé par `(x, y, ψ, vx)` et δ : plus aucune dynamique raide. Dans la bande, la part `(1 − κ)` ramène progressivement vers la variété, ce qui amortit le mode de lacet (mesure de §6.4).

**S-D2.** Là où `κ > 0`, on a `vx ≥ v_bas`, donc `max(vx, v_bas) = vx` : **le plancher ne change rien à la physique**, il n'agit que là où la contribution dynamique est multipliée par 0. Avec `max(vx, 1e-6)`, l'évaluation dynamique à `vx` minuscule donnerait des dérivées énormes (`k = 95/vx` ≈ 10⁸ s⁻¹) : les étages intermédiaires de RK4 peuvent déborder en `inf` (bien plus tôt en FP32, au-delà de 3,4·10³⁸), et `0 × inf = NaN`. Avec `v_bas`, l'évaluation est celle d'un état régulier à 1 m/s, toujours finie.

Mesuré en retirant le plancher : les 6 cas de `test_finite_at_standstill_and_in_band` échouent, et `test_closed_loop_short` aussi (l'état est `NaN` dès le premier pas depuis l'arrêt).

**S-D3.** Avec `vx < 0`, `atan((vy + l_f r)/vx)` change de signe : la convention « `α > 0` pousse vers la gauche » s'inverse et les forces de pneu poussent du mauvais côté. Il faudrait réécrire les angles de dérive avec `|vx|` et traiter le changement de sens, ce qui ajoute des cas et des branches pour une situation qui n'arrive pas en course. Le prix : la voiture ne peut pas se dégager en reculant après un tête-à-queue ou un contact avec un mur. Pour un robot de navigation (nav2), ce serait inacceptable.

**S-E1.** À `Δt = 40` ms : `vx > 95,2 × 0,04 / 2 = 1,9` m/s, **au-dessus** de la bande de mélange : le modèle dynamique serait utilisé seul entre 1,5 et 1,9 m/s en étant instable. Avec 2 sous-pas de 10 ms : 0,48 m/s (mesuré 0,475).

**S-E2.** À 4 évaluations chacun, RK4 ×1 est de 100 à 1000 fois plus précis qu'Euler ×4 (table de §6.3 : 0,00 cm contre 2,5 à 4,1 cm à haute vitesse). Attention, ce n'est **pas** un argument de stabilité : Euler ×4 est stable jusqu'à 0,24 m/s, RK4 ×1 seulement jusqu'à 0,68 m/s. Les deux sont sous `v_bas`, la stabilité ne départage donc pas ; c'est la précision qui décide.

**S-E3.**
1. **Le véhicule simulé est la vérité du projet.** S'il est mal intégré, la « réalité » contre laquelle on compare les contrôleurs (figure de la partie G, expériences H3 et H4) contient des artefacts numériques de 10 à 20 cm par horizon, et les triplets `(state, control, next_state)` qui serviront à entraîner le modèle appris du jalon 2 seraient faux.
2. **L'étape 4 mesurera « RK4 → Euler » comme une optimisation.** Pour juger son effet sur la qualité, il faut une référence précise. Et la parité NumPy/CUDA demande que la référence soit bien définie, pas qu'elle soit approximative.
En H5, le contrôleur corrige les erreurs à 50 Hz parce que le modèle des rollouts et le véhicule simulé sont presque identiques : la boucle fermée **masque** l'erreur, elle ne la supprime pas.

**S-E4.** Non. Le nombre de sous-pas est le même pour tous les threads : boucle uniforme, sans divergence. Si c'est une constante de compilation (template), le compilateur peut dérouler la boucle.

**S-F1.** `tire_force` tombe dans la dernière branche (Pacejka) pour toute valeur inconnue : la simulation tournerait avec un autre pneu que celui demandé, sans aucun message. La validation au chargement transforme une erreur silencieuse en erreur immédiate (même raisonnement que S-A1 de l'étape 1).

**S-F2.** Non. La forme du tableau est fixée par la config (`cfg.state_dim`), elle est la même pour tous les rollouts et toutes les itérations. En CUDA, c'est le choix du kernel à compiler : un paramètre de template `template <class Model>` ou deux kernels distincts. Aucun thread ne prend une branche différente de son voisin.

**S-F3.** À 7 m/s, `a_lat = v² tan δ / L` dépasse μg dès `tan δ > 9,81 × 0,33 / 49 = 0,066`, soit δ ≈ 0,066 rad : moins que σ_δ = 0,1. Presque tous les échantillons bruités sont pénalisés, et de façon quadratique, donc la dispersion des coûts explose et l'ESS s'effondre (6 à 21 à λ = 3). À 5 m/s, le seuil était déjà à 0,13 rad (journal de l'étape 1).

**S-F4.** Dans le modèle cinématique, `v` est la norme de la vitesse du CdG, dirigée selon `ψ + β`. En dérapage, `vx = v cos β` sous-estime cette norme (de 4 % à β = 16°). Donner `hypot(vx, vy)` respecte la définition de l'état du modèle. L'écart est petit, mais il n'y a aucune raison de fausser l'observation.

**S-F5.** Sous `v_bas`, `step` **est** le modèle cinématique (κ = 0, `test_blend_is_kinematic_below_low_speed`) : le test comparerait le cinématique à lui-même. Mesuré : à 1,2 m/s (κ = 0,4), l'écart n'est que de 0,22 cm. Le test croisé doit être fait au-dessus de `blend_speed_high`, là où le modèle dynamique est seul.

**S-G1.** Parce que le véhicule est presque neutre. En régime établi, `δ − L r / vx = α_f − α_r` (aux petits angles), et `α_f − α_r = K_us · a_y / g` avec `K_us = 0,0119` rad/g : 0,7° d'écart par g d'accélération latérale. Un véhicule franchement sous-vireur verrait `α_f` grimper bien au-dessus de `α_r`.

**S-G2.** Si le contrôleur cinématique avait une ESS effondrée, on pourrait attribuer son échec à un mauvais λ et non au modèle. Avec des ESS comparables (250 à 326 contre 407 à 592), les deux contrôleurs sont dans un régime d'estimation sain, et la différence de résultat vient de ce qu'ils prédisent, c'est-à-dire du modèle.

**S-G3.** Avec un λ grand, les poids sont presque uniformes et la commande est une moyenne de beaucoup de séquences : elle réagit moins, accélère moins franchement (vitesse moyenne 4,5 à 4,7 m/s contre 5,5 pour le dynamique à λ = 3) et braque moins fort. À plus faible accélération latérale, le modèle cinématique redevient presque juste (partie A) : l'erreur de modèle pèse moins. Le contrôleur est sauvé par sa timidité, pas par son modèle.

## S-H : résultats mesurés de la partie H

Mesurés avec la solution de référence, la config de 7.1 et une seule modification à la fois (`pixi run sweep H1 … H9`, graine 0). « off » est le nombre de pas hors piste, la gigue est `mean(abs(Δδ))` en rad par pas, β max est calculé au-dessus de 1,5 m/s.

| variante | t (s) | off | max \|d\| | ESS méd | ESS p5 | gigue | v moy | β max (°) |
|---|---|---|---|---|---|---|---|---|
| **référence** | **9,02** | **0** | **0,54** | **484** | **13** | **0,023** | **5,46** | **16,4** |
| H1 λ = 1 | 8,56 | 0 | 0,63 | 208 | 1 | 0,050 | 5,78 | 19,7 |
| H1 λ = 10 | 10,52 | 0 | 0,41 | 784 | 343 | 0,013 | 4,69 | 10,5 |
| H2 pneus linéaires (modèle et réalité) | 8,86 | 0 | 0,41 | 429 | 12 | 0,022 | 5,47 | 10,8 |
| H2 pneus tanh (modèle et réalité) | 9,16 | 0 | 0,55 | 420 | 15 | 0,022 | 5,40 | 16,7 |
| H3 rollouts linéaires, réalité Pacejka | 9,22 | 0 | 0,61 | 358 | 12 | 0,022 | 5,38 | 16,2 |
| H3 rollouts tanh, réalité Pacejka | 9,04 | 0 | 0,53 | 467 | 14 | 0,023 | 5,44 | 16,1 |
| H4 réalité μ = 0,8 | 9,62 | **28** | 0,81 | 331 | 3 | 0,025 | 5,25 | 26,6 |
| H4 μ = 0,8 partout | 9,32 | 0 | 0,64 | 311 | 14 | 0,025 | 5,39 | 24,4 |
| H5 Euler partout | 9,06 | 0 | 0,55 | 466 | 14 | 0,022 | 5,46 | 17,1 |
| H5 rollouts Euler, réalité RK4 | 9,14 | 0 | 0,51 | 390 | 14 | 0,022 | 5,38 | 16,5 |
| H6 T = 15 | 9,86 | **152** | 1,35 | 423 | 21 | 0,023 | 5,39 | 43,8 |
| H6 T = 50 | 8,18 | 0 | 0,50 | 195 | 4 | 0,055 | 5,87 | 27,4 |
| H6 T = 50, λ = 6 | 8,42 | 0 | 0,55 | 366 | 16 | 0,031 | 5,63 | 19,6 |
| H7 σ_δ = 0,05 | 9,08 | 0 | 0,60 | 563 | 19 | 0,016 | 5,53 | 16,9 |
| H7 σ_δ = 0,2 | 10,00 | 0 | 0,46 | 39 | 3 | 0,044 | 4,94 | 10,9 |
| H8 `v_ref` = 9 | 8,60 | 0 | 0,64 | 25 | 3 | 0,047 | 5,84 | 22,8 |
| H8 `v_ref` = 9, λ = 10 | 9,62 | 0 | 0,55 | 334 | 150 | 0,015 | 5,19 | 15,4 |
| H8 `v_ref` = 9, λ = 10, T = 50 | **8,04** | 0 | 0,64 | 125 | 19 | 0,033 | 5,92 | 25,8 |
| H9 rollouts cinématiques | 10,58 | **29** | 0,83 | 274 | 14 | 0,056 | 4,88 | 39,5 |
| H9 cinématique + adhérence | 10,40 | **50** | 1,07 | 6 | 1 | 0,067 | 4,90 | 22,2 |
| H9 cinématique + adhérence, λ = 10 | 10,96 | 0 | 0,46 | 264 | 61 | 0,013 | 4,54 | 10,9 |

Compléments sur 5 graines (0 à 4), pour les résultats de type « sort / ne sort pas » :

| variante | pas hors piste par graine | tours propres | temps (s) |
|---|---|---|---|
| référence (dynamique, λ = 3) | [0, 0, 0, 0, 0] | 5/5 | 8,68 à 9,18 |
| H1 λ = 1 | [0, 0, 0, 0, 0] | 5/5 | 8,36 à 8,70, mais `max \|d\|` de 0,61 à 0,65 |
| H2 linéaire partout | [0, 0, 0, 0, 0] | 5/5 | 8,32 à 8,88 |
| H2 Pacejka C = 1,9 partout | [0, 0, 0, 0, 0] | 5/5 | 8,66 à 9,08 |
| H3 rollouts linéaires, réalité Pacejka | [0, 0, 5, 23, 37] | 2/5 | 8,72 à 9,22 |
| H3 rollouts tanh, réalité Pacejka | [0, 0, 0, 0, 0] | 5/5 | 8,70 à 9,22 |
| H3 rollouts tanh, réalité Pacejka C = 1,9 | [0, 0, 0, 0, 0] | 5/5 | 8,68 à 9,10 |
| H4 réalité μ = 0,8 | [28, 25, 31, 31, 84] | **0/5** | 9,14 à 9,74 |
| H4 μ = 0,8 partout | [0, 0, 0, 0, 0] | 5/5 | 9,18 à 9,54 |
| H5 rollouts Euler, réalité RK4 | [0, 0, 0, 0, 0] | 5/5 | 8,66 à 9,22 |
| H6 T = 50 | [0, 0, 0, 0, 0] | 5/5 | 8,14 à 8,18 |
| H9 cinématique pur, λ = 10 | [0, 0, 0, 0, 0] | 5/5 | 10,62 à 10,94 |
| H9 cinématique + adhérence, λ = 10 | [0, 0, 0, 0, 0] | 5/5 | 10,86 à 11,08 |
| *idem à `v_ref` = 8, λ = 10 :* dynamique | [0, 0, 0, 0, 0] | 5/5 | 9,38 à 10,04 |
| cinématique pur | [0, 0, 26, 35, 5] | 2/5 | 10,22 à 11,18 |
| cinématique + adhérence | [0, 0, 0, 101, 0] | 4/5 | 10,24 à 11,26 |
| *à `v_ref` = 8, λ = 3 :* dynamique | [0, 0, 0, 0, 0] | 5/5 | 8,24 à 8,86 |
| cinématique pur | [111, 63, 867, 106, 783] | 0/5 (2 tête-à-queue, la voiture repart à l'envers) | — |
| cinématique + adhérence | [37, 68, 40, 43, 0] | 1/5 | 9,90 à 10,28 |

**Ce qu'il faut en retenir.**

- **H1, λ.** Le même compromis qu'à l'étape 1, à une autre échelle. λ = 1 est plus rapide (8,56 s) mais vit dangereusement : ESS p5 = 1, `max |d|` jusqu'à 0,65 m sur 5 graines, c'est-à-dire la limite. λ = 10 est très doux mais 1,5 s plus lent. La gigue suit la loi `σ/√ESS` de la théorie §4.2 en ordre de grandeur : `0,1/√13 = 0,028` contre 0,023 mesuré à λ = 3, `0,1/√343 = 0,005` contre 0,013 à λ = 10 (la gigue mesure aussi les vraies variations de braquage dans les virages, d'où le plancher). Sur un vrai robot, λ = 3.
- **H2, pneus.** Avec des pneus linéaires partout, la voiture va plus vite (8,32 à 8,88 s) et **dépasse μg** (accélération latérale max de 10,2 à 12,3 m/s² selon la graine) : rien ne l'en empêche, c'est un monde sans limite d'adhérence. tanh et Pacejka C = 1,5 donnent le même tour à 0,15 s près, ce qu'annonçait le tableau de §4.4 : avec ces coefficients, les deux courbes se superposent dans la zone où roule la voiture. Même avec C = 1,9 (pic à 30°), rien ne change mesurablement. **La branche descendante de Pacejka n'a pas d'effet visible ici.** C'est un résultat, à noter dans le journal ; il ne prouve pas qu'elle est inutile (avec des pneus identifiés à pic étroit, ce serait différent).
- **H3, modèle de pneu trop optimiste dans le contrôleur.** Des rollouts à pneus linéaires sur une réalité Pacejka sortent 3 fois sur 5 : c'est le mode d'échec de la théorie §7.9, version atténuée de la figure comparative (le modèle linéaire est juste aux petits angles, faux seulement au-delà de 14°). Des rollouts tanh sur une réalité Pacejka ne sortent jamais : l'erreur de modèle est trop petite pour compter. Conséquence pratique pour CUDA : si tanh est moins cher, il est un candidat sérieux pour les rollouts.
- **H4, la pluie.** Surestimer μ de 25 % (contrôleur à 1,0, réalité à 0,8) fait sortir la voiture **à chaque graine**. Le même contrôleur avec le bon μ passe 5 fois sur 5, 0,3 s plus lent. C'est la phrase de la théorie §6.7 : « une erreur de 10 % sur μ fait la différence entre une dérive maîtrisée et une sortie ». Une erreur sur μ est bien plus grave qu'une erreur sur la forme de la courbe (H3 tanh).
- **H5, intégrateur.** Aucune différence mesurable en boucle fermée, même avec des rollouts Euler sur une réalité RK4 (5/5). Le temps d'itération, lui, passe de 14,9 à 7,3 ms. Voir S-E3 pour ce qu'il faut en conclure (et ne pas en conclure).
- **H6, horizon.** À 7 m/s, T = 15 donne 0,3 s, soit 2,1 m de piste vue, alors que freiner de 7 à 4,6 m/s demande `(49 − 21,5)/8 = 3,4` m à 4 m/s² : la voiture découvre les virages trop tard et sort 152 pas (117 avec la graine 1). T = 50 (1 s, 7 m) est **plus rapide** (8,18 s, 5/5 graines) : la voiture freine juste ce qu'il faut. Mais l'ESS baisse (195 / 4) : le coût est sommé sur plus de pas, son échelle grandit, et λ doit suivre (λ = 6 : ESS 366 / 16). C'est la leçon de l'étape 1, qui revient à chaque changement : l'échelle du coût dépend de T. Pour l'étape 5 (T = 50 sur le Jetson), il faudra re-régler λ.
- **H7, σ_δ.** σ_δ = 0,05 : gigue plus faible (0,016), même tour. σ_δ = 0,2 : l'ESS s'effondre (39 / 3), et la voiture ralentit (10,00 s, v moyenne 4,94) : beaucoup de rollouts braquent trop et sortent, et les meilleurs survivants sont ceux qui roulent prudemment. Une exploration trop large rend prudent.
- **H8, demander plus vite.** `v_ref = 9` à λ = 3 effondre l'ESS (25 / 3) : le terme de vitesse pèse `((v − 9)/1)²` par pas, l'échelle du coût monte encore. Avec λ = 10, l'ESS revient mais la voiture est **plus lente** qu'à `v_ref = 7` (9,62 s) : λ = 10 est mou. Le meilleur tour mesuré de toute l'étape, 8,04 s, vient de `v_ref = 9`, λ = 10 **et** T = 50 : à cette vitesse, c'est l'horizon qui limite, pas la consigne.
- **H9, le cinématique.** Voir la partie G. À réglage égal (λ = 3), il sort ; rendu timide (λ = 10), il passe à 7 m/s mais 2 s plus lentement, et ressort à 8 m/s. À 8 m/s et λ = 3, le cinématique pur part deux fois en tête-à-queue (la progression devient négative : la voiture repart à l'envers sur la piste).

## Code de référence

Ce code est celui qui a produit tous les chiffres du document : `pixi run track && pixi run test && pixi run sim && pixi run compare && pixi run sweep H1 H2 H3 H4 H5 H6 H7 H8 H9` donnent les résultats des parties E à H (55 tests passent). Il part du code du dépôt à la fin de l'étape 1 (ton code, pas celui des solutions de `step_1.md`, qui diffère un peu : `SimLog`, `simulate`, `_readonly`…).

**Inchangés** : `python/mppi/controller.py`, `python/mppi/track.py`, `python/generate_track.py`. C'est le résultat attendu de la section 7.5.

### `pixi.toml` (ajout)

```toml
# STEP 2
compare = { cmd = "python bench/compare_models.py", depends-on = ["track"] }
```

### `cuda/include/mppi/config.hpp` (ajout du commentaire)

```cpp
// Parses config/mppi.yaml, the same file the Python side reads.
//
// Fields added in step 2 (to parse in step 3):
//   vehicle.mass, vehicle.izz, vehicle.cornering_stiffness_front,
//   vehicle.cornering_stiffness_rear, vehicle.tire_model (linear|tanh|pacejka),
//   vehicle.pacejka_c, vehicle.pacejka_e, vehicle.blend_speed_low,
//   vehicle.blend_speed_high, vehicle.integrator (euler|rk4), vehicle.substeps
// Removed in step 2: vehicle.kinematic_blend_speed
#pragma once
```

### `config/mppi.yaml`

```yaml
# Single source of truth for both the Python reference and the C++/CUDA path.
# SI units and radians everywhere. psi is measured from the x axis.
# State order is fixed: [x, y, psi, v] kinematic, [x, y, psi, vx, vy, r] dynamic.

model: dynamic            # kinematic | dynamic
state_dim: 6              # 4 for kinematic, 6 for dynamic
control_dim: 2            # [a, delta]

mppi:
  num_samples: 1024       # K, raised to 8192 once the CUDA path is validated
  horizon: 30             # T, raised to 50
  dt: 0.02
  lambda: 3.0             # était 0.3 à l'étape 1, re-réglé pour v_ref = 7 (partie G)
  gamma: 0.0              # poids du terme de correction, théorie §4.5
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
  width: 0.30             # largeur hors tout, pour la marge de sortie de piste
  mu: 1.0                 # coefficient d'adhérence, pour le terme d'adhérence
  mass: 3.5
  izz: 0.04
  cornering_stiffness_front: 4.0   # C_S (1/rad) : F_y = mu * C_S * F_z * alpha, convention CommonRoad
  cornering_stiffness_rear: 4.2
  tire_model: pacejka     # linear | tanh | pacejka
  pacejka_c: 1.5          # forme ; B = C_S / C (pente à l'origine identique au linéaire)
  pacejka_e: 0.0
  blend_speed_low: 1.0    # m/s : cinématique pur en dessous (remplace kinematic_blend_speed)
  blend_speed_high: 1.5   # m/s : dynamique pur au-dessus
  integrator: rk4         # euler | rk4, partie dynamique seulement
  substeps: 1             # sous-pas par intervalle dt

cost:
  w_lateral: 1.0
  w_progress: 10.0        # était 1.0
  w_offtrack: 100.0
  w_control: 0.01
  w_adhesion: 10.0        # utilisé seulement par le modèle cinématique
  w_speed: 1.0
  lateral_scale: 0.4      # d0 (m)
  speed_scale: 1.0        # v0 (m/s)
  v_ref: 7.0              # était 3.0 ; vitesse de référence (m/s)
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

### `python/mppi/config.py`

```python
"""Loader for config/mppi.yaml.

Same file is parsed by cuda/include/mppi/config.hpp. Any field added here must
be added there, otherwise the Python/CUDA parity check compares two different
problems.
"""
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

STATE_DIM = {"kinematic": 4, "dynamic": 6}
TIRE_MODELS = ("linear", "tanh", "pacejka")
INTEGRATORS = ("euler", "rk4")


@dataclass(frozen=True)
class Vehicle:
    wheelbase: float
    lf: float
    lr: float
    width: float
    mu: float
    mass: float
    izz: float
    cornering_stiffness_front: float   # C_S, 1/rad, normalisée par la charge (CommonRoad)
    cornering_stiffness_rear: float
    tire_model: str                    # linear | tanh | pacejka
    pacejka_c: float
    pacejka_e: float
    blend_speed_low: float             # m/s, cinématique pur en dessous
    blend_speed_high: float            # m/s, dynamique pur au-dessus
    integrator: str                    # euler | rk4, pour la partie dynamique
    substeps: int                      # sous-pas par intervalle de commande


@dataclass(frozen=True)
class MppiParams:
    num_samples: int
    horizon: int
    dt: float
    lam: float
    gamma: float
    noise_std: np.ndarray   # (control_dim,)
    seed: int


@dataclass(frozen=True)
class Bounds:
    u_min: np.ndarray   # [a_min, delta_min]
    u_max: np.ndarray   # [a_max, delta_max]


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


def _readonly(values):
    """float64 array that cannot be modified in place."""
    arr = np.array(values, dtype=np.float64)
    arr.setflags(write=False)
    return arr


def load_config(path) -> Config:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    root = path.resolve().parent.parent   # config/mppi.yaml -> repo root

    model = raw["model"]
    if model not in STATE_DIM:
        raise ValueError(f"model={model!r} unknown, expected one of {list(STATE_DIM)}")
    if raw["state_dim"] != STATE_DIM[model]:
        raise ValueError(
            f"state_dim={raw['state_dim']} but model {model!r} needs {STATE_DIM[model]}"
        )

    m = raw["mppi"]
    noise_std = _readonly(m["noise_std"])
    if noise_std.shape != (raw["control_dim"],):
        raise ValueError(
            f"noise_std has {noise_std.size} entries, control_dim={raw['control_dim']}"
        )

    v = raw["vehicle"]
    if not math.isclose(v["lf"] + v["lr"], v["wheelbase"]):
        raise ValueError(
            f"lf + lr = {v['lf'] + v['lr']} but wheelbase = {v['wheelbase']}"
        )

    if v["tire_model"] not in TIRE_MODELS:
        raise ValueError(f"tire_model={v['tire_model']!r}, expected one of {TIRE_MODELS}")
    if v["integrator"] not in INTEGRATORS:
        raise ValueError(f"integrator={v['integrator']!r}, expected one of {INTEGRATORS}")
    if not 0.0 < v["blend_speed_low"] < v["blend_speed_high"]:
        raise ValueError("need 0 < blend_speed_low < blend_speed_high")
    if not (isinstance(v["substeps"], int) and v["substeps"] >= 1):
        raise ValueError(f"substeps must be an integer >= 1, got {v['substeps']!r}")

    b = raw["control_bounds"]
    u_min = _readonly([b["a_min"], b["delta_min"]])
    u_max = _readonly([b["a_max"], b["delta_max"]])
    if not (u_min < u_max).all():
        raise ValueError(
            f"control_bounds: min must be < max, got u_min={u_min}, u_max={u_max}"
        )

    t = raw["track"]

    # --- construction ---
    return Config(
        model=model,
        state_dim=raw["state_dim"],
        control_dim=raw["control_dim"],
        mppi=MppiParams(
            num_samples=m["num_samples"],
            horizon=m["horizon"],
            dt=m["dt"],
            lam=m["lambda"],   # "lambda" is a Python keyword
            gamma=m["gamma"],
            noise_std=noise_std,
            seed=m["seed"],
        ),
        bounds=Bounds(u_min=u_min, u_max=u_max),
        vehicle=Vehicle(**v),   # les clés YAML sont exactement les noms des champs
        cost=CostParams(**raw["cost"]),   # YAML keys match the field names exactly
        track=TrackParams(
            grid_resolution=t["grid_resolution"],
            resample_step=t["resample_step"],
            margin=t["margin"],
            npz_path=root / t["npz_path"],
            bin_path=root / t["bin_path"],
        ),
        max_steps=raw["sim"]["max_steps"],
    )
```

### `python/mppi/dynamics.py`

```python
"""Vehicle dynamics.

Frame conventions, fixed for the whole project:

- Right-handed world frame, x forward, y left, psi measured from the x axis,
  counter-clockwise positive.
- SI units and radians everywhere. No degrees, ever.
- Kinematic state order: [x, y, psi, v]
- Dynamic state order:   [x, y, psi, vx, vy, r]
  vx, vy are body-frame velocities of the CG, r is the yaw rate.
- Control order: [a, delta], longitudinal acceleration and steering angle.

Public interface, mirrored by dynamics.cuh as a __device__ function:

    step(state, control, dt, vehicle) -> state

The model is selected by the size of the last axis of `state`
(STATE_DIM in config.py), a static choice, like a template parameter in CUDA.

Project choices, to be reproduced by the CUDA kernel:
- the vehicle never reverses: v (kinematic) or vx (dynamic) >= 0 after a step;
- controls are not clipped here, the controller applies the bounds;
- dynamic model: single-track at the CG, tire forces in the CommonRoad /
  F1TENTH normalized convention F_y = mu * C_S * F_z * alpha at small alpha;
- below blend_speed_high the dynamic step is blended with a kinematic step
  written in the same 6-component state (theory 6.5); every branch is a
  clip/where so that the CUDA version has no state-dependent `if`.
"""
import numpy as np

from .config import STATE_DIM, Vehicle

G = 9.81   # m/s²


# ---------------------------------------------------------------------------
# Modèle cinématique, état [x, y, psi, v] (étape 1, inchangé)
# ---------------------------------------------------------------------------

def step_kinematic(state: np.ndarray, control: np.ndarray, dt: float, vehicle: Vehicle) -> np.ndarray:
    """state (..., 4), control (..., 2) -> (..., 4). Kinematic bicycle at the CG, explicit Euler."""
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
# Pneus
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Modèle dynamique, état [x, y, psi, vx, vy, r]
# ---------------------------------------------------------------------------

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


def step(state: np.ndarray, control: np.ndarray, dt: float, vehicle: Vehicle) -> np.ndarray:
    """Model chosen by the state size: kinematic (4) or dynamic (6)."""
    if state.shape[-1] == STATE_DIM["dynamic"]:
        return step_dynamic(state, control, dt, vehicle)
    return step_kinematic(state, control, dt, vehicle)
```

### `python/mppi/cost.py`

```python
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

    state (..., state_dim), control (..., 2) -> (...,). Vectorized over leading axes,
    typically K.
    """
    c, veh, u_max = cfg.cost, cfg.vehicle, cfg.bounds.u_max
    d, _ = lookup(track, state[..., 0], state[..., 1])   # s inutile ici
    v = state[..., 3]   # v (cinématique) ou vx (dynamique) : même indice

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

    cost = lateral + offtrack + speed + effort
    if cfg.model == "kinematic":   # choix statique (config), pas une branche sur l'état
        # Rustine du cinématique, qui n'a pas de limite d'adhérence (Q-D4 de l'étape 1).
        # Le modèle dynamique sature ses pneus lui-même : terme supprimé.
        a_lat = v**2 * np.abs(np.tan(control[..., 1])) / veh.wheelbase
        cost = cost + c.w_adhesion * np.maximum(0.0, a_lat / (veh.mu * G) - 1.0) ** 2
    return cost


def terminal_cost(state_T: np.ndarray, s0: float, track: Track, cfg: Config) -> np.ndarray:
    """Progress reward on the last state x_T. state_T (..., state_dim) -> (...,).

    s0 is the progress of the rollout start x0, shared by all K rollouts.
    Normalized by the distance covered at v_ref over the horizon: ~ -w_progress
    for a rollout driven at v_ref.
    """
    c, m = cfg.cost, cfg.mppi
    _, s_T = lookup(track, state_T[..., 0], state_T[..., 1])
    progress = wrap(s_T - s0, track.length)   # sinon la ligne de départ casse tout (Q-C4)
    return -c.w_progress * progress / (c.v_ref * m.horizon * m.dt)
```

### `python/run_sim.py`

```python
"""Closed-loop simulation with the NumPy controller.

From step 2 on, the simulated vehicle (the "plant") is always the dynamic
model. The controller rolls out the model named in the config: dynamic
(perfect model) or kinematic (model mismatch, theory 7.9).

Writes the trajectory plot and the time series to results/figures/, and the
(state, control, next_state) triplets of the plant to results/trajectories/.
"""
import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # pas de fenêtre : le script tourne aussi en SSH
import matplotlib.pyplot as plt
import numpy as np

from mppi import track as trk
from mppi.config import STATE_DIM, Config, Vehicle, load_config
from mppi.controller import MPPI
from mppi.dynamics import G, axle_loads, slip_angles, step_dynamic, tire_force

BEAM_PROGRESS = 12.0   # m : on photographie le faisceau à l'entrée du premier virage
BEAM_SIZE = 200        # nombre de rollouts tracés, les plus lourds


@dataclass
class SimLog:
    states: np.ndarray       # (N, 6)  x_k du véhicule simulé, toujours dynamique
    controls: np.ndarray     # (N, 2)  u_k
    next_states: np.ndarray  # (N, 6)  x_{k+1}
    ess: np.ndarray          # (N,)
    rho: np.ndarray          # (N,)    coût minimal de l'itération
    d: np.ndarray            # (N,)    écart latéral de x_{k+1}
    t_iter: np.ndarray       # (N,)    durée de command(), en secondes
    progress: float          # distance parcourue le long de la piste (m)
    beam: tuple[np.ndarray, np.ndarray] | None   # (X, w) d'une itération, pour la figure


def observe(x: np.ndarray, cfg: Config) -> np.ndarray:
    """Plant state (6,) -> the state the controller works with (state_dim,).

    A kinematic controller gets the speed norm: it has no notion of vy or r.
    """
    if cfg.state_dim == STATE_DIM["dynamic"]:
        return x
    return np.array([x[0], x[1], x[2], np.hypot(x[3], x[4])])


def simulate(cfg: Config, track: trk.Track, plant: Vehicle | None = None) -> SimLog:
    """Runs MPPI in closed loop from a standstill on the start line, until one lap or max_steps.

    plant: parameters of the simulated vehicle, cfg.vehicle by default (perfect
    model). Passing other ones simulates a model error (part H).
    """
    plant = cfg.vehicle if plant is None else plant
    x = np.array([*track.centerline[0], track.heading[0], 0.0, 0.0, 0.0])
    ctrl = MPPI(cfg, track)

    states, controls, nexts, ess, rho, d_log, t_iter = [], [], [], [], [], [], []
    beam = None
    progress = 0.0
    _, s_prev = trk.lookup(track, x[0], x[1])

    for _ in range(cfg.max_steps):
        tic = time.perf_counter()
        u, info = ctrl.command(observe(x, cfg))
        t_iter.append(time.perf_counter() - tic)   # on ne chronomètre que le contrôleur

        # Le "vrai" système est le modèle dynamique, quel que soit le modèle des rollouts
        x_next = step_dynamic(x, u, cfg.mppi.dt, plant)

        d, s = trk.lookup(track, x_next[0], x_next[1])
        progress += trk.wrap(s - s_prev, track.length)   # wrap : franchir la ligne ne fait pas -48 m
        s_prev = s

        states.append(x); controls.append(u); nexts.append(x_next)
        ess.append(info["ess"]); rho.append(info["rho"]); d_log.append(float(d))
        if beam is None and progress >= BEAM_PROGRESS:
            beam = (info["X"], info["w"])

        x = x_next
        if progress >= track.length or not np.all(np.isfinite(x)):
            break

    return SimLog(np.array(states), np.array(controls), np.array(nexts), np.array(ess),
                  np.array(rho), np.array(d_log), np.array(t_iter), float(progress), beam)


def chassis(log: SimLog, cfg: Config) -> dict[str, np.ndarray]:
    """Slip angles, body slip and lateral acceleration of the plant along the run."""
    veh = cfg.vehicle
    alpha_f, alpha_r = slip_angles(log.states, log.controls[:, 1], veh)
    fzf, fzr = axle_loads(veh)
    fyf = tire_force(alpha_f, fzf, veh.cornering_stiffness_front, veh)
    fyr = tire_force(alpha_r, fzr, veh.cornering_stiffness_rear, veh)
    vx, vy = log.states[:, 3], log.states[:, 4]
    return {
        "alpha_f": alpha_f, "alpha_r": alpha_r,
        "beta": np.arctan2(vy, np.maximum(vx, 1e-3)),            # dérive du véhicule
        "a_lat": (fyf * np.cos(log.controls[:, 1]) + fyr) / veh.mass,
        "speed": np.hypot(vx, vy),
    }


def print_summary(log: SimLog, cfg: Config, track: trk.Track) -> None:
    d_max = cfg.cost.track_half_width - 0.5 * cfg.vehicle.width
    duration = len(log.states) * cfg.mppi.dt
    lap = log.progress >= track.length
    off = np.abs(log.d) > d_max
    ch = chassis(log, cfg)
    p5, p50, p95 = np.percentile(log.ess, [5, 50, 95])

    print(f"controller model {cfg.model}, plant dynamic ({cfg.vehicle.tire_model} tires)")
    print(f"lap {'done' if lap else 'NOT done'} in {duration:.2f} s, "
          f"progress {log.progress:.1f}/{track.length:.1f} m")
    print(f"offtrack steps {off.sum()}, max |d| {np.abs(log.d).max():.3f} m (limit {d_max:.2f})")
    print(f"v mean {ch['speed'].mean():.2f} max {ch['speed'].max():.2f} m/s, "
          f"max a_lat {np.abs(ch['a_lat']).max():.1f} m/s2 (mu g = {cfg.vehicle.mu * G:.1f})")
    fast = log.states[:, 3] > cfg.vehicle.blend_speed_high   # dérive sans objet en dessous
    deg = lambda k: np.degrees(np.abs(ch[k][fast]).max()) if fast.any() else 0.0
    print(f"above {cfg.vehicle.blend_speed_high} m/s: max |beta| {deg('beta'):.1f} deg, "
          f"max |alpha_f| {deg('alpha_f'):.1f} deg, max |alpha_r| {deg('alpha_r'):.1f} deg")
    print(f"ESS median {p50:.0f} p5 {p5:.0f} p95 {p95:.0f} / K={cfg.mppi.num_samples}")
    print(f"iteration time median {1e3 * np.median(log.t_iter):.1f} ms")


def save_trajectories(log: SimLog, cfg: Config, path: Path) -> None:
    """(state, control, next_state) triplets of the plant, reused for the learned model."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, state=log.states, control=log.controls, next_state=log.next_states,
             ess=log.ess, rho=log.rho, dt=cfg.mppi.dt)


def draw_track(ax, track: trk.Track, hw: float) -> None:
    normal = np.column_stack([-np.sin(track.heading), np.cos(track.heading)])
    for side in (-1, 1):
        edge = track.centerline + side * hw * normal
        ax.plot(*np.vstack([edge, edge[:1]]).T, "k", lw=1)   # edge[:1] referme la boucle
    ax.plot(*track.centerline.T, "k--", lw=0.5)
    ax.set_aspect("equal")


def plot_trajectory(log: SimLog, cfg: Config, track: trk.Track, path: Path) -> None:
    """Track edges, driven path colored by body slip angle, and the rollout beam at one instant."""
    fig, ax = plt.subplots(figsize=(9, 6))
    zoom = ax.inset_axes([0.35, 0.3, 0.3, 0.4]) if log.beam is not None else None
    beta = np.degrees(chassis(log, cfg)["beta"])
    lim = max(np.abs(beta).max(), 1.0)

    for a in filter(None, (ax, zoom)):
        draw_track(a, track, cfg.cost.track_half_width)
        sc = a.scatter(log.states[:, 0], log.states[:, 1], c=beta, s=2,
                       cmap="coolwarm", vmin=-lim, vmax=lim)

    if zoom is not None:
        X, w = log.beam
        for i in np.argsort(w)[-BEAM_SIZE:]:
            zoom.plot(X[i, :, 0], X[i, :, 1], color=plt.cm.viridis(w[i] / w.max()), lw=0.5, alpha=0.6)
        lo, hi = X[..., :2].min(axis=(0, 1)) - 0.3, X[..., :2].max(axis=(0, 1)) + 0.3
        zoom.set_xlim(lo[0], hi[0]); zoom.set_ylim(lo[1], hi[1])
        zoom.set_xticks([]); zoom.set_yticks([])
        zoom.set_title("rollouts, colored by weight", fontsize=8)
        ax.indicate_inset_zoom(zoom, edgecolor="gray")

    fig.colorbar(sc, ax=ax, label="body slip beta (deg)")
    ax.set_title(f"Step 2, MPPI with {cfg.model} rollouts, dynamic plant")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_series(log: SimLog, cfg: Config, path: Path) -> None:
    """ESS, d, speed, yaw rate, slip angles and steering against time."""
    d_max = cfg.cost.track_half_width - 0.5 * cfg.vehicle.width
    t = np.arange(len(log.states)) * cfg.mppi.dt
    ch = chassis(log, cfg)
    alpha_sat = np.degrees(1.0 / cfg.vehicle.cornering_stiffness_front)   # fin de la zone linéaire

    fig, axs = plt.subplots(6, 1, figsize=(9, 13), sharex=True)
    axs[0].plot(t, log.ess); axs[0].set_ylabel("ESS")
    axs[0].axhline(cfg.mppi.num_samples, color="gray", ls=":")
    axs[1].plot(t, log.d)
    for sign in (-1, 1):
        axs[1].axhline(sign * d_max, color="r")
    axs[1].set_ylabel("d (m)")
    axs[2].plot(t, log.states[:, 3], label="vx")
    axs[2].plot(t, log.states[:, 4], label="vy")
    axs[2].axhline(cfg.cost.v_ref, color="gray", ls=":")
    axs[2].set_ylabel("m/s"); axs[2].legend(loc="upper right")
    axs[3].plot(t, log.states[:, 5]); axs[3].set_ylabel("r (rad/s)")
    axs[4].plot(t, np.degrees(ch["alpha_f"]), label="alpha_f")
    axs[4].plot(t, np.degrees(ch["alpha_r"]), label="alpha_r")
    axs[4].plot(t, np.degrees(ch["beta"]), label="beta", lw=0.8)
    for sign in (-1, 1):
        axs[4].axhline(sign * alpha_sat, color="gray", ls=":")
    axs[4].set_ylabel("deg"); axs[4].legend(loc="upper right")
    axs[5].plot(t, log.controls[:, 1]); axs[5].set_ylabel("delta (rad)")
    axs[5].set_xlabel("t (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/mppi.yaml")
    cfg = load_config(ap.parse_args().config)
    track = trk.load(cfg.track.npz_path)
    results = cfg.track.npz_path.parent.parent   # results/track/costmap.npz -> results/

    log = simulate(cfg, track)
    print_summary(log, cfg, track)

    fig_dir = results / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    save_trajectories(log, cfg, results / "trajectories" / f"step2_{cfg.model}.npz")
    plot_trajectory(log, cfg, track, fig_dir / f"step2_{cfg.model}_trajectory.png")
    plot_series(log, cfg, fig_dir / f"step2_{cfg.model}_series.png")
    print(f"figures in {fig_dir}")


if __name__ == "__main__":
    main()
```

### `python/tests/test_step1.py` (début du fichier seulement)

Le reste du fichier est inchangé.

```python
import dataclasses
from pathlib import Path

import numpy as np
import pytest

from mppi import track as trk
from mppi.config import STATE_DIM, load_config
from mppi.controller import MPPI, S_MAX, rollout_costs
from mppi.cost import G, stage_cost, terminal_cost
from mppi.dynamics import step

# Le YAML est passé en dynamique à l'étape 2 : ces tests gardent le cinématique
CFG = dataclasses.replace(load_config(Path(__file__).resolve().parents[2] / "config" / "mppi.yaml"),
                          model="kinematic", state_dim=STATE_DIM["kinematic"])
VEH = CFG.vehicle
```

`from mppi.dynamics import step` fonctionne toujours : `step` choisit le cinématique pour un état à 4 composantes.

### `python/tests/test_step2.py`

```python
"""Step 2 tests: dynamic single-track model, tires, blend, and the 6-state controller."""
import dataclasses
from pathlib import Path

import numpy as np
import pytest

from mppi import dynamics as dyn
from mppi import track as trk
from mppi.config import load_config
from mppi.controller import MPPI, rollout_costs

CFG = load_config(Path(__file__).resolve().parents[2] / "config" / "mppi.yaml")
VEH = CFG.vehicle
DT = CFG.mppi.dt
TIRES = ("linear", "tanh", "pacejka")


def with_vehicle(**kw):
    return dataclasses.replace(VEH, **kw)


def on_manifold(vx, delta, veh=VEH):
    """Dynamic state at the origin, heading +x, with (vy, r) on the kinematic manifold."""
    tan_d = np.tan(delta)
    return np.array([0.0, 0.0, 0.0, vx, veh.lr / veh.wheelbase * vx * tan_d, vx * tan_d / veh.wheelbase])


def run(state, control, n, veh=VEH, f=dyn.step):
    for _ in range(n):
        state = f(state, control, DT, veh)
    return state


# --- Configuration -----------------------------------------------------------

def test_config_is_dynamic():
    assert CFG.model == "dynamic" and CFG.state_dim == 6


def test_cornering_stiffness_convention():
    """C_S is normalized by the load: C_alpha = mu * C_S * F_z, ~75 / 66 N/rad (theory 6.3)."""
    fzf, fzr = dyn.axle_loads(VEH)
    assert fzf + fzr == pytest.approx(VEH.mass * dyn.G)
    assert VEH.mu * VEH.cornering_stiffness_front * fzf == pytest.approx(74.9, abs=0.1)
    assert VEH.mu * VEH.cornering_stiffness_rear * fzr == pytest.approx(65.6, abs=0.1)


# --- Pneus --------------------------------------------------------------------

@pytest.mark.parametrize("tire", TIRES)
def test_tire_same_slope_at_origin(tire):
    fz, c_s = 18.7, VEH.cornering_stiffness_front
    alpha = 1e-4
    f = dyn.tire_force(np.array(alpha), fz, c_s, with_vehicle(tire_model=tire))
    assert f == pytest.approx(VEH.mu * c_s * fz * alpha, rel=1e-6)


@pytest.mark.parametrize("tire", ("tanh", "pacejka"))
def test_tire_saturates_at_mu_fz(tire):
    fz = 18.7
    alpha = np.linspace(-np.pi / 2, np.pi / 2, 2001)
    f = dyn.tire_force(alpha, fz, VEH.cornering_stiffness_front, with_vehicle(tire_model=tire))
    assert np.abs(f).max() <= VEH.mu * fz * (1 + 1e-12)
    assert np.abs(f).max() >= 0.95 * VEH.mu * fz
    assert np.allclose(f, -f[::-1])          # force impaire


# --- Modèle dynamique ---------------------------------------------------------

def test_straight_line_stays_straight():
    x = run(on_manifold(3.0, 0.0), np.zeros(2), 50)
    assert np.allclose(x[[1, 2, 4, 5]], 0.0)
    assert x[0] == pytest.approx(3.0) and x[3] == pytest.approx(3.0)


def test_turns_left_with_positive_steer():
    x = run(on_manifold(3.0, 0.0), np.array([0.0, 0.1]), 25)
    assert x[1] > 0 and x[2] > 0 and x[5] > 0


@pytest.mark.parametrize("tire", TIRES)
def test_left_right_symmetry(tire):
    """Mirroring y, psi, vy, r and delta mirrors the result exactly."""
    veh = with_vehicle(tire_model=tire)
    rng = np.random.default_rng(0)
    states = rng.normal(size=(64, 6)) * [1, 1, 0.5, 0, 0.5, 1]
    states[:, 3] = rng.uniform(0, 8, 64)
    controls = rng.uniform(CFG.bounds.u_min, CFG.bounds.u_max, (64, 2))
    flip_x, flip_u = np.array([1, -1, -1, 1, -1, -1]), np.array([1, -1])
    a = dyn.step(states * flip_x, controls * flip_u, DT, veh)
    b = dyn.step(states, controls, DT, veh) * flip_x
    assert np.allclose(a, b, atol=1e-12)


def test_steady_state_cornering():
    """Steady turn: (F_yf cos(delta) + F_yr) / m = vx * r, zero yaw moment, and
    r matches the understeer formula r = vx delta / (L + K_us vx^2 / g)."""
    veh = with_vehicle(tire_model="linear")
    vx, delta = 3.0, 0.05
    fzf, fzr = dyn.axle_loads(veh)
    x = on_manifold(vx, 0.0)
    for _ in range(250):   # 5 s, vx maintenu constant par la commande a
        alpha_f, _ = dyn.slip_angles(x, delta, veh)
        fyf = dyn.tire_force(alpha_f, fzf, veh.cornering_stiffness_front, veh)
        a = fyf * np.sin(delta) / veh.mass - x[4] * x[5]
        x = dyn.step(x, np.array([a, delta]), DT, veh)

    alpha_f, alpha_r = dyn.slip_angles(x, delta, veh)
    fyf = dyn.tire_force(alpha_f, fzf, veh.cornering_stiffness_front, veh)
    fyr = dyn.tire_force(alpha_r, fzr, veh.cornering_stiffness_rear, veh)
    assert x[3] == pytest.approx(vx, rel=1e-3)
    assert (fyf * np.cos(delta) + fyr) / veh.mass == pytest.approx(x[3] * x[5], rel=1e-6)
    assert veh.lf * fyf * np.cos(delta) - veh.lr * fyr == pytest.approx(0.0, abs=1e-6)
    k_us = 1 / veh.cornering_stiffness_front - 1 / veh.cornering_stiffness_rear
    assert x[5] == pytest.approx(vx * delta / (veh.wheelbase + k_us * vx**2 / dyn.G), rel=0.01)


def test_cross_check_with_kinematic_model():
    """Permanent non-regression: at low speed and small steer, the dynamic model
    (here 2 m/s, above the blend band, so purely dynamic) follows the kinematic one.

    Expected gap: understeer K_us vx^2 / (g L) = 1.5 % of the heading (0.009 rad
    out of 0.61), plus the slip build-up lag and the steering drag on vx:
    6.8 cm and 0.011 rad measured over 4 m. Tolerances x1.5 to x2.
    """
    vx, delta, n = 2.0, 0.05, 100
    beta = np.arctan(VEH.lr / VEH.wheelbase * np.tan(delta))
    k = run(np.array([0.0, 0.0, 0.0, vx / np.cos(beta)]), np.array([0.0, delta]), n)
    d = run(on_manifold(vx, delta), np.array([0.0, delta]), n)
    assert np.hypot(k[0] - d[0], k[1] - d[1]) < 0.10
    assert abs(k[2] - d[2]) < 0.02


def test_blend_is_kinematic_below_low_speed():
    """Below blend_speed_low the step is exactly the kinematic one in 6 states."""
    x = on_manifold(0.9 * VEH.blend_speed_low, 0.2)
    u = np.array([0.5, 0.2])
    assert np.array_equal(dyn.step(x, u, DT, VEH), dyn.step_kinematic6(x, u, DT, VEH))


def test_blend_is_continuous():
    u = np.array([1.0, 0.3])
    for v in (VEH.blend_speed_low, VEH.blend_speed_high):
        lo = dyn.step(on_manifold(v - 1e-7, 0.3), u, DT, VEH)
        hi = dyn.step(on_manifold(v + 1e-7, 0.3), u, DT, VEH)
        assert np.allclose(lo, hi, atol=1e-5)


@pytest.mark.parametrize("tire", TIRES)
@pytest.mark.parametrize("integrator", ("euler", "rk4"))
def test_finite_at_standstill_and_in_band(tire, integrator):
    """vx from 0 to 2 m/s with arbitrary vy, r: finite, no warning (0 * NaN trap)."""
    veh = with_vehicle(tire_model=tire, integrator=integrator)
    rng = np.random.default_rng(1)
    n = 401
    states = np.zeros((n, 6))
    states[:, 3] = np.linspace(0.0, 2.0, n)
    states[:, 4] = rng.normal(0, 0.5, n)
    states[:, 5] = rng.normal(0, 2.0, n)
    controls = rng.uniform(CFG.bounds.u_min, CFG.bounds.u_max, (n, 2))
    with np.errstate(all="raise"):
        out = run(states, controls, 50, veh)
    assert np.all(np.isfinite(out))


def test_never_reverses():
    x = dyn.step(on_manifold(0.01, 0.0), np.array([-4.0, 0.0]), DT, VEH)
    assert x[3] == 0.0 and x[4] == 0.0 and x[5] == 0.0


def test_integrator_stable_at_blend_low():
    """Config guard: the pure dynamic step must be stable where the blend starts using it."""
    x = on_manifold(VEH.blend_speed_low, 0.0)
    x[5] = 0.1
    for _ in range(100):
        x = dyn.step_dynamic_only(x, np.zeros(2), DT, VEH)
    assert abs(x[5]) < 0.1


def test_batch_matches_single():
    rng = np.random.default_rng(0)
    states = rng.normal(size=(8, 6))
    states[:, 3] = rng.uniform(0, 6, 8)
    controls = rng.normal(size=(8, 2)) * 0.1
    batch = dyn.step(states, controls, DT, VEH)
    loop = np.stack([dyn.step(states[i], controls[i], DT, VEH) for i in range(8)])
    assert batch.shape == (8, 6)
    assert np.allclose(batch, loop)


# --- Contrôleur avec l'état à 6 composantes ----------------------------------

@pytest.fixture(scope="module")
def track():
    return trk.load(CFG.track.npz_path)


def start_state(track, vx=2.0):
    return np.array([*track.centerline[0], track.heading[0], vx, 0.0, 0.0])


def test_rollout_costs_6_states(track):
    T = CFG.mppi.horizon
    eps = np.random.default_rng(0).normal(size=(64, T, 2)) * CFG.mppi.noise_std
    S1, _, X1 = rollout_costs(start_state(track), np.zeros((T, 2)), eps, track, CFG)
    S2, _, X2 = rollout_costs(start_state(track), np.zeros((T, 2)), eps, track, CFG)
    assert X1.shape == (64, T + 1, 6)
    assert np.array_equal(S1, S2) and np.all(np.isfinite(S1))


def test_closed_loop_short(track):
    """One second from a standstill with the dynamic model: stays on track, speeds up."""
    ctrl, x = MPPI(CFG, track), start_state(track, vx=0.0)
    d_max = CFG.cost.track_half_width - 0.5 * VEH.width
    for _ in range(50):
        u, _ = ctrl.command(x)
        x = dyn.step(x, u, DT, VEH)
        d, _ = trk.lookup(track, x[0], x[1])
        assert abs(d) < d_max
    assert x[3] > 2.0
```

### `bench/compare_models.py`

```python
"""Step 2 exit figure: MPPI with kinematic rollouts vs MPPI with dynamic rollouts.

Same plant (the dynamic model), same track, same v_ref, same lambda, K, T,
Sigma and seeds. Only the model inside the rollouts changes (theory 7.9).

    pixi run compare                 # 5 seeds, figure + table
"""
import argparse
import dataclasses
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from mppi import track as trk
from mppi.config import STATE_DIM, Config, load_config
from run_sim import SimLog, chassis, draw_track, simulate

SEEDS = range(5)


def variants(base: Config) -> dict[str, Config]:
    """The three controllers compared. Only the rollout model and the adhesion patch differ."""
    kin = dataclasses.replace(base, model="kinematic", state_dim=STATE_DIM["kinematic"])
    return {
        "dynamic": dataclasses.replace(base, model="dynamic", state_dim=STATE_DIM["dynamic"]),
        "kinematic": dataclasses.replace(kin, cost=dataclasses.replace(base.cost, w_adhesion=0.0)),
        "kinematic + adhesion term": kin,
    }


def with_seed(cfg: Config, seed: int) -> Config:
    return dataclasses.replace(cfg, mppi=dataclasses.replace(cfg.mppi, seed=seed))


def run(cfg: Config) -> SimLog:
    return simulate(cfg, trk.load(cfg.track.npz_path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config" / "mppi.yaml"))
    base = load_config(ap.parse_args().config)
    track = trk.load(base.track.npz_path)
    d_max = base.cost.track_half_width - 0.5 * base.vehicle.width
    cfgs = variants(base)

    jobs = [(name, s, with_seed(cfg, s)) for name, cfg in cfgs.items() for s in SEEDS]
    with ProcessPoolExecutor() as pool:
        logs = dict(zip([(n, s) for n, s, _ in jobs], pool.map(run, [c for *_, c in jobs])))

    print(f"v_ref {base.cost.v_ref}, lambda {base.mppi.lam}, K {base.mppi.num_samples}, "
          f"T {base.mppi.horizon}, plant dynamic, seeds {list(SEEDS)}")
    print("| controller | clean laps | off-track steps per seed | lap time (s) | ESS median |")
    print("|---|---|---|---|---|")
    for name in cfgs:
        ls = [logs[name, s] for s in SEEDS]
        off = [int((np.abs(l.d) > d_max).sum()) for l in ls]
        clean = sum(o == 0 and l.progress >= track.length for o, l in zip(off, ls))
        t = [len(l.states) * base.mppi.dt for l in ls]
        ess = [np.median(l.ess) for l in ls]
        print(f"| {name} | {clean}/{len(ls)} | {off} | {min(t):.2f} to {max(t):.2f} | "
              f"{min(ess):.0f} to {max(ess):.0f} |")

    # Figure : seed 0, une piste par contrôleur, points hors piste en rouge
    fig, axs = plt.subplots(1, len(cfgs), figsize=(6 * len(cfgs), 5))
    for ax, (name, cfg) in zip(axs, cfgs.items()):
        log = logs[name, 0]
        draw_track(ax, track, base.cost.track_half_width)
        speed = chassis(log, cfg)["speed"]
        sc = ax.scatter(log.states[:, 0], log.states[:, 1], c=speed, s=2, cmap="plasma",
                        vmin=0, vmax=base.cost.v_ref + 1)
        off = np.abs(log.d) > d_max
        ax.scatter(log.next_states[off, 0], log.next_states[off, 1], s=8, c="red", label="off track")
        ax.set_title(f"{name} rollouts: {off.sum()} off-track steps\n"
                     f"lap {len(log.states) * base.mppi.dt:.2f} s (seed 0)")
        ax.legend(loc="lower right", fontsize=8)
    fig.colorbar(sc, ax=axs, label="speed (m/s)", shrink=0.8)
    fig.suptitle(f"Same dynamic plant, v_ref = {base.cost.v_ref} m/s, lambda = {base.mppi.lam}")
    out = base.track.npz_path.parent.parent / "figures" / "step2_comparison.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"figure: {out}")


if __name__ == "__main__":
    main()
```

### `bench/sweep.py`

Les expériences G2 à G6 de l'étape 1 ont été retirées du dictionnaire pour la lisibilité ; garde-les si tu veux, mais avec le YAML de l'étape 2 elles ne reproduisent plus les chiffres de `step_1.md` (elles tournent en dynamique à `v_ref = 7`).

```python
"""Parameter experiments: part G of step 1, part H of step 2.

    pixi run sweep H1          # one experiment
    pixi run sweep H1 H3 H5    # several
    pixi run sweep             # all of them

Write down your prediction before running each experiment.
"""
import argparse
import dataclasses
import sys
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from mppi import track as trk
from mppi.config import STATE_DIM, Config, Vehicle, load_config
from run_sim import chassis, simulate

ConfigFn = Callable[[Config], Config]
# (nom, modification de la config du contrôleur, modification des paramètres du véhicule simulé)
Variant = tuple[str, ConfigFn, dict]


def with_mppi(**kw) -> ConfigFn:
    """Config -> copy of Config with some cfg.mppi fields replaced."""
    return lambda c: dataclasses.replace(c, mppi=dataclasses.replace(c.mppi, **kw))


def with_cost(**kw) -> ConfigFn:
    return lambda c: dataclasses.replace(c, cost=dataclasses.replace(c.cost, **kw))


def with_vehicle(**kw) -> ConfigFn:
    """Vehicle parameters used by the rollouts (the controller's model)."""
    return lambda c: dataclasses.replace(c, vehicle=dataclasses.replace(c.vehicle, **kw))


def kinematic(c: Config) -> Config:
    return dataclasses.replace(c, model="kinematic", state_dim=STATE_DIM["kinematic"])


def chain(*fs: ConfigFn) -> ConfigFn:
    """Applies several modifications in a row."""
    def apply(c: Config) -> Config:
        for f in fs:
            c = f(c)
        return c
    return apply


SAME = lambda c: c   # noqa: E731


def both(**kw) -> tuple[ConfigFn, dict]:
    """Same change in the rollouts and in the simulated vehicle: perfect model."""
    return with_vehicle(**kw), kw


EXPERIMENTS: dict[str, list[Variant]] = {
    # Étape 1 (avec le YAML de l'étape 2, ces lignes ne reproduisent plus les chiffres de step_1.md)
    "G1": [(f"lambda={lam}", with_mppi(lam=lam), {}) for lam in (1.0, 0.3, 0.1, 0.03)],
    "G7": [("seed=1", with_mppi(seed=1), {})],
    # Étape 2
    "H1": [(f"lambda={lam}", with_mppi(lam=lam), {}) for lam in (1.0, 10.0)],
    "H2": [(f"tires {t}, both", *both(tire_model=t)) for t in ("linear", "tanh")],
    "H3": [("rollouts linear, plant pacejka", with_vehicle(tire_model="linear"), {}),
           ("rollouts tanh, plant pacejka", with_vehicle(tire_model="tanh"), {})],
    "H4": [("plant mu=0.8", SAME, {"mu": 0.8}), ("mu=0.8, both", *both(mu=0.8))],
    "H5": [("euler, both", *both(integrator="euler")),
           ("rollouts euler, plant rk4", with_vehicle(integrator="euler"), {})],
    "H6": [("T=15", with_mppi(horizon=15), {}), ("T=50", with_mppi(horizon=50), {}),
           ("T=50, lambda=6", with_mppi(horizon=50, lam=6.0), {})],
    "H7": [(f"sigma_delta={s}", with_mppi(noise_std=np.array([0.5, s])), {}) for s in (0.05, 0.2)],
    "H8": [("v_ref=9", with_cost(v_ref=9.0), {}),
           ("v_ref=9, lambda=10", chain(with_cost(v_ref=9.0), with_mppi(lam=10.0)), {}),
           ("v_ref=9, lambda=10, T=50", chain(with_cost(v_ref=9.0), with_mppi(lam=10.0, horizon=50)), {})],
    "H9": [("kinematic rollouts", chain(kinematic, with_cost(w_adhesion=0.0)), {}),
           ("kinematic + adhesion", kinematic, {}),
           ("kinematic + adhesion, lambda=10", chain(kinematic, with_mppi(lam=10.0)), {})],
}


def run(job: tuple[Config, Vehicle]) -> dict[str, float | bool]:
    """One closed-loop lap of controller cfg on the simulated vehicle plant."""
    cfg, plant = job
    track = trk.load(cfg.track.npz_path)
    log = simulate(cfg, track, plant)
    d_max = cfg.cost.track_half_width - 0.5 * cfg.vehicle.width
    fast = log.states[:, 3] > cfg.vehicle.blend_speed_high
    beta = np.abs(chassis(log, cfg)["beta"][fast])
    return {
        "lap": log.progress >= track.length,
        "t": len(log.states) * cfg.mppi.dt,
        "progress": log.progress,
        "ess_med": float(np.median(log.ess)),
        "ess_p5": float(np.percentile(log.ess, 5)),
        "jitter": float(np.abs(np.diff(log.controls[:, 1])).mean()),   # rad/pas, théorie §7.5
        "dmax": float(np.abs(log.d).max()),
        "off": int((np.abs(log.d) > d_max).sum()),
        "v_mean": float(np.hypot(log.states[:, 3], log.states[:, 4]).mean()),
        "beta_max": float(np.degrees(beta.max())) if beta.size else 0.0,
    }


def print_row(name: str, r: dict[str, float | bool]) -> None:
    lap = "yes" if r["lap"] else f"NO ({r['progress']:.1f} m)"
    print(f"| {name} | {lap} | {r['t']:.2f} | {r['off']} | {r['dmax']:.2f} | {r['ess_med']:.0f} "
          f"| {r['ess_p5']:.0f} | {r['jitter']:.3f} | {r['v_mean']:.2f} | {r['beta_max']:.1f} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiments", nargs="*", help=f"among {', '.join(EXPERIMENTS)}, default: all")
    ap.add_argument("--config", default=str(ROOT / "config" / "mppi.yaml"))
    args = ap.parse_args()
    unknown = set(args.experiments) - set(EXPERIMENTS)
    if unknown:
        ap.error(f"unknown experiments {sorted(unknown)}, expected among {list(EXPERIMENTS)}")
    base = load_config(args.config)

    variants: list[Variant] = [("reference", SAME, {})]
    for name in args.experiments or EXPERIMENTS:
        variants += [(f"{name} {label}", f, p) for label, f, p in EXPERIMENTS[name]]

    # Les runs sont indépendants : un processus par variante
    with ProcessPoolExecutor() as pool:
        jobs = [(f(base), dataclasses.replace(base.vehicle, **p)) for _, f, p in variants]
        results = pool.map(run, jobs)
        print("| variant | lap | t (s) | off | max abs(d) | ESS med | ESS p5 | jitter | v mean | max beta (deg) |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for (name, *_), r in zip(variants, results):
            print_row(name, r)


if __name__ == "__main__":
    main()
```

### `bench/integrators.py (mesures de la partie E, S-E)`

Sortie attendue : les tableaux de §5.2, §6.3, §6.4 et §6.5.

```python
"""Step 2, part E: stiffness, stability and accuracy of the integrators.

    pixi run python bench/integrators.py
"""
import dataclasses
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from mppi import dynamics as dyn
from mppi.config import load_config

VEH = load_config(ROOT / "config" / "mppi.yaml").vehicle
DT = 0.02
VARIANTS = [("euler", 1), ("euler", 2), ("euler", 4), ("rk4", 1), ("rk4", 2)]


def eigenvalues():
    """Linearized (vy, r) system at constant vx, linear tires (theory 6.3)."""
    fzf, fzr = dyn.axle_loads(VEH)
    cf = VEH.mu * VEH.cornering_stiffness_front * fzf
    cr = VEH.mu * VEH.cornering_stiffness_rear * fzr
    m, iz, lf, lr = VEH.mass, VEH.izz, VEH.lf, VEH.lr
    print(f"C_alpha front {cf:.1f} N/rad, rear {cr:.1f} N/rad, yaw coefficient "
          f"{(lf**2 * cf + lr**2 * cr) / iz:.1f}")
    for vx in (0.5, 1.0, 1.5, 3.0, 5.0, 8.0):
        a = np.array([[-(cf + cr) / (m * vx), -(lf * cf - lr * cr) / (m * vx) - vx],
                      [-(lf * cf - lr * cr) / (iz * vx), -(lf**2 * cf + lr**2 * cr) / (iz * vx)]])
        print(f"vx {vx}: eigenvalues {np.round(np.linalg.eigvals(a), 1)}")


def unstable_below(integrator, substeps, tire):
    """Largest vx at which a yaw-rate perturbation does not decay, pure dynamic model."""
    # Bande de mélange abaissée : sinon le plancher vx_safe = max(vx, 1.0) masque l'instabilité
    veh = dataclasses.replace(VEH, integrator=integrator, substeps=substeps, tire_model=tire,
                              blend_speed_low=0.01, blend_speed_high=0.02)
    vxs = np.linspace(0.05, 3.0, 2951)
    s = np.zeros((len(vxs), 6))
    s[:, 3], s[:, 5] = vxs, 0.1
    with np.errstate(all="ignore"):
        for _ in range(100):
            s = dyn.step_dynamic_only(s, np.zeros((len(vxs), 2)), DT, veh)
        bad = ~(np.abs(s[:, 5]) < 0.1)        # NaN compte comme instable
    return vxs[bad].max() if bad.any() else 0.0


def accuracy():
    """Error after one horizon (30 steps) against RK4 with 200 sub-steps."""
    ref = dataclasses.replace(VEH, integrator="rk4", substeps=200)
    for vx, delta, a in [(1.2, 0.3, 0.0), (2.0, 0.3, 0.0), (5.0, 0.1, 0.0), (5.0, 0.3, 0.0), (7.0, 0.4, -2.0)]:
        u = np.array([a, delta])

        def run(veh):
            s, rs = np.array([0, 0, 0, vx, 0, 0.0]), []
            for _ in range(30):
                s = dyn.step_dynamic(s, u, DT, veh)
                rs.append(s[5])
            return s, np.array(rs)

        s_ref, r_ref = run(ref)
        cells = []
        for integ, n in VARIANTS:
            s, r = run(dataclasses.replace(VEH, integrator=integ, substeps=n))
            cells.append(f"{100 * np.hypot(*(s[:2] - s_ref[:2])):.2f} cm / {np.abs(r - r_ref).max():.3f}")
        print(f"({vx}, {delta}, {a}) | " + " | ".join(cells))


def timing():
    rng = np.random.default_rng(0)
    x = np.zeros((1024, 6)); x[:, 3] = 5.0
    u = rng.normal(size=(1024, 2)) * [0.5, 0.1]
    for integ, n in VARIANTS:
        veh = dataclasses.replace(VEH, integrator=integ, substeps=n)
        tic = time.perf_counter()
        for _ in range(300):
            dyn.step_dynamic(x, u, DT, veh)
        print(f"{integ} x{n}: {(time.perf_counter() - tic) / 300 * 1e3:.3f} ms per step, K=1024")


if __name__ == "__main__":
    eigenvalues()
    for integ, n in VARIANTS:
        print(f"{integ} x{n}: unstable below {unstable_below(integ, n, 'linear'):.2f} m/s (linear), "
              f"{unstable_below(integ, n, 'pacejka'):.2f} m/s (pacejka)")
    z = np.linspace(-4.0, 0.0, 400001)
    g = 1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24
    print(f"RK4 real-axis limit k dt = {-z[np.abs(g) <= 1].min():.3f}")
    print("case (vx, delta, a) | " + " | ".join(f"{i} x{n}" for i, n in VARIANTS))
    accuracy()
    timing()
```
