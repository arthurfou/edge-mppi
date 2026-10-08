# Étape 2 : réponses et prédictions

Règle (comme à l'étape 1) : écrire la réponse AVANT de lire la section
Solutions de `step_2.md`, puis noter l'écart.

## Partie A : pourquoi le modèle cinématique ne suffit plus

Paramètres utilisés (`config/mppi.yaml`) : L = 0,33 m, l_r = 0,18 m,
δ_max = 0,4 rad, μ = 1,0, g = 9,81 m/s².

### Q-A1 : vitesses limites

Limite d'adhérence : a_lat ≤ μ g = 9,81 m/s², donc v_max = √(μ g R).

| Virage | R (m) | v_max (m/s) |
|---|---|---|
| Plus serré de la piste (R_min) | 2,19 | 4,64 |
| Cercle à braquage max (CdG) | 0,80 | 2,80 |

Rayon du cercle à braquage max, au CdG :
β = atan(l_r/L · tan 0,4) = 0,227 rad (13,0°),
R = L / (cos β · tan δ) = 0,801 m (0,78 m à l'essieu arrière).

Virage serré à 7 m/s : a_lat = v²/R = 49 / 2,19 = 22,4 m/s² = 2,3 g.
Le modèle cinématique l'accepte sans broncher : rien dans ses équations ne
borne v²/R.

Étape 1 à v_ref = 3 : max a_lat = 6,3 m/s² = 0,64 μg, sous la limite.
Le modèle cinématique y était donc valable (les pneus réels auraient tenu).
Remarque : sur la ligne centrale, 3 m/s dans R = 2,19 ne donne que
4,1 m/s². Les 6,3 mesurés viennent de la trajectoire réelle, localement plus
serrée que la ligne centrale (coupe de virage, corrections de braquage).

### Q-A2 : pourquoi la rustine d'adhérence ne remplace pas un modèle

1. Elle mesure la mauvaise grandeur : a_lat = v² tan δ / L est
   l'accélération que **le modèle cinématique** prédit, pas celle que les
   pneus fournissent. Près de la limite, la voiture glisse et tourne moins
   que ce que dit la formule.
2. Elle ne dit rien de l'après-limite : un rollout qui dépasse μg paie une
   pénalité, mais sa trajectoire prédite reste sur le cercle. Le contrôleur
   ne voit pas la voiture partir à l'extérieur, donc ne peut pas planifier
   le rattrapage.
3. (bonus) Elle dégrade le réglage : le terme croît vite avec v, étale les
   coûts des rollouts et effondre l'ESS (journal de l'étape 1 : médiane − min
   de 0,7 à 57 à v_ref = 5).

### Q-A3 : prédiction (à vérifier sur la figure de la partie G)

Le contrôleur cinématique demande un virage à 2 g. Le véhicule dynamique ne
peut fournir qu'environ 1 g : les pneus avant saturent en premier
(sous-virage), la voiture tourne moins que prévu, β et α grandissent, et elle
part vers l'**extérieur** du virage. Le contrôleur, qui croit encore être sur
son cercle, braque davantage, ce qui n'apporte plus de force (pneu saturé).
Prédiction : sortie de piste à l'extérieur des virages les plus serrés pour
le cinématique pur ; le cinématique + rustine ralentit avant les virages et
passe parfois.

**Vérification (partie G) :**

### Point de contrôle A

- Le virage serré (R = 2,19 m) n'est plus faisable au-delà de ≈ 4,6 m/s.
  À v_ref = 7, il faut donc freiner nettement avant ce virage.
- β = −15° : la vitesse du CdG fait un angle de 15° avec l'axe de la voiture,
  vers la droite (β = atan(vy/vx) < 0, donc vy < 0). Dans un virage à gauche,
  le nez pointe vers l'intérieur et la voiture glisse vers l'extérieur :
  c'est un dérapage, au-delà du seuil |β| > 10° du critère de sortie.

---

## Partie H : prédictions puis mesures

Règle : remplir **Prédiction** et **Pourquoi** AVANT de lancer
`pixi run sweep Hx`. Ensuite seulement, remplir **Mesure** et **Écart**.
Un sens suffit (↑, ↓, =, « sort », « ne sort pas ») et une ligne de pourquoi.
Ne lire S-H dans `step_2.md` qu'après.

Rappels sur l'outil (`bench/sweep.py`) :

- chaque variante modifie **le contrôleur** (`with_vehicle`, `with_mppi`,
  `with_cost`, `kinematic`) et/ou **le véhicule simulé** (dict `plant`) ;
  `both(...)` modifie les deux (modèle parfait) ;
- la colonne `lap` vaut « yes » même après une sortie : le critère qui compte
  est `off = 0` ;
- pour un résultat « sort / ne sort pas », relancer sur 5 graines :
  `pixi run sweep Hx --seeds 5` (tours propres = tour fini et `off = 0`).

### Référence (config de 7.1, graine 0, déjà mesurée)

`v_ref = 7`, λ = 3, σ = [0,5 ; 0,1], T = 30, K = 1024, Pacejka C = 1,5, RK4.

| t (s) | off | max abs(d) | ESS méd | ESS p5 | gigue | v moy | β max (°) |
|---|---|---|---|---|---|---|---|
| 9,02 | 0 | 0,54 | 484 | 13 | 0,023 | 5,46 | 16,4 |

Sur 5 graines : 5/5 tours propres, 8,68 à 9,18 s.

Rappel de la partie G (`pixi run sweep lambda-scan`) : λ = 0,3 et 1 passent
à 2 cm du bord avec ESS p5 = 1 ; λ = 10 est sûr mais 1,5 s plus lent.

---

## H1 : λ ∈ {1, 10}

`pixi run sweep H1` (puis `--seeds 5`)

**Question.** Temps au tour, ESS, gigue, `max |d|` ? Lequel choisirais-tu sur
un vrai robot ? Relie l'ESS et la gigue à la théorie §4.1 et §4.2
(`σ/√ESS`) : calcule la gigue attendue à ESS = 13 et à ESS = 343.

**Prédiction :**

**Pourquoi :**

**Mesure :**

**Écart :**

---

## H2 : pneus linéaires, puis tanh, dans le modèle ET dans la réalité

`pixi run sweep H2` (puis `--seeds 5`)

**Question.** La voiture va-t-elle plus vite avec des pneus linéaires ? Que
devient `a_lat` max ? tanh ≠ Pacejka ? Un modèle linéaire autorise-t-il des
accélérations latérales supérieures à μg ? Si oui, la voiture simulée
sort-elle pour autant ? (Ici, la réalité **et** le modèle sont linéaires.)

**Prédiction :**

**Pourquoi :**

**Mesure :**

**Écart :**

---

## H3 : rollouts linéaires (puis tanh), réalité Pacejka

`pixi run sweep H3 --seeds 5`

**Question.** Le contrôleur sort-il de la piste ? Quel est le mode d'échec
de la théorie §7.9 qui correspond ?

**Prédiction :**

**Pourquoi :**

**Mesure :**

**Écart :**

---

## H4 : réalité à μ = 0,8, contrôleur à μ = 1 ; puis les deux à 0,8

`pixi run sweep H4 --seeds 5`

**Question.** Effet d'une surestimation de 25 % de l'adhérence ? C'est la
pluie : que dit la théorie §6.7 de l'erreur sur μ ?

**Prédiction :**

**Pourquoi :**

**Mesure :**

**Écart :**

---

## H5 : Euler ×1 dans les deux ; puis rollouts Euler, réalité RK4

`pixi run sweep H5`

**Question.** Le tour change-t-il ? Le temps de calcul ? Si la boucle fermée
ne voit pas la différence, est-ce que l'intégrateur n'a pas d'importance ?
(Q-E3.)

Note : `sweep` ne mesure pas le temps d'itération, et les runs en parallèle
le faussent. Pour le temps, comparer `pixi run sim` avec
`integrator: euler` puis `rk4` dans le YAML.

**Prédiction :**

**Pourquoi :**

**Mesure :**

**Écart :**

---

## H6 : T = 15 ; T = 50 ; T = 50 avec λ = 6

`pixi run sweep H6`

**Question.** Horizon court à 7 m/s ? Horizon long : ESS ? Calcule la
distance vue à 7 m/s avec T = 15, et la distance de freinage de 7 à 4,6 m/s
(vitesse max du virage serré) à 4 m/s² (théorie §4.3). Pour T = 50 : pourquoi
l'ESS baisse-t-elle, et pourquoi remonter λ ?

**Prédiction :**

**Pourquoi :**

**Mesure :**

**Écart :**

---

## H7 : σ_δ = 0,05 ; σ_δ = 0,2

`pixi run sweep H7`

**Question.** Gigue, ESS, temps au tour ? Un σ_δ plus grand explore plus :
pourquoi pourrait-il **ralentir** la voiture ?

**Prédiction :**

**Pourquoi :**

**Mesure :**

**Écart :**

---

## H8 : `v_ref = 9` ; puis λ = 10 ; puis λ = 10 et T = 50

`pixi run sweep H8`

**Question.** Aller plus vite en demandant plus ? Pourquoi demander 9 m/s ne
donne-t-il pas un tour plus rapide à λ = 10 ?

**Prédiction :**

**Pourquoi :**

**Mesure :**

**Écart :**

---

## H9 : rollouts cinématiques ; + adhérence ; + adhérence et λ = 10

`pixi run sweep H9 --seeds 5`

**Question.** Retrouve et prolonge la figure de la partie G. Pourquoi la
rustine d'adhérence est-elle **plus** nuisible à `v_ref = 7` qu'à
`v_ref = 3` ?

**Prédiction :**

**Pourquoi :**

**Mesure :**

**Écart :**
