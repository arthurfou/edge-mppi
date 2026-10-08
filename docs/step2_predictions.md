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
