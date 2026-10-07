# Étape 1, partie G : prédictions puis mesures

Règle : remplir la colonne **Prédiction** et la ligne **Pourquoi** AVANT de lancer
`pixi run sweep Gx`. Ensuite seulement, remplir **Mesure** et **Écart**.

Pour les prédictions, un sens suffit (↑, ↓, =, « ≈ 1 », « non ») : pas besoin
de chiffres exacts. C'est le raisonnement qui compte.

## Référence (config de base, déjà mesurée)

λ = 0,3, σ = [0,5, 0,1], γ = 0, T = 30, v_ref = 3, w_speed = 1, seed = 0.

| Tour | t (s) | ESS méd | ESS p5 | Gigue (rad/pas) | max abs(d) (m) | v moy (m/s) |
|---|---|---|---|---|---|---|
| oui | 16,12 | 444 | 104 | 0,037 | 0,06 | 2,99 |

**Le fil rouge de toute la partie G.** Les poids valent `w = exp(−(S − ρ)/λ)`.
Ce qui compte n'est ni λ seul ni S seul, mais le **rapport entre la dispersion
des coûts et λ**. Si les coûts des rollouts diffèrent de 0,7 et que λ = 0,3,
le rapport vaut environ 2 : beaucoup de rollouts gardent un poids notable, et
l'ESS est élevée. Si les coûts diffèrent de 57 avec λ = 0,3, le rapport vaut
190, `exp(−190) ≈ 10⁻⁸³`, et seul le meilleur rollout compte (ESS = 1). Presque
toutes les expériences ci-dessous se lisent avec cette grille.

---

## G1 : λ ∈ {1 ; 0,3 ; 0,1 ; 0,03}

**Pourquoi (ta justification) :**

| λ | | Tour | ESS méd | ESS p5 | Gigue | max abs(d) |
|---|---|---|---|---|---|---|
| 1 | Prédiction | | | | | |
| 1 | Mesure | oui | 777 | 337 | 0,027 | 0,14 |
| 0,1 | Prédiction | | | | | |
| 0,1 | Mesure | oui | 140 | 53 | 0,049 | 0,06 |
| 0,03 | Prédiction | | | | | |
| 0,03 | Mesure | oui | 9 | 1 | 0,109 | 0,05 |

**Quel λ choisir, et pourquoi :**

**Écart / ce que j'ai appris :**
$\lambda$ trop bas sélectionne moins de bonnes trajectoires. Cela fait qu'on utilise pas trop les autres rollouts. On se retrouve a avoir plus de jitter sans conséquence aberrante sur le temps de tour (tous finissent).

> **Correction.** C'est juste, avec une nuance de vocabulaire : un λ bas ne
> sélectionne pas *moins de bonnes* trajectoires, il en sélectionne *moins tout
> court*. Il concentre le poids sur les toutes meilleures, et à λ = 0,03 sur
> une seule (ESS p5 = 1).
>
> **Pourquoi la gigue monte.** Quand un seul rollout porte le poids,
> `U ← U + eps_meilleur` : la commande recopie le bruit de ce rollout, qui est
> du bruit blanc, différent à chaque pas. Avec beaucoup de rollouts pondérés,
> leurs bruits se moyennent et s'annulent en partie. C'est la loi des grands
> nombres : la moyenne de N bruits indépendants a un écart-type divisé par √N.
>
> **Ce que tu n'as pas noté : l'autre côté du compromis.** À λ = 1, la gigue
> est la plus faible, mais `max abs(d)` passe de 0,06 à 0,14 m (× 2,3). Si λ
> est trop grand, même les mauvais rollouts gardent du poids, et la moyenne est
> tirée vers eux : le suivi se dégrade. À la limite λ → ∞, tous les poids sont
> égaux, et la commande ne bouge plus (moyenne de bruits ≈ 0).
>
> | λ | Effet |
> |---|---|
> | trop petit | ESS ≈ 1, la commande recopie le bruit → forte gigue |
> | trop grand | ESS ≈ K, la commande est la moyenne de tout → suivi mou |
>
> **Choix : λ = 0,3**, avec une ESS ≈ 40 % de K, un bon suivi et une gigue
> modérée. La règle de la théorie §4.6 est une ESS minimale (p5) entre 1 % et
> 10 % de K, et une ESS médiane de quelques centaines à K = 1024.
>
> **Pourquoi le temps au tour ne bouge pas :** à 3 m/s, la piste est facile, et
> même un contrôleur bruité y arrive. Cette tâche ne permet pas de discriminer
> les réglages par le temps au tour : c'est pour ça qu'on regarde l'ESS et la
> gigue.

---

## G2 : w_speed = 0

**Pourquoi :**

| | Tour (sinon distance) | ESS méd | Gigue | v moy |
|---|---|---|---|---|
| Prédiction | | | | |
| Mesure | **non, 24,2 m en 60 s** | 1023 | 0,018 | 0,40 |

**Écart / ce que j'ai appris :**
w_speed est le poids du terme de vitesse dans stage_cost, la fonction de coût. Aucun coût => On va avoir une vitesse qui ne grossit pas forcément et ne pousse pas le robot à finir

> **Correction.** La conclusion est la bonne (la voiture ne finit pas), mais
> le mécanisme mérite d'être précisé, parce qu'il reste un terme qui pousse à
> avancer : la **progression** du coût terminal. Pourquoi ne suffit-il pas ?
>
> 1. **À l'arrêt, le coût est plat.** Avec σ_a = 0,5 m/s² sur 0,6 s, un
>    rollout parti de v ≈ 0 avance de quelques centimètres au plus. La
>    progression vaut `−10 × 0,05 / 1,8 ≈ −0,3` pour le meilleur et ≈ 0 pour
>    les autres. Les coûts sont presque égaux, donc tous les poids sont égaux
>    (ESS = 1023 ≈ K), et la mise à jour est la moyenne de 1024 bruits ≈ 0. La
>    commande ne bouge plus.
> 2. **Ce n'est pas un problème de λ.** Une ESS ≈ K ressemble au symptôme d'un
>    λ trop grand, mais ici c'est le **coût** qui n'a plus d'information.
>    Baisser λ n'y changerait presque rien. Il faut ajouter un terme qui
>    distingue les rollouts, et c'est le rôle du terme de vitesse : à l'arrêt,
>    il coûte 9 par pas, et le moindre gain de vitesse se voit.
>
> C'est exactement S-D1 : la voiture a démarré, parcouru 24 m, puis s'est
> arrêtée dans un virage où elle avait dû freiner, et n'est jamais repartie
> (v moy = 0,4 m/s sur 60 s). Le terme de vitesse rend le coût informatif
> **partout**, y compris à l'arrêt.

---

## G3 : v_ref = 5, puis 7 (λ = 0,3 inchangé)

**Pourquoi :**

| v_ref | | Tour | t (s) | ESS méd | ESS p5 | Gigue | max abs(d) |
|---|---|---|---|---|---|---|---|
| 5 | Prédiction | | | | | | |
| 5 | Mesure | oui | 9,56 | **1** | 1 | 0,116 | 0,25 |
| 7 | Prédiction | | | | | | |
| 7 | Mesure | oui | 7,56 | **1** | 1 | 0,125 | 0,56 |

**Écart / ce que j'ai appris :**
Bah on finit plus vite vu que la vitesse de référence qu'on doit atteindre est plus grande. C'est marrant parce que on l'atteint pour v_ref = 5 mais pas pour v_ref = 7. Malheureusement on tombe à ESS = 1.

Je comprends pas pour G3 pourquoi il y a un ESS de 1. C'est normal qu'il aille plus vite vu qu'on augmente la vitesse de référence qu'on veut lui donner mais pq l'ess devient 1 ?

> **Correction : pourquoi l'ESS tombe à 1.** λ n'a pas changé, mais
> **l'échelle des coûts**, si. J'ai mesuré, en régime établi, l'écart entre le
> coût médian et le coût minimal des 1024 rollouts, puis décomposé terme par
> terme :
>
> | | (S médian − S min) | ÷ λ | Terme dominant de la dispersion |
> |---|---|---|---|
> | v_ref = 3 | 0,7 | 2,4 | latéral (0,37) et progression (0,28) |
> | v_ref = 5 | **57** | **190** | **adhérence (35)**, latéral (2,8) |
>
> C'est le **terme d'adhérence** qui explose. Il pénalise
> `a_lat = v² · tan δ / L` au-delà de μg = 9,81 m/s² :
> - à 3 m/s, la limite est atteinte pour δ ≈ 0,35 rad. Avec un bruit
>   σ_δ = 0,1, presque aucun rollout n'y arrive, et le terme vaut 0 partout ;
> - à 5 m/s, la limite tombe à δ ≈ 0,13 rad. Elle est **dans le bruit** : une
>   bonne partie des rollouts la dépassent, et paient `10 × (ratio − 1)²` à
>   chaque pas. Les coûts s'étalent sur des dizaines d'unités.
>
> Avec une dispersion de 57 et λ = 0,3, le 2e meilleur rollout pèse déjà
> `exp(−quelques unités / 0,3)`, ce qui est négligeable. Il ne reste qu'un
> rollout : ESS = 1, et la gigue triple (0,116 contre 0,037), pour la même
> raison qu'à λ = 0,03 en G1.
>
> **La leçon (la plus importante de la partie G) :** λ n'a pas de valeur
> « bonne » dans l'absolu. **Il se règle par rapport à la dispersion des
> coûts**, et doit être revérifié dès qu'on change un poids, une consigne ou
> le bruit (théorie §4.1). À v_ref = 5, λ = 0,3 est l'équivalent de λ = 0,002 à
> v_ref = 3.
>
> **Sur « on l'atteint pour v_ref = 5 mais pas pour 7 »** : à v_ref = 7, la
> vitesse moyenne n'est que de 6,18 m/s, parce que la voiture **doit** ralentir
> dans les virages (l'adhérence l'y oblige) et que le départ arrêté compte dans
> la moyenne. Elle frôle aussi la sortie : `max abs(d) = 0,56` pour une limite
> de 0,65.
>
> **Bonus, mesuré.** À v_ref = 5 :
>
> | λ | ESS méd | ESS p5 | Gigue | max abs(d) | t (s) |
> |---|---|---|---|---|---|
> | 0,3 | 1 | 1 | 0,116 | 0,25 | 9,56 |
> | 1 | 19 | 3 | 0,045 | 0,17 | 10,62 |
> | **3** | **238** | **126** | **0,018** | 0,18 | 11,18 |
> | 10 | 562 | 406 | 0,012 | 0,25 | 11,72 |
> | 30 | 932 | 524 | 0,014 | 0,34 | 13,40 |
>
> **λ ≈ 2** remet l'ESS médiane vers 100, et λ = 3 donne un excellent
> compromis. On retrouve le même rapport dispersion/λ qu'à v_ref = 3. Note
> aussi que la voiture est plus lente avec un λ mieux réglé (11,2 s contre
> 9,6 s) : à ESS = 1, elle roulait plus vite qu'elle ne le « voulait »
> (voir G6).

---

## G4 : σ_δ = 0,02, puis 0,3 (σ_a = 0,5 inchangé)

**Pourquoi :**

| σ_δ | | ESS méd | ESS p5 | Gigue | max abs(d) |
|---|---|---|---|---|---|
| 0,02 | Prédiction | | | | |
| 0,02 | Mesure | 790 | **22** | 0,009 | 0,13 |
| 0,3 | Prédiction | | | | |
| 0,3 | Mesure | 66 | 6 | 0,057 | 0,05 |

**Écart / ce que j'ai appris :**
0,02 : peu de jitter et ESS med plus grand car on a besoin de plus de trajectoires pour trouver le bon chemin vu qu'on s'éloigne peu de la ligne droite.

> **Correction.** Les observations sont justes (peu de gigue, ESS médiane plus
> grande), mais la cause est inversée. L'ESS n'est pas une quantité que MPPI
> « choisit » parce qu'il en aurait besoin : c'est une **conséquence** de la
> dispersion des coûts (voir le fil rouge).
>
> **σ_δ = 0,02, faisceau étroit :**
> - les rollouts braquent presque tous pareil, donc ils suivent presque la même
>   trajectoire, ont des coûts presque égaux, et des poids presque égaux :
>   l'ESS médiane monte (790) ;
> - la commande moyenne est faite de petits bruits, donc la gigue est très
>   faible (0,009) ;
> - **mais l'ESS p5 s'effondre (22)** : dans les virages, la séquence nominale
>   ne braque pas assez, et presque aucun rollout ne braque assez **non plus**,
>   car le bruit est trop petit pour explorer. Les quelques rollouts qui
>   s'en approchent prennent tout le poids. Le suivi se dégrade
>   (`max abs(d)` = 0,13 contre 0,06) : la voiture prend les virages en retard.
>
> **σ_δ = 0,3, faisceau large :**
> - les rollouts explorent beaucoup, et beaucoup **sortent de piste** (coût de
>   100+) ou dépassent l'adhérence. La dispersion des coûts explose, donc
>   l'ESS chute (66, et 6 au p5) ;
> - la commande moyenne contient de gros bruits, donc la gigue monte (0,057).
>
> **La leçon :** σ règle **l'exploration**. Trop petit, on ne trouve pas les
> bonnes commandes quand elles sont loin de U. Trop grand, on gaspille la
> plupart des rollouts hors piste. On le choisit **en regardant le faisceau**
> (théorie §4.2), puis on revérifie λ, puisque modifier σ modifie la
> dispersion des coûts.

---

## G5 : γ = λ (= 0,3)

**Pourquoi :**

| | t (s) | ESS méd | Gigue | v moy |
|---|---|---|---|---|
| Prédiction | | | | |
| Mesure | 17,04 | 423 | **0,008** | 2,82 |

**Écart / ce que j'ai appris :**

> **Correction (pas de note de ta part).** Le terme `γ Σ Uᵀ Σ⁻¹ ε` pénalise
> les perturbations qui **éloignent encore la commande de zéro**. Avec γ = λ,
> c'est la valeur exacte de la théorie, et le contrôleur optimise en fait
> `S + (λ/2) Σ Uᵀ Σ⁻¹ U` : il est tiré vers `a = 0` et `δ = 0`.
>
> - **Gigue divisée par 4,6** (0,008 contre 0,037) : les variations brusques de
>   δ sont pénalisées. C'est le meilleur résultat de toute la partie G sur ce
>   critère.
> - **ESS p5 bien meilleure** (341 contre 104) : le terme lisse aussi le
>   paysage de coût.
> - **Mais la voiture est plus lente** (17,04 s et 2,82 m/s contre 16,12 s et
>   2,99 m/s) : tenir 3 m/s demande une accélération non nulle au démarrage et
>   en sortie de virage, et le terme la freine. Dans un long virage, il
>   résisterait aussi au braquage nécessaire, et la voiture élargirait ses
>   trajectoires (`max abs(d)` = 0,10 contre 0,06).
>
> **La leçon :** la théorie dit γ = λ, mais la pratique prend γ ≪ λ (nav2 :
> λ = 0,3, γ = 0,015, théorie §4.5). Garder γ = 0 à l'étape 1 est défendable.
> Si la gigue pose problème sur la vraie voiture, un petit γ est un levier
> efficace.

---

## G6 : T = 10, v_ref = 5

**Pourquoi :**

| | Tour | t (s) | v moy | max abs(d) | Pas hors piste |
|---|---|---|---|---|---|
| Prédiction | | | | | |
| Mesure | oui | 6,92 | **7,40** | **0,65** | 0 |

**Écart / ce que j'ai appris :**
On ne sélectionne plus qu'une trajectoire et on va super vite mais je comprends pas pourquoi

> **Correction.** Deux effets se combinent. Je les ai vérifiés par la mesure,
> parce que l'explication de S-G (« la normalisation de la progression donne
> un poids énorme ») **ne tient pas** sur mes mesures : la progression est le
> terme qui pèse le moins dans la décision.
>
> **1. ESS = 1, pour la même raison qu'en G3** : à v_ref = 5, le terme
> d'adhérence étale les coûts (dispersion ≈ 104, soit 350 fois λ).
>
> **2. Avec un horizon de 0,2 s, l'accélération ne change presque pas le
> coût.** Sur 10 pas, un bruit σ_a = 0,5 m/s² modifie la vitesse d'environ
> 0,1 m/s au plus. L'effet sur le terme de vitesse est de l'ordre de 1, noyé
> dans la dispersion de 104 due au braquage. Le « meilleur » rollout est donc
> choisi presque uniquement sur son braquage, et son accélération est
> **quasiment au hasard**.
>
> Ce que ça donne dans le temps (trace mesurée de `U_a`, l'accélération
> nominale) :
>
> | t (s) | v (m/s) | U_a (m/s²) | |
> |---|---|---|---|
> | 1 | 3,7 | ≈ +3,8 | normal : on accélère vers v_ref |
> | 2 | 5,8 | ≈ +1 à +2 | v_ref dépassée, il faudrait freiner |
> | 4 | 8,4 | ≈ +1 à +4 | toujours positif |
> | 6 | 11,5 | ≈ +3 à +4 | à la butée a_max = 4 |
> | 6,9 | ≈ 12,4 | | fin du tour |
>
> Pendant le démarrage, la commande nominale a pris une forte accélération
> (+3,8 m/s², légitime). Ensuite, il faudrait la ramener à 0, soit un
> changement de 3,8 = 8 σ_a. MPPI ne peut le faire que par petites
> corrections cohérentes d'une itération à l'autre. Avec ESS = 1 et un signal
> noyé, ces corrections sont une marche au hasard : la mesure donne un
> freinage moyen de −0,04 m/s² par itération seulement, alors que **tous**
> les termes du coût préfèrent freiner. La voiture **ne cesse jamais
> d'accélérer** : 7,4 m/s de moyenne et 12,4 m/s à la fin, pour une consigne
> de 5.
>
> **Pourquoi elle ne sort pas quand même** (`max abs(d)` = 0,65 = d_max pile) :
> le terme de sortie de piste, lui, est assez fort (100+) pour dominer la
> sélection. Les rollouts qui sortent sont éliminés, et la voiture rase le
> bord sans le franchir. Avec un tour de plus, elle finirait par sortir.
>
> **La leçon :** un horizon court rend le contrôleur **myope**. Il ne voit pas
> les conséquences de ses commandes lentes, comme l'accélération (qui agit
> sur la vitesse, qui agit sur la position), ni les virages à venir (théorie
> §4.3). L'horizon doit couvrir au moins la distance de freinage :
> à 5 m/s et 4 m/s², il faut 1,25 s, soit T ≥ 63 pas à dt = 0,02. Même T = 30
> (0,6 s) est juste : à v_ref = 5 et T = 30, la voiture atteint 7,4 m/s dans
> la dernière ligne droite, par le même mécanisme, plus lentement.

---

## G7 : seed = 1

**Pourquoi :**

| | Le comportement change-t-il ? | t (s) | ESS méd | Gigue |
|---|---|---|---|---|
| Prédiction | | | | |
| Mesure | presque pas | 16,10 | 439 | 0,036 |

**Écart / ce que j'ai appris :**
Toujours le même résultat et c'est attendu car on a toujours la même seed.

> **Correction : attention, c'est l'inverse.** La graine **a changé** : la
> référence utilise `seed = 0`, cette variante `seed = 1`. Le bruit tiré est
> donc **complètement différent** à chaque pas, et les 1024 rollouts de chaque
> itération ne sont plus les mêmes.
>
> Le résultat est quand même presque identique (16,10 s contre 16,12 s, ESS
> 439 contre 444, gigue 0,036 contre 0,037), et c'est **ça** l'information :
> le comportement ne tient pas à un tirage chanceux. Le réglage est
> **robuste** au hasard.
>
> Si changer la graine changeait beaucoup le résultat, ça voudrait dire que la
> performance dépend de la chance, et qu'aucune autre mesure du tableau ne
> serait fiable : chaque chiffre serait un tirage parmi d'autres. C'est pour
> ça qu'on fait cette expérience **en dernier** : elle valide les six autres.
>
> Seule différence visible : l'ESS p5 (217 contre 104). Le 5e centile capture
> les pires instants, qui sont rares et donc plus sensibles au tirage. Les
> comparaisons d'ESS p5 entre variantes ne sont significatives que si l'écart
> est grand.
>
> **Ce qui serait identique à coup sûr :** relancer avec la **même** graine.
> Le résultat serait reproductible au bit près. C'est ce qu'on exige pour la
> comparaison NumPy/CUDA de l'étape 3.

---

## Résultats bruts (`pixi run sweep`)

| variant             | lap         | t (s) | ESS med | ESS p5 | jitter | max abs(d) | off | v mean |
| ------------------- | ----------- | ----- | ------- | ------ | ------ | ---------- | --- | ------ |
| reference           | yes         | 16.12 | 444     | 104    | 0.037  | 0.06       | 0   | 2.99   |
| G1 lambda=1.0       | yes         | 16.44 | 777     | 337    | 0.027  | 0.14       | 0   | 2.93   |
| G1 lambda=0.3       | yes         | 16.12 | 444     | 104    | 0.037  | 0.06       | 0   | 2.99   |
| G1 lambda=0.1       | yes         | 16.24 | 140     | 53     | 0.049  | 0.06       | 0   | 2.97   |
| G1 lambda=0.03      | yes         | 16.32 | 9       | 1      | 0.109  | 0.05       | 0   | 2.96   |
| G2 w_speed=0        | NO (24.2 m) | 60.00 | 1023    | 422    | 0.018  | 0.20       | 0   | 0.40   |
| G3 v_ref=5.0        | yes         | 9.56  | 1       | 1      | 0.116  | 0.25       | 0   | 5.01   |
| G3 v_ref=7.0        | yes         | 7.56  | 1       | 1      | 0.125  | 0.56       | 0   | 6.18   |
| G4 sigma_delta=0.02 | yes         | 16.14 | 790     | 22     | 0.009  | 0.13       | 0   | 2.99   |
| G4 sigma_delta=0.3  | yes         | 16.80 | 66      | 6      | 0.057  | 0.05       | 0   | 2.87   |
| G5 gamma=lambda     | yes         | 17.04 | 423     | 341    | 0.008  | 0.10       | 0   | 2.82   |
| G6 T=10, v_ref=5    | yes         | 6.92  | 1       | 1      | 0.055  | 0.65       | 0   | 7.40   |
| G7 seed=1           | yes         | 16.10 | 439     | 217    | 0.036  | 0.09       | 0   | 3.00   |

## Synthèse : que fait chaque bouton

| Paramètre | Trop petit | Trop grand | Comment le régler |
|---|---|---|---|
| λ | ESS ≈ 1, la commande recopie le bruit, forte gigue | ESS ≈ K, la moyenne est molle, mauvais suivi | ESS p5 entre 1 et 10 % de K. **Relatif à la dispersion des coûts** : à revérifier après tout changement |
| σ (bruit) | exploration étroite, retard dans les virages (ESS p5 bas) | rollouts gaspillés hors piste, forte gigue | en regardant le faisceau, puis revérifier λ |
| γ | (0 : aucun effet visible ici) | commande tirée vers 0 : douce mais lente, virages élargis | γ ≪ λ, ou 0 |
| T (horizon) | myope : ne voit ni les virages ni l'effet de l'accélération | coûteux en calcul (étape 3) | couvrir au moins la distance de freinage |
| w_speed | coût plat à l'arrêt, la voiture s'arrête | — | assez fort pour que le coût soit informatif partout |
| seed | — | — | ne doit rien changer : c'est le test de robustesse |
