# MPPI : commande prédictive par intégrale de chemin

Ce document présente MPPI (*Model Predictive Path Integral control*) à partir de zéro, en vue de son implémentation pour un véhicule de type voiture, d'abord en NumPy puis sous forme de kernel CUDA sur Jetson Orin Nano. Il suppose une bonne maîtrise des probabilités, de l'optimisation et de l'analyse, et aucune connaissance en théorie du contrôle : chaque terme du domaine est défini avant d'être employé.

Les exemples numériques utilisent les paramètres cibles du projet : $K = 8192$ trajectoires échantillonnées, un horizon de $T = 50$ pas et un pas de temps $\Delta t = 20$ ms, soit une seconde de prédiction et une fréquence de contrôle de 50 Hz. Les paramètres du véhicule sont ceux de `config/mppi.yaml` (gabarit F1TENTH), et les conventions de repère suivent l'en-tête de `python/mppi/dynamics.py`.

## Table des matières

- [Notations](#notations)
- [1. Situer MPPI](#1-situer-mppi)
  - [1.1 État, commande, modèle](#11-état-commande-modèle)
  - [1.2 Contrôleur, boucle ouverte, boucle fermée](#12-contrôleur-boucle-ouverte-boucle-fermée)
  - [1.3 Le problème de contrôle optimal](#13-le-problème-de-contrôle-optimal)
  - [1.4 Horizon glissant](#14-horizon-glissant)
  - [1.5 Contraintes non holonomes](#15-contraintes-non-holonomes)
  - [1.6 Quatre familles de contrôleurs](#16-quatre-familles-de-contrôleurs)
  - [1.7 Origine de MPPI](#17-origine-de-mppi)
- [2. L'algorithme](#2-lalgorithme)
  - [2.1 Ce que l'algorithme conserve](#21-ce-que-lalgorithme-conserve)
  - [2.2 Une itération pas à pas](#22-une-itération-pas-à-pas)
  - [2.3 Pseudocode](#23-pseudocode)
  - [2.4 Indépendants entre eux, séquentiels dans le temps](#24-indépendants-entre-eux-séquentiels-dans-le-temps)
  - [2.5 Conséquences pour le kernel CUDA](#25-conséquences-pour-le-kernel-cuda)
- [3. Les fondements théoriques](#3-les-fondements-théoriques)
  - [3.1 Le problème stochastique](#31-le-problème-stochastique)
  - [3.2 L'équation de Hamilton-Jacobi-Bellman](#32-léquation-de-hamilton-jacobi-bellman)
  - [3.3 Transformation logarithmique et formule de Feynman-Kac](#33-transformation-logarithmique-et-formule-de-feynman-kac)
  - [3.4 Lecture informationnelle : énergie libre et divergence KL](#34-lecture-informationnelle--énergie-libre-et-divergence-kl)
  - [3.5 De la distribution optimale à une mise à jour calculable](#35-de-la-distribution-optimale-à-une-mise-à-jour-calculable)
  - [3.6 Pourquoi l'exponentielle](#36-pourquoi-lexponentielle)
- [4. Les paramètres et leur effet](#4-les-paramètres-et-leur-effet)
  - [4.1 La température](#41-la-température)
  - [4.2 La covariance du bruit](#42-la-covariance-du-bruit)
  - [4.3 L'horizon](#43-lhorizon)
  - [4.4 Le nombre d'échantillons](#44-le-nombre-déchantillons)
  - [4.5 Les poids du coût et le terme de correction](#45-les-poids-du-coût-et-le-terme-de-correction)
  - [4.6 Ordre de réglage](#46-ordre-de-réglage)
- [5. La fonction de coût](#5-la-fonction-de-coût)
  - [5.1 Structure](#51-structure)
  - [5.2 Les termes usuels du suivi de piste](#52-les-termes-usuels-du-suivi-de-piste)
  - [5.3 Normaliser les termes](#53-normaliser-les-termes)
  - [5.4 Les pièges classiques](#54-les-pièges-classiques)
- [6. Les modèles de dynamique véhicule](#6-les-modèles-de-dynamique-véhicule)
  - [6.1 Repères et grandeurs](#61-repères-et-grandeurs)
  - [6.2 Le bicycle cinématique](#62-le-bicycle-cinématique)
  - [6.3 Le bicycle dynamique](#63-le-bicycle-dynamique)
  - [6.4 Modèles de pneu et saturation d'adhérence](#64-modèles-de-pneu-et-saturation-dadhérence)
  - [6.5 Le problème de la basse vitesse](#65-le-problème-de-la-basse-vitesse)
  - [6.6 Ce que les deux modèles ignorent](#66-ce-que-les-deux-modèles-ignorent)
  - [6.7 Pourquoi MPPI brille quand les pneus décrochent](#67-pourquoi-mppi-brille-quand-les-pneus-décrochent)
- [7. Les modes d'échec](#7-les-modes-déchec)
- [8. Variantes et état de l'art](#8-variantes-et-état-de-lart)
- [9. Ressources](#9-ressources)
- [Annexe : chiffres de référence](#annexe--chiffres-de-référence)

---

## Notations

| Symbole | Signification | Valeur dans le projet |
|---|---|---|
| $x_t \in \mathbb{R}^n$ | état à l'instant $t$ | $n = 4$ (cinématique) ou $n = 6$ (dynamique) |
| $u_t \in \mathbb{R}^m$ | commande à l'instant $t$ | $m = 2$, $u = (a, \delta)$ |
| $F$ | modèle discret, $x_{t+1} = F(x_t, u_t)$ | `step(state, control, dt)` |
| $\Delta t$ | pas de temps | 0,02 s |
| $T$ | horizon, en nombre de pas | 50 |
| $K$ | nombre de trajectoires échantillonnées | 8192 |
| $U = (u_0, \dots, u_{T-1})$ | séquence de commande nominale | tableau $T \times m$ |
| $\varepsilon^k_t$ | bruit de la trajectoire $k$ au pas $t$ | $\mathcal{N}(0, \Sigma)$ |
| $\Sigma$ | covariance du bruit | $\mathrm{diag}(0{,}5^2,\ 0{,}1^2)$ |
| $S^k$ | coût total de la trajectoire $k$ | scalaire |
| $\lambda$ | température | 1,0 dans le fichier actuel |
| $w^k$ | poids normalisé de la trajectoire $k$ | $\sum_k w^k = 1$ |

L'indice bas $t$ désigne toujours le temps, l'exposant $k$ l'échantillon. Les positions dans le plan sont notées $p_x, p_y$ pour ne pas les confondre avec l'état $x$ ; le code les appelle `x` et `y`.

---

## 1. Situer MPPI

### 1.1 État, commande, modèle

Un système que l'on veut piloter est décrit à chaque instant par son **état** : le plus petit ensemble de grandeurs qui, connu à un instant donné et complété par les commandes futures, suffit à prédire toute l'évolution future. Pour la voiture décrite par le modèle cinématique, l'état est $x = (p_x, p_y, \psi, v)$ : la position dans le plan, le **cap** $\psi$ (angle entre l'axe longitudinal du véhicule et l'axe $x$ du repère monde) et la vitesse. Le choix de l'état relève de la modélisation : le modèle dynamique de la section 6 ajoute la vitesse latérale et la vitesse de lacet, parce qu'une voiture qui glisse garde la mémoire de sa rotation et de sa dérive, ce qu'un état à quatre composantes ne peut pas représenter.

La **commande** (on dit aussi *entrée*, ou *action* en apprentissage par renforcement) est la grandeur que le contrôleur choisit. Ici $u = (a, \delta)$, l'accélération longitudinale et l'**angle de braquage** des roues avant. Les **actionneurs** sont les organes physiques qui réalisent la commande : le moteur et son variateur pour $a$, le servomoteur de direction pour $\delta$. Une commande est toujours bornée par ce que l'actionneur peut produire ; le fichier de configuration impose $a \in [-4, 4]$ m/s² et $\delta \in [-0{,}4 ;\ 0{,}4]$ rad, soit environ 23 degrés.

Le **modèle de dynamique** décrit comment l'état évolue sous l'effet de la commande. La physique le donne en temps continu, sous forme d'équation différentielle $\dot{x} = f(x, u)$. Un calculateur travaille en temps discret : la commande est maintenue constante pendant chaque intervalle $\Delta t$ (on parle de **bloqueur d'ordre zéro**, *zero-order hold*), et l'on intègre l'équation sur cet intervalle pour obtenir le **modèle discret** $x_{t+1} = F(x_t, u_t)$. Le schéma le plus simple est celui d'Euler explicite, $F(x, u) = x + \Delta t\, f(x, u)$ ; Runge-Kutta d'ordre 4 est plus précis pour quatre évaluations de $f$. Dans le dépôt, $F$ est la fonction `step(state, control, dt)`.

Une **trajectoire** est la suite des états et des commandes sur une fenêtre de temps. Simuler le modèle à partir d'un état initial sous une séquence de commandes donnée s'appelle un **rollout** ; MPPI en exécute $K = 8192$ à chaque itération. Enfin, l'état réel n'est jamais mesuré directement : des capteurs (odométrie, centrale inertielle, lidar) alimentent un **estimateur d'état** (filtre de Kalman, localisation) qui fournit une estimation $\hat{x}$. Ce document suppose cette estimation disponible.

### 1.2 Contrôleur, boucle ouverte, boucle fermée

Un **contrôleur** (ou *loi de commande* ; en apprentissage, une *politique*) est un algorithme qui produit la commande à partir de l'information disponible. La distinction la plus importante du domaine sépare deux façons de s'en servir.

En **boucle ouverte**, on calcule à l'avance toute la séquence de commandes et on l'applique sans regarder ce qui se passe. Cela ne fonctionne que si le modèle est parfait. Un biais de braquage de 0,01 rad (un demi-degré, typique d'un servomoteur mal centré) suffit à ruiner le plan : à 5 m/s, l'écart latéral vaut à peu près $\tfrac{1}{2}\,\tfrac{v^2}{L}\,\delta\, t^2$ avec l'empattement $L = 0{,}33$ m, soit 38 cm après une seconde et 1,5 m après deux secondes, bien au-delà de la demi-largeur de piste de 0,8 m.

En **boucle fermée**, le contrôleur mesure l'état en permanence et corrige la commande en fonction de l'écart observé. On parle de **rétroaction** (*feedback*). La valeur visée s'appelle la **consigne** ou **référence**, et l'écart entre référence et mesure l'**erreur de suivi**. La boucle fermée compense les erreurs de modèle et les perturbations, mais elle introduit un risque propre : une correction trop forte ou trop tardive fait osciller le système, voire diverger. On dit qu'un système bouclé est **stable** si les petites perturbations ne s'amplifient pas. Le **retard** entre la mesure et l'application de la commande (la latence) est l'ennemi principal de la stabilité, ce qui explique en partie l'importance du budget de 20 ms.

### 1.3 Le problème de contrôle optimal

Le contrôle optimal formule le pilotage comme une optimisation. On se donne un **coût instantané** $c(x, u)$ qui mesure ce qui est indésirable à chaque pas (s'écarter de la piste, braquer brutalement), un **coût terminal** $\phi(x)$ qui évalue l'état final, et un **horizon** $T$, nombre de pas sur lesquels on anticipe. Le problème s'écrit :

$$
\min_{u_0, \dots, u_{T-1}} \; \phi(x_T) + \sum_{t=0}^{T-1} c(x_t, u_t)
\quad \text{sous} \quad x_{t+1} = F(x_t, u_t), \quad x_0 = \hat{x}, \quad u_t \in \mathcal{U}.
$$

Les **contraintes** se répartissent en deux catégories. Une **contrainte dure** doit être respectée exactement, comme les bornes des actionneurs ($u_t \in \mathcal{U}$). Une **contrainte souple** est remplacée par une pénalité dans le coût, qu'on peut violer si le bénéfice le justifie. Dans MPPI, les bornes de commande sont dures (on sature les commandes échantillonnées), tandis que « rester sur la piste » ou « éviter l'obstacle » sont presque toujours souples.

Avec nos paramètres, la variable de décision est un vecteur de $T \times m = 100$ réels. L'état n'est pas une variable libre : il est entièrement déterminé par $\hat{x}$ et les commandes. La difficulté vient de ce que $F$ est non linéaire et que le coût peut être non convexe, non différentiable, voire discontinu.

La **fonction valeur** $V_t(x)$ est le coût optimal restant à payer depuis l'état $x$ à l'instant $t$ (*cost-to-go*). Elle vérifie le **principe de Bellman** :

$$
V_T(x) = \phi(x), \qquad V_t(x) = \min_{u \in \mathcal{U}} \big[ c(x, u) + V_{t+1}(F(x, u)) \big].
$$

Un sous-chemin d'un chemin optimal est optimal. Si l'on connaissait $V$ partout, la commande optimale en tout état serait immédiate. Mais calculer $V$ sur tout l'espace d'état (programmation dynamique) est hors de portée en dimension 6 : une grille de 100 valeurs par axe représente $10^{12}$ cellules. On se contente donc de résoudre le problème depuis l'état courant, ce qu'on appelle **optimisation de trajectoire**. La section 3 reprend ce cadre avec du bruit, et c'est là que la forme de MPPI apparaît.

### 1.4 Horizon glissant

La solution du problème précédent est une séquence de commandes, donc un plan en boucle ouverte. La **commande prédictive** (MPC, *Model Predictive Control*) le transforme en contrôleur bouclé par une idée simple : à chaque pas de temps, on mesure l'état, on résout le problème sur les $T$ pas à venir, on applique uniquement la première commande $u_0$, on jette le reste, et on recommence au pas suivant avec la nouvelle mesure. La fenêtre d'optimisation avance avec le temps sans jamais être atteinte ; on parle d'**horizon glissant** ou **horizon fuyant** (*receding horizon*). À 50 Hz avec $T = 50$, le contrôleur regarde toujours exactement une seconde devant lui.

Chaque commande appliquée dépend de la dernière mesure : c'est bien une boucle fermée, et les erreurs de modèle sont corrigées à chaque pas. Comme deux problèmes successifs se recouvrent sur $T-1$ pas, la solution précédente décalée d'un pas constitue un excellent point de départ pour la suivante. On parle de **démarrage à chaud** (*warm start*). MPPI repose entièrement sur ce mécanisme.

### 1.5 Contraintes non holonomes

Une **contrainte holonome** est une relation portant sur la seule configuration du système, de la forme $g(q) = 0$ : un pendule dont la longueur est fixe, par exemple. Elle réduit le nombre de degrés de liberté. Une **contrainte non holonome** porte sur les vitesses et ne peut pas s'intégrer en une relation sur les positions.

La voiture en fournit l'exemple type. Si les roues roulent sans glisser, la vitesse du milieu de l'essieu arrière est alignée avec l'axe du véhicule :

$$
\dot{p}_x \sin\psi - \dot{p}_y \cos\psi = 0.
$$

Cette relation n'interdit aucune configuration : toute position et tout cap $(p_x, p_y, \psi)$ sont atteignables, comme le montre un créneau. Elle interdit en revanche certaines vitesses instantanées : une voiture ne se déplace pas latéralement. Le système possède trois degrés de liberté de configuration pour deux commandes indépendantes ; on dit qu'il est **sous-actionné**. Pour se décaler de 30 cm latéralement, il faut braquer, avancer, contre-braquer : une manœuvre étalée dans le temps, qui exige d'anticiper.

Cette propriété a des conséquences théoriques fortes. Le théorème de Brockett (1983) montre qu'aucun retour d'état continu et indépendant du temps $u = k(x)$ ne stabilise un tel système en une configuration fixe. Concrètement, la linéarisation du modèle à l'arrêt est non commandable : à $v = 0$, aucun braquage ne change la position latérale. Autour d'une trajectoire en mouvement, les linéarisations redeviennent commandables et le suivi fonctionne, mais la géométrie des manœuvres doit être traitée par l'anticipation, ce que fait naturellement un contrôleur prédictif.

À la limite d'adhérence, la contrainte cesse d'être vérifiée : les pneus glissent et la voiture se déplace effectivement de côté. Le modèle dynamique de la section 6 abandonne cette contrainte et modélise les forces qui la font respecter en temps normal.

### 1.6 Quatre familles de contrôleurs

**Le PID.** Le correcteur proportionnel-intégral-dérivé calcule la commande à partir de l'erreur de suivi $e(t)$ seule :

$$
u(t) = K_p\, e(t) + K_i \int_0^t e(s)\, ds + K_d\, \dot{e}(t).
$$

Le terme proportionnel corrige l'écart présent, l'intégral annule les biais persistants, le dérivé amortit. On règle trois gains, sans aucun modèle. Pour une voiture, on bouclerait par exemple le braquage sur l'écart latéral à la ligne centrale et l'accélération sur l'écart de vitesse. Le PID est imbattable sur des boucles simples et rapides (le servomoteur de direction contient probablement le sien), mais il échoue dès que la tâche demande plus. Il ne voit que l'erreur présente et n'anticipe pas un virage qui arrive. Il traite chaque commande séparément, alors que vitesse et braquage sont couplés : on ne prend pas une épingle à la même vitesse qu'une courbe large. Il ignore les bornes des actionneurs, et son terme intégral continue d'accumuler quand la commande sature (phénomène d'emballement, *windup*). Enfin, des gains réglés à 2 m/s ne conviennent plus à 8 m/s. Les contrôleurs géométriques classiques pour véhicules, *pure pursuit* et *Stanley*, ajoutent une anticipation par un point visé en avant, mais restent des lois réactives sans notion d'optimalité.

**Le LQR.** Le régulateur linéaire quadratique suppose un modèle linéaire $x_{t+1} = A x_t + B u_t$ et un coût quadratique $\sum_t x_t^\top Q x_t + u_t^\top R u_t$, où les matrices symétriques $Q \succeq 0$ et $R \succ 0$ pèsent respectivement l'écart d'état et l'effort de commande. La solution optimale est un retour d'état linéaire $u_t = -K x_t$, où le gain $K$ s'obtient hors ligne en résolvant l'équation de Riccati :

$$
P = Q + A^\top P A - A^\top P B (R + B^\top P B)^{-1} B^\top P A, \qquad K = (R + B^\top P B)^{-1} B^\top P A.
$$

C'est élégant, optimal et quasi gratuit à l'exécution. Pour un système non linéaire, on **linéarise** le modèle autour d'une trajectoire de référence et on régule l'écart à cette référence. Le LQR échoue dans trois situations. Quand l'état s'éloigne du point de linéarisation, le modèle linéaire devient faux ; c'est précisément ce qui se passe quand un pneu sature, puisque la force latérale cesse de croître avec l'angle. Il ne connaît aucune contrainte, et une commande de braquage de 0,6 rad calculée par $-Kx$ sera simplement écrêtée à 0,4 sans que le contrôleur en tienne compte. Et il n'accepte qu'un coût quadratique, ce qui exclut « rester entre les bords de la piste » ou « ne pas entrer dans une cellule occupée ».

**La MPC classique.** La commande prédictive, telle qu'on la pratique en industrie, résout numériquement le problème de la section 1.3 à chaque pas, en horizon glissant. Avec un modèle linéaire et un coût quadratique, chaque résolution est un programme quadratique (QP), rapide et fiable ; c'est la forme dominante dans l'industrie des procédés et l'automobile. Avec un modèle non linéaire, on obtient un programme non linéaire (NLP) résolu par programmation quadratique séquentielle (SQP) ou par points intérieurs (bibliothèques acados, IPOPT, FORCES Pro). Les méthodes iLQR et DDP, qui itèrent des LQR autour de la trajectoire courante, relèvent de la même famille. Toutes utilisent les dérivées premières, souvent secondes, du modèle et du coût. Leur force est de traiter explicitement les contraintes dures et de converger en quelques itérations quand le problème est régulier. Leurs faiblesses découlent du recours aux gradients : il faut un modèle et un coût différentiables, une carte de coût tabulée ou une fonction indicatrice « hors piste » posent problème ; la solution est un minimum local qui dépend de l'initialisation, et un obstacle rend le problème non convexe ; le temps de résolution varie d'un pas à l'autre et peut exploser quand le solveur peine à converger. Ces méthodes fonctionnent pourtant très bien en course, pourvu que le problème soit formulé avec soin : Liniger, Domahidi et Morari ont piloté des voitures à l'échelle 1:43 au plus près de la limite avec une MPC non linéaire à gradients (2015).

**L'apprentissage de politique.** L'apprentissage par renforcement ou par imitation entraîne hors ligne un réseau de neurones $\pi_\theta(x)$ qui renvoie directement la commande. L'exécution se réduit à une passe avant du réseau, très rapide. Le coût est ailleurs : il faut des millions d'interactions, en pratique dans un simulateur, et la politique apprise souffre de l'écart entre simulation et réalité (*sim-to-real gap*). L'objectif est figé dans les poids : changer la fonction de coût ou le tracé impose de réentraîner. Face à un état jamais rencontré à l'entraînement, le comportement n'est pas prévisible, et les garanties formelles sont rares. On peut voir l'apprentissage de politique comme une optimisation amortie : tout le calcul est fait avant le déploiement. La MPC fait l'inverse et optimise en ligne, à chaque pas, pour l'état effectivement rencontré.

**MPPI.** MPPI conserve la structure de la MPC (modèle, coût, horizon glissant, démarrage à chaud) et remplace le solveur à gradients par une méthode de Monte-Carlo. À chaque pas, on perturbe aléatoirement la séquence nominale $K$ fois, on simule les $K$ trajectoires avec le modèle, on les évalue par le coût, et on déplace la séquence nominale vers une moyenne des perturbations pondérée par $\exp(-S/\lambda)$. Aucune dérivée n'est nécessaire : le coût peut être une indicatrice lue dans une grille, le modèle peut contenir des saturations, des tables, des branchements ou un réseau de neurones. Les $K$ simulations sont indépendantes, donc massivement parallélisables. Le prix à payer est un grand nombre d'évaluations du modèle (409 600 par itération avec nos paramètres), une solution bruitée, des contraintes seulement souples, et une efficacité qui se dégrade quand la dimension de la commande augmente.

| | Modèle requis | Coût admissible | Contraintes | Calcul en ligne | Point de rupture typique |
|---|---|---|---|---|---|
| PID | aucun | erreur de suivi | non | négligeable | couplages, anticipation, saturation |
| LQR | linéaire | quadratique | non | négligeable | non-linéarité, bornes |
| MPC à gradients | différentiable | différentiable | dures | variable, parfois élevé | coûts non lisses, minima locaux |
| Politique apprise | simulateur pour l'entraînement | quelconque, figé | apprises | faible | hors distribution, changement d'objectif |
| MPPI | simulable | quelconque | souples | élevé mais constant | dimension de commande, échelle du coût |

### 1.7 Origine de MPPI

MPPI est né au Georgia Institute of Technology, dans le laboratoire d'Evangelos Theodorou, principalement à travers la thèse de Grady Williams. Sa base théorique est plus ancienne. Hilbert Kappen avait montré en 2005 qu'une classe de problèmes de contrôle stochastique devient linéaire après un changement de variable logarithmique, et que la solution s'exprime alors comme une espérance sur des trajectoires aléatoires : une intégrale de chemin. Emanuel Todorov avait obtenu des résultats voisins en temps discret (problèmes de décision markoviens linéairement solubles). Theodorou, Buchli et Schaal en avaient tiré PI² (2010), un algorithme d'apprentissage par renforcement pour la robotique.

Le pas franchi par Williams, Aldrich et Theodorou (article publié en 2017 dans le *Journal of Guidance, Control, and Dynamics*, prépublié en 2015) consiste à utiliser ce résultat en ligne, en horizon glissant, et à exploiter le fait que l'espérance se calcule par des simulations indépendantes, donc sur GPU. La démonstration expérimentale a eu lieu sur AutoRally, une plateforme de recherche construite à Georgia Tech : un véhicule tout-terrain à l'échelle 1:5, d'une vingtaine de kilogrammes, embarquant un ordinateur équipé d'un GPU, roulant sur une piste en terre. L'article de l'ICRA 2016 (*Aggressive Driving with Model Predictive Path Integral Control*) montre le véhicule en conduite agressive, en glissade contrôlée dans les virages. L'article de l'ICRA 2017 remplace le modèle physique par un réseau de neurones appris sur des données de conduite, et celui des *IEEE Transactions on Robotics* (2018) reformule toute la théorie en termes de divergence de Kullback-Leibler.

Ce terrain d'essai n'a rien d'un hasard, il concentre les conditions où MPPI se distingue. En conduite agressive sur terre, la dynamique est dominée par la saturation des pneus, très non linéaire et mal représentée par une linéarisation. La tâche s'exprime naturellement par un coût non lisse : une carte de coût de la piste, où sortir des bords coûte très cher. La commande n'a que deux dimensions, ce qui rend l'échantillonnage efficace sur un horizon de quelques secondes. Il faut réagir à plusieurs dizaines de hertz, et un GPU grand public embarqué suffit à simuler des milliers de trajectoires dans ce délai. Enfin la théorie des intégrales de chemin fournissait une règle de mise à jour justifiée, là où une recherche aléatoire ad hoc aurait laissé le choix de la pondération à l'intuition.

---

## 2. L'algorithme

### 2.1 Ce que l'algorithme conserve

D'une itération à l'autre, MPPI ne conserve qu'un seul objet : la **séquence nominale** $U = (u_0, \dots, u_{T-1}) \in \mathbb{R}^{T \times m}$, sa meilleure estimation courante des commandes optimales pour la seconde à venir. Tout le reste (bruit, trajectoires simulées, coûts, poids) est temporaire et recalculé à chaque pas.

Le principe tient en une phrase : perturber $U$ aléatoirement $K$ fois, simuler chaque perturbation, évaluer chaque trajectoire, puis déplacer $U$ vers les perturbations qui ont mené à un faible coût, avec un poids qui décroît exponentiellement avec ce coût.

Les ordres de grandeur, avec $K = 8192$, $T = 50$ et $m = 2$ :

- le tenseur de bruit contient $8192 \times 50 \times 2 = 819\,200$ scalaires, soit 3,1 Mio en float32 ;
- chaque itération évalue le modèle $K \times T = 409\,600$ fois, et le coût autant de fois ;
- stocker toutes les trajectoires d'état du modèle dynamique demanderait $8192 \times 51 \times 6$ float32, soit 9,6 Mio, alors que MPPI n'a besoin que des $K$ coûts finaux (32 Kio) ;
- à 50 Hz, cela représente environ 20 millions d'évaluations du modèle par seconde.

### 2.2 Une itération pas à pas

**Étape 0, état initial.** L'estimateur fournit $\hat{x}$, point de départ commun des $K$ rollouts. Une subtilité compte ici, car le calcul prend du temps. Si l'itération commence à l'instant $t_k$ et prend près de 20 ms, la commande calculée n'est disponible qu'à $t_k + 20$ ms, alors que l'état a déjà changé. La solution propre consiste à appliquer à $t_k$ la commande calculée à l'itération précédente, à prédire l'état à $t_{k+1}$ en simulant un pas de modèle sous cette commande, et à optimiser à partir de cet état prédit. Le retard d'un pas devient alors connu et intégré au modèle, au lieu d'être subi. La section 2.3 donne les deux variantes.

**Étape 1, échantillonnage du bruit.** On tire $K \times T$ vecteurs indépendants $\varepsilon^k_t \sim \mathcal{N}(0, \Sigma)$. Avec $\Sigma$ diagonale, cela revient à tirer des gaussiennes centrées réduites et à les multiplier par $\sigma_a = 0{,}5$ m/s² et $\sigma_\delta = 0{,}1$ rad. Ce bruit est indépendant d'un pas de temps à l'autre (bruit blanc) ; la section 4.2 explique pourquoi c'est souvent un mauvais choix et comment le corriger. Certaines implémentations tirent une fraction des échantillons autour de zéro, ou autour d'une commande de freinage, au lieu de les centrer sur $U$, pour garder en permanence une alternative sûre dans la population.

**Étape 2, commandes perturbées.** Chaque trajectoire reçoit la séquence $v^k_t = u_t + \varepsilon^k_t$, saturée aux bornes des actionneurs. Après saturation, on recalcule le bruit effectif $\varepsilon^k_t \leftarrow v^k_t - u_t$. Sans cette correction, la mise à jour de l'étape 6 moyennerait des perturbations qui n'ont jamais été simulées et pousserait $U$ hors des bornes. Avec elle, la nouvelle séquence $\sum_k w^k v^k_t$ est une combinaison convexe de commandes admissibles, donc admissible elle aussi.

**Étape 3, rollouts.** Pour chaque $k$, on part de $x^k_0 = \hat{x}$ et on applique $x^k_{t+1} = F(x^k_t, v^k_t)$ pour $t = 0, \dots, T-1$. Les $K$ rollouts n'échangent aucune information.

**Étape 4, coût par trajectoire.** On accumule le long de chaque rollout :

$$
S^k = \phi(x^k_T) + \sum_{t=1}^{T} c(x^k_t, v^k_{t-1}) + \gamma \sum_{t=0}^{T-1} u_t^\top \Sigma^{-1} \varepsilon^k_t.
$$

Le coût d'état est évalué sur $x_1, \dots, x_T$ : l'état $x_0$ est le même pour tous les échantillons et ne les départage pas. Le dernier terme, où $\gamma$ vaut $\lambda$ en théorie, est une correction d'échantillonnage préférentiel dont la section 3.5 donne l'origine exacte. Il pénalise les perturbations qui éloignent encore davantage la commande de zéro.

**Étape 5, pondération exponentielle.** On calcule $\rho = \min_k S^k$, puis

$$
\tilde{w}^k = \exp\!\left(-\frac{S^k - \rho}{\lambda}\right), \qquad \eta = \sum_{j=1}^{K} \tilde{w}^j, \qquad w^k = \frac{\tilde{w}^k}{\eta}.
$$

La soustraction de $\rho$ ne change pas les poids normalisés, puisqu'elle multiplie tous les $\tilde{w}^k$ par la même constante $e^{\rho/\lambda}$. Elle est indispensable numériquement. En float32, $e^{-z}$ devient inférieur au plus petit nombre normal représentable dès $z \approx 87{,}3$, et nul dès $z \approx 103{,}3$ (ou dès 87,3 si les nombres sous-normaux sont désactivés, ce que font souvent les options de compilation agressives). Un ensemble de coûts compris entre 150 et 400 avec $\lambda = 1$ donnerait donc $\eta = 0$ et une division par zéro. Après soustraction, le meilleur échantillon a toujours un poids brut de 1.

On en profite pour calculer la **taille effective d'échantillon** (ESS, *effective sample size*) :

$$
\mathrm{ESS} = \frac{1}{\sum_k (w^k)^2} \in [1, K].
$$

Elle vaut $K$ si tous les poids sont égaux et 1 si un seul échantillon porte tout le poids. C'est l'indicateur de santé le plus utile de tout l'algorithme ; il doit être journalisé à chaque itération.

**Étape 6, moyenne pondérée.** La séquence nominale devient

$$
u_t \leftarrow u_t + \sum_{k=1}^{K} w^k \varepsilon^k_t, \qquad t = 0, \dots, T-1.
$$

Chaque pas de temps reçoit sa propre moyenne, mais avec les mêmes poids : une trajectoire est bonne ou mauvaise dans son ensemble. Certaines implémentations lissent ensuite $U$ le long du temps, par exemple avec un filtre de Savitzky-Golay dans `nav2_mppi_controller`, ce qui échange un peu de réactivité contre une commande plus douce.

**Étape 7, application.** On envoie $u_0$ aux actionneurs. Les $T-1$ autres commandes ne seront jamais appliquées telles quelles, elles servent de point de départ à l'itération suivante.

**Étape 8, décalage.** On décale la séquence d'un pas, $u_t \leftarrow u_{t+1}$ pour $t = 0, \dots, T-2$, et on complète la dernière case, en général en recopiant l'avant-dernière. Recopier convient mieux qu'insérer une commande nulle : en plein virage, la meilleure hypothèse pour le pas qui entre dans l'horizon est de garder le braquage. Ce décalage est le démarrage à chaud de la section 1.4. Il fait qu'une itération par pas de temps suffit : l'optimisation se poursuit d'un pas sur l'autre, et l'effort cumulé sur une commande donnée correspond à 50 itérations, une par pas de temps entre son entrée dans l'horizon et son application.

### 2.3 Pseudocode

La version de base applique la commande dès qu'elle est calculée. Les indices suivent la convention NumPy.

```python
# ---------------------------------------------------------------------------
# Paramètres (config/mppi.yaml)
# ---------------------------------------------------------------------------
K, T, dt     = 8192, 50, 0.02
m            = 2                                # (a, delta)
lam          = ...                              # température, voir section 4.1
sigma        = array([0.5, 0.1])                # écarts-types du bruit sur (a, delta)
Sigma_inv    = diag(1.0 / sigma**2)
gamma        = ...                              # poids de la correction, gamma <= lam
u_min        = array([-4.0, -0.4])
u_max        = array([ 4.0,  0.4])
S_MAX        = 1e6                              # coût de repli pour un rollout non fini

U = zeros((T, m))                               # séquence nominale, conservée entre itérations

# ---------------------------------------------------------------------------
# Boucle de contrôle, cadencée à dt
# ---------------------------------------------------------------------------
tous les dt secondes:

    # 0. État initial commun à tous les rollouts
    x0 = estimer_etat()                         # vecteur de taille state_dim

    # 1. Bruit blanc gaussien, tiré à neuf à chaque itération
    eps = normal(size=(K, T, m)) * sigma        # (K, T, m)

    # 2. Commandes perturbées, saturées, puis bruit effectif
    V   = clip(U[None, :, :] + eps, u_min, u_max)   # (K, T, m)
    eps = V - U[None, :, :]                     # ce qui a réellement été simulé

    # 3 et 4. Rollouts et coûts
    S = zeros(K)
    pour k dans 0..K-1, EN PARALLÈLE:           # aucun échange entre rollouts
        x = x0
        pour t dans 0..T-1, SÉQUENTIELLEMENT:   # x_{t+1} dépend de x_t
            x     = step(x, V[k, t], dt)
            S[k] += cout_etape(x, V[k, t])
            S[k] += gamma * U[t] @ Sigma_inv @ eps[k, t]
        S[k] += cout_terminal(x)
        si non isfinite(S[k]):                  # rollout divergent (section 7.7)
            S[k] = S_MAX

    # 5. Poids exponentiels, calculés relativement au meilleur coût
    rho = min(S)
    w   = exp(-(S - rho) / lam)                 # dans ]0, 1], le meilleur vaut 1
    w   = w / sum(w)
    ess = 1.0 / sum(w**2)                       # à journaliser

    # 6. Moyenne pondérée des perturbations, pas de temps par pas de temps
    U = U + tensordot(w, eps, axes=1)           # (T, m), reste dans les bornes

    # 7. Application de la première commande
    envoyer_aux_actionneurs(U[0])

    # 8. Décalage d'un pas pour l'itération suivante
    U[:-1] = U[1:]
    U[-1]  = U[-2]
```

La variante avec compensation de latence ne change que le début et la fin de l'itération :

```python
u_suivante = zeros(m)

tous les dt secondes, à l'instant t_k:
    x_mes = estimer_etat()
    envoyer_aux_actionneurs(u_suivante)         # appliquée sur [t_k, t_k + dt]
    x0 = step(x_mes, u_suivante, dt)            # état prédit à t_{k+1}

    # étapes 1 à 6 inchangées ; U[0] porte désormais sur l'intervalle [t_{k+1}, t_{k+2}]

    u_suivante = U[0]                           # sera appliquée au prochain tick
    U[:-1] = U[1:]
    U[-1]  = U[-2]
```

La commande a exactement un pas de retard, mais ce retard est déterministe et le contrôleur le connaît. La variante de base subit au contraire un retard variable, compris entre zéro et un pas selon la durée du calcul, que le modèle ignore.

### 2.4 Indépendants entre eux, séquentiels dans le temps

Les deux boucles imbriquées des étapes 3 et 4 n'ont pas la même nature, et toute l'implémentation GPU découle de cette asymétrie.

Selon l'axe $k$, les rollouts sont **indépendants**. Une fois l'état initial et le tenseur de bruit fixés, la trajectoire $k$ est une fonction déterministe de $(\hat{x}, v^k_0, \dots, v^k_{T-1})$ seuls. Aucun rollout ne lit ce qu'un autre a calculé. Les échanges n'interviennent qu'après, à l'étape 5, quand le minimum et la normalisation couplent tous les coûts. Le calcul des $K$ trajectoires est donc parallèle au sens le plus fort : on peut le répartir sur autant de fils d'exécution que l'on veut, dans n'importe quel ordre, sans verrou ni communication.

Selon l'axe $t$, le calcul est **séquentiel**. L'état $x_{t+1}$ est obtenu à partir de $x_t$, qui dépend lui-même de $x_{t-1}$ : le modèle est une récurrence, une chaîne de Markov déterministe une fois le bruit tiré. On ne peut pas calculer le pas 30 avant le pas 29. C'est la même contrainte que dans un réseau récurrent, et elle tient à la nature même d'une simulation physique.

Ces deux axes autorisent deux ordres de boucles, qui produisent exactement les mêmes résultats. La version NumPy place la boucle sur $t$ à l'extérieur : 50 itérations Python, dont chacune applique `step` à un tableau d'états de forme $(K, n)$ en une seule opération vectorisée. Le kernel CUDA fait l'inverse : un fil d'exécution par rollout, et à l'intérieur de chaque fil une boucle C ordinaire sur les 50 pas. Pour le contrôle de parité entre les deux implémentations, il faut garder en tête que les deux codes parcourent le même calcul dans un ordre différent ; les résultats doivent coïncider trajectoire par trajectoire.

### 2.5 Conséquences pour le kernel CUDA

**Le grain naturel est le rollout.** On lance un fil par trajectoire, soit 8192 fils. Avec des blocs de 256 fils, cela fait 32 blocs ; $K$ étant une puissance de deux, aucun fil de garde n'est nécessaire, mais il vaut mieux garder le test `if (k >= K) return;` pour le jour où $K$ changera.

**Fusionner dynamique et coût dans la même boucle.** Chaque fil garde son état (4 à 6 flottants) et son coût accumulé dans des variables locales, que le compilateur place dans des registres. Les seuls accès à la mémoire globale sont la lecture des 100 valeurs de bruit du rollout, la lecture de la carte de coût à chaque pas, et l'écriture d'un unique flottant à la fin. Aucune trajectoire n'est matérialisée. La conception inverse, plus proche de NumPy, lance un kernel par pas de temps : 50 lancements successifs, chacun lisant et écrivant un tableau de $K$ états en mémoire globale. Elle est plus simple à écrire et à vérifier, mais multiplie les lancements et les transferts mémoire. La table d'optimisation de `bench/results.md` (lignes 0 et 1) est prévue pour mesurer précisément cet écart.

**Garder des boucles de longueur fixe.** Tous les fils exécutent exactement $T$ pas. Un rollout qui sort de piste ne doit pas s'interrompre par un `break` : on le « gèle » en multipliant sa mise à jour par un masque, ce qui garde un flot de contrôle identique pour tous les fils. Le coût d'un branchement dépendant des données varie selon l'architecture GPU, et la règle sûre est de préférer des mélanges arithmétiques (`mix`, `clamp`, fonctions lisses) quand ils sont simples.

**Les étapes 5 et 6 sont légères.** Le calcul des poids ne porte que sur $K$ flottants, 32 Kio. La mise à jour $\sum_k w^k \varepsilon^k_t$ représente 819 200 multiplications-additions : elle se fait dans un second kernel (un fil par couple $(t, j)$ sommant sur $K$) ou sur le CPU. Sur le Jetson, CPU et GPU partagent la même mémoire physique LPDDR5 : il n'y a pas de bus PCIe à traverser, et des tampons en mémoire partagée (*zero-copy*) évitent toute copie. Cette architecture diffère de celle d'un GPU discret, et les proportions entre temps de calcul et temps d'accès mémoire mesurées sur une carte discrète ne se transposent pas.

**Le générateur aléatoire est un poste à part entière.** Il faut 819 200 gaussiennes par itération. Deux approches coexistent. La première tire tout le tenseur d'un coup sur le GPU avec cuRAND. La seconde utilise dans le kernel un générateur à compteur (de type Philox), dont la sortie est une fonction pure de la clé (graine, itération, $k$, $t$, composante) : le bruit n'a plus besoin d'être stocké, il peut être régénéré à l'identique dans le kernel de mise à jour. Pour la vérification de parité, le plus simple reste de générer le bruit une fois côté hôte avec une graine fixe et de le fournir aux deux implémentations, ce que prévoit `bench/check_parity.py`.

**La précision diffère entre NumPy et CUDA.** NumPy calcule par défaut en float64, le kernel en float32. Sur 50 pas, les écarts d'arrondi s'accumulent. En régime normal ils restent négligeables, mais à la limite d'adhérence la dynamique devient sensible aux conditions initiales et deux trajectoires numériquement voisines peuvent diverger nettement. Il faut donc forcer float32 côté NumPy pour la parité et comparer les coûts $S^k$ avec une tolérance relative.

**Ordre de grandeur du budget.** Une évaluation du modèle dynamique avec pneus et du coût représente de l'ordre de la centaine d'opérations flottantes. Sur 409 600 évaluations, cela fait quelques dizaines de millions d'opérations par itération, ce qui reste modeste pour le GPU Ampere de 1024 cœurs de l'Orin Nano. Il est probable que le temps se concentre ailleurs que dans l'arithmétique : accès mémoire, génération du bruit, lancements de kernels, synchronisations CPU-GPU et gigue du système d'exploitation. C'est une estimation, et seule la mesure au profileur tranchera. Le budget de 20 ms couvre toute l'itération, de la lecture de l'état à l'envoi de la commande.

---

## 3. Les fondements théoriques

La pondération $\exp(-S/\lambda)$ peut sembler arbitraire : pourquoi une exponentielle, pourquoi ce paramètre, pourquoi moyenner au lieu de garder la meilleure trajectoire ? Cette section montre que la règle de MPPI est la solution exacte d'un problème de contrôle optimal stochastique précis, puis explique les approximations qui mènent à l'algorithme. Deux chemins y conduisent : l'un passe par les équations aux dérivées partielles et la formule de Feynman-Kac, l'autre par la divergence de Kullback-Leibler. Ils aboutissent au même résultat, et chacun éclaire un aspect différent.

### 3.1 Le problème stochastique

Un problème de **contrôle stochastique** est un problème de contrôle optimal où la dynamique contient du bruit, de sorte qu'on optimise une espérance. On considère, en temps continu, un système dont la commande est perturbée par un bruit blanc qui entre par le même canal qu'elle :

$$
dx = f(x, t)\, dt + G(x, t)\big(u\, dt + dw\big), \qquad dw \sim \mathcal{N}(0, \Sigma\, dt).
$$

Ici $w$ est un mouvement brownien de covariance $\Sigma$, $f$ la dynamique libre (sans commande) et $G \in \mathbb{R}^{n \times m}$ la manière dont la commande agit sur l'état. Cette forme affine en la commande est une hypothèse de la théorie ; la section 3.4 montre que le cadre discret s'en passe. L'hypothèse que le bruit entre par le canal de la commande est naturelle : un actionneur ne réalise jamais exactement la consigne. Elle est surtout celle qui correspond à l'algorithme, qui perturbe les commandes.

Le coût à minimiser sur $[t, t_f]$ sépare un coût d'état $q$, arbitraire, et un coût de commande quadratique :

$$
J(x, t; u) = \mathbb{E}\left[ \phi(x_{t_f}) + \int_t^{t_f} \Big( q(x_s) + \tfrac{1}{2} u_s^\top R\, u_s \Big) ds \right],
\qquad V(x, t) = \inf_u J(x, t; u).
$$

Aucune régularité n'est demandée à $q$ au-delà de la mesurabilité : ce sera la carte de coût de la piste.

### 3.2 L'équation de Hamilton-Jacobi-Bellman

Le principe de Bellman sur un intervalle infinitésimal donne

$$
V(x, t) = \min_u \mathbb{E}\Big[ \big(q + \tfrac{1}{2} u^\top R u\big) dt + V(x + dx,\, t + dt) \Big].
$$

On développe $V(x + dx, t + dt)$ par la formule d'Itô. Comme $dx$ contient un terme d'ordre $\sqrt{dt}$, le terme de second ordre contribue à l'ordre $dt$ :

$$
\mathbb{E}\big[V(x + dx, t + dt)\big] = V + \Big( \partial_t V + \nabla V^\top (f + G u) + \tfrac{1}{2} \operatorname{tr}\big( G \Sigma G^\top \nabla^2 V \big) \Big) dt + o(dt).
$$

En reportant et en divisant par $dt$, on obtient l'équation de **Hamilton-Jacobi-Bellman** (HJB) :

$$
-\partial_t V = \min_u \Big[ q + \tfrac{1}{2} u^\top R u + \nabla V^\top (f + G u) \Big] + \tfrac{1}{2} \operatorname{tr}\big( G \Sigma G^\top \nabla^2 V \big).
$$

Le minimum en $u$ d'une forme quadratique strictement convexe s'obtient en annulant le gradient : $R u + G^\top \nabla V = 0$, d'où

$$
u^\ast = -R^{-1} G^\top \nabla V.
$$

En substituant, $\tfrac{1}{2} u^{\ast\top} R u^\ast + \nabla V^\top G u^\ast = -\tfrac{1}{2} \nabla V^\top G R^{-1} G^\top \nabla V$, et l'équation devient

$$
-\partial_t V = q + \nabla V^\top f - \tfrac{1}{2} \nabla V^\top G R^{-1} G^\top \nabla V + \tfrac{1}{2} \operatorname{tr}\big( G \Sigma G^\top \nabla^2 V \big),
\qquad V(x, t_f) = \phi(x).
$$

C'est une EDP parabolique non linéaire, à cause du terme quadratique en $\nabla V$, posée en dimension $n$. La résoudre sur une grille en dimension 6 est exclu. Tout l'intérêt de la suite est de faire disparaître ce terme non linéaire.

### 3.3 Transformation logarithmique et formule de Feynman-Kac

On pose, pour un $\lambda > 0$ à choisir,

$$
V = -\lambda \log \Psi, \qquad \text{c'est-à-dire} \qquad \Psi = e^{-V/\lambda}.
$$

La fonction $\Psi \in\, ]0, 1]$ (si les coûts sont positifs) mesure la « désirabilité » d'un état : proche de 1 quand le coût restant est faible, proche de 0 quand il est élevé. On calcule

$$
\partial_t V = -\lambda \frac{\partial_t \Psi}{\Psi}, \qquad
\nabla V = -\lambda \frac{\nabla \Psi}{\Psi}, \qquad
\nabla^2 V = -\lambda \left( \frac{\nabla^2 \Psi}{\Psi} - \frac{\nabla \Psi \nabla \Psi^\top}{\Psi^2} \right).
$$

En reportant dans HJB, deux termes quadratiques en $\nabla \Psi$ apparaissent. Le terme non linéaire donne $-\tfrac{\lambda^2}{2}\, \nabla \Psi^\top G R^{-1} G^\top \nabla \Psi / \Psi^2$, et la partie hessienne donne $+\tfrac{\lambda}{2}\, \nabla \Psi^\top G \Sigma G^\top \nabla \Psi / \Psi^2$. Ils se compensent exactement dès que $\lambda\, G R^{-1} G^\top = G \Sigma G^\top$, ce qui est garanti par la condition

$$
\boxed{\;\lambda R^{-1} = \Sigma\;}
$$

Sous cette condition, il reste, après multiplication par $-\Psi / \lambda$, une équation **linéaire** en $\Psi$ :

$$
-\partial_t \Psi = -\frac{q}{\lambda} \Psi + f^\top \nabla \Psi + \tfrac{1}{2} \operatorname{tr}\big( G \Sigma G^\top \nabla^2 \Psi \big),
\qquad \Psi(x, t_f) = e^{-\phi(x)/\lambda}.
$$

La condition $\lambda R^{-1} = \Sigma$ mérite qu'on s'y arrête. Elle impose $R = \lambda \Sigma^{-1}$ : commander coûte cher dans les directions où le bruit est faible et peu cher dans celles où il est fort. On peut la lire ainsi : il est coûteux de pousser le système dans une direction où le hasard ne le pousse jamais. Elle a aussi une conséquence pratique qui reviendra dans tout le document : le même scalaire $\lambda$ relie l'amplitude du bruit, le coût de la commande et, on va le voir, la température de la pondération. Ces trois quantités ne se règlent pas indépendamment.

L'équation linéaire obtenue est une équation de Kolmogorov rétrograde avec un terme d'absorption $-q\Psi/\lambda$. La **formule de Feynman-Kac** en donne la solution sous forme d'espérance :

$$
\Psi(x, t) = \mathbb{E}\left[ \exp\!\left( -\frac{1}{\lambda} \Big( \phi(\tilde{x}_{t_f}) + \int_t^{t_f} q(\tilde{x}_s)\, ds \Big) \right) \,\middle|\, \tilde{x}_t = x \right],
$$

où $\tilde{x}$ suit la dynamique **non commandée** $d\tilde{x} = f\, dt + G\, dw$. La preuve tient en une ligne. On pose $M_s = \exp\big(-\tfrac{1}{\lambda}\int_t^s q(\tilde{x}_r)\, dr\big)\, \Psi(\tilde{x}_s, s)$. La formule d'Itô donne une dérive proportionnelle à $-\tfrac{q}{\lambda}\Psi + \partial_s \Psi + f^\top \nabla \Psi + \tfrac{1}{2}\operatorname{tr}(G\Sigma G^\top \nabla^2 \Psi)$, nulle d'après l'EDP. $M$ est donc une martingale, et $\Psi(x, t) = M_t = \mathbb{E}[M_{t_f}]$, qui est exactement l'expression annoncée.

En notant $S(\tau) = \phi(\tilde{x}_{t_f}) + \int_t^{t_f} q(\tilde{x}_s)\, ds$ le coût d'état d'un chemin $\tau$, on obtient

$$
V(x, t) = -\lambda \log \mathbb{E}_{\tau}\left[ e^{-S(\tau)/\lambda} \right].
$$

Ce résultat est remarquable. La valeur d'un problème de contrôle optimal non linéaire s'obtient sans résoudre d'EDP ni optimiser quoi que ce soit : il suffit de simuler le système **sans commande**, de laisser le bruit explorer, et de moyenner $e^{-S/\lambda}$ sur les chemins obtenus. La quantité $-\lambda \log \mathbb{E}[e^{-S/\lambda}]$ est un **minimum doux** (*soft-min*) : quand $\lambda \to 0$ elle tend vers le plus petit coût atteignable par le bruit, quand $\lambda \to \infty$ vers le coût moyen $\mathbb{E}[S]$. Le nom d'intégrale de chemin vient de la physique : l'intégrale de chemin de Feynman en mécanique quantique a la même structure formelle, une somme sur tous les chemins pondérée par l'exponentielle d'une action.

Reste à obtenir la commande. D'après la section 3.2 et la condition $\lambda R^{-1} = \Sigma$,

$$
u^\ast = -R^{-1} G^\top \nabla V = \lambda R^{-1} G^\top \frac{\nabla \Psi}{\Psi} = \Sigma G^\top \frac{\nabla \Psi}{\Psi}.
$$

Le gradient de $\Psi$ s'estime lui aussi à partir des chemins, grâce à une intégration par parties gaussienne. Sur le premier intervalle, $\tilde{x}_{t+dt} = x + f\, dt + G\, dw$, et au premier ordre

$$
\mathbb{E}\big[ \Psi(\tilde{x}_{t+dt}, t + dt)\, dw \big] \approx \mathbb{E}\big[ \big(\Psi + \nabla \Psi^\top G\, dw\big)\, dw \big] = \Sigma G^\top \nabla \Psi\, dt,
$$

puisque $\mathbb{E}[dw] = 0$ et $\mathbb{E}[dw\, dw^\top] = \Sigma\, dt$. Par la propriété de la tour, $\Psi(\tilde{x}_{t+dt}, t+dt)$ peut être remplacé par $e^{-S/\lambda}$ sous l'espérance, et le facteur $e^{-q(x)dt/\lambda}$ du premier intervalle vaut 1 au premier ordre. On obtient

$$
\boxed{\; u^\ast(x, t)\, dt = \frac{\mathbb{E}\big[ e^{-S(\tau)/\lambda}\, dw_t \big]}{\mathbb{E}\big[ e^{-S(\tau)/\lambda} \big]} \;}
$$

C'est MPPI sous forme continue : on simule depuis l'état courant des chemins bruités sans commande, on pondère chacun par $e^{-S/\lambda}$, et la commande optimale à l'instant présent est la moyenne pondérée du premier incrément de bruit. Un incrément de bruit qui a mené à un futur peu coûteux tire la commande dans sa direction. La justification rigoureuse des approximations au premier ordre en $dt$ se trouve chez Kappen (2005) et Theodorou, Buchli et Schaal (2010).

L'algorithme discret de la section 2 applique ce résultat avec deux modifications. Il échantillonne autour de la séquence nominale $U$ au lieu de zéro, ce qui est beaucoup plus efficace mais demande une correction d'échantillonnage préférentiel. Et il met à jour tous les pas de temps de la séquence d'un coup. La lecture informationnelle rend ces deux points transparents.

### 3.4 Lecture informationnelle : énergie libre et divergence KL

On se place maintenant en temps discret, sans hypothèse de forme sur la dynamique. Soit $V = (v_0, \dots, v_{T-1}) \in \mathbb{R}^{Tm}$ une séquence de commandes effectivement appliquées. Le modèle $F$ étant déterministe et $x_0$ fixé, la trajectoire est une fonction de $V$, et l'on note $S(V) = \phi(x_T) + \sum_t c(x_t)$ son coût d'état. On définit deux lois sur les séquences :

- la loi de référence $\mathbb{P}$, où les $v_t$ sont i.i.d. de loi $\mathcal{N}(0, \Sigma)$, qui modélise le système livré au seul bruit ;
- la loi commandée $\mathbb{Q}_U$, où $v_t \sim \mathcal{N}(u_t, \Sigma)$ indépendamment, qui modélise le système sous la commande $U$ avec le même bruit.

Leurs densités sont $p(V) \propto \exp(-\tfrac{1}{2}\sum_t v_t^\top \Sigma^{-1} v_t)$ et $q_U(V) \propto \exp(-\tfrac{1}{2}\sum_t (v_t - u_t)^\top \Sigma^{-1} (v_t - u_t))$. Leur divergence de Kullback-Leibler se calcule directement :

$$
D_{\mathrm{KL}}(\mathbb{Q}_U \parallel \mathbb{P}) = \mathbb{E}_{\mathbb{Q}_U}\left[ \log \frac{q_U}{p} \right]
= \mathbb{E}_{\mathbb{Q}_U}\left[ \sum_t \Big( u_t^\top \Sigma^{-1} v_t - \tfrac{1}{2} u_t^\top \Sigma^{-1} u_t \Big) \right]
= \frac{1}{2} \sum_t u_t^\top \Sigma^{-1} u_t.
$$

Par conséquent

$$
\mathbb{E}_{\mathbb{Q}_U}[S] + \lambda\, D_{\mathrm{KL}}(\mathbb{Q}_U \parallel \mathbb{P}) = \mathbb{E}_{\mathbb{Q}_U}\left[ S(V) + \sum_t \tfrac{1}{2} u_t^\top (\lambda \Sigma^{-1}) u_t \right].
$$

Le membre de droite est le problème de contrôle stochastique de la section 3.1 avec $R = \lambda \Sigma^{-1}$. La condition qui semblait technique en temps continu devient ici une identité : **le coût quadratique de la commande est exactement $\lambda$ fois la divergence KL entre le système commandé et le système libre**. En temps continu, le théorème de Girsanov donne le même résultat, $D_{\mathrm{KL}} = \mathbb{E}\big[\tfrac{1}{2}\int u^\top \Sigma^{-1} u\, dt\big]$.

On élargit alors le problème : au lieu de chercher une commande $U$, on cherche une loi quelconque $\mathbb{Q}$ sur les séquences, absolument continue par rapport à $\mathbb{P}$, qui minimise $\mathbb{E}_{\mathbb{Q}}[S] + \lambda D_{\mathrm{KL}}(\mathbb{Q} \parallel \mathbb{P})$. Ce problème a une solution explicite, connue sous le nom de **principe variationnel de Gibbs** (ou formule de Donsker-Varadhan). On définit la loi $\mathbb{Q}^\ast$ par sa densité

$$
\frac{d\mathbb{Q}^\ast}{d\mathbb{P}}(V) = \frac{e^{-S(V)/\lambda}}{\eta}, \qquad \eta = \mathbb{E}_{\mathbb{P}}\big[ e^{-S/\lambda} \big].
$$

Pour toute loi $\mathbb{Q}$,

$$
D_{\mathrm{KL}}(\mathbb{Q} \parallel \mathbb{Q}^\ast) = \mathbb{E}_{\mathbb{Q}}\left[ \log \frac{d\mathbb{Q}}{d\mathbb{P}} + \frac{S}{\lambda} + \log \eta \right]
= D_{\mathrm{KL}}(\mathbb{Q} \parallel \mathbb{P}) + \frac{\mathbb{E}_{\mathbb{Q}}[S]}{\lambda} + \log \eta,
$$

donc

$$
\mathbb{E}_{\mathbb{Q}}[S] + \lambda D_{\mathrm{KL}}(\mathbb{Q} \parallel \mathbb{P}) = \lambda D_{\mathrm{KL}}(\mathbb{Q} \parallel \mathbb{Q}^\ast) \underbrace{- \lambda \log \mathbb{E}_{\mathbb{P}}\big[e^{-S/\lambda}\big]}_{\mathcal{F}}.
$$

La divergence KL étant positive et nulle seulement si les lois coïncident, le minimum est atteint en $\mathbb{Q} = \mathbb{Q}^\ast$ et vaut $\mathcal{F}$. La quantité $\mathcal{F}$ est l'**énergie libre** ; c'est l'analogue discret de la fonction valeur $V = -\lambda \log \Psi$ de la section 3.3. Aucun contrôleur, quel qu'il soit, ne fait mieux que $\mathcal{F}$ sur ce critère.

Le vocabulaire vient de la physique statistique : $\mathbb{Q}^\ast$ est une distribution de Boltzmann où $S$ joue le rôle de l'énergie et $\lambda$ celui de la température, d'où le nom du paramètre. Pour quelqu'un qui vient de l'apprentissage automatique, la structure est aussi celle de l'inférence bayésienne : $\mathbb{P}$ est un a priori, $e^{-S/\lambda}$ une vraisemblance, $\mathbb{Q}^\ast$ l'a posteriori, et $\mathbb{E}_{\mathbb{Q}}[S] + \lambda D_{\mathrm{KL}}(\mathbb{Q} \parallel \mathbb{P})$ est, au facteur $\lambda$ près, l'opposé de l'ELBO de l'inférence variationnelle. Cette correspondance porte un nom dans la littérature, *control as inference*.

### 3.5 De la distribution optimale à une mise à jour calculable

La loi $\mathbb{Q}^\ast$ est optimale, mais inutilisable telle quelle : c'est une loi sur $\mathbb{R}^{100}$, non gaussienne, éventuellement multimodale, dont on ne sait pas tirer d'échantillons. Un actionneur ne sait de toute façon appliquer qu'une commande. MPPI procède en deux approximations.

**Projection sur les lois gaussiennes.** On cherche la loi commandée $\mathbb{Q}_U$ la plus proche de $\mathbb{Q}^\ast$ au sens de $D_{\mathrm{KL}}(\mathbb{Q}^\ast \parallel \mathbb{Q}_U)$ :

$$
D_{\mathrm{KL}}(\mathbb{Q}^\ast \parallel \mathbb{Q}_U) = \text{cste} - \mathbb{E}_{\mathbb{Q}^\ast}[\log q_U(V)] = \text{cste} + \mathbb{E}_{\mathbb{Q}^\ast}\left[ \tfrac{1}{2} \sum_t (v_t - u_t)^\top \Sigma^{-1} (v_t - u_t) \right].
$$

C'est une fonction quadratique strictement convexe de chaque $u_t$, minimale en

$$
u_t^\ast = \mathbb{E}_{\mathbb{Q}^\ast}[v_t].
$$

La meilleure commande gaussienne est la moyenne de la distribution optimale.

**Échantillonnage préférentiel.** On ne sait pas échantillonner $\mathbb{Q}^\ast$, mais on sait échantillonner $\mathbb{Q}_{\hat{U}}$ autour de la séquence nominale courante $\hat{U}$. On réécrit l'espérance :

$$
\mathbb{E}_{\mathbb{Q}^\ast}[v_t] = \mathbb{E}_{\mathbb{Q}_{\hat{U}}}\left[ \frac{q^\ast(V)}{q_{\hat{U}}(V)}\, v_t \right],
\qquad \frac{q^\ast(V)}{q_{\hat{U}}(V)} = \frac{e^{-S(V)/\lambda}}{\eta} \cdot \frac{p(V)}{q_{\hat{U}}(V)}.
$$

Avec $V = \hat{U} + \mathcal{E}$, où $\mathcal{E} = (\varepsilon_0, \dots, \varepsilon_{T-1})$ est le bruit tiré,

$$
\frac{p(V)}{q_{\hat{U}}(V)} = \exp\left( -\tfrac{1}{2} \sum_t (\hat{u}_t + \varepsilon_t)^\top \Sigma^{-1} (\hat{u}_t + \varepsilon_t) + \tfrac{1}{2} \sum_t \varepsilon_t^\top \Sigma^{-1} \varepsilon_t \right)
= \exp\left( -\sum_t \Big( \tfrac{1}{2} \hat{u}_t^\top \Sigma^{-1} \hat{u}_t + \hat{u}_t^\top \Sigma^{-1} \varepsilon_t \Big) \right).
$$

Le facteur en $\hat{u}_t^\top \Sigma^{-1} \hat{u}_t$ est le même pour tous les échantillons et disparaît à la normalisation. Il reste

$$
\frac{q^\ast(V)}{q_{\hat{U}}(V)} \propto \exp\left( -\frac{1}{\lambda} \Big[ S(\hat{U} + \mathcal{E}) + \lambda \sum_t \hat{u}_t^\top \Sigma^{-1} \varepsilon_t \Big] \right).
$$

En remplaçant l'espérance par une moyenne sur $K$ échantillons et en normalisant les poids empiriquement (échantillonnage préférentiel autonormalisé), on obtient exactement la mise à jour de la section 2.2 :

$$
u_t \leftarrow \sum_k w^k (\hat{u}_t + \varepsilon^k_t) = \hat{u}_t + \sum_k w^k \varepsilon^k_t,
\qquad w^k = \frac{e^{-\tilde{S}^k/\lambda}}{\sum_j e^{-\tilde{S}^j/\lambda}},
\qquad \tilde{S}^k = S^k + \lambda \sum_t \hat{u}_t^\top \Sigma^{-1} \varepsilon^k_t.
$$

Le terme $\lambda \sum_t \hat{u}_t^\top \Sigma^{-1} \varepsilon^k_t$ de l'étape 4 n'est donc pas un ajout heuristique. Il compense le fait qu'on échantillonne autour de $\hat{U}$ alors que la référence $\mathbb{P}$ est centrée en zéro, et rétablit la préférence pour les petites commandes encodée dans $\mathbb{P}$. Les implémentations remplacent souvent $\lambda$ par un $\gamma$ plus petit dans ce terme (section 4.5).

Trois propriétés de cet estimateur comptent en pratique. Il est biaisé à $K$ fini, avec un biais en $O(1/K)$, mais convergent. Sa variance est gouvernée par la taille effective d'échantillon plutôt que par $K$ (section 4.1). Et répéter la mise à jour à état initial fixé est une itération de point fixe qui rapproche $\hat{U}$ de la moyenne de $\mathbb{Q}^\ast$ ; en temps réel, le démarrage à chaud étale ces itérations sur les pas de temps successifs.

Le sens de la divergence a une conséquence directe sur le comportement. $D_{\mathrm{KL}}(\mathbb{Q}^\ast \parallel \mathbb{Q}_U)$ pénalise lourdement les régions où $\mathbb{Q}^\ast$ a de la masse et $\mathbb{Q}_U$ n'en a pas : la gaussienne projetée doit couvrir toute la masse de $\mathbb{Q}^\ast$. Si $\mathbb{Q}^\ast$ est bimodale, par exemple parce qu'on peut contourner un obstacle par la gauche ou par la droite, la gaussienne qui couvre les deux modes est centrée entre eux, c'est-à-dire sur l'obstacle. La divergence dans l'autre sens choisirait un seul mode, mais elle n'admet pas de mise à jour explicite aussi simple. Ce défaut structurel est traité en section 7.4, et plusieurs variantes de la section 8 existent précisément pour le corriger.

### 3.6 Pourquoi l'exponentielle

On pourrait imaginer d'autres règles : garder les 5 % meilleures trajectoires et les moyenner (c'est la méthode de l'entropie croisée, CEM), pondérer par $1/S$, par le rang. Quatre arguments montrent que l'exponentielle est la forme juste pour le problème posé.

**Elle est l'unique solution du problème régularisé.** Le principe de Gibbs dit plus qu'une inégalité : la fonctionnelle $\mathbb{Q} \mapsto \mathbb{E}_{\mathbb{Q}}[S] + \lambda D_{\mathrm{KL}}(\mathbb{Q} \parallel \mathbb{P})$ est strictement convexe et $\mathbb{Q}^\ast \propto e^{-S/\lambda}\, \mathbb{P}$ est son unique minimiseur. Toute autre pondération optimise un autre critère, ou aucun. Garder les meilleures trajectoires revient à remplacer $e^{-S/\lambda}$ par une indicatrice de seuil : la règle a ses mérites empiriques, mais elle ne résout pas ce problème de contrôle.

**Elle respecte la structure temporelle.** Le coût est additif dans le temps, $S = \sum_t c_t$, et l'exponentielle transforme cette somme en produit : $e^{-S/\lambda} = \prod_t e^{-c_t/\lambda}$. Comme $\mathbb{P}$ est un produit de lois indépendantes par pas de temps, $\mathbb{Q}^\ast$ restreinte aux commandes à partir du pas $s$, conditionnellement aux commandes antérieures, est la distribution optimale du même problème posé depuis l'état atteint au pas $s$. C'est la cohérence de Bellman : la solution restreinte à un sous-horizon reste la solution du sous-problème. Exiger qu'une pondération continue $g$ vérifie $g(a + b) = g(a)\, g(b)$ pour que cette propriété tienne, c'est poser l'équation fonctionnelle de Cauchy, dont les seules solutions continues positives sont les exponentielles $g(s) = e^{-s/\lambda}$.

**Elle interpole entre les deux stratégies naïves.** Quand $\lambda \to 0$, tout le poids se concentre sur la meilleure trajectoire : MPPI devient une recherche aléatoire qui prend l'argmin. Quand $\lambda \to \infty$, tous les poids s'égalisent et la mise à jour est la moyenne du bruit, qui tend vers zéro. Entre les deux, l'exponentielle utilise l'écart de coût en valeur : un échantillon plus mauvais de 0,1 et un autre plus mauvais de 100 ne sont pas traités de la même manière, alors que la sélection d'une élite les confond dès qu'ils sont tous deux dans l'élite ou tous deux dehors. Cette sensibilité à l'échelle du coût est une force, et aussi la source du problème de réglage de la section 4.1.

**Elle réalise une descente de gradient régularisée sans calculer de gradient.** C'est l'argument le plus éclairant pour comprendre le rôle de $\lambda$ et $\Sigma$. Plaçons-nous dans la limite $K \to \infty$ et supposons le coût localement quadratique autour de $\hat{U}$ à l'échelle du bruit : $S(\hat{U} + \mathcal{E}) \approx S(\hat{U}) + g^\top \mathcal{E} + \tfrac{1}{2} \mathcal{E}^\top H \mathcal{E}$, où $\mathcal{E} \in \mathbb{R}^{Tm}$ et $\Sigma$ désigne ici la covariance diagonale par blocs de tout le vecteur de bruit. La mise à jour est la moyenne de $\mathcal{E}$ sous la densité

$$
\propto \exp\left( -\tfrac{1}{2} \mathcal{E}^\top \Sigma^{-1} \mathcal{E} - \tfrac{1}{\lambda} \big( g^\top \mathcal{E} + \tfrac{1}{2} \mathcal{E}^\top H \mathcal{E} \big) \right),
$$

qui est une gaussienne de précision $\Sigma^{-1} + H/\lambda$ (supposée définie positive) et de moyenne

$$
\Delta U = -\left( \Sigma^{-1} + \frac{H}{\lambda} \right)^{-1} \frac{g}{\lambda} = -\big( \lambda \Sigma^{-1} + H \big)^{-1} g.
$$

C'est un pas de Newton amorti, de type Levenberg-Marquardt, avec l'amortissement $\lambda \Sigma^{-1}$, c'est-à-dire la matrice $R$ du coût de commande. Si le coût est localement linéaire ($H = 0$), on obtient $\Delta U = -\Sigma g / \lambda$ : un pas de gradient préconditionné par $\Sigma$, de longueur inversement proportionnelle à $\lambda$. MPPI estime ce pas uniquement à partir d'évaluations du coût, sans jamais dériver ni le modèle ni le coût, et le calcul reste valable quand ceux-ci ne sont pas dérivables, pourvu qu'ils soient réguliers « en moyenne » à l'échelle du bruit. Dans ce régime, $\Sigma / \lambda$ joue le rôle d'un pas d'apprentissage, et l'on retrouve le couplage entre $\lambda$ et $\Sigma$ imposé par la condition $\lambda R^{-1} = \Sigma$.

---

## 4. Les paramètres et leur effet

### 4.1 La température

**Ce qu'elle contrôle.** La température $\lambda$ fixe la sélectivité de la pondération. Un échantillon dont le coût dépasse le minimum de $\Delta S$ reçoit un poids relatif $e^{-\Delta S/\lambda}$. Avec $\lambda = 1$, un surcoût de 1 divise le poids par 2,7, un surcoût de 5 par 150 et un surcoût de 10 par 22 000. Tout se joue donc dans le rapport entre $\lambda$ et la dispersion des coûts **au voisinage du minimum**.

**Taille effective d'échantillon et échelle.** Supposons que les coûts des échantillons suivent une loi normale d'écart-type $s$. Les poids bruts $e^{-S/\lambda}$ suivent alors une loi log-normale et, pour $K$ grand,

$$
\frac{\mathrm{ESS}}{K} \approx \frac{\mathbb{E}[e^{-S/\lambda}]^2}{\mathbb{E}[e^{-2S/\lambda}]} = \frac{e^{-2\mu/\lambda + s^2/\lambda^2}}{e^{-2\mu/\lambda + 2 s^2/\lambda^2}} = e^{-(s/\lambda)^2}.
$$

Avec $K = 8192$ :

| $s / \lambda$ | $\mathrm{ESS}/K$ | ESS |
|---|---|---|
| 0,5 | 0,78 | 6380 |
| 1 | 0,37 | 3014 |
| 1,5 | 0,11 | 863 |
| 2 | 0,018 | 150 |
| 2,5 | 0,0019 | 16 |
| 3 | 0,00012 | environ 1, l'approximation ne vaut plus |

La leçon de ce tableau est la raideur de la transition : en divisant $\lambda$ par trois, on passe de 3000 échantillons effectifs à un seul. L'hypothèse gaussienne est grossière (la distribution réelle des coûts est asymétrique, bornée à gauche, avec une queue droite lourde due aux sorties de piste), mais l'ordre de grandeur est représentatif. Ce qui compte, c'est la queue gauche : les échantillons à +1000 reçoivent un poids nul et n'influencent rien, alors qu'ils gonflent l'écart-type. Pour estimer $s$ en pratique, il vaut mieux mesurer la dispersion des meilleurs 10 % des coûts qu'utiliser l'écart-type global.

**L'interaction avec l'échelle du coût.** Les poids ne dépendent que de $(S^k - \rho)/\lambda$. Multiplier tous les termes du coût par une constante $c$ équivaut exactement à diviser $\lambda$ par $c$. Une valeur de $\lambda$ n'a donc aucun sens isolément : elle ne se lit que rapportée à l'échelle du coût. C'est la source d'erreur la plus fréquente avec MPPI, parce que l'échelle du coût change silencieusement au gré de décisions d'écriture qui paraissent anodines.

Prenons les poids actuels du fichier de configuration, avec un coût d'écart latéral $w_{\text{lat}}\, d_t^2$ sommé sur les 50 pas et $w_{\text{lat}} = 1$. Des trajectoires sur la piste dont l'écart latéral moyen varie de 0,1 à 0,3 m ont des coûts latéraux compris entre 0,5 et 4,5 ; une dispersion $s$ de l'ordre de 1 donne, avec $\lambda = 1$, une ESS de l'ordre de 3000. Trois variantes d'écriture du même coût :

- on décide d'écrire le coût comme une intégrale, $\sum_t c_t\, \Delta t$ : toute l'échelle est divisée par 50, $s/\lambda \approx 0{,}02$, les poids sont uniformes et le contrôleur devient inerte ;
- la grille stocke l'écart latéral en centimètres au lieu de mètres : le terme est multiplié par $10^4$, $s/\lambda \approx 10^4$, un seul échantillon porte tout le poids et la commande devient erratique ;
- on passe $w_{\text{offtrack}}$ de 100 à 1000 : sur une ligne droite au milieu de la piste, aucun échantillon ne sort et rien ne change ; en virage serré près du bord, la moitié des échantillons sortent et la dispersion au voisinage du minimum explose. La température effective dépend alors de la situation.

Le même $\lambda = 1$ produit donc un contrôleur correct, inerte ou effondré selon une convention d'écriture.

**Trop petit.** La pondération se concentre sur quelques échantillons, l'ESS tombe à l'unité ou à la dizaine. La mise à jour recopie essentiellement le bruit du meilleur échantillon, qui change d'une itération à l'autre. Symptômes : commande de braquage qui saute à haute fréquence, trajectoire en zigzag, comportement qui change nettement quand on change la graine du générateur aléatoire.

**Trop grand.** Les poids sont presque uniformes, l'ESS approche $K$, et la mise à jour vaut à peu près la moyenne du bruit, dont l'écart-type est $\sigma/\sqrt{K}$ (0,001 rad pour le braquage). La séquence nominale ne bouge presque plus. Symptômes : le véhicule réagit tard ou pas du tout, élargit les virages, fonce vers un obstacle que les rollouts voient pourtant clairement.

**Comment le régler.** On journalise l'ESS à chaque itération et on vise une plage de l'ordre de 1 % à 10 % de $K$, soit 80 à 800 échantillons effectifs pour $K = 8192$. D'après le modèle gaussien, cela correspond à $\lambda$ entre $s/2{,}1$ et $s/1{,}5$. On règle $\lambda$ après avoir normalisé le coût (section 5.3), jamais avant, et on le revérifie à chaque modification d'un poids.

Une alternative robuste consiste à adapter $\lambda$ à chaque itération pour maintenir l'ESS dans la plage visée. L'ESS est une fonction croissante de $\lambda$ : c'est l'exponentielle de l'entropie de Rényi d'ordre 2 d'une distribution de Gibbs, qui croît avec la température. Une dichotomie d'une vingtaine d'étapes, chacune sur 8192 exponentielles, coûte donc une fraction de milliseconde. Ce choix a deux contreparties. Le problème optimisé change d'une itération à l'autre, ce qui efface l'interprétation de la section 3. Et quand le coût est plat (tous les échantillons équivalents), l'adaptation fait chuter $\lambda$ pour recréer artificiellement de la sélectivité, ce qui amplifie le bruit ; il faut donc borner $\lambda$. La normalisation min-max des coûts, $(S - \min S)/(\max S - \min S)$, est plus simple mais dépend de l'échantillon le plus mauvais, en général un rollout sorti de piste sans intérêt : elle est à éviter.

### 4.2 La covariance du bruit

**Ce qu'elle contrôle.** $\Sigma$ fixe la zone explorée autour de la séquence nominale. Avec $\sigma_a = 0{,}5$ m/s² et $\sigma_\delta = 0{,}1$ rad, l'exploration couvre environ 6 % de la plage d'accélération (8 m/s²) et 12,5 % de la plage de braquage (0,8 rad) par écart-type. D'après la section 3.6, $\Sigma/\lambda$ joue aussi le rôle d'un pas d'apprentissage : plus $\Sigma$ est grand, plus la mise à jour peut être ample.

**Le bruit résiduel dans la commande.** Même si le coût ne distinguait pas les échantillons, la moyenne pondérée du bruit ne serait pas nulle. Pour des poids indépendants du bruit, sa variance vaut $\sigma^2 \sum_k (w^k)^2 = \sigma^2/\mathrm{ESS}$. Pour le braquage, cela donne un bruit de commande de $0{,}1/\sqrt{100} = 0{,}01$ rad avec une ESS de 100, et de $0{,}1/\sqrt{4} = 0{,}05$ rad (près de 3 degrés) avec une ESS de 4. Ce bruit s'ajoute à chaque itération, à 50 Hz. Le compromis est donc direct : un grand $\sigma$ explore mieux mais injecte davantage de bruit dans la commande, sauf si l'ESS reste élevée.

**La saturation.** Quand la commande nominale est proche d'une borne, une grande partie des échantillons est écrêtée. En virage serré avec $u_\delta = 0{,}35$ rad, la probabilité qu'un échantillon dépasse 0,4 rad est $\mathbb{P}(\varepsilon > 0{,}05) = \mathbb{P}(Z > 0{,}5) \approx 31\,\%$. Ces échantillons ne portent plus d'information d'un côté, et l'exploration devient asymétrique. La fraction d'échantillons saturés est une grandeur à journaliser.

**La corrélation temporelle.** Le bruit blanc de l'étape 1 est tiré indépendamment à chaque pas de 20 ms. Le véhicule filtre fortement ces variations rapides : il intègre le braquage en cap, puis le cap en position. Un calcul aux petits angles le montre. Avec le modèle cinématique à vitesse constante $v$, l'écart latéral en fin d'horizon dû au bruit de braquage vaut

$$
y_T \approx \frac{v^2 \Delta t^2}{L} \sum_{j=0}^{T-1} (T - 1 - j)\, \varepsilon_j.
$$

Pour un bruit blanc, l'écart-type est $\sigma_\delta \frac{v^2 \Delta t^2}{L} \sqrt{\sum_{i=0}^{T-1} i^2}$, avec $\sum_{i=0}^{49} i^2 = 40\,425$. À 5 m/s, $\frac{v^2 \Delta t^2}{L} = 0{,}0303$ m/rad et l'écart-type vaut $0{,}1 \times 0{,}0303 \times 201 \approx 0{,}61$ m. Pour un bruit parfaitement corrélé (le même écart de braquage sur toute la séquence), la somme devient $\sum_i i = 1225$ et l'écart-type formel vaut 3,7 m. L'approximation aux petits angles ne tient plus à cette amplitude, puisqu'un braquage constant de 0,1 rad fait tourner la voiture de 1,5 rad en une seconde, mais l'ordre de grandeur reste : à amplitude égale, le bruit blanc explore six fois moins de positions latérales qu'un bruit corrélé, tout en produisant des séquences de commande hachées. On y remédie en tirant un bruit coloré (filtré passe-bas dans le temps) ou en paramétrant la séquence par quelques points de contrôle interpolés. Six nœuds de spline sur 50 pas ramènent la dimension explorée de 100 à 12, et à nombre d'échantillons égal, l'échantillonnage devient beaucoup plus efficace.

**Trop petit.** Les rollouts forment un faisceau étroit autour de la trajectoire nominale. Le contrôleur ne découvre pas de manœuvre nouvelle, reste bloqué derrière un obstacle, et converge lentement quand la situation change brutalement.

**Trop grand.** La majorité des rollouts sortent de piste ou saturent, l'ESS s'effondre, la commande devient bruitée. Le faisceau de trajectoires ressemble à un éventail qui couvre toute la piste.

**Comment la régler.** On part d'écarts-types de 10 à 20 % de la plage de chaque commande et on regarde le faisceau. Tracer les $K$ rollouts d'une itération, colorés par leur poids, est le diagnostic le plus utile de tout le projet : on doit voir un faisceau qui couvre la largeur de piste utile sans la déborder massivement, et un sous-ensemble de trajectoires lourdes qui dessine un comportement sensé. On vérifie ensuite la fraction de saturation et le bruit de la commande appliquée. Enfin on se souvient que modifier $\Sigma$ modifie la dispersion des coûts, donc l'ESS : $\lambda$ est à revérifier.

### 4.3 L'horizon

**Ce qu'il contrôle.** L'horizon $T \Delta t$ fixe la distance d'anticipation. Avec 50 pas de 20 ms, le contrôleur voit une seconde devant lui, soit $v \times 1$ s de piste. Il faut comparer cette distance à ce que le véhicule doit pouvoir faire dans l'horizon, et d'abord s'arrêter. Avec la décélération maximale de 4 m/s², l'arrêt depuis la vitesse $v$ prend $v/4$ secondes sur $v^2/8$ mètres.

| Vitesse (m/s) | Distance vue (m) | Distance d'arrêt (m) | Durée d'arrêt (s) |
|---|---|---|---|
| 2 | 2 | 0,5 | 0,5 |
| 4 | 4 | 2 | 1,0 |
| 6 | 6 | 4,5 | 1,5 |
| 8 | 8 | 8 | 2,0 |
| 10 | 10 | 12,5 | 2,5 |

Jusqu'à 8 m/s, un obstacle qui entre dans l'horizon est encore évitable par un freinage maximal, même si au-delà de 4 m/s l'arrêt complet ne tient plus dans l'horizon et le contrôleur ne voit pas la fin de sa propre manœuvre. Au-dessus de 8 m/s, il découvre l'obstacle trop tard. Ce calcul ignore la latence et la limite d'adhérence, qui aggravent la situation.

**Trop court.** Le contrôleur est myope : il freine trop tard, entre trop vite dans les virages et ne voit pas qu'une ligne serrée à l'entrée d'un virage compromet la sortie. Un coût terminal bien choisi, qui estime le coût restant au-delà de l'horizon (vitesse compatible avec la courbure à venir, progression), compense partiellement.

**Trop long.** Le temps de calcul croît linéairement avec $T$. Les erreurs du modèle s'accumulent et la fin de la trajectoire prédite devient fictive. Le bruit s'accumule aussi : l'écart latéral dû au bruit blanc croît comme $T^{3/2}$ d'après le calcul de la section 4.2, donc une part croissante des rollouts sort de piste en fin d'horizon, ce qui fait baisser l'ESS. Enfin, les derniers pas de la séquence sont mal optimisés, puisque leur effet sur le coût est faible et bruité.

**Pas de temps et horizon.** On pourrait garder une seconde d'horizon avec $\Delta t = 40$ ms et $T = 25$, pour moitié moins de calcul. Mais la commande serait deux fois moins fine, et l'intégration d'Euler du modèle dynamique deviendrait instable en dessous de 1,9 m/s au lieu de 0,95 m/s (section 6.5). On peut dissocier les deux : garder une séquence de commande à 20 ms et intégrer le modèle avec plusieurs sous-pas par intervalle, ou l'inverse, une séquence à 40 ms interpolée. Quel que soit le choix, si la période de contrôle vaut $\Delta t$, une itération complète doit tenir dans $\Delta t$.

**Comment le régler.** On part de la distance d'arrêt ou de la distance de freinage avant le virage le plus exigeant à la vitesse maximale visée, on en déduit une durée minimale, et on vérifie que les trajectoires tracées restent plausibles en fin d'horizon. Symptômes d'un horizon trop court : freinages tardifs, sorties à l'extérieur des virages rapides. Symptômes d'un horizon trop long : ESS basse, fin de faisceau éclatée, dépassement du budget de latence.

### 4.4 Le nombre d'échantillons

**Ce qu'il contrôle.** $K$ fixe la précision de l'estimation de Monte-Carlo. L'erreur statistique sur la mise à jour décroît en $1/\sqrt{\mathrm{ESS}}$, et à température fixée l'ESS est à peu près proportionnelle à $K$ (dans le modèle gaussien, $\mathrm{ESS} \approx K e^{-(s/\lambda)^2}$). Passer de 1024 à 8192 échantillons divise donc le bruit de la commande par $\sqrt{8} \approx 2{,}8$, pour un calcul huit fois plus lourd.

**Les manœuvres rares.** $K$ détermine aussi la probabilité de découvrir une manœuvre qui n'occupe qu'une petite région de l'espace des perturbations. Si une manœuvre utile correspond à une région de probabilité $p$ sous la loi d'échantillonnage, la probabilité qu'au moins un échantillon y tombe est $1 - (1-p)^K \approx 1 - e^{-pK}$. Pour $p = 10^{-4}$, c'est 10 % avec $K = 1024$, 56 % avec $K = 8192$ et plus de 99,8 % avec $K = 65\,536$. Le démarrage à chaud aide : une manœuvre découverte une fois est conservée dans la séquence nominale et affinée aux itérations suivantes.

**Trop petit.** Commande bruitée, symptômes identiques à ceux d'une température trop basse, et sensibilité à la graine aléatoire.

**Trop grand.** Aucun défaut de contrôle, seulement de la latence. Or une latence qui dépasse le budget se paie en stabilité, ce qui annule le bénéfice.

**Comment le régler.** On fixe tous les autres paramètres et on trace une métrique de qualité (temps au tour, écart latéral moyen, nombre de sorties de piste, bruit haute fréquence du braquage) en fonction de $K \in \{256, 512, \dots, 16\,384\}$. La courbe atteint un plateau ; le bon $K$ est le début du plateau, sous réserve de tenir le budget. Les puissances de deux simplifient le découpage en blocs CUDA. On garde en tête que le bruit sur la commande dépend de l'ESS et non de $K$ : augmenter $K$ ne sert à rien si $\lambda$ maintient l'ESS à 3.

### 4.5 Les poids du coût et le terme de correction

**Ce qu'ils contrôlent.** Les poids relatifs des termes du coût fixent le compromis recherché : suivre la ligne centrale ou couper les virages, aller vite ou rester doux. Leur échelle globale n'a pas de sens propre, puisqu'elle se confond avec $\lambda$ (section 4.1). Il faut donc séparer deux questions : le rapport entre les termes, qui définit le comportement voulu, et l'échelle d'ensemble rapportée à $\lambda$, qui définit la qualité de l'estimation.

**Symptômes de poids mal équilibrés.** Un terme de progression trop fort fait couper les virages et frôler les bords. Un terme latéral trop fort rend la voiture timide et oscillante autour de la ligne centrale. Une régularisation de commande trop forte donne une voiture paresseuse qui sous-braque dans les virages longs. Une pénalité de sortie trop faible laisse passer des raccourcis hors piste. Chacun de ces comportements est lisible directement sur la trajectoire suivie, à condition de ne modifier qu'un poids à la fois.

**Le poids $\gamma$ de la correction.** La théorie impose $\gamma = \lambda$ dans le terme $\gamma \sum_t u_t^\top \Sigma^{-1} \varepsilon_t$. Avec cette valeur, le contrôleur optimise effectivement $S + \tfrac{\lambda}{2} \sum_t u_t^\top \Sigma^{-1} u_t$ : il est tiré vers les commandes nulles, $a = 0$ et $\delta = 0$. En ligne droite c'est inoffensif ; dans un virage long, qui demande un braquage constant non nul, le contrôleur résiste au braquage nécessaire. En pratique, les implémentations utilisent un $\gamma$ nettement plus petit que $\lambda$. Dans `nav2_mppi_controller`, les valeurs par défaut documentées sont de l'ordre de 0,3 pour la température et 0,015 pour `gamma`. Le symptôme d'un $\gamma$ trop grand est une voiture qui élargit systématiquement les virages longs alors que le coût latéral devrait l'en empêcher. Avec $\gamma = 0$, on perd la correction d'échantillonnage préférentiel, sans conséquence visible en général quand le bruit est petit devant les commandes.

### 4.6 Ordre de réglage

Les paramètres interagissent, mais un ordre de réglage limite les allers-retours. On fixe d'abord $T$ et $\Delta t$ à partir des distances d'arrêt et du budget de latence. On normalise ensuite chaque terme du coût par son échelle caractéristique (section 5.3), sans encore toucher aux poids relatifs. On choisit $\Sigma$ en regardant le faisceau de trajectoires. On règle $\lambda$ pour placer l'ESS dans la plage visée. On ajuste enfin les poids relatifs un par un, en observant la métrique qu'ils sont censés influencer, et en revérifiant l'ESS après chaque changement. $K$ se fixe en dernier, par la courbe de plateau.

---

## 5. La fonction de coût

### 5.1 Structure

Le coût d'une trajectoire a la forme générale

$$
S = \phi(x_T) + \sum_{t=1}^{T} c(x_t, u_{t-1}) + \gamma \sum_{t=0}^{T-1} u_t^\top \Sigma^{-1} \varepsilon_t.
$$

MPPI ne demande aucune propriété mathématique au coût : ni continuité, ni convexité, ni dérivabilité. Il en demande une autre, plus subtile : que les écarts de coût entre échantillons voisins renseignent sur la direction à suivre. Un coût discontinu fonctionne très bien s'il reste informatif ; un coût lisse mais plat sur toute la région explorée ne fournit aucune information.

Deux propriétés du cadre discret pèsent sur la conception. Le coût n'est évalué qu'aux instants $t \Delta t$ : à 8 m/s, deux évaluations successives sont séparées de 16 cm, et ce qui se passe entre les deux est invisible. Et le coût est évalué 409 600 fois par itération : chaque terme doit être peu coûteux, idéalement une lecture dans une grille précalculée et quelques opérations arithmétiques. C'est la raison du choix de la grille dans le projet : chaque cellule stocke l'écart latéral signé $d$ à la ligne centrale et la progression curviligne $s$.

### 5.2 Les termes usuels du suivi de piste

**L'écart latéral.** Le terme de base est $(d_t / d_0)^2$, où $d_t$ est l'écart latéral signé lu dans la grille et $d_0$ une échelle caractéristique (par exemple 0,2 m). Il maintient la voiture près de la ligne de référence. Pour un comportement de course, on veut au contraire laisser le véhicule utiliser la largeur de piste : on peut alors remplacer le carré par une zone morte, nulle tant que $|d_t|$ reste sous un seuil, ou supprimer ce terme au profit de la seule pénalité de sortie.

**La progression.** Pour aller vite, on récompense la distance curviligne parcourue. La forme la plus simple est terminale :

$$
\phi(x_T) = -w_s\, \frac{\operatorname{wrap}(s_T - s_0)}{v_{\text{ref}}\, T \Delta t},
$$

normalisée par la distance que parcourrait le véhicule à une vitesse de référence, de sorte qu'une progression « normale » vaut $-w_s$. La fonction $\operatorname{wrap}$ ramène la différence dans $[-\ell/2, \ell/2[$, où $\ell$ est la longueur du tour. Sans elle, une trajectoire qui franchit la ligne de départ voit $s$ passer de $\ell$ à 0 et reçoit une progression de $-\ell$ : toutes les trajectoires qui franchissent la ligne sont éliminées, et le véhicule freine devant elle. Pour un simple suivi à vitesse imposée, on remplace ce terme par un coût de vitesse $((v_t - v_{\text{ref}})/v_0)^2$.

**L'erreur de cap.** Le terme $(\operatorname{wrap}(\psi_t - \psi_{\text{piste}}(s_t)) / \psi_0)^2$ aligne le véhicule sur la direction de la piste. Il aide à stabiliser le suivi à basse vitesse, mais il demande de stocker le cap de la piste dans la grille.

**La sortie de piste et les obstacles.** On combine une indicatrice et un terme gradué :

$$
c_{\text{hors}}(x_t) = w_{\text{hors}}\, \mathbb{1}\big[|d_t| > d_{\max}\big] \left( 1 + \frac{|d_t| - d_{\max}}{d_0} \right),
$$

avec $d_{\max}$ égal à la demi-largeur de piste (0,8 m) moins la demi-largeur du véhicule, puisque $d_t$ est mesuré au centre de gravité alors que ce sont les roues qui sortent. L'indicatrice crée le saut de coût qui rend la sortie inacceptable, le terme gradué garde une information de direction quand tous les échantillons sont sortis (section 5.4). Pour une collision, on ajoute souvent un mécanisme de gel : le rollout s'arrête sur place et paie la pénalité pour tous les pas restants, ce qui empêche une trajectoire de traverser un mur et d'en ressortir avec un coût modéré.

**La régularisation de commande.** Deux termes coexistent : l'amplitude $\sum_j (u_{t,j}/u_{0,j})^2$, normalisée par exemple par les bornes des actionneurs, et la variation $\sum_j ((u_{t,j} - u_{t-1,j})/\Delta u_{0,j})^2$, normalisée par la variation maximale physiquement possible en un pas. Le second est le plus utile : il lisse la commande appliquée et reflète la réalité d'un servomoteur, qui ne braque pas instantanément.

**La limite d'adhérence.** Avec le modèle cinématique, qui ignore la saturation des pneus, il faut interdire par le coût les virages physiquement impossibles. L'accélération latérale prédite vaut $v^2 \tan\delta / L$ et doit rester sous $\mu g$ :

$$
c_{\text{adh}} = w_{\text{adh}} \max\!\left(0,\; \frac{|v^2 \tan\delta / L|}{\mu g} - 1\right)^2.
$$

Avec le modèle dynamique, la saturation est dans le modèle et ce terme devient inutile.

### 5.3 Normaliser les termes

Les termes du coût ont des unités différentes : des mètres carrés pour l'écart latéral, des mètres pour la progression, des radians carrés pour le braquage. Un poids $w_{\text{lat}} = 1$ n'a pas de sens tant qu'on n'a pas dit « 1 par mètre carré, par pas ». Le poids porte une unité, et sa valeur numérique dépend de conventions arbitraires : unités, sommation par pas ou intégrale, coût terminal ou cumulé.

Pour rendre ces conventions visibles, on peut calculer le **taux de change** entre deux termes. Avec les poids du fichier actuel, un écart latéral sommé par pas ($w_{\text{lat}} = 1$ m⁻²) et une progression terminale ($w_{\text{progress}} = 1$ m⁻¹), un mètre de progression vaut autant qu'un écart latéral constant $d$ tel que $50\, d^2 = 1$, soit 14 cm tenus pendant tout l'horizon. Si l'on écrit maintenant l'écart latéral comme une intégrale ($\sum_t d_t^2 \Delta t$), le même mètre de progression vaut un écart d'un mètre tenu pendant une seconde, ce qui dépasse la demi-largeur de piste. Le passage de l'un à l'autre ne change aucun poids dans le fichier et transforme un contrôleur prudent en contrôleur qui coupe tout.

Normaliser consiste à diviser chaque grandeur par une échelle caractéristique choisie selon le sens physique, de sorte que chaque terme normalisé vaille environ 1 à la limite de l'acceptable :

$$
c(x_t, u_t) = w_d \left(\frac{d_t}{d_0}\right)^2 + w_v \left(\frac{v_t - v_{\text{ref}}}{v_0}\right)^2 + w_u \sum_j \left(\frac{u_{t,j}}{u_{0,j}}\right)^2 + w_{\Delta u} \sum_j \left(\frac{u_{t,j} - u_{t-1,j}}{\Delta u_{0,j}}\right)^2 + c_{\text{hors}}(x_t).
$$

Les poids deviennent sans dimension et directement comparables : $w_d = 2$ et $w_v = 1$ signifie qu'un écart latéral de $d_0$ est deux fois plus grave qu'un écart de vitesse de $v_0$. L'échelle d'ensemble du coût est alors connue à l'avance (quelques unités à quelques dizaines pour une trajectoire correcte sur 50 pas), et le réglage de $\lambda$ devient reproductible. C'est la condition pour que le réglage de la section 4.1 garde un sens quand on modifie un poids.

### 5.4 Les pièges classiques

**Les coûts qui se compensent.** Un terme négatif (une récompense de progression) peut racheter un terme positif. Si la pénalité de sortie de piste est plus faible que le gain de progression d'un raccourci, MPPI prend le raccourci, et il a raison au regard du coût qu'on lui a donné. Avec $w_{\text{offtrack}} = 100$ par pas hors piste et une progression maximale d'une dizaine de mètres sur l'horizon, un seul pas hors piste coûte plus que tout ce que la progression peut rapporter : la compensation est impossible. Si la même pénalité est écrite en intégrale, elle tombe à 2 par pas hors piste, et un raccourci de trois pas qui fait gagner un mètre devient rentable dès que la progression pèse plus de 6 par mètre. La sommation masque aussi les pics : un coût moyen correct peut cacher un passage bref mais inacceptable, par exemple un pas de temps à 30 cm d'un obstacle. Enfin une récompense non bornée crée des comportements imprévus : sans la fonction $\operatorname{wrap}$, une trajectoire qui recule à travers la ligne de départ gagne un tour entier de progression.

**Les minima locaux.** MPPI est une méthode locale : il échantillonne autour de la séquence nominale et ne voit que ce que le bruit atteint. Un coût présentant un col (contourner un obstacle exige d'abord de s'écarter de la ligne idéale, donc de payer un surcoût temporaire) peut piéger la séquence nominale d'un côté du col. Le démarrage à chaud renforce l'effet : une fois engagé dans une solution, le contrôleur y reste. La grille de piste crée un piège plus sournois. Dans une épingle ou quand deux portions de piste sont proches, le point de ligne centrale le plus proche d'une cellule peut appartenir à l'autre portion. L'écart latéral et la progression lus dans la grille sautent alors brutalement, et des trajectoires qui coupent vers la portion voisine reçoivent une progression fictive. Il faut vérifier la grille visuellement dans ces zones et, si besoin, résoudre l'ambiguïté par continuité de $s$ le long du rollout.

**Les pénalités trop dures.** Une pénalité énorme et constante crée un plateau. Tant qu'une partie des échantillons reste sur la piste, cela fonctionne. Quand tous les échantillons sortent, après une perturbation ou une mauvaise initialisation, ils ont tous le même coût, les poids deviennent uniformes et le contrôleur ne sait plus dans quelle direction revenir. C'est la raison du terme gradué de la section 5.2. La dureté pose aussi un problème quand la séquence nominale longe le bord : la moitié des échantillons sortent, reçoivent un poids nul, et l'ESS est divisée par deux ou plus, avec le bruit de commande qui en résulte. S'y ajoute un problème de précision : en float32, l'écart entre deux nombres représentables vaut 0,0625 autour de $10^6$ et 4 autour de $5 \times 10^7$. Une pénalité de $10^6$ par pas cumulée sur 50 pas absorbe complètement les différences de coût d'ordre unité entre échantillons. Enfin, un obstacle fin peut être « traversé » entre deux évaluations : à 8 m/s, le véhicule avance de 16 cm par pas, plus que trois cellules de la grille à 5 cm. Pour des obstacles fins, il faut dilater l'obstacle dans la grille d'au moins un pas de déplacement, ou tester le segment entre deux états successifs.

---

## 6. Les modèles de dynamique véhicule

### 6.1 Repères et grandeurs

On utilise deux repères. Le **repère monde** est fixe, avec $x$ vers l'avant de la position de départ, $y$ vers la gauche, et le cap $\psi$ compté positivement dans le sens trigonométrique depuis l'axe $x$. Le **repère véhicule** est attaché au centre de gravité, avec $x$ vers l'avant du véhicule et $y$ vers sa gauche. Dans ce repère, $v_x$ est la vitesse longitudinale, $v_y$ la vitesse latérale et $r = \dot{\psi}$ la **vitesse de lacet**, la vitesse de rotation autour de l'axe vertical. L'**angle de dérive du véhicule** $\beta = \arctan(v_y / v_x)$ mesure l'écart entre la direction du vecteur vitesse et l'axe du véhicule : nul quand la voiture roule droit, grand quand elle glisse.

La **modélisation bicycle** (on dit aussi *single-track*) fusionne les deux roues de chaque essieu en une roue unique placée sur l'axe du véhicule. Elle néglige le transfert de charge entre gauche et droite et les différences de braquage entre les roues avant, mais conserve l'essentiel du comportement plan. Les paramètres sont la masse $m = 3{,}5$ kg, le moment d'inertie de lacet $I_z = 0{,}04$ kg·m², et les distances du centre de gravité à l'essieu avant $l_f = 0{,}15$ m et à l'essieu arrière $l_r = 0{,}18$ m, avec l'empattement $L = l_f + l_r = 0{,}33$ m.

### 6.2 Le bicycle cinématique

Le modèle cinématique suppose que les roues roulent sans glisser : la vitesse de chaque roue est alignée avec son plan. Écrit au milieu de l'essieu arrière, il donne, avec l'état $x = (p_x, p_y, \psi, v)$ du projet :

$$
\dot{p}_x = v \cos\psi, \qquad
\dot{p}_y = v \sin\psi, \qquad
\dot{\psi} = \frac{v}{L} \tan\delta, \qquad
\dot{v} = a.
$$

La troisième équation vient de la géométrie : les normales aux deux roues se coupent au centre instantané de rotation, à la distance $R = L / \tan\delta$ de l'essieu arrière, et $\dot{\psi} = v / R$. Écrit au centre de gravité, le modèle fait apparaître l'angle de dérive géométrique $\beta = \arctan\big( \tfrac{l_r}{L} \tan\delta \big)$ :

$$
\dot{p}_x = v \cos(\psi + \beta), \qquad
\dot{p}_y = v \sin(\psi + \beta), \qquad
\dot{\psi} = \frac{v \cos\beta}{L} \tan\delta, \qquad
\dot{v} = a.
$$

Les deux formes décrivent le même mouvement en des points différents. La seconde est celle qu'il faut utiliser si l'on veut, à terme, mélanger le modèle cinématique avec le modèle dynamique, qui est écrit au centre de gravité (section 6.5).

Ce modèle est exact tant que les pneus ne glissent pas, c'est-à-dire à basse vitesse et faible accélération latérale. Il ignore tout le reste : la dérive des pneus, l'inertie de lacet (la rotation s'établit instantanément quand on braque), le sous-virage et le survirage, la saturation d'adhérence, le transfert de charge. Sa conséquence la plus dangereuse est qu'il n'a aucune limite d'accélération latérale. Au braquage maximal $\delta = 0{,}4$ rad, le rayon de virage vaut $R = 0{,}33 / \tan 0{,}4 = 0{,}78$ m. À 8 m/s, l'accélération latérale prédite est $v^2/R = 82$ m/s², plus de 8 g, alors qu'un pneu avec un coefficient d'adhérence $\mu \approx 1$ plafonne vers 9,8 m/s². À ce braquage, le modèle cinématique devient faux dès $v = \sqrt{\mu g R} \approx 2{,}8$ m/s. Un MPPI qui utilise ce modèle à haute vitesse planifie des virages impossibles, et la voiture réelle part tout droit. Il faut alors ajouter la pénalité d'adhérence de la section 5.2, ou passer au modèle dynamique.

### 6.3 Le bicycle dynamique

Le modèle dynamique applique les lois de Newton au véhicule, avec l'état $x = (p_x, p_y, \psi, v_x, v_y, r)$. La position évolue selon la vitesse exprimée dans le repère monde :

$$
\dot{p}_x = v_x \cos\psi - v_y \sin\psi, \qquad
\dot{p}_y = v_x \sin\psi + v_y \cos\psi, \qquad
\dot{\psi} = r.
$$

Les vitesses obéissent au principe fondamental de la dynamique écrit dans le repère véhicule, qui tourne à la vitesse $r$. Dans un repère tournant, l'accélération vaut $\dot{\mathbf{v}} + \boldsymbol{\omega} \times \mathbf{v}$, avec $\boldsymbol{\omega} \times \mathbf{v} = (-r v_y,\ r v_x)$, d'où

$$
\dot{v}_x = a - \frac{F_{yf} \sin\delta}{m} + v_y r, \qquad
\dot{v}_y = \frac{F_{yf} \cos\delta + F_{yr}}{m} - v_x r, \qquad
\dot{r} = \frac{l_f F_{yf} \cos\delta - l_r F_{yr}}{I_z}.
$$

On a supposé ici que la commande $a$ représente directement la force de traction divisée par la masse ; $F_{yf}$ et $F_{yr}$ sont les forces latérales des pneus avant et arrière. Les termes $v_y r$ et $-v_x r$ sont les termes d'entraînement dus à la rotation du repère : en virage stabilisé, $\dot{v}_y = 0$ et les forces latérales fournissent exactement l'accélération centripète $v_x r$.

Toute la physique intéressante se trouve dans les forces de pneu. Un pneu ne produit une force latérale que s'il glisse légèrement : la direction de la vitesse de son point de contact s'écarte de son plan d'un petit angle, l'**angle de dérive du pneu** $\alpha$. Pour la roue avant, braquée de $\delta$, dont le centre se déplace à la vitesse $(v_x,\ v_y + l_f r)$ dans le repère véhicule, et pour la roue arrière, à la vitesse $(v_x,\ v_y - l_r r)$ :

$$
\alpha_f = \delta - \arctan\!\left(\frac{v_y + l_f r}{v_x}\right), \qquad
\alpha_r = -\arctan\!\left(\frac{v_y - l_r r}{v_x}\right).
$$

La convention de signe est choisie pour qu'un angle positif produise une force positive : en ligne droite, un braquage à gauche ($\delta > 0$) donne $\alpha_f > 0$, une force vers la gauche, et une vitesse de lacet positive.

**Le modèle de pneu linéaire.** Pour de petits angles, la force est proportionnelle à l'angle : $F_y = C_\alpha\, \alpha$, où $C_\alpha$ est la **rigidité de dérive** en N/rad. Les coefficients `cornering_stiffness_front: 4.0` et `cornering_stiffness_rear: 4.2` du fichier de configuration sont trop petits pour être des rigidités en N/rad : ils suivent la convention du modèle *single-track* de CommonRoad, reprise par le simulateur F1TENTH, où la rigidité est normalisée par la charge verticale, $F_y = \mu\, C_S\, F_z\, \alpha$. Les charges statiques par essieu valent

$$
F_{zf} = m g \frac{l_r}{L} = 18{,}7 \text{ N}, \qquad F_{zr} = m g \frac{l_f}{L} = 15{,}6 \text{ N},
$$

d'où, avec $\mu = 1$, des rigidités de dérive de 74,9 N/rad à l'avant et 65,6 N/rad à l'arrière. Cette interprétation est à confirmer au moment d'écrire `dynamics.py`, et le coefficient d'adhérence $\mu$ manque pour l'instant dans la configuration.

Ces valeurs permettent de lire le comportement du véhicule. Le **gradient de sous-virage** $K_{us} = F_{zf}/C_{\alpha f} - F_{zr}/C_{\alpha r} = 1/C_{Sf} - 1/C_{Sr}$ vaut $0{,}25 - 0{,}238 = 0{,}012$ rad par g d'accélération latérale. Il est positif, donc le véhicule est légèrement **sous-vireur** : quand l'accélération latérale augmente, il faut braquer un peu plus que ne le prédit la géométrie, et le train avant atteint sa limite le premier. Un véhicule **survireur** ($K_{us} < 0$) verrait au contraire l'arrière décrocher le premier et tendrait au tête-à-queue. Avec 0,7 degré par g, ce véhicule est presque neutre.

Les valeurs donnent aussi les échelles de temps de la dynamique, qui compteront pour l'intégration numérique. En linéarisant le couple $(v_y, r)$ à $v_x$ constant, le terme dominant de l'équation de lacet est

$$
\dot{r} \approx -\frac{l_f^2 C_{\alpha f} + l_r^2 C_{\alpha r}}{I_z\, v_x}\, r + \dots = -\frac{95{,}3}{v_x}\, r + \dots
$$

et celui de l'équation latérale $\dot{v}_y \approx -\frac{C_{\alpha f} + C_{\alpha r}}{m\, v_x}\, v_y = -\frac{40{,}1}{v_x}\, v_y$. Le couplage entre les deux est faible : à $v_x = 1$ m/s, les valeurs propres du système linéarisé valent $-40{,}3$ et $-95{,}0$ s⁻¹. La constante de temps du mode de lacet vaut donc environ $v_x / 95$ secondes : 10 ms à 1 m/s, et elle raccourcit encore quand la vitesse diminue. C'est l'origine du problème de la section 6.5.

### 6.4 Modèles de pneu et saturation d'adhérence

Le modèle linéaire prédit une force qui croît sans limite avec l'angle de dérive. Un pneu réel ne transmet pas plus que $\mu F_z$ : au-delà, il glisse. Avec le modèle linéaire normalisé et $\mu = 1$, cette limite est atteinte dès $\alpha = \mu / C_S = 0{,}25$ rad pour le pneu avant.

**Pacejka simplifié.** La « formule magique » de Pacejka est le modèle de pneu empirique de référence. Sa forme simplifiée pour la force latérale s'écrit

$$
F_y = D \sin\!\Big( C \arctan\big( B\alpha - E\,(B\alpha - \arctan(B\alpha)) \big) \Big), \qquad D = \mu F_z.
$$

Chaque coefficient a un rôle lisible. $D$ est la force maximale. $C$ fixe la forme : pour $C > 1$, la courbe passe par un maximum puis redescend vers $D \sin(C\pi/2)$ quand $\alpha$ augmente. $B$ étire la courbe horizontalement, et la pente à l'origine vaut $BCD$, qu'on identifie à la rigidité de dérive : $B = C_S / (\mu C)$ dans la convention normalisée. $E$ ajuste la courbure près du maximum ; avec $E = 0$, le maximum est atteint en $\alpha_{\text{pic}} = \tan(\pi/2C)/B$. Les valeurs de $C$ pour la force latérale se situent typiquement entre 1,2 et 1,9, mais $B$, $C$ et $\mu$ doivent être identifiés sur le véhicule réel, sur la surface réelle : les pneus d'un modèle réduit sur un sol lisse n'ont rien à voir avec ceux d'une voiture sur asphalte.

**Une alternative lisse et sans branche.** Pour un kernel GPU, la saturation tangente hyperbolique est commode :

$$
F_y = \mu F_z \tanh\!\left( \frac{C_S\, \alpha}{\mu} \right).
$$

Elle a la même pente que le modèle linéaire à l'origine, sature exactement à $\mu F_z$, reste dérivable et ne coûte qu'une fonction. Elle n'a en revanche pas de branche descendante : au-delà du pic, le pneu réel perd de la force alors que ce modèle la maintient. Un véhicule simulé ainsi pardonne donc davantage les glissades que le véhicule réel, ce qui pousse le contrôleur vers des dérives que la réalité ne tolérera pas.

**Le cercle d'adhérence.** Un pneu ne dispose que d'une adhérence totale $\mu F_z$, à partager entre la force longitudinale (accélérer, freiner) et la force latérale (tourner) : $F_x^2 + F_y^2 \leq (\mu F_z)^2$. La capacité latérale restante vaut $\sqrt{(\mu F_z)^2 - F_x^2}$. À titre d'illustration, si toute la traction passait par l'essieu arrière et sans tenir compte du transfert de charge, accélérer à 4 m/s² demanderait $F_x = 14$ N à l'arrière, dont la capacité est de 15,6 N : il ne resterait que 6,9 N de force latérale disponible, moins de la moitié. Accélérer fort en virage fait décrocher l'arrière. Le châssis F1TENTH étant à quatre roues motrices, la répartition réelle est différente, mais le phénomène demeure. Le **transfert de charge longitudinal** l'accentue : accélérer charge l'arrière et décharge l'avant de $m\, a\, h / L$, où $h$ est la hauteur du centre de gravité, ce qui modifie les $F_z$ et donc les capacités.

### 6.5 Le problème de la basse vitesse

Le modèle dynamique se comporte mal quand $v_x$ tend vers zéro, pour trois raisons de nature différente.

**Une singularité algébrique.** Les angles de dérive contiennent $\arctan(\cdot / v_x)$. À $v_x \to 0$, le moindre $v_y$ ou $r$ produit un angle de dérive proche de $\pm\pi/2$ et des forces maximales. En marche arrière, le signe s'inverse. Utiliser `atan2` évite la division par zéro mais pas l'absurdité physique.

**Une raideur numérique.** Les constantes de temps calculées en section 6.3 sont proportionnelles à $v_x$. Le schéma d'Euler explicite appliqué à $\dot{r} = -k r$ n'est stable que si $|1 - k \Delta t| < 1$, soit $\Delta t < 2/k$. Avec $k = 95{,}3 / v_x$ et $\Delta t = 20$ ms, la stabilité exige $v_x > 95{,}3 \times 0{,}02 / 2 \approx 0{,}95$ m/s. En dessous, le lacet simulé oscille avec une amplitude qui double à chaque pas, et le rollout diverge vers l'infini en quelques dizaines de pas. Avec $\Delta t = 40$ ms, le seuil monte à 1,9 m/s.

**Une invalidité physique.** Le modèle de force proportionnelle à l'angle de dérive suppose un pneu qui roule. À très basse vitesse, le pneu se comporte comme une contrainte cinématique : il ne glisse pas latéralement, et les modèles de pneu en régime établi ne s'appliquent plus.

**La bascule vers le cinématique.** La solution standard, utilisée notamment par l'équipe AMZ de l'ETH Zurich en Formula Student, mélange les deux modèles selon la vitesse. On définit une plage de transition $[v_{\text{bas}}, v_{\text{haut}}]$, par exemple $[1{,}0 ;\ 1{,}5]$ m/s (le fichier de configuration ne contient pour l'instant qu'un seuil unique, `kinematic_blend_speed: 1.5`), et le coefficient

$$
\kappa = \operatorname{clamp}\!\left( \frac{v_x - v_{\text{bas}}}{v_{\text{haut}} - v_{\text{bas}}},\ 0,\ 1 \right).
$$

L'état suivant est la combinaison $x_{t+1} = \kappa\, F_{\text{dyn}}(x_t, u_t) + (1 - \kappa)\, F_{\text{cin}}(x_t, u_t)$. Pour que ce mélange ait un sens, les deux modèles doivent partager l'état à six composantes et le même point de référence, le centre de gravité. Le modèle cinématique exprimé dans cet état intègre $p_x, p_y, \psi, v_x$ comme en section 6.2 puis replace $(v_y, r)$ sur la variété cinématique, c'est-à-dire sur les valeurs qu'impose le roulement sans glissement :

$$
v_y = \frac{l_r}{L}\, v_x \tan\delta, \qquad r = \frac{v_x}{L} \tan\delta.
$$

Le mélange est continu, sans branchement, donc adapté au kernel. Il comporte un piège propre au calcul sans branche : on évalue aussi $F_{\text{dyn}}$ quand $\kappa = 0$, et si cette évaluation produit un infini ou un NaN, alors $0 \times \text{NaN} = \text{NaN}$ contamine l'état. Les dénominateurs du modèle dynamique doivent donc utiliser une vitesse bornée, $\max(v_x, v_{\text{bas}})$, pour rester finis même quand leur contribution est annulée. Comme le seuil de stabilité d'Euler (0,95 m/s) se situe sous $v_{\text{bas}}$, le modèle dynamique n'est jamais utilisé seul dans sa zone instable.

Deux compléments améliorent la robustesse. Intégrer plusieurs sous-pas par intervalle de commande abaisse le seuil de stabilité proportionnellement. Traiter implicitement les termes raides (Euler semi-implicite) le supprime : pour le lacet, $r_{t+1} = (r_t + \Delta t\, \tau_t)/(1 + \Delta t\, k)$, où $\tau_t$ regroupe les termes non raides, est stable pour tout $\Delta t$.

### 6.6 Ce que les deux modèles ignorent

Aucun des deux modèles ne représente les actionneurs. La commande $\delta$ y est appliquée instantanément, alors qu'un servomoteur a une vitesse de rotation limitée et un retard. De même $a$ n'est pas une grandeur que l'on commande : le variateur reçoit une consigne de puissance ou de courant, et l'accélération résultante dépend de la vitesse, de la batterie et de l'adhérence. Un modèle de premier ordre pour la direction, $\dot{\delta} = (\delta_{\text{cmd}} - \delta)/\tau$, qui ajoute $\delta$ à l'état, corrige l'écart le plus visible. Les modèles ignorent aussi la suspension, le roulis, le transfert de charge latéral, la latence des capteurs et la variation de l'adhérence selon la surface. MPPI optimise contre le modèle qu'on lui donne ; tout ce que le modèle ignore, le contrôleur l'ignore aussi (section 7.9).

### 6.7 Pourquoi MPPI brille quand les pneus décrochent

À la limite d'adhérence, l'application qui associe une séquence de commandes à une trajectoire change de nature. Tant que les pneus sont dans leur zone linéaire, braquer davantage fait tourner davantage, et une linéarisation autour de la trajectoire courante représente fidèlement l'effet d'une petite correction. Au-delà du pic de la courbe de Pacejka, braquer davantage réduit la force latérale : la dérivée change de signe. Un solveur à gradients qui linéarise en ce point reçoit une information locale exacte mais trompeuse pour la manœuvre, car la bonne réponse à une glissade de l'arrière est souvent un contre-braquage franc, loin dans une direction que le gradient local ne suggère pas. Le hessien du problème devient indéfini, le solveur doit être régularisé, et sa convergence en temps borné n'est plus garantie.

Le problème devient aussi multimodal. Pour prendre un virage à la limite, il existe une ligne en adhérence et une ligne en dérive contrôlée, et entre les deux une zone de trajectoires qui finissent en tête-à-queue. Une méthode locale à gradients choisit un mode selon son initialisation. MPPI évalue à chaque itération des milliers de trajectoires dispersées, dont certaines tombent dans chaque mode, et la pondération favorise celles qui fonctionnent.

MPPI n'utilise le modèle qu'en simulation directe. Les saturations, les `clamp`, la bascule cinématique, une table de Pacejka interpolée ou un réseau de neurones ne posent aucun problème de dérivabilité. Le coût peut rester une indicatrice de sortie de piste lue dans une grille. Sur AutoRally, le modèle était précisément un réseau de neurones appris à partir de données de glisse, et le coût une carte de piste.

Le même raisonnement fixe la limite de la méthode. MPPI est exactement aussi bon que son modèle, et c'est à la limite d'adhérence que le modèle est le plus incertain : $\mu$ varie avec la surface, la température et l'usure des pneus, et une erreur de 10 % sur $\mu$ fait la différence entre une dérive maîtrisée et une sortie. Les méthodes à gradients bien formulées pilotent aussi à la limite ; l'avantage de MPPI est d'y parvenir avec des modèles et des coûts arbitraires, au prix d'un calcul massif mais constant, là où une MPC à gradients demande un travail de formulation soigné pour chaque changement de modèle ou de coût.

---

## 7. Les modes d'échec

Chaque mode est décrit par ce qu'on observe, la cause probable, la façon de confirmer le diagnostic et les remèdes. Presque tous se diagnostiquent avec les mêmes instruments, qu'il vaut mieux mettre en place dès la version NumPy : journal par itération de l'ESS, de $\rho = \min S$, de la médiane de $S$, de la fraction d'échantillons saturés et de la fraction d'échantillons hors piste ; tracé des $K$ rollouts d'une itération colorés par leur poids ; série temporelle de la commande appliquée et de son spectre ; comparaison entre la trajectoire prédite par la séquence nominale et la trajectoire effectivement suivie.

### 7.1 Effondrement des poids

**Symptôme.** La commande de braquage saute d'une itération à l'autre, la trajectoire zigzague, et le comportement change nettement avec la graine du générateur. L'ESS journalisée tombe sous la dizaine, souvent à 1 ou 2.

**Cause.** La température est trop basse par rapport à la dispersion des coûts près du minimum. Cela arrive après une modification de l'échelle du coût (unité, intégrale, nouveau terme), quand une pénalité très dure coupe la population en deux près d'un bord, ou quand $\Sigma$ est si grande que seuls quelques échantillons restent raisonnables.

**Diagnostic.** Histogramme de $(S^k - \rho)/\lambda$ : si le deuxième meilleur échantillon est déjà à plus de 5 ou 10 unités du premier, les poids sont effondrés. Vérifier si l'effondrement est permanent (échelle) ou localisé près des bords (pénalité dure).

**Remède.** Normaliser le coût puis relever $\lambda$ jusqu'à une ESS de 1 à 10 % de $K$, ou adapter $\lambda$ à l'ESS. Adoucir les pénalités par un terme gradué. Réduire $\Sigma$ si la plupart des échantillons sont inutilisables.

### 7.2 Poids uniformes et commande inerte

**Symptôme.** Le véhicule réagit à peine : il continue tout droit à l'entrée d'un virage, élargit, ou fonce vers un obstacle que les rollouts traversent pourtant. La séquence nominale varie très peu d'une itération à l'autre. L'ESS dépasse la moitié de $K$.

**Cause.** Soit $\lambda$ est trop grand devant l'échelle du coût, soit le coût est plat sur toute la zone explorée. Le second cas est fréquent : tous les échantillons sont hors piste avec une pénalité constante, ou l'objectif est au-delà de l'horizon et seul un coût terminal constant le représente, ou encore la carte de coût est mal alignée et renvoie la même valeur partout.

**Diagnostic.** Écart-type des coûts des 10 % meilleurs échantillons, comparé à $\lambda$. Si cet écart-type est lui-même quasi nul, le problème vient du coût et non de la température.

**Remède.** Baisser $\lambda$ dans le premier cas. Dans le second, rendre le coût informatif : terme gradué hors piste, coût terminal qui dépend de la distance à l'objectif, vérification de l'alignement entre la grille et le repère du véhicule.

### 7.3 Exploration insuffisante et blocage

**Symptôme.** Le véhicule reste bloqué derrière un obstacle, ou suit une ligne visiblement mauvaise sans jamais en changer. Il met plusieurs secondes à s'adapter à un changement de situation. Le faisceau des rollouts est étroit et ne montre aucune alternative.

**Cause.** Bruit trop faible, bruit blanc qui explore mal les déplacements latéraux (section 4.2), démarrage à chaud qui enferme la séquence nominale dans un minimum local, ou saturation qui supprime l'exploration d'un côté.

**Diagnostic.** Tracer le faisceau. Relancer depuis le même état avec une séquence nominale réinitialisée à zéro : si le contrôleur trouve alors une meilleure solution, c'est le démarrage à chaud qui piège.

**Remède.** Augmenter $\Sigma$, passer à un bruit coloré ou à une paramétrisation par points de contrôle, réserver une fraction des échantillons à des séquences de référence (commande nulle, freinage maximal, sortie d'un contrôleur simple comme *pure pursuit*), ou réinitialiser la séquence nominale quand le coût minimal dépasse un seuil.

### 7.4 Moyenne de deux modes

**Symptôme.** Face à un obstacle centré, le véhicule hésite entre la gauche et la droite, puis va droit dessus ou le percute par le côté. La commande de braquage oscille autour de zéro à l'approche de l'obstacle.

**Cause.** La distribution optimale est bimodale, et la mise à jour calcule une moyenne, qui tombe entre les deux modes. C'est la conséquence directe du sens de la divergence KL minimisée (section 3.5). Les échantillons qui passent à gauche et ceux qui passent à droite ont des poids comparables, et leur moyenne est une trajectoire qui ne passe ni d'un côté ni de l'autre.

**Diagnostic.** Faisceau coloré par les poids : deux groupes de trajectoires lourdes, de part et d'autre de l'obstacle, et une trajectoire nominale entre les deux.

**Remède.** Baisser la température près des obstacles pour que le meilleur mode l'emporte nettement. Briser la symétrie par le coût (préférence de côté, progression qui favorise la corde du prochain virage). Utiliser une variante multimodale (Stein-MPPI, mélanges de gaussiennes, Tsallis-MPPI, section 8). Le démarrage à chaud aide aussi, car une fois qu'un côté domine légèrement, la séquence nominale s'y déplace et les échantillons de l'autre côté perdent du poids.

### 7.5 Oscillations de la commande

**Symptôme.** Deux formes distinctes. À haute fréquence, le braquage vibre de pas en pas, avec un spectre qui monte jusqu'à la fréquence de Nyquist de 25 Hz ; le servomoteur chauffe et le véhicule tremble. À basse fréquence, de l'ordre du hertz, le véhicule serpente autour de sa ligne.

**Cause.** Les oscillations rapides viennent du bruit d'estimation : $\sigma/\sqrt{\mathrm{ESS}}$ injecté à chaque itération, aggravé par le bruit blanc et l'absence de coût sur les variations de commande. Les oscillations lentes sont un phénomène de boucle fermée : latence non compensée, dynamique d'actionneur absente du modèle (le servomoteur réel suit la consigne avec retard, le contrôleur surcorrige), poids latéral trop fort par rapport à la régularisation, ou erreur de décalage de la séquence (appliquer $U[1]$ au lieu de $U[0]$, oublier le décalage, décaler deux fois).

**Diagnostic.** Rejouer l'expérience en simulation avec une graine fixe. Si l'oscillation lente existe en simulation avec un modèle parfait et sans latence, elle vient du coût ou du décalage ; si elle n'apparaît qu'avec le véhicule réel ou avec une latence simulée, elle vient du retard ou de l'actionneur. Pour les oscillations rapides, vérifier l'ESS au moment des oscillations.

**Remède.** Pour le bruit rapide : relever l'ESS, ajouter un coût sur la variation de commande, filtrer la séquence nominale (Savitzky-Golay), colorer le bruit, ou passer à Smooth-MPPI. Pour les oscillations lentes : compensation de latence de la section 2.3, modèle de premier ordre pour la direction, rééquilibrage des poids, et test unitaire du décalage.

### 7.6 Coupe de virage

**Symptôme.** Le véhicule prend l'intérieur des virages, mord sur la bordure ou la franchit avec les roues intérieures, alors que les rollouts affichés restent apparemment sur la piste.

**Cause.** Plusieurs causes coexistent souvent. La récompense de progression pèse trop par rapport au coût latéral, et la trajectoire la plus courte passe à l'intérieur ; c'est le comportement optimal pour le coût donné. La pénalité de sortie est évaluée au centre de gravité avec la demi-largeur de piste entière, alors que les roues sortent 15 cm plus tôt. Le coût n'est évalué que tous les 16 cm à 8 m/s, et la trajectoire entre deux évaluations coupe le coin intérieur d'un virage serré. La grille de piste renvoie un écart latéral ambigu à l'intérieur des épingles (section 5.4). Enfin, un coût qui vise un point de référence situé loin en avant sur la trajectoire attire le véhicule en ligne droite vers ce point.

**Diagnostic.** Comparer la ligne suivie à la ligne centrale et aux bords en tenant compte de la largeur du véhicule. Tracer la valeur de $d$ lue dans la grille le long de la trajectoire effective et chercher des sauts. Refaire l'essai en réduisant le poids de progression.

**Remède.** Réduire la marge utile ($d_{\max}$ = demi-largeur de piste moins demi-largeur du véhicule, plus une marge de sécurité), rééquilibrer progression et écart latéral par le taux de change de la section 5.3, dilater les bords dans la grille, ou tester le segment entre états successifs.

### 7.7 Rollouts qui divergent

**Symptôme.** La commande devient NaN ou infinie, ou le véhicule fait un écart brutal inexpliqué, souvent au démarrage ou à l'arrêt. Certains coûts journalisés sont NaN ou astronomiques.

**Cause.** Instabilité de l'intégration d'Euler du modèle dynamique à basse vitesse (sous 0,95 m/s à 20 ms), division par $v_x$ proche de zéro, NaN multiplié par un coefficient de mélange nul, ou forces de pneu évaluées hors domaine. Un seul NaN suffit à tout contaminer : les comparaisons avec NaN étant toujours fausses, le minimum peut valoir NaN selon l'implémentation, tous les poids deviennent NaN, et la séquence nominale entière avec eux.

**Diagnostic.** Compter les coûts non finis à chaque itération. Enregistrer l'état initial et le bruit de l'itération fautive, puis rejouer le rollout concerné pas à pas en NumPy.

**Remède.** Bascule cinématique à basse vitesse, dénominateurs bornés, intégration semi-implicite ou sous-pas. Remplacer tout coût non fini par une grande valeur finie. Vérifier que la séquence nominale est finie avant d'envoyer la commande, et prévoir une commande de repli (freinage) sinon.

### 7.8 Latence et décalage mal gérés

**Symptôme.** Le contrôleur marche bien en simulation synchrone et se dégrade sur le véhicule : léger serpentement, réactions en retard, qualité qui dépend de la charge du processeur.

**Cause.** En simulation, le calcul est instantané et l'état ne bouge pas pendant l'optimisation. Sur le véhicule, l'état a évolué de près d'un pas quand la commande est appliquée : à 5 m/s, le véhicule a avancé de 10 cm à l'aveugle. Si la durée du calcul varie, ce retard varie aussi. Un décalage de séquence erroné produit des effets semblables.

**Diagnostic.** Mesurer l'horodatage de l'estimation d'état et celui de l'envoi de la commande. Introduire en simulation un retard égal au temps de calcul mesuré et vérifier que le problème réapparaît.

**Remède.** Compensation de latence de la section 2.3, qui rend le retard constant et connu. Cadencer la boucle sur une horloge fixe au lieu d'enchaîner les itérations dès qu'elles terminent.

### 7.9 Exploitation des erreurs du modèle

**Symptôme.** Les trajectoires prédites sont belles et le coût faible, mais le véhicule réel ne les suit pas : il sous-vire hors du virage, glisse plus que prévu ou freine trop tard. L'écart entre trajectoire prédite et trajectoire suivie grandit systématiquement dans les mêmes situations.

**Cause.** MPPI optimise contre son modèle et trouve toute erreur du modèle qui rend une trajectoire artificiellement bon marché. L'exemple type est le modèle cinématique à haute vitesse, qui autorise des virages à 8 g (section 6.2). Plus subtilement, un modèle de pneu sans branche descendante rend les dérives trop faciles, et un $\mu$ surestimé autorise des vitesses de passage intenables.

**Diagnostic.** Enregistrer à chaque itération la trajectoire prédite par la séquence nominale et la comparer à la trajectoire réelle sur la même seconde. Localiser les situations où l'erreur est maximale.

**Remède.** Améliorer le modèle là où il se trompe, prendre un $\mu$ prudent, pénaliser les régions où le modèle est peu fiable (accélération latérale élevée, grande dérive), ou utiliser une variante robuste (Tube-MPPI, RMPPI).

---

## 8. Variantes et état de l'art

**Tsallis-MPPI** (Wang et al., RSS 2021). Remplace la divergence KL par une divergence de Tsallis, ce qui transforme la pondération exponentielle en une famille de pondérations à support fini qui éliminent entièrement les mauvais échantillons, et donne un intermédiaire réglable entre MPPI et la méthode de l'entropie croisée.

**Smooth-MPPI** (Kim et al., IEEE RA-L 2022). Échantillonne le bruit sur la dérivée de la commande et intègre pour obtenir la commande, ce qui produit des séquences lisses par construction et supprime le besoin de filtrer a posteriori.

**MPPI avec contraintes.** Tube-MPPI (Williams et al., RSS 2018) associe à MPPI un contrôleur de suivi qui maintient le véhicule proche de la trajectoire nominale malgré les perturbations ; RMPPI (Gandhi et al., IEEE RA-L 2021) en tire des garanties de performance bornées ; Shield-MPPI (Yin et al., IEEE RA-L 2023) ajoute des fonctions barrières de commande qui corrigent la commande pour respecter une contrainte de sécurité dure, là où MPPI de base n'offre que des pénalités.

**MPPI avec modèle appris.** Williams et al. (ICRA 2017) remplacent le modèle physique par un réseau de neurones entraîné sur des données de conduite, ce qui capture des effets impossibles à modéliser à la main (terrain meuble, glisse) ; les approches hybrides apprennent seulement le résidu entre le modèle physique et la réalité, ce qui conserve un comportement sensé hors des données.

**Log-MPPI** (Mohamed et al., IEEE RA-L 2022). Échantillonne un bruit normal-log-normal à queues plus lourdes, qui explore plus loin dans les environnements encombrés sans augmenter $K$.

**Stein Variational MPC** (Lambert et al., CoRL 2020). Représente la distribution de commande par un ensemble de particules mises à jour par descente de gradient de Stein, ce qui préserve plusieurs modes au lieu de les moyenner.

**Biased-MPPI** (Trevisan et Alonso-Mora, IEEE RA-L 2024). Mélange aux échantillons gaussiens des séquences fournies par d'autres contrôleurs (freinage, suivi géométrique), avec une correction des poids qui garde la cohérence théorique, et corrige le manque d'exploration dans les situations où la séquence nominale est piégée.

**DMD-MPC** (Wagener et al., RSS 2019). Reformule MPPI comme une descente en miroir dynamique en ligne, ce qui unifie MPPI, CEM et d'autres méthodes par échantillonnage et donne un pas de mise à jour réglable.

**CoVO-MPC** (Yi et al., L4DC 2024). Analyse la vitesse de convergence de MPPI et en déduit une covariance d'échantillonnage optimale, adaptée au problème au lieu d'être fixée à la main.

**Bruit coloré et paramétrisations réduites.** Tirer un bruit corrélé dans le temps ou n'échantillonner que quelques points de contrôle d'une spline corrige l'inefficacité du bruit blanc (section 4.2) ; la bibliothèque MPPI-Generic propose ces options.

---

## 9. Ressources

### 9.1 Articles, dans l'ordre de lecture

1. **Williams, Drews, Goldfain, Rehg, Theodorou.** *Information-Theoretic Model Predictive Control: Theory and Applications to Autonomous Driving.* IEEE Transactions on Robotics, 2018. L'article de référence : la dérivation par la divergence KL de la section 3.4, l'algorithme complet, le terme de correction et les expériences sur AutoRally. À lire en entier ; la partie sur le réseau de neurones peut être survolée en première lecture.

2. **Williams, Aldrich, Theodorou.** *Model Predictive Path Integral Control: From Theory to Parallel Computation.* Journal of Guidance, Control, and Dynamics, 2017. La dérivation par HJB et Feynman-Kac de la section 3.3 et la première implémentation GPU. Les sections théoriques sont denses en calcul stochastique ; la partie sur le passage au GPU est courte et directement utile pour le kernel.

3. **Williams, Drews, Goldfain, Rehg, Theodorou.** *Aggressive Driving with Model Predictive Path Integral Control.* ICRA 2016. Court, expérimental, montre ce que la méthode donne sur un vrai véhicule. Utile pour la motivation, sans théorie nouvelle par rapport aux deux précédents.

4. **Kappen.** *Linear Theory for Control of Nonlinear Stochastic Systems.* Physical Review Letters, 2005. La transformation logarithmique à l'origine de tout. Quatre pages. Son article long *Path integrals and symmetry breaking for optimal control theory* (J. Stat. Mech., 2005) peut être sauté, sauf intérêt pour le lien avec la physique statistique.

5. **Theodorou, Todorov.** *Relative Entropy and Free Energy Dualities: Connections to Path Integral and KL Control.* IEEE CDC, 2012. L'équivalence entre les deux lectures de la section 3. Recommandé si l'on veut comprendre pourquoi les deux dérivations donnent le même résultat.

6. **Williams et al.** *Robust Sampling Based Model Predictive Control with Sparse Objective Information.* RSS 2018. Tube-MPPI, et une discussion honnête des échecs de MPPI de base quand le coût est peu informatif. Utile après une première implémentation.

Peuvent être sautés en première approche : Theodorou, Buchli et Schaal (JMLR 2010) sur PI², orienté apprentissage de politique paramétrée ; Todorov (PNAS 2009) sur les MDP linéairement solubles, élégant mais redondant après les articles ci-dessus ; les variantes de la section 8, à consulter seulement quand un mode d'échec précis les motive.

### 9.2 Dynamique du véhicule

**Rajamani.** *Vehicle Dynamics and Control*, Springer, 2e édition, 2012. Les chapitres 2 et 3 couvrent les modèles bicycle cinématique et dynamique avec les conventions de signe expliquées. C'est la référence la plus directe pour la section 6.

**Liniger, Domahidi, Morari.** *Optimization-Based Autonomous Racing of 1:43 Scale RC Cars.* Optimal Control Applications and Methods, 2015. Un modèle bicycle dynamique avec Pacejka simplifié, identifié sur des voitures miniatures, et une MPC à gradients qui fonctionne à la limite. Utile à la fois pour le modèle et pour savoir contre quoi comparer MPPI.

**Kabzan et al.** *AMZ Driverless: The Full Autonomous Racing System.* Journal of Field Robotics, 2020. La section sur le modèle de véhicule décrit la bascule entre modèles cinématique et dynamique à basse vitesse.

**Althoff, Würsching.** *CommonRoad: Vehicle Models.* Documentation technique du projet CommonRoad. Définit le modèle *single-track* avec rigidités normalisées que reprend le simulateur F1TENTH, donc la convention probable des paramètres du fichier de configuration.

**Pacejka.** *Tire and Vehicle Dynamics*, Butterworth-Heinemann. La référence sur les modèles de pneu ; seul le chapitre sur la formule magique est utile ici.

### 9.3 Implémentations open source

**`nav2_mppi_controller`** (dépôt `ros-navigation/navigation2`, ROS 2). L'implémentation la plus utilisée en production, et la référence de l'étape 7 du projet. Elle diffère du projet sur des points qui comptent pour la comparaison. Elle tourne sur CPU, en C++ vectorisé, sans GPU. Ses modèles de mouvement (différentiel, omnidirectionnel, Ackermann) échantillonnent des consignes de vitesse et les intègrent avec des limites d'accélération, sans modèle de pneu : elle délègue la dynamique à un contrôleur de bas niveau. Les valeurs par défaut documentées sont de l'ordre de 1000 échantillons et 56 pas de 50 ms, soit 2,8 s d'horizon et 56 000 évaluations par itération, sept fois moins que les 409 600 du projet. À vérifier dans la version utilisée pour la comparaison. Ce qui vaut la lecture : l'optimiseur (échantillonnage, pondération avec `temperature` et `gamma`, filtre de Savitzky-Golay, décalage), l'architecture en « critiques » où chaque terme du coût est un greffon séparé, et la documentation des paramètres, qui décrit symptômes et réglages. Ce qu'on peut sauter : l'intégration avec la pile de navigation (plugins, cartes de coût ROS, cycle de vie des nœuds).

**AutoRally** (dépôt `AutoRally/autorally`). L'implémentation d'origine, en C++ et CUDA pour ROS 1, avec les modèles physique et neuronal du véhicule. Elle n'est plus maintenue et son code est daté, mais c'est l'ancêtre direct du kernel du projet et la seule implémentation qui correspond exactement aux articles.

**MPPI-Generic** (Vlahov et al., 2024, laboratoire ACDS de Georgia Tech). Bibliothèque CUDA moderne issue du même laboratoire, avec plusieurs modèles de dynamique, bruit coloré, Tube-MPPI et RMPPI. La plus utile pour voir comment structurer un kernel MPPI générique et quelles optimisations mémoire ont été retenues.

**`pytorch_mppi`** (dépôt `UM-ARM-Lab/pytorch_mppi`). Une implémentation Python compacte et lisible, vectorisée avec PyTorch. Utile comme deuxième référence pour la version NumPy, notamment pour le traitement des bornes et du terme de correction. Sa structure vectorisée par pas de temps est l'opposée de celle du kernel.

### 9.4 Contexte général, facultatif

**Åström, Murray.** *Feedback Systems: An Introduction for Scientists and Engineers*, Princeton University Press, disponible gratuitement en ligne. Pour acquérir le vocabulaire de base du contrôle (boucle fermée, stabilité, PID, retard) si le besoin s'en fait sentir. Les chapitres 1, 3 et 10 suffisent.

**Rawlings, Mayne, Diehl.** *Model Predictive Control: Theory, Computation, and Design*, Nob Hill, 2e édition, disponible gratuitement en ligne. La référence sur la MPC à gradients. Le chapitre 1 donne le cadre général ; le reste n'est utile que pour une comparaison approfondie avec les méthodes d'optimisation classiques.

---

## Annexe : chiffres de référence

| Grandeur | Valeur | Section |
|---|---|---|
| Horizon de prédiction | $50 \times 20$ ms = 1 s | 1.4 |
| Taille du tenseur de bruit | 819 200 scalaires, 3,1 Mio en float32 | 2.1 |
| Évaluations du modèle par itération | 409 600 | 2.1 |
| Trajectoires complètes du modèle dynamique, si stockées | 9,6 Mio | 2.1 |
| Seuil de sous-dépassement de $e^{-z}$ en float32 | $z \approx 87$ (normal), $103$ (sous-normal) | 2.2 |
| ESS visée pour $K = 8192$ | 80 à 800 | 4.1 |
| $\lambda$ correspondant (coûts gaussiens d'écart-type $s$) | entre $s/2{,}1$ et $s/1{,}5$ | 4.1 |
| Bruit résiduel sur $\delta$, ESS = 100 / ESS = 4 | 0,01 rad / 0,05 rad | 4.2 |
| Écart latéral exploré à 5 m/s, bruit blanc / corrélé | 0,61 m / environ 3,7 m | 4.2 |
| Vitesse maximale d'arrêt garanti dans l'horizon | 8 m/s | 4.3 |
| Rayon de virage à $\delta = 0{,}4$ rad | 0,78 m | 6.2 |
| Vitesse de validité du modèle cinématique à ce braquage ($\mu = 1$) | 2,8 m/s | 6.2 |
| Rigidités de dérive avant / arrière ($\mu = 1$) | 74,9 / 65,6 N/rad | 6.3 |
| Gradient de sous-virage | 0,012 rad/g | 6.3 |
| Seuil de stabilité d'Euler du lacet, $\Delta t$ = 20 ms / 40 ms | 0,95 m/s / 1,9 m/s | 6.5 |
| Distance parcourue par pas à 8 m/s | 16 cm | 5.4 |
