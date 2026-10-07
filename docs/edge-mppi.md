# MPPI accéléré GPU sur cible embarquée — Jalon 1

**Arthur Fournier — septembre 2026**

---

## 1. Phrase de positionnement

> Contrôle prédictif temps réel accéléré GPU sur cible embarquée.

Cette phrase est la boussole du projet. Chaque décision technique se juge par rapport à elle : si une option ne sert ni le temps réel, ni l'accélération GPU, ni la contrainte embarquée, elle sort du périmètre.

## 2. Objectif

Implémenter un contrôleur MPPI (Model Predictive Path Integral) pour un véhicule de type voiture, l'accélérer par un kernel CUDA écrit à la main, et le déployer sur un NVIDIA Jetson Orin Nano en tenant un budget temps réel strict.

Le projet vise trois compétences que mon parcours ne couvre pas encore :

- **Contrôle robotique** — MPPI, dynamique véhicule, conception de fonction de coût
- **Programmation GPU bas niveau** — CUDA C++, profiling Nsight, optimisation mémoire
- **Déploiement embarqué** — cible ARM + GPU contraint, budget de latence, enveloppe de puissance

Ces trois axes se recoupent exactement avec le périmètre des équipes de robotique embarquée (NVIDIA Isaac, Wayve, Waabi, ANYbotics).

## 3. Pourquoi MPPI

MPPI est un algorithme d'optimisation en ligne. À chaque pas de temps, il échantillonne K séquences de commandes bruitées, déroule la dynamique du véhicule sur un horizon T, évalue un coût par trajectoire, puis pondère les séquences en `exp(-coût/λ)` pour produire la commande appliquée.

Trois propriétés le rendent idéal ici :

- **Parallélisme massif et irrégulier.** Les rollouts sont indépendants entre eux mais séquentiels dans le temps. Ni cuBLAS ni PyTorch n'exploitent bien ce motif — c'est précisément le cas où écrire un kernel à la main apporte un gain réel et mesurable.
- **État de petite dimension.** 4 à 6 variables selon le modèle, donc des milliers de trajectoires tiennent dans le budget temps réel d'une carte embarquée.
- **Origine automobile.** L'algorithme vient du projet AutoRally (Georgia Tech), conçu pour du pilotage agressif de voiture RC. Le cas d'usage est directement lisible par le secteur de la conduite autonome.

**MPPI ne s'entraîne pas.** Aucun dataset n'est nécessaire : la dynamique est écrite à la main, l'environnement est généré, tout est produit en interne au projet.

## 4. Périmètre du jalon 1

### Dans le périmètre

- Modèle bicycle cinématique puis dynamique (avec modèle de pneu)
- Implémentation MPPI de référence en Python/NumPy
- Kernel CUDA fusionné, optimisé et profilé
- Déploiement sur Jetson Orin Nano en hardware-in-the-loop
- Caractérisation systématique latence / qualité de contrôle
- Comparaison avec une baseline externe de qualité production

### Hors périmètre

- **Aucun robot physique.** Le Jetson est utilisé comme *cible de calcul embarquée*, pas comme cerveau de robot. La simulation tourne sur le PC de développement, le contrôleur sur le Jetson, les deux en boucle fermée via Ethernet. C'est du hardware-in-the-loop, la pratique standard avant tout déploiement réel.
- **Aucun apprentissage.** Pas de réseau de neurones, pas de RL. Ces éléments arrivent aux jalons suivants.

### Le mot « dynamique » — clarification

Deux sens différents circulent et il faut les tenir séparés :

| Terme | Sens | Jalon |
|---|---|---|
| Modèle **dynamique** | Physique écrite à la main : forces de pneu, dérive, saturation d'adhérence | Jalon 1, étape 2 |
| Modèle de **dynamique appris** | Réseau de neurones entraîné à prédire l'état suivant | Jalon 2 |

## 5. Spécifications cibles

| Paramètre | Valeur |
|---|---|
| Plateforme simulée | Véhicule Ackermann, paramètres F1TENTH |
| Modèle final | Bicycle dynamique, état `[x, y, ψ, vx, vy, r]` |
| Échantillons K | 8192 |
| Horizon T | 50 pas |
| Pas de temps dt | 20 ms |
| **Budget de latence** | **< 20 ms par itération de contrôle** |
| Cible matérielle | Jetson Orin Nano Super 8 Go |
| Machine de développement | PC Ubuntu + RTX 3060 (architecture Ampere, comme le Jetson) |

Le budget de 20 ms est le critère central du projet. Toute optimisation se mesure par rapport à lui.

## 6. Décisions d'architecture

Ces choix sont pris dès le départ **parce qu'ils conditionnent les jalons suivants**. Les ignorer imposerait une réécriture.

### 6.1 Interface de dynamique isolée

`step(state, control, dt) -> state` doit être une fonction device isolée derrière une interface propre. Le modèle sera remplacé trois fois : cinématique → dynamique → appris (jalon 2). Le swap doit rester mécanique.

### 6.2 Dimension d'état paramétrable

`state_dim` est une constante de configuration, jamais un `4` écrit en dur. Le passage cinématique (4) → dynamique (6) → appris (variable) doit se faire sans chasse aux valeurs magiques.

### 6.3 Configuration unique partagée

Un seul fichier YAML contenant horizon, dt, K, λ, variances, bornes de commande et poids du coût, lu **à l'identique** par la version Python et la version C++. C'est ce qui rend les comparaisons valides.

### 6.4 Sauvegarde systématique des trajectoires

Dès l'étape 6, enregistrer tous les triplets `(état, commande, état suivant)`. Coût nul, et cela constitue directement le dataset d'entraînement du jalon 2.

### 6.5 Plateforme figée

Un seul véhicule, un seul jeu de paramètres, pour tous les jalons. Changer de plateforme en cours de route casse la comparabilité du benchmark final.

### 6.6 Représentation du circuit portable

Le circuit est précalculé en **grille 2D** (écart latéral signé + progression curviligne par cellule), pas en objet spline interrogé à l'exécution. Deux raisons : un kernel CUDA ne peut pas appeler une bibliothèque Python, et une grille se place naturellement en texture memory à l'étape d'optimisation.

## 7. Étapes détaillées

Chaque étape a un critère de sortie explicite. Tant qu'il n'est pas vérifié, on ne passe pas à la suivante. `git tag` à chaque fin d'étape.

---

### Étape 0 — Fondations *(½ journée)*

Repo `mppi-cuda`, structure `python/`, `cuda/`, `bench/`, `docs/`. CMake dès le départ. Fichier de configuration YAML unique. Conventions de repère écrites en commentaire en tête du fichier de dynamique : ψ mesuré depuis l'axe x, radians et SI partout, ordre exact des composantes de l'état.

**Sortie :** repo initialisé, CMake qui compile un hello-world CUDA, README avec la phrase de positionnement et le budget de 20 ms.

---

### Étape 1 — MPPI cinématique en Python *(2-3 jours)*

**Modèle.** Bicycle cinématique, état `[x, y, ψ, v]`, commandes `[a, δ]`, intégration Euler à dt = 20 ms. Contraintes `|δ| ≤ 0.4 rad`, accélération bornée.

**Contrôleur.** Implémentation vectorisée NumPy complète : échantillonnage du bruit `(K, T, 2)`, rollouts, coût par trajectoire, pondération `w = exp(-(S - S_min)/λ)`, moyenne pondérée, décalage de la séquence nominale d'un pas.

**Coût.** Écart latéral à la centerline + progression + pénalité de sortie de piste + régularisation de commande. **Normaliser les termes** pour qu'ils soient dans le même ordre de grandeur, sinon le réglage de λ devient impossible.

**Circuit.** Généré à la main : ~10 points de contrôle, spline périodique (`scipy.interpolate.splprep`), rééchantillonnage à pas constant en abscisse curviligne, remplissage de la grille par KD-tree. Sauvegarde en `.npy` et en binaire brut pour le C++ (header : dimensions, bornes du domaine, row-major, float32).

**Critère de sortie :** tour complet sans sortie de piste à K = 1024, T = 30. Plot de trajectoire et courbe de coût. λ et variance tunés à la main, avec compréhension qualitative de leur effet.

**Piège connu :** une non-convergence vient presque toujours d'un λ mal calibré par rapport à l'échelle du coût.

---

### Étape 2 — Modèle dynamique en Python *(3-4 jours)*

**Modèle.** Bicycle dynamique, état `[x, y, ψ, vx, vy, r]`. Angles de dérive avant/arrière, modèle de pneu linéaire (`Fy = -C·α`) puis Pacejka simplifié pour obtenir la saturation d'adhérence.

**Points critiques.**
- Bascule sur le modèle cinématique sous 1,5 m/s avec interpolation douce sur une bande. Le modèle de pneu diverge quand la vitesse longitudinale tend vers zéro (angle de dérive non défini).
- Intégration RK4 ou dt réduit : le modèle dynamique est plus raide, Euler peut diverger.

**Validation croisée.** À faible vitesse et faible braquage, les deux modèles doivent produire des trajectoires quasi identiques. Conserver ce test comme non-régression permanente.

**Critère de sortie :** tour à vitesse élevée avec dérapage visible en sortie de virage, géré par le contrôleur. Figure comparative cinématique vs dynamique sur le même circuit à la même vitesse cible — le cinématique doit échouer là où le dynamique passe. Première figure de résultat du projet.

---

### Étape 3 — Portage CUDA naïf *(4-5 jours)*

Objectif : **correction, pas performance.** Aucune optimisation à ce stade.

Un thread par trajectoire, boucle sur l'horizon dans le kernel, dynamique en fonction `__device__`, coût accumulé au vol, écriture d'un seul float par thread. Réduction sur CPU pour l'instant.

**Méthode de validation — le cœur de l'étape.** Générer le tenseur de bruit en Python, le sauvegarder, l'injecter dans les deux implémentations, puis comparer les coûts par trajectoire un à un. Ils doivent coïncider à ~1e-4 près en FP32. Une seule divergence signale un bug de dynamique. Infiniment plus rapide à débugger que de comparer des trajectoires finales.

**Critère de sortie :** les K coûts CUDA correspondent aux K coûts NumPy sur bruit fixé, et la boucle de contrôle C++ complète boucle le circuit comme la version Python. Temps par itération relevé comme baseline.

---

### Étape 4 — Optimisation du kernel *(1-2 semaines)*

**L'étape la plus importante du projet.** Une optimisation à la fois, mesure après chacune, une ligne de tableau à chaque fois.

Dans l'ordre de rentabilité :

1. **Kernel fusionné rollout + coût** — état du véhicule maintenu en registres pendant tout l'horizon, mémoire globale touchée uniquement en entrée et en sortie. Aucune trajectoire n'est matérialisée. C'est l'essentiel du gain.
2. **cuRAND en device** — génération du bruit dans le kernel avec un état par thread, plutôt qu'un tenseur `K × T × 2` préparé sur le host et transféré.
3. **Réduction en primitives warp** — `__shfl_down_sync` ou coopérative groups pour min des coûts, somme des exponentielles, moyenne pondérée. Faire la version naïve d'abord, puis optimiser et mesurer.
4. **Cost map en texture memory** — les accès sont irréguliers et non coalescés par nature. Un objet texture donne le cache spatial 2D et l'interpolation bilinéaire gratuite en hardware.
5. **Layout structure-of-arrays** — pour tout ce qui reste en mémoire globale, afin que les threads d'un warp lisent des adresses contiguës.
6. **FP32 strict + intrinsèques rapides** — `__sinf`, `__expf`, `__atanf` pour la trigonométrie du modèle de pneu. Quantifier l'impact sur la qualité de trajectoire.

**Instrumentation.** Nsight Compute pour occupancy, throughput mémoire, efficacité des accès. Pour chaque optimisation, noter *pourquoi* un gain était attendu et s'il a eu lieu. **Les optimisations sans effet sont aussi à documenter** — cela montre qu'on mesure au lieu de croire.

**Critère de sortie :** tableau de 6-7 lignes (temps par itération, accélération cumulée), chaque ligne explicable. Savoir dire si le kernel est memory-bound ou compute-bound, et pourquoi.

**⚠️ Cette étape doit être écrite à la main.** C'est la seule compétence rare que le projet démontre, et c'est exactement ce qu'un entretien technique creusera.

---

### Étape 5 — Déploiement Jetson *(3-4 jours)*

Build **natif sur la carte**, pas de cross-compilation (économie de plusieurs heures).

**Boucle hardware-in-the-loop.** Simulation sur le PC, contrôleur sur le Jetson, socket TCP, **simulation synchronisée sur le contrôleur** (pas de temps réel côté sim, on attend la réponse) — sinon les mesures ne veulent rien dire.

**Spécificités Jetson à exploiter.**
- La mémoire est **physiquement unifiée** : CPU et GPU partagent le même pool LPDDR5. `cudaHostAlloc` en mode mapped ou la mémoire managée élimine de vraies copies. Mesurer la différence — c'est un point que peu de gens connaissent, et il n'existe pas sur la RTX 3060.
- `nvpmodel` pour basculer entre 7 W / 15 W / 25 W / MAXN, `jetson_clocks`, surveillance par `tegrastats`.

**Mesure de latence côté Jetson uniquement**, avec des events CUDA, jamais un chronomètre incluant le réseau.

**Critère de sortie :** tour de circuit complet en HIL, budget de 20 ms tenu à K = 8192, T = 50, modèle dynamique. En cas d'échec, savoir exactement quel paramètre réduire et de combien.

---

### Étape 6 — Caractérisation *(4-5 jours)*

Ce qui transforme le projet en artefact citable.

**Balayages de latence.** En fonction de K (256 → 32768), de T, du modèle (cinématique vs dynamique), du mode de puissance. **p50 et p99, pas seulement la moyenne** — le jitter compte autant que le temps moyen en temps réel.

**Comparaison inter-plateformes.** Mêmes optimisations sur RTX 3060 et sur Orin Nano. Les gains diffèrent (rapport calcul/bande passante, nombre de SM, mémoire unifiée) : c'est un résultat, pas un problème.

**Métriques de contrôle.** Temps au tour, erreur latérale RMS, taux de succès sur 50 essais avec obstacles aléatoires.

**La courbe la plus intéressante : qualité de contrôle en fonction du budget de calcul.** Elle montre où le GPU achète réellement de la performance de contrôle, et pas seulement des FLOPS.

**⚠️ Sauvegarder toutes les trajectoires** `(état, commande, état suivant)` pendant ces campagnes — dataset du jalon 2.

**Critère de sortie :** 5-6 figures propres, générées par script reproductible.

---

### Étape 7 — Baseline externe *(2-3 jours)*

Faire tourner `nav2_mppi_controller` (ROS 2, C++, CPU, qualité production) sur le même circuit et comparer. **Lire son code avant** — c'est une implémentation industrielle qui contient des astuces réelles.

C'est la comparaison qui a du poids : « X fois plus rapide que mon NumPy » ne vaut rien, « X fois plus rapide que l'implémentation de référence de Nav2 à qualité de contrôle égale » est un vrai résultat. Bénéfice secondaire : première exposition à ROS 2, qui manque au CV.

---

### Étape 8 — Packaging *(2 jours)*

- README avec la phrase de positionnement, le tableau d'optimisations, les figures, une GIF du contrôleur en action, les instructions de reproduction
- PDF de 2 pages (contexte, méthode, résultats, deux figures)
- Repo tagué par étape pour rendre la progression visible

## 8. Calendrier

| Charge hebdo | Durée totale |
|---|---|
| 20 h+ | 6-8 semaines |
| 10-20 h | 10-12 semaines |
| 5-10 h | 12-16 semaines |

**Le projet peut figurer sur le CV en « en cours » dès aujourd'hui.** Il devient présentable en entretien à la fin de l'étape 3 (CUDA qui tourne et validé), et pleinement défendable à la fin de l'étape 5.

**Priorisation si le temps manque :** étapes 1→5 obligatoires, étape 7 très rentable, étape 6 réductible à trois figures, étape 2 reportable après le CUDA en dernier recours.

## 9. Suite (aperçu)

Le jalon 1 est autonome et se suffit à lui-même. Les jalons suivants réutilisent la plateforme, le harnais d'évaluation et le pipeline de déploiement :

- **Jalon 2** — remplacer la dynamique analytique par un réseau appris (2-3 semaines)
- **Jalon 3** — politique entraînée par RL en simulation parallèle, déployée sur la même cible (un semestre)
- **Jalon 4** — benchmark comparatif des trois approches sur la même tâche

Détail complet dans le fichier `02_contexte_et_jalons.md`.
