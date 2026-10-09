# Étape 3 : portage CUDA naïf, validé contre NumPy

TP autonome, suite de `docs/step_2.md`. Si tu n'as jamais fait de CUDA, **lis d'abord `docs/cuda.md`** (sections 1 à 8). Ce document y renvoie (« fiche §4 ») au lieu de tout répéter. La théorie de fond reste dans `docs/mppi-theory.md` (§2.4 et §2.5 surtout).

**Règle du jeu (inchangée).** Les questions **Q** se répondent par écrit avant de regarder les [Solutions](#solutions). Les blocs de code de l'énoncé sont des squelettes et des rappels d'API. Les points de contrôle (✔) disent quoi vérifier avant de continuer.

**Durée estimée.** 4 à 5 jours (plan `edge-mppi.md`). Les parties A et B (apprendre CUDA, écrire la config C++) prennent un jour ; la parité (D, E) est le cœur.

**D'où viennent les chiffres.** La solution de référence (fin du document) a été compilée avec la chaîne pixi (nvcc 12.9, gcc 14) et exécutée **sur le chemin CPU seulement**, sur la machine WSL2 sans GPU. Les chiffres « mesurés » portent donc sur le C++ compilé pour le CPU. Tout ce qui concerne le GPU (temps de kernel, parité GPU) est **à mesurer par toi** sur la machine distante : le document dit ce que tu dois attendre qualitativement, sans inventer de valeur.

## Sommaire

- [0. Objectif et critère de sortie](#0-objectif-et-critère-de-sortie)
- [1. Préparer le terrain](#1-préparer-le-terrain)
- [2. Partie A : premiers kernels](#2-partie-a--premiers-kernels)
- [3. Partie B : la configuration et la piste en C++](#3-partie-b--la-configuration-et-la-piste-en-c)
- [4. Partie C : dynamique et coût en `__host__ __device__`](#4-partie-c--dynamique-et-coût-en-__host__-__device__)
- [5. Partie D : le kernel et le chemin CPU](#5-partie-d--le-kernel-et-le-chemin-cpu)
- [6. Partie E : la parité, cœur de l'étape](#6-partie-e--la-parité-cœur-de-létape)
- [7. Partie F : la boucle de contrôle C++](#7-partie-f--la-boucle-de-contrôle-c)
- [8. Partie G : mesurer la baseline](#8-partie-g--mesurer-la-baseline)
- [9. Dépannage](#9-dépannage)
- [10. Clôturer l'étape](#10-clôturer-létape)
- [Solutions](#solutions)

---

## 0. Objectif et critère de sortie

### 0.1 Ce que tu vas construire

Le contrôleur MPPI de l'étape 2, réécrit en C++/CUDA :

- un **kernel** où chaque thread déroule un rollout complet (T pas de dynamique, coût accumulé au vol) et écrit **un seul float**, son coût `S[k]` ;
- tout le reste (bruit, poids, moyenne pondérée, décalage) sur le **CPU**, en C++ ;
- une boucle fermée C++ (contrôleur + véhicule simulé) qui fait le tour de piste ;
- un **contrôle de parité** : mêmes `x0`, `U`, bruit → les K coûts CUDA comparés un par un aux K coûts NumPy.

**Correction, pas performance.** Aucune optimisation à cette étape. Le bruit est même tiré sur le CPU et copié à chaque itération, ce qui est lent : c'est la baseline que l'étape 4 améliorera, ligne par ligne.

### 0.2 Critère de sortie

- [ ] `pixi run build` compile tout, sans avertissement ;
- [ ] `pixi run check` : sur tous les cas de parité, **aucun écart inexpliqué** entre les coûts NumPy et les coûts CUDA (définition précise en partie E), chemin CPU **et** chemin GPU ;
- [ ] le contrôle de parité **échoue** quand on injecte un bug (au moins deux bugs testés) ;
- [ ] `pixi run sim-cuda` : la boucle de contrôle C++, rollouts sur GPU, boucle le circuit sans sortie, avec des métriques comparables à NumPy sur 5 graines ;
- [ ] `compute-sanitizer` ne signale aucune erreur ;
- [ ] la baseline mesurée (ligne 0 de `bench/results.md`) à K = 8192, T = 50, avec la décomposition par phase (bruit, copies, kernel, mise à jour) ;
- [ ] une entrée dans `docs/journal.md` ;
- [ ] `git tag step-3`.

### 0.3 Vue d'ensemble des fichiers

| Fichier | Rôle | Partie |
|---|---|---|
| `pixi.toml` | `yaml-cpp`, tâches `check`, `sim-cuda`, `bench` | 1 |
| `CMakeLists.txt` | bibliothèque `mppi_core`, exécutables | 1, D |
| `cuda/exercises/*.cu` | premiers kernels (jetables) | A |
| `cuda/include/mppi/cuda_check.hpp` | `CUDA_CHECK`, `DeviceBuffer<T>` | A |
| `cuda/include/mppi/hd.hpp` | macro `HD` = `__host__ __device__` | C |
| `cuda/include/mppi/config.hpp`, `cuda/src/config.cpp` | lecture du YAML | B |
| `cuda/include/mppi/track.cuh`, `cuda/src/track.cpp` | grille de coût, `lookup`, `wrap` | B |
| `cuda/include/mppi/npy.hpp` | lecture/écriture `.npy` | B |
| `cuda/include/mppi/dynamics.cuh` | `Kinematic`, `Dynamic` (= `dynamics.py`) | C |
| `cuda/include/mppi/cost.cuh` | `stage_cost`, `terminal_cost`, `rollout_cost` | C |
| `cuda/include/mppi/rollouts.hpp`, `cuda/src/mppi_kernel.cu` | le kernel et sa boucle CPU jumelle | D |
| `cuda/include/mppi/controller.hpp`, `cuda/src/controller.cpp` | `Controller` : bruit, rollouts, poids, mise à jour | F |
| `cuda/apps/parity.cpp`, `bench/check_parity.py` | contrôle de parité | E |
| `cuda/apps/sim.cpp`, `python/run_sim_cuda.py` | boucle fermée C++ | F |
| `cuda/apps/bench.cpp`, `bench/run_bench.py` | mesure de latence | G |

**Inchangés** : tout `python/mppi/` (c'est la référence), les tests Python.

---

## 1. Préparer le terrain

### 1.1 Où tourne quoi

- **Machine WSL2 (locale)** : pas de GPU. `pixi run build` y fonctionne (nvcc compile sans carte), le chemin CPU du C++ y tourne, la parité CPU aussi. Tout programme qui touche au GPU y échoue avec `CUDA driver version is insufficient for CUDA runtime version`.
- **Machine GPU distante** (VS Code Remote-SSH) : tout. Vérifie d'abord `nvidia-smi` (modèle et version CUDA du pilote, qui doit être ≥ 12.9 pour le toolkit de pixi) et `pixi run build && ./build/hello_cuda` (le test de l'étape 0).

**Q-1.1.** `nvidia-smi --query-gpu=name,compute_cap --format=csv` sur la machine distante : quel GPU, quelle compute capability ? Est-elle dans `CMAKE_CUDA_ARCHITECTURES` ? Sinon, que se passe-t-il au lancement d'un kernel (fiche §8.2) ?

### 1.2 Dépendance : yaml-cpp

Le C++ doit lire **le même** `config/mppi.yaml` que Python (décision §6.3 du plan). Bibliothèque standard pour ça : `yaml-cpp`, disponible sur conda-forge pour linux-64.

```bash
pixi add yaml-cpp
```

### 1.3 Tâches pixi

`build`, `check` et `bench` existent depuis l'étape 0. À ajuster sous `# STEP 3` :

```toml
check = { cmd = "python bench/check_parity.py --config config/mppi.yaml", env = { PYTHONPATH = "python" }, depends-on = ["build", "track"] }
bench = { cmd = "python bench/run_bench.py --config config/mppi.yaml", env = { PYTHONPATH = "python" }, depends-on = ["build", "track"] }
sim-cuda = { cmd = "python python/run_sim_cuda.py --config config/mppi.yaml", env = { PYTHONPATH = "python" }, depends-on = ["build", "track"] }
```

Les arguments en plus passent au script : `pixi run check --backend cpu` sur la machine sans GPU.

### 1.4 L'architecture en une image

```
                    config/mppi.yaml  ◄── lu par les deux côtés
                   ╱                ╲
     python/mppi (référence)      cuda/ (C++/CUDA)
     rollout_costs(x0,U,eps)      Controller::rollout_costs(x0,U,eps)
             │                         │            │
             │                  rollouts_cpu    rollouts_gpu ── kernel, un thread par k
             │                         └─────┬──────┘
             │                     rollout_cost<Model>(k, ...)   ← UNE fonction __host__ __device__
             │                               │
             ▼                               ▼
          S_py (K,)  ◄──── comparés un par un ────►  S_cpu (K,), S_gpu (K,)
                     bench/check_parity.py
```

L'idée centrale : la physique et le coût sont écrits **une seule fois**, en fonctions `__host__ __device__` (fiche §6.5), et compilés pour le CPU **et** pour le GPU. Trois implémentations à comparer : NumPy (float64), C++ CPU (float32), CUDA GPU (float32).

- NumPy ≠ C++ CPU → bug dans le **portage** (maths, indices, conventions). Déboguable sans GPU, avec `printf` et `gdb`.
- C++ CPU = NumPy mais GPU ≠ → bug dans la **mécanique CUDA** (copies, indice de thread, tailles).

**Q-1.2.** Pourquoi ne pas écrire directement le kernel, comme le dit le plan, et comparer GPU contre NumPy ? Qu'apporte le chemin CPU intermédiaire ?

✔ **Point de contrôle 1.** `pixi add yaml-cpp` fait, `pixi run build` compile toujours `hello_cuda`, et sur la machine distante `./build/hello_cuda` affiche ton GPU.

---

## 2. Partie A : premiers kernels

Avant le projet, deux petits programmes dans `cuda/exercises/` (non ajoutés au CMake, compilés à la main). Lis la fiche §1 à §5 d'abord.

```bash
pixi run nvcc -std=c++17 -arch=sm_89 -Icuda/include -o build/ex1 cuda/exercises/ex1_saxpy.cu
./build/ex1
```

(**Toujours via `pixi run`** : fiche §7.4.)

### 2.1 Exercice 1 : SAXPY

`y = a·x + y` sur un million de floats, un thread par élément (fiche §3). Puis :

1. écris `CUDA_CHECK` (fiche §8.1) dans `cuda/include/mppi/cuda_check.hpp` et entoure **chaque** appel ;
2. provoque les erreurs : `block = 2048` ; `grid = n` au lieu de `(n + block - 1) / block` mais sans garde `if (i < n)` ; copie `cudaMemcpy(y.data(), d_y, n, ...)` (octets oubliés). Note ce que tu observes pour chacune ;
3. lance la version sans garde sous `pixi run compute-sanitizer ./build/ex1`.

**Q-A1.** Avec n = 1 000 000 et des blocs de 256, combien de blocs ? Combien de threads ne font rien ? Que se passe-t-il exactement si on oublie la garde ?

**Q-A2.** Laquelle des trois erreurs ne déclenche **aucun** message sans `compute-sanitizer` ni vérification du résultat ? Pourquoi est-ce la plus dangereuse ?

### 2.2 Exercice 2 : un thread par rollout, en miniature

Le thread `k` intègre `ṙ = −k_k r` par Euler pendant T = 30 pas (le mode de lacet de l'étape 2, §5.2), avec `k_k = 95,2 / vx_k`, `vx_k = 0,5 + 0,01 k`, K = 1024 threads. Il écrit **un seul float** : `r_T`. Le CPU compare à la formule exacte `r_T = r₀ (1 − k dt)^T` et compte les threads instables (`|1 − k dt| > 1`).

C'est la structure exacte du kernel MPPI : boucle sur t **dans** le thread, état dans une variable locale (un registre), une écriture en mémoire globale à la fin.

**Q-A3.** Combien de threads instables attends-tu (seuil d'Euler de l'étape 2, 0,95 m/s) ? Les threads instables ralentissent-ils les autres ?

**Q-A4.** Écris le kernel en fonction `__device__ float decay_one(float k, ...)` appelée par le kernel, puis rends-la `__host__ __device__` et appelle-la aussi depuis une boucle CPU. Les résultats CPU et GPU sont-ils **bit à bit** identiques ? Pourquoi (fiche §6.3) ?

✔ **Point de contrôle A.** Les deux exercices tournent sur la machine distante. Tu sais expliquer `blockIdx.x * blockDim.x + threadIdx.x`, pourquoi `cudaMemcpy` attend le kernel, et à quoi sert `cudaGetLastError()`.

---

## 3. Partie B : la configuration et la piste en C++

### 3.1 Deux niveaux de configuration

```cpp
struct Params {                 // ce que le kernel lit : copié par valeur au lancement
    ModelKind model;  int state_dim, K, T;
    float dt, lambda, gamma, noise_std[2], u_min[2], u_max[2];
    VehicleParams vehicle;      // floats et enums seulement
    CostParams cost;
};
struct Config {                 // tout, côté hôte
    Params p;
    unsigned long long seed;
    std::string track_bin_path;
    int max_steps;
};
Config load_config(const std::string& path, const std::vector<std::string>& overrides = {});
```

`Params` est passé **par valeur** au kernel (fiche §5.1) : il ne doit contenir **aucun** `std::string`, `std::vector` ni pointeur hôte. Les chaînes du YAML (`tire_model: pacejka`) deviennent des `enum class`.

### 3.2 yaml-cpp en 6 lignes

```cpp
#include <yaml-cpp/yaml.h>
YAML::Node raw = YAML::LoadFile(path);
float dt = raw["mppi"]["dt"].as<float>();
std::string tire = raw["vehicle"]["tire_model"].as<std::string>();
YAML::Node std_ = raw["mppi"]["noise_std"];          // séquence : std_.size(), std_[0].as<float>()
if (!raw["mppi"]["lambda"]) throw std::runtime_error("missing key mppi.lambda");
```

Piège : `raw["clé_inexistante"].as<float>()` lève une exception au message obscur. Écris un petit `at(node, "clé")` qui lève une erreur avec le nom de la clé.

**Validation** : la même que `config.py` (`model`/`state_dim` cohérents, `tire_model` et `integrator` connus, `0 < blend_speed_low < blend_speed_high`, `substeps ≥ 1`, `u_min < u_max`, `lf + lr = wheelbase`). Une faute de frappe doit échouer au chargement, des deux côtés.

**Les surcharges `--set`.** Le contrôle de parité teste plusieurs variantes (pneu tanh, Euler, cinématique…) sans écrire un YAML par cas : `load_config(path, {"vehicle.tire_model=tanh", "mppi.horizon=50"})` modifie l'arbre YAML **avant** de le lire. Piège de yaml-cpp : pour descendre dans l'arbre, `node = node["vehicle"]` **écrit** dans `node` au lieu de le re-pointer ; il faut `node.reset(node["vehicle"])`.

**Q-B1.** Pourquoi la surcharge doit-elle échouer si la clé n'existe pas déjà dans le YAML ?

### 3.3 La piste

`track.save` (étape 1) écrit `results/track/costmap.bin` : un en-tête de 32 octets (`"<4sIiiffff"` : magic `MPPG`, version, nx, ny, x_min, y_min, res, length), puis la grille `(ny, nx, 2)` en float32 row-major. En C++ :

```cpp
struct TrackView {              // ce que lit un rollout, passé par valeur au kernel
    const float* grid;          // pointeur HÔTE sur le chemin CPU, DEVICE sur le chemin GPU
    int nx, ny;
    float x_min, y_min, res, length;
};
HD void lookup(const TrackView& tr, float x, float y, float& d, float& s);   // = track.lookup
HD float wrap(float ds, float length);                                       // = track.wrap
```

`lookup` : cellule la plus proche, indices bornés (`floorf`, puis `min`/`max`), `cell = grid + (iy*nx + ix)*2`.

**Q-B2.** `wrap` en Python : `(ds + L/2) % L − L/2`. Le `%` de Python et le `fmodf` du C n'ont pas le même comportement pour un argument négatif. Lequel renvoie quoi pour `(−3) % 10` ? Corrige `fmodf` pour retrouver Python. Note bien ce point : la partie E y reviendra.

### 3.4 Échanger des tableaux avec NumPy : `.npy`

Pour la parité et les logs de simulation, le C++ lit et écrit des `.npy`, que NumPy ouvre directement. Le format est simple : `\x93NUMPY`, version, longueur d'en-tête sur 2 octets, un dictionnaire Python en texte (`{'descr': '<f4', 'fortran_order': False, 'shape': (1024, 30, 2), }`) complété par des espaces jusqu'à un multiple de 64 octets, puis les données brutes. Un lecteur/écrivain minimal (float32/float64, ordre C) tient en 80 lignes (`npy.hpp`, en solution). Tu peux le recopier : ce n'est pas le sujet de l'étape.

✔ **Point de contrôle B.** Un petit `main` charge la config et la piste et affiche `K`, `T`, `lambda`, `tire_model`, `nx × ny` et `length` : les mêmes valeurs que Python (`length 48.31 m, grid 454x279`).

---

## 4. Partie C : dynamique et coût en `__host__ __device__`

### 4.1 Les règles du portage

`dynamics.cuh` et `cost.cuh` sont la traduction **ligne à ligne** de `dynamics.py` et `cost.py`. Règles :

1. **Même ordre d'opérations** que Python quand c'est raisonnable : ça facilite la comparaison ligne à ligne en cas d'écart.
2. **FP32 strict** : littéraux `0.5f`, fonctions `sinf`, `atanf`, `tanhf`, `fmaxf` (fiche §6.2).
3. **Pas de `if` sur l'état** : `fmaxf`, `fminf`, mélange par `kappa`, comme en NumPy. Les `switch` sur `tire_model` et `integrator` sont permis (uniformes sur le warp).
4. **Taille d'état connue à la compilation** : le modèle est un **type**, paramètre de template (fiche §6.4).

```cpp
struct Kinematic {
    static constexpr int DIM = 4;
    HD static void step(const float* s, float a, float delta, float dt,
                        const VehicleParams& v, float* out);
};
struct Dynamic {
    static constexpr int DIM = 6;
    HD static void derivative(const float* s, float a, float delta, const VehicleParams& v, float* ds);
    HD static void step_dynamic_only(...);   // substeps Euler ou RK4
    HD static void step_kinematic6(...);
    HD static void step(...);                // mélange kappa, vx >= 0
};
```

Les tableaux locaux (`float k1[DIM]`) restent en registres tant que toutes les boucles qui les indexent ont des bornes connues à la compilation (le compilateur les déroule). `DIM` étant `constexpr`, c'est le cas.

**Q-C1.** L'interface Python est `step(state, control, dt, vehicle)` qui choisit le modèle selon `state.shape[-1]`. En CUDA, pourquoi un template plutôt qu'un `if (state_dim == 6)` dans le kernel ? (Deux raisons : divergence, et taille des tableaux locaux.)

### 4.2 Le coût d'un rollout

La fonction centrale, appelée par le kernel **et** par la boucle CPU :

```cpp
template <class Model>
HD float rollout_cost(int k, const float* x0, const float* U, const float* eps,
                      const TrackView& tr, const Params& p);
```

Elle reproduit exactement `rollout_costs` de `controller.py` pour **un** k :

1. `x = x0` (tableau local `float x[Model::DIM]`), `s0` = progression de `x0` ;
2. pour t = 0 … T−1 : `v_t = clip(U[t] + eps[k, t])`, bruit effectif `v_t − U[t]` (pour le terme γ), `x = step(x, v_t)`, `S += stage_cost(x, v_t)` ;
3. `S += terminal_cost(x_T, s0)` ;
4. `S += γ · Σ_t U_t Σ⁻¹ ε_t` (avec le bruit **effectif**) ;
5. `S = isfinite(S) ? S : S_MAX`.

Indices (ordre C, fiche §12) : `U[t*2 + j]`, `eps[(k*T + t)*2 + j]`.

Le terme d'adhérence n'existe que pour le modèle cinématique : `if constexpr (Model::DIM == Kinematic::DIM)`.

**Q-C2.** Dans `stage_cost` Python, `offtrack = w * (excess > 0) * (1 + excess/d0)`. En C++, `excess > 0.0f ? w * (1 + excess/d0) : 0.0f`. Est-ce une branche qui diverge ? Que génère le compilateur pour une expression aussi courte ?

**Q-C3.** Le kernel lira `U` et `x0` : identiques pour tous les threads. Et `eps` : différent pour chaque thread. Lequel de ces accès est coalescé (fiche §5.3) ?

✔ **Point de contrôle C.** `dynamics.cuh` et `cost.cuh` compilent dans un `.cu`. Relis-les côte à côte avec les fichiers Python : chaque ligne Python a sa jumelle.

---

## 5. Partie D : le kernel et le chemin CPU

### 5.1 Le kernel

Une fois `rollout_cost` écrite, le kernel tient en trois lignes :

```cpp
template <class Model>
__global__ void rollout_kernel(const float* x0, const float* U, const float* eps,
                               TrackView track, Params p, float* S) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= p.K) return;
    S[k] = rollout_cost<Model>(k, x0, U, eps, track, p);
}
```

Et sa jumelle CPU :

```cpp
for (int k = 0; k < p.K; ++k) S[k] = rollout_cost<Model>(k, x0, U, eps, track, p);
```

Deux fonctions d'interface, déclarées dans `rollouts.hpp` (du C++ pur, appelable depuis un `.cpp`) et définies dans `mppi_kernel.cu` :

```cpp
void rollouts_cpu(const float* x0, const float* U, const float* eps,
                  const TrackView& track, const Params& p, float* S);          // pointeurs hôte
void rollouts_gpu(const float* d_x0, const float* d_U, const float* d_eps,
                  const TrackView& d_track, const Params& p, float* d_S);      // pointeurs device
```

`rollouts_gpu` choisit le template selon `p.model`, calcule la grille (`(K + 255) / 256` blocs de 256), lance, et vérifie `cudaGetLastError()`.

### 5.2 La grille sur le GPU

`TrackView` contient un pointeur. Pour le GPU, il faut copier la grille (454 × 279 × 2 floats ≈ 1 Mo) en mémoire device **une fois** au démarrage, et passer au kernel un `TrackView` dont `grid` est le pointeur device. Les autres champs (nx, ny, x_min…) sont les mêmes.

### 5.3 CMake

```cmake
find_package(CUDAToolkit REQUIRED)
find_package(yaml-cpp CONFIG REQUIRED)

add_library(mppi_core STATIC
  cuda/src/config.cpp cuda/src/track.cpp cuda/src/controller.cpp cuda/src/mppi_kernel.cu)
target_include_directories(mppi_core PUBLIC ${CMAKE_SOURCE_DIR}/cuda/include)
target_link_libraries(mppi_core PUBLIC CUDA::cudart yaml-cpp::yaml-cpp)
target_compile_options(mppi_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

add_executable(mppi_parity cuda/apps/parity.cpp)
target_link_libraries(mppi_parity PRIVATE mppi_core)
```

`controller.cpp` est un `.cpp` : il appelle `cudaMalloc` et `rollouts_gpu`, mais ne contient pas de kernel (fiche §7.1).

**Q-D1.** Ressources du kernel de référence (`pixi run cuobjdump --dump-resource-usage build/libmppi_core.a`) : 72 registres par thread pour `Dynamic`, 36 pour `Kinematic`, 32 octets de pile, 0 mémoire partagée. Combien de threads de ce kernel dynamique peuvent être actifs en même temps sur un SM (65 536 registres) ? Avec K = 1024 et des blocs de 256, combien de SM de la RTX 4000 Ada (48) travaillent ? Et avec K = 8192 ?

✔ **Point de contrôle D.** `pixi run build` compile `mppi_core` sans avertissement. `cuobjdump -sass build/libmppi_core.a | grep -cE "DADD|DMUL|DFMA"` donne 0 pour les kernels (aucun calcul en double).

---

## 6. Partie E : la parité, cœur de l'étape

### 6.1 Le protocole

`bench/check_parity.py`, pour chaque cas :

1. construit la config Python du cas (`dataclasses.replace`) et les mêmes surcharges `--set` pour le C++ ;
2. tire `x0`, `U` et le bruit `(K, T, 2)` en NumPy, les **convertit en float32** et les sauve en `.npy` dans `results/parity/caseN/` ;
3. lance `build/mppi_parity --dir ... --backend cpu|gpu|both --set ...`, qui écrit `S_cpu.npy`, `S_gpu.npy` ;
4. calcule `S_py = rollout_costs(...)` en float64 **sur les entrées float32 reconverties** (les deux côtés partent exactement des mêmes nombres) ;
5. compare rollout par rollout : `err_k = |S_cpp − S_py| / max(1, |S_py|)`.

Les cas doivent couvrir chaque branche du code : départ arrêté (`vx = 0`, plancher `vx_safe`), virage en dérapage à 6 m/s, dans la bande de mélange (1,2 m/s), hors piste, modèle cinématique (terme d'adhérence), tanh + Euler ×2 + γ ≠ 0, pneus linéaires, K = 8192 et T = 50.

**Q-E1.** Pourquoi `max(1, |S_py|)` et pas `|S_py|` au dénominateur ?

### 6.2 Ce que tu vas observer : la surprise

Mesuré sur le chemin CPU de la référence :

| cas | erreur max | erreur p99 | rollouts > 1e-4 |
|---|---|---|---|
| standstill | 2,8e-7 | 2,3e-7 | 0 |
| cornering | 6,7e-4 | 1,8e-6 | 1 |
| in the band | 3,4e-7 | 2,8e-7 | 0 |
| off track | 3,3e-7 | 1,8e-7 | 0 |
| kinematic | 4,7e-3 | 1,1e-6 | 1 |
| tanh, euler ×2, γ | 6,7e-4 | 1,7e-5 | 1 |
| linear tires | 3,8e-4 | 1,7e-6 | 1 |
| K = 8192, T = 50 | 6,1e-2 | 1,2e-6 | 9 |

99 % des rollouts coïncident à 1e-6 près, mais **quelques-uns** sont loin, jusqu'à 6 % d'écart. Le plan dit « une seule divergence signale un bug de dynamique ». Ce n'est pas un bug.

**Q-E2.** Avant de lire la suite : propose une explication. Indice : qu'est-ce qui, dans le coût, n'est **pas continu** en fonction de la position ?

### 6.3 L'explication et le critère

`lookup` prend la **cellule la plus proche** : `d` et `s` sont constants par morceaux et sautent quand la position franchit un bord de cellule (tous les 5 cm). Un rollout qui passe à quelques micromètres d'un bord peut tomber dans une cellule en float32 et dans la voisine en float64. Son coût saute (et si `|d|` franchit `d_max`, la pénalité hors piste de 100 s'ajoute ou disparaît).

Vérifié sur la référence : **chaque** rollout hors tolérance a un état à moins de 2 µm d'un bord de cellule. Et NumPy seul, avec `x0` décalé de 10 µm, produit **plus** d'écarts > 1e-4 que le C++ (6 contre 1 sur le cas « cornering », 47 contre 9 sur K = 8192) : le coût lui-même est sensible à ce niveau, le portage n'y est pour rien.

D'où le critère du contrôle de parité :

- un rollout avec `err > 1e-4` est **expliqué** si sa trajectoire NumPy passe à moins de `EDGE_EPS = 1e-5` m d'un bord de cellule ; sinon il est **inexpliqué** ;
- le contrôle **échoue** s'il y a un seul rollout inexpliqué, ou si plus de 1 % des rollouts sont « expliqués » (un vrai bug ne doit pas se cacher derrière l'excuse).

```python
def edge_distance(X, track):
    """(K, T+1, n) -> (K,) plus petite distance (m) de x_1..x_T à un bord de cellule."""
    g = (X[:, 1:, :2] - [track.x_min, track.y_min]) / track.res
    return np.abs(g - np.round(g)).min(axis=(1, 2)) * track.res
```

**Q-E3.** L'erreur de position en float32 après 50 pas est de l'ordre du micromètre. Pourquoi `EDGE_EPS = 1e-5` m et pas 1e-3 m ? Que deviendrait le contrôle avec 1e-3 ?

**Q-E4.** L'interpolation bilinéaire de la grille (étape 4, texture) rend `d` et `s` continus. Que deviendraient ces écarts ? Pourquoi ne peut-on pas la mettre seulement côté CUDA ?

### 6.4 Tester le test

Un contrôle qui ne peut pas échouer ne protège de rien (même idée qu'à l'étape 2, partie D). Injecte un bug, recompile, relance `pixi run check --backend cpu`, puis remets le code :

1. **oubli du `clip`** : `vt[j] = u + e;` au lieu de `fminf(fmaxf(u + e, u_min), u_max)` ;
2. **`wrap` sans correction du modulo** (Q-B2) ;
3. un bug de ton choix dans la dynamique (un signe, `lf` au lieu de `lr`).

Mesuré sur la référence. Le bug 1 est détecté : 33 à 234 rollouts inexpliqués selon le cas. Le bug 2, lui, **passe inaperçu** avec les 8 cas ci-dessus : aucun rollout ne franchit la ligne de départ, donc `ds` n'est jamais négatif. Il a fallu ajouter un cas « start line » (départ 1 m avant la ligne, à 6 m/s), après quoi le bug donne 1012 rollouts inexpliqués sur 1024.

**Q-E5.** Quelle leçon générale tires-tu du bug 2 sur la conception des cas de test ?

### 6.5 Le chemin GPU

Une fois le CPU vert, `pixi run check` (sans `--backend`) sur la machine distante compare aussi le GPU. À quoi t'attendre : même ordre de grandeur que le CPU (p99 ~1e-6), avec quelques rollouts « edge », **pas forcément les mêmes** que le CPU : `nvcc` fusionne les `a*b + c` en FMA et les `sinf`/`atanf` du GPU n'arrondissent pas comme ceux de la libm (fiche §6.3).

Si le GPU donne n'importe quoi alors que le CPU est vert : le bug est dans la mécanique (indice de thread, taille de copie, `TrackView` avec un pointeur hôte, tableaux non copiés). `pixi run compute-sanitizer ./build/mppi_parity --dir results/parity/case0 --backend gpu` en premier.

**Q-E6.** Lance la parité GPU avec `--fmad=false` (ajout temporaire dans `target_compile_options`). L'erreur p99 change-t-elle ? Le nombre de cas « edge » ?

✔ **Point de contrôle E.** `pixi run check` : 0 inexpliqué partout, CPU et GPU. Au moins deux bugs injectés détectés. `compute-sanitizer` propre.

---

## 7. Partie F : la boucle de contrôle C++

### 7.1 La classe `Controller`

Même algorithme que `MPPI.command` en Python (théorie §2.3) :

```cpp
class Controller {
public:
    Controller(const Config& cfg, const TrackHost& track, Backend backend);   // alloue tout ici
    std::array<float, 2> command(const float* x0, Diagnostics* diag);
    void rollout_costs(const float* x0, const float* U, const float* eps, float* S, Diagnostics* diag);
private:
    Params p_;  Backend backend_;
    TrackView track_host_, track_dev_;
    std::vector<float> U_, eps_, S_;  std::vector<double> w_;
    std::mt19937_64 rng_;  std::normal_distribution<float> normal_;
    DeviceBuffer<float> d_grid_, d_x0_, d_U_, d_eps_, d_S_;    // alloués une fois
};
```

`command` :

1. tire le bruit `(K, T, 2)` sur le CPU ;
2. `rollout_costs` : sur GPU, copie `x0`, `U`, `eps` → kernel → copie `S` ;
3. poids softmin en **double** sur le CPU (`rho = min S`, `w = exp(−(S − rho)/λ)`, normalisation, ESS) ;
4. `U += Σ_k w_k (clip(U + eps_k) − U)` ;
5. renvoie `U[0]`, décale, recopie la dernière commande.

**Règle : aucune allocation dans `command`.** `cudaMalloc` coûte des centaines de µs ; tout est alloué dans le constructeur. Le `DeviceBuffer<T>` (RAII, fiche §5.2) libère tout seul.

**Q-F1.** Le chemin CPU ne doit appeler **aucune** fonction CUDA (pas même dans le constructeur). Pourquoi ? (Pense à la machine WSL2.)

**Q-F2.** Pourquoi calculer les poids en `double` sur le CPU alors que les coûts sont en `float` ?

### 7.2 La boucle fermée

`cuda/apps/sim.cpp` reproduit `simulate()` de `run_sim.py` : à chaque pas, `observe` (norme de la vitesse pour un contrôleur cinématique), `command`, véhicule simulé `Dynamic::step` (la même fonction `HD`, appelée sur le CPU), progression par `lookup` + `wrap`, arrêt au tour complet ou sur un état non fini. Il écrit ses logs en `.npy` (`states`, `controls`, `next_states`, `ess`, `rho`, `t_iter_ms`, `t_kernel_ms`).

Le C++ ne connaît pas la ligne centrale : l'état initial lui est donné par `python/run_sim_cuda.py` (`--x0 x0.npy`). Ce script relit ensuite les logs, reconstruit un `SimLog` et réutilise `summarize`, `plot_trajectory` et `plot_series` de `run_sim.py` : **le même code de mesure** pour NumPy et C++.

Différence voulue avec NumPy : le véhicule simulé C++ est en float32, et le bruit vient de `std::mt19937_64` au lieu du générateur PCG64 de NumPy. Les deux tours ne peuvent donc pas être identiques : on compare des **distributions** (plusieurs graines), pas une trajectoire.

### 7.3 Résultats de référence (chemin CPU)

`pixi run sim-cuda --backend cpu --numpy`, config de l'étape 2 (`v_ref = 7`, λ = 3, K = 1024, T = 30) :

```
                            C++ cpu            NumPy
 lap                       ✔ 8.70 s         ✔ 9.02 s
 off-track steps                  0                0
 max |d| (m)                   0.58             0.54
 speed mean / max (m/s)  5.64 / 7.02      5.46 / 6.72
 max |β| (deg)                 23.4             16.4
 ESS median / p5           572 / 21         484 / 13
 iteration time (ms)           13.8             26.8
```

Sur 10 graines (`--set mppi.seed=s`) :

| | tours propres | temps (s) | max \|d\| | β max (°) |
|---|---|---|---|---|
| NumPy | 10/10 | 8,58 à 9,18 | 0,52 à 0,65 | 15,1 à 42,9 |
| C++ (CPU) | 9/10 | 8,48 à 8,92 | 0,55 à 0,67 | 18,4 à 49,6 |

La graine 4 du C++ sort d'**un** pas (0,67 m pour une limite de 0,65) ; la graine 5 de NumPy frôle la limite (0,65) sans sortir. Même comportement des deux côtés : à 7 m/s et λ = 3, le contrôleur vit près du bord (étape 2, H1). Ce n'est pas une différence d'implémentation.

**Q-F3.** Comment montrerais-tu plus rigoureusement que les deux contrôleurs sont « les mêmes » en boucle fermée, malgré des générateurs aléatoires différents ? (Deux idées : une sur le bruit, une sur la statistique.)

✔ **Point de contrôle F.** `pixi run sim-cuda` (GPU) boucle le tour, et 5 graines donnent des métriques dans les mêmes plages que NumPy.

---

## 8. Partie G : mesurer la baseline

### 8.1 L'outil

`cuda/apps/bench.cpp` lance `command()` en boucle depuis un état fixe, jette 10 itérations d'échauffement (fiche §9), et affiche p50 et p99 de chaque phase : bruit (CPU), rollouts (dont copie H→D, kernel, copie D→H, mesurés par événements CUDA), poids + mise à jour, total. `bench/run_bench.py` balaie K.

```bash
pixi run bench                                   # GPU, T = 50, K = 256 … 32768
./build/mppi_bench --backend gpu --set mppi.num_samples=8192 --set mppi.horizon=50
```

### 8.2 Ce qui a été mesuré (CPU), ce que tu dois mesurer (GPU)

Chemin CPU, machine WSL2, un seul cœur :

| K, T | bruit | rollouts | poids + MàJ | total |
|---|---|---|---|---|
| 1024, 30 | 1,6 ms | 18,3 ms | 0,07 ms | 20,0 ms |
| 8192, 50 | **21,7 ms** | 246 ms | 1,0 ms | 269 ms |

Le chiffre important : à la taille cible, **tirer le bruit sur le CPU prend à lui seul 21,7 ms**, plus que tout le budget de 20 ms. Quel que soit le temps du kernel, la baseline GPU ne tiendra pas le budget, et ce n'est pas la faute du kernel.

**À mesurer sur la machine distante** et à reporter dans la ligne 0 de `bench/results.md` (K = 8192, T = 50) : total p50/p99, et la décomposition. Prédis d'abord :

**Q-G1.** Taille du bruit à K = 8192, T = 50 en octets ? À ~10 Go/s effectifs sur PCIe, combien de temps pour la copie H→D ?

**Q-G2.** Le kernel exécute 8192 × 50 évaluations de `Dynamic::step` (4 dérivées RK4, une douzaine de fonctions transcendantes chacune) plus le coût. Ordre de grandeur des opérations ? Combien de temps sur un GPU qui fait ~10 TFLOP/s FP32 si le calcul était parfait ? Pourquoi le temps réel sera-t-il plus grand (occupation, accès non coalescés, divergence de `sinf` sur grands arguments) ?

**Q-G3.** Classe les phases de la plus coûteuse à la moins coûteuse, **avant** de mesurer. Quelle ligne de l'étape 4 attaque la première ?

**Q-G4.** Le premier lancement du kernel est nettement plus lent que les suivants. Pourquoi (fiche §9) ? Mesure-le en passant `--warmup 0 --iters 3`.

### 8.3 Ce que la baseline doit contenir

Dans `bench/results.md`, ligne 0 : temps total p50 (et p99 dans le journal) sur la RTX 4000 Ada à K = 8192, T = 50, modèle dynamique. Note dans le journal ce qui tournait sur le nœud pendant la mesure (`nvidia-smi`). Dans le journal, la décomposition par phase, et le balayage K de `pixi run bench`.

✔ **Point de contrôle G.** Ligne 0 remplie, décomposition dans le journal, et tu sais dire quelle phase domine.

---

## 9. Dépannage

| Symptôme | Cause probable | Vérification |
|---|---|---|
| Des centaines d'erreurs dans `type_traits`, `c++config.h`, `user-defined literal operator not found` | `nvcc` lancé hors pixi : gcc du système avec les en-têtes de pixi | toujours `pixi run nvcc` / `pixi run build` (fiche §7.4) |
| `CUDA driver version is insufficient for CUDA runtime version` | machine sans GPU (WSL2), ou pilote plus ancien que CUDA 12.9 | `nvidia-smi` ; sur WSL2, `--backend cpu` |
| `no kernel image is available for execution on the device` | GPU absent de `CMAKE_CUDA_ARCHITECTURES` | Q-1.1 ; `rm -rf build` après modification |
| `Could not find a package configuration file provided by "yaml-cpp"` | `pixi add yaml-cpp` oublié, ou `cmake` lancé hors pixi | `pixi run build` |
| `undefined reference to mppi::...` | fichier absent de `add_library`, ou fonction déclarée sans définition | `CMakeLists.txt` |
| `identifier "__host__" is undefined` dans un `.cpp` | en-tête avec code device inclus par `g++` | macro `HD` (fiche §6.5) |
| `calling a __host__ function from a __global__ function is not allowed` | fonction appelée dans le kernel sans `__device__`, ou fonction `std::` non disponible côté device | `HD` partout ; `sinf` au lieu de `std::sin` |
| `illegal memory access` | `TrackView` avec le pointeur hôte de la grille sur le GPU ; indice `eps` faux ; buffer trop petit | `compute-sanitizer` ; chemin CPU d'abord |
| Tous les coûts GPU valent 0 ou des déchets, sans erreur | erreur non vérifiée, copie dans le mauvais sens, taille en éléments au lieu d'octets | `CUDA_CHECK` partout, `cudaGetLastError()` |
| CPU vert, GPU tout faux | mécanique CUDA, pas les maths | §6.5 |
| Parité : tous les cas ont des écarts ~1e-3 | un `float` vs `double` dans un paramètre, une constante (G = 9.81) différente, ordre des termes du coût, `noise_std` mal lu | compare la config affichée des deux côtés |
| Parité : seulement le cas « start line » | `wrap` (Q-B2) | |
| Parité : seulement « standstill » ou « in the band » | plancher `vx_safe`, `kappa`, `step_kinematic6` | |
| Parité : seulement « kinematic » | terme d'adhérence, `observe` | |
| Parité : quelques rollouts > 1e-4, tous « edge » | normal (§6.3) | |
| `mppi_sim` beaucoup plus lent que `mppi_bench` | `cudaMalloc` dans `command` ; build `Debug` | constructeur ; `CMAKE_BUILD_TYPE=Release` |
| Instructions `DFMA`/`DMUL` dans le SASS du kernel | littéral sans `f`, `sin` au lieu de `sinf` | fiche §6.2 |

---

## 10. Clôturer l'étape

1. `pixi run build && pixi run test && pixi run check && pixi run sim-cuda && pixi run bench` passent sur la machine distante.
2. Coche le critère de sortie (section 0.2).
3. Ligne 0 de `bench/results.md`.
4. Entrée du journal :
   - **Measured** : tableau de parité CPU et GPU, bugs injectés et détectés, bilan `sim-cuda` sur 5 graines face à NumPy, baseline et décomposition par phase, ressources du kernel (registres) ;
   - **No effect / reverted** : par exemple `--fmad=false` (Q-E6) ;
   - les champs YAML lus par le C++ (tous, désormais) ; nouvelle dépendance `yaml-cpp` ;
   - les questions que tu t'es posées, avec leurs réponses.
5. Coche `Step 3` dans le README.
6. Commit, `git tag step-3`.

---
---

# Solutions

Ne lis une solution qu'après avoir écrit ta propre réponse.

## Réponses aux questions

**S-1.1.** `NVIDIA RTX 4000 Ada Generation, 8.9`, dans la liste (`89`). Un GPU absent de la liste et **plus récent** (une RTX 50xx, 12.0) : le binaire contient le SASS sm_89 et le PTX de compute_89, que le pilote compile à la volée (JIT) ; ça marche, avec un premier lancement lent. Un GPU **plus ancien** que 8.9 (une RTX 3060, 8.6) : le PTX ne se compile pas vers une architecture plus ancienne, aucun code exécutable, `no kernel image is available`. Remède : ajouter son numéro à `CMAKE_CUDA_ARCHITECTURES` et effacer `build/`.

**S-1.2.** Trois raisons. (1) Le chemin CPU se débogue sans GPU (machine WSL2), avec `printf` et `gdb`. (2) Il sépare les deux familles de bugs : maths du portage (CPU ≠ NumPy) et mécanique CUDA (GPU ≠ CPU). (3) Il ne coûte rien : la même fonction `rollout_cost` sert aux deux, il n'y a pas de deuxième implémentation à maintenir.

**S-A1.** 3907 blocs (1 000 000 / 256 = 3906,25, arrondi au-dessus) ; 3907 × 256 − 1 000 000 = 192 threads sans travail. Sans garde, ces 192 threads lisent et écrivent après la fin des tableaux : comportement indéfini. Souvent rien de visible (l'allocation est arrondie par le pilote), parfois `illegal memory access`, parfois la corruption silencieuse d'un autre buffer. `compute-sanitizer` le signale à coup sûr.

**S-A2.** La copie de `n` octets au lieu de `n * sizeof(float)` : elle copie un quart du tableau, ne lève aucune erreur, et les trois quarts restants gardent leurs anciennes valeurs (ici 2,0). Seule la vérification du résultat la détecte. Le débordement sans garde est aussi souvent silencieux. `block = 2048` est la seule bruyante (`invalid configuration argument`), **à condition** de vérifier `cudaGetLastError()`.

**S-A3.** Instable si `k dt > 2`, soit `vx < 95,2 × 0,02 / 2 = 0,952` m/s, soit `0,5 + 0,01 k < 0,952`, soit k ≤ 45 : 46 threads. Ils ne ralentissent personne : tous les threads exécutent les mêmes T pas, quelle que soit la valeur de r (pas de branche), et un `inf` se calcule aussi vite qu'un nombre fini. C'est la même raison pour laquelle un rollout MPPI qui diverge ne coûte rien de plus, et pourquoi on le neutralise après coup (`S_MAX`) au lieu de l'arrêter par un `break`.

**S-A4.** Souvent identiques ici, car la boucle ne fait que `r += dt * (−rate * r)`, mais pas garanti : `nvcc` peut fusionner en FMA côté GPU (un arrondi au lieu de deux), et `95.2f / (0.5f + 0.01f * k)` peut être arrondi différemment. Écart attendu de l'ordre de 1e-7 relatif. Bit à bit identique n'est **jamais** un critère entre CPU et GPU.

**S-B1.** Une surcharge `vehicle.tire_modle=tanh` (faute de frappe) créerait une nouvelle clé ignorée : le C++ tournerait avec Pacejka, le Python (via `dataclasses.replace`) planterait ou ferait autre chose, et le contrôle de parité comparerait deux problèmes différents sans le dire. Même principe que la validation de `config.py` (S-F1 de l'étape 2).

**S-B2.** Python : `(−3) % 10 = 7` (modulo « plancher », résultat du signe du diviseur). C : `fmodf(−3, 10) = −3` (troncature, signe du dividende). Correction : `r = fmodf(ds + L/2, L); if (r < 0) r += L; return r − L/2;`. Le `if` porte sur l'état, mais c'est un `if` d'une instruction que le compilateur transforme en sélection sans saut. Sans cette correction, tout rollout qui franchit la ligne de départ calcule une progression fausse d'une longueur de tour (48 m).

**S-C1.** (1) Le `if` serait uniforme (même valeur pour tous les threads) donc sans divergence, mais il resterait un test à chaque pas. (2) Surtout, la taille du tableau d'état serait inconnue à la compilation : `float x[state_dim]` est impossible, il faudrait `float x[6]` partout, avec des boucles à borne variable que le compilateur ne déroule pas, et le tableau passerait en mémoire **locale** (lente) au lieu des registres. Avec un template, `DIM` est une constante : boucles déroulées, état en registres, et le choix est fait une fois, à l'hôte, avant le lancement. C'est l'équivalent CUDA de « le modèle est choisi par la config, pas par l'état » (S-F2 de l'étape 2).

**S-C2.** Elle peut diverger au sens où deux threads d'un warp peuvent prendre des valeurs différentes de la condition, mais pour une expression aussi courte le compilateur génère les deux calculs et une instruction de **sélection** (`SEL`/`FSEL`), sans saut : aucun coût de divergence. La divergence coûte quand les branches sont longues (des dizaines d'instructions) ou contiennent des boucles.

**S-C3.** `U[t*2 + j]` et `x0` : tous les threads d'un warp lisent **la même** adresse, le matériel sert la valeur à tout le warp en une transaction (*broadcast*), c'est efficace. `eps[(k*T + t)*2 + j]` : les threads `k` et `k+1` lisent à `T × 2 × 4 = 240` octets d'écart (T = 30) : **non coalescé**, une transaction par thread. C'est la ligne 5 (SoA) de l'étape 4. La grille de coût : adresses dépendant de la position de chaque rollout, irrégulières par nature (ligne 4, texture).

**S-D1.** 65 536 / 72 = 910 threads, arrondis par l'allocation des registres (par warp, par granules) : en pratique 768 threads (24 warps, 3 blocs de 256) sur les 1536 possibles d'un SM Ada (8.9), soit 50 % d'occupation théorique. Avec K = 1024 : 4 blocs, donc **4 SM sur 48** travaillent, le GPU est inactif à plus de 90 %. À K = 8192 : 32 blocs, un par SM, donc **32 SM sur 48**, chacun à un tiers de sa capacité (1 bloc sur 3 possibles). Il faut K = 48 × 256 = 12 288 pour donner un bloc à chaque SM, et K = 48 × 768 = 36 864 pour les remplir. Conclusion : jusqu'à K = 8192, la RTX 4000 Ada est mal utilisée quel que soit le kernel ; cela se verra dans le balayage en K (temps presque constant de 256 à ~12 288). Des blocs de 128 répartissent K = 8192 sur les 48 SM (64 blocs) : une ligne possible du tableau de l'étape 4.

**S-E1.** Certains coûts sont proches de zéro (la récompense de progression terminale est négative et compense les termes positifs). Une erreur relative sur un nombre proche de 0 est énorme pour un écart absolu insignifiant. `max(1, |S|)` donne une erreur relative pour les grands coûts et absolue pour les petits. Les coûts typiques sont O(1) à O(1000) : l'échelle 1 est l'unité naturelle du coût normalisé (étape 1, partie D).

**S-E2.** `lookup` (cellule la plus proche) : le coût est constant par morceaux en position, avec des sauts aux bords de cellule. Toute différence d'arrondi, même de 1e-7 m, peut faire basculer une cellule. Voir §6.3.

**S-E3.** Probabilité qu'un rollout passe « par hasard » à moins de ε d'un bord : environ `2 × 2ε/res` par coordonnée et par pas, sur 2T coordonnées-pas. Avec ε = 1e-5, res = 0,05, T = 30 : ~5 % des rollouts. Avec ε = 1e-3 : la quasi-totalité des rollouts serait « explicable », et un vrai bug passerait. ε doit être juste au-dessus de l'erreur de position float32 attendue (~1e-6 m). Le plafond de 1 % de rollouts « expliqués » ferme la porte restante.

**S-E4.** Avec une interpolation bilinéaire, `d` et `s` deviennent continus en position (mais pas dérivables aux bords) : un écart de position de 1e-6 m donne un écart de coût de l'ordre de 1e-6 × pente, et les « edge » disparaissent (sauf au seuil hors piste, qui reste un saut de `w_offtrack` : il faudrait aussi le lisser). On ne peut pas la mettre seulement côté CUDA : la parité comparerait deux fonctions de coût différentes. Il faut changer `track.lookup` en Python en même temps, et refaire les réglages de l'étape 2 (le coût change). C'est une décision de l'étape 4, à mesurer.

**S-E5.** Un cas de test n'exerce que les branches que ses trajectoires traversent. Il faut un cas par **régime** du code (arrêt, bande, dynamique pur, hors piste, ligne de départ, chaque modèle de pneu, chaque intégrateur), et vérifier la couverture en injectant des bugs dans chaque fonction (*mutation testing*). Le plan disait « une divergence signale un bug » ; la réciproque est fausse : l'absence de divergence ne prouve rien sur le code que les cas n'exécutent pas.

**S-E6.** Non mesuré (pas de GPU). Attendu : la p99 change de l'ordre de 1e-7, les cas « edge » changent d'identité. Si tout devient vert sans FMA alors qu'il y avait des écarts inexpliqués, le code est sensible aux arrondis à un endroit inattendu (une soustraction de nombres proches, par exemple) : à regarder. Ne garde pas `--fmad=false` : c'est un outil de diagnostic.

**S-F1.** Sur une machine sans GPU, le premier appel CUDA (même `cudaEventCreate` ou `cudaMalloc` de 4 octets) échoue avec `driver version is insufficient`, et `CUDA_CHECK` arrête le programme. Le chemin CPU doit tourner partout, y compris en CI (étape `ci.md`) : c'est lui qui teste les maths.

**S-F2.** `exp(−(S − ρ)/λ)` pour des écarts de coût jusqu'à ~1000 et λ = 3 sous-passe la plage du float (`exp(−87)` ≈ 1e-38, plus petit float normal) : beaucoup de poids deviennent 0 ou dénormaux en float, alors qu'en double ils restent représentables jusqu'à `exp(−708)`. La somme de 8192 poids et l'ESS sont aussi plus justes en double. Sur CPU, le double ne coûte rien. Sur GPU (étape 4), il faudra raisonner en float : soustraire ρ est alors indispensable, et les poids nuls n'ont pas d'importance (ils ne contribuent pas).

**S-F3.** (1) **Même bruit** : faire écrire le bruit par NumPy dans un fichier et le faire lire par le C++ à chaque itération (mode « replay ») ; la boucle fermée doit alors suivre la trajectoire NumPy pendant des centaines de pas, jusqu'à ce qu'un écart d'arrondi bascule une cellule de la grille. (2) **Statistique** : 20 à 50 graines de chaque côté, et comparer les distributions (temps au tour, `max |d|`, ESS) avec un test de Kolmogorov-Smirnov ou simplement les moyennes et intervalles de confiance. Avec 10 graines, temps moyen 8,81 s (NumPy) contre 8,74 s (C++) pour un écart-type ~0,2 s : différence non significative.

**S-G1.** 8192 × 50 × 2 × 4 octets = 3,28 Mo. À 10 Go/s : ~0,33 ms. En mémoire paginable (`std::vector`), le pilote passe par un tampon intermédiaire : souvent 1 ms ou plus. À mesurer. La copie D→H de S (32 Ko) est négligeable, mais elle **attend** la fin du kernel (synchronisation).

**S-G2.** 409 600 pas × (4 dérivées × ~60 opérations + ~40 pour le reste et le coût) ≈ 409 600 × 300 ≈ 1,2e8 opérations, plus des fonctions transcendantes qui en coûtent chacune 10 à 40. Disons 2e8 à 5e8. À 10 TFLOP/s : 20 à 50 µs si tout était parfait. Le temps réel sera bien plus grand (probablement de l'ordre de la milliseconde, à mesurer) : 50 % d'occupation théorique au mieux, 32 blocs pour 28 SM (une seconde vague avec 4 blocs : la moitié du temps, le GPU est presque vide), lectures du bruit non coalescées, accès irréguliers à la grille, et la boucle séquentielle de 50 pas qui limite le parallélisme d'instructions.

**S-G3.** Prédiction raisonnable : bruit CPU (~22 ms, mesuré) ≫ copie H→D (~0,5 à 1 ms) ≳ kernel (~1 ms ?) > mise à jour CPU (~1 ms, mesuré) ≫ copie D→H. La première optimisation rentable est donc **cuRAND en device** (ligne 2 de l'étape 4), qui supprime le tirage CPU **et** la copie. Le plan met « kernel fusionné » en ligne 1 : le kernel de l'étape 3 est déjà fusionné (rollout + coût dans le même thread, état en registres). Il faudra en tenir compte en écrivant le tableau de l'étape 4 : ta ligne 1 sera peut-être « déjà fait ».

**S-G4.** Le premier lancement charge le module du kernel sur le GPU (chargement paresseux depuis CUDA 12.2 : les kernels sont chargés à leur premier appel, pas au démarrage), et le contexte CUDA, les caches et les fréquences sont « froids ». Ce coût unique ne doit pas entrer dans les statistiques, d'où l'échauffement. En contrôle temps réel, il faut le payer **avant** la première commande : un appel à vide au démarrage du contrôleur.

## Code de référence

C'est le code qui a produit les chiffres « mesurés » de ce document (chemin CPU). Il compile sans avertissement avec `pixi run build` (nvcc 12.9, gcc 14). Le chemin GPU compile mais n'a pas été exécuté.

**Inchangés** : `python/mppi/*`, `python/tests/*`, `config/mppi.yaml`.

### `pixi.toml` (ajouts)

```toml
[dependencies]
yaml-cpp = ">=0.8.0,<0.9"

[tasks]
# STEP 3 (check et bench remplacent ceux de l'étape 0)
check = { cmd = "python bench/check_parity.py --config config/mppi.yaml", env = { PYTHONPATH = "python" }, depends-on = ["build", "track"] }
bench = { cmd = "python bench/run_bench.py --config config/mppi.yaml", env = { PYTHONPATH = "python" }, depends-on = ["build", "track"] }
sim-cuda = { cmd = "python python/run_sim_cuda.py --config config/mppi.yaml", env = { PYTHONPATH = "python" }, depends-on = ["build", "track"] }
```

### `CMakeLists.txt`

```cmake
cmake_minimum_required(VERSION 3.24)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 89 = RTX 4000 Ada Generation (cluster GPU node)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

project(edge_mppi LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)   # CUDA::cudart, for the .cpp files that call the runtime
find_package(yaml-cpp CONFIG REQUIRED)

add_executable(hello_cuda cuda/tests/hello_cuda.cu)
target_include_directories(hello_cuda PRIVATE ${CMAKE_SOURCE_DIR}/cuda/include)

# STEP 3
# The model, the cost and the controller, shared by every executable
add_library(mppi_core STATIC
  cuda/src/config.cpp
  cuda/src/track.cpp
  cuda/src/controller.cpp
  cuda/src/mppi_kernel.cu
)
target_include_directories(mppi_core PUBLIC ${CMAKE_SOURCE_DIR}/cuda/include)
target_link_libraries(mppi_core PUBLIC CUDA::cudart yaml-cpp::yaml-cpp)
# Line numbers in device code, for compute-sanitizer and Nsight. No effect on speed.
target_compile_options(mppi_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

add_executable(mppi_parity cuda/apps/parity.cpp)
target_link_libraries(mppi_parity PRIVATE mppi_core)

add_executable(mppi_sim cuda/apps/sim.cpp)
target_link_libraries(mppi_sim PRIVATE mppi_core)

add_executable(mppi_bench cuda/apps/bench.cpp)
target_link_libraries(mppi_bench PRIVATE mppi_core)
```

### `cuda/exercises/ex1_saxpy.cu`

```cpp
// Exercise 1: y = a * x + y on N floats, one thread per element.
#include <cstdio>
#include <vector>

#include "mppi/cuda_check.hpp"

__global__ void saxpy(int n, float a, const float* x, float* y) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

int main() {
    const int n = 1000000;
    std::vector<float> x(n, 1.0f), y(n, 2.0f);

    float *d_x = nullptr, *d_y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, y.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    const int block = 256;
    const int grid = (n + block - 1) / block;   // 3907 blocks, the last one partly idle
    saxpy<<<grid, block>>>(n, 3.0f, d_x, d_y);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(y.data(), d_y, n * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));

    int wrong = 0;
    for (float v : y) wrong += (v != 5.0f);
    std::printf("y[0] = %.1f, %d wrong values\n", y[0], wrong);
    return wrong == 0 ? 0 : 1;
}
```

### `cuda/exercises/ex2_decay.cu`

```cpp
// Exercise 2: "one thread per rollout" in miniature.
// Thread k integrates dr/dt = -k_k r with explicit Euler for T steps (the yaw
// mode of step 2, section 5.2), k_k = 95.2 / vx_k with vx_k = 0.5 + 0.01 k.
// Writes one float per thread. The host checks against the closed form
// r_T = r0 (1 - k dt)^T, and counts the unstable threads (|1 - k dt| > 1).
#include <cmath>
#include <cstdio>
#include <vector>

#include "mppi/cuda_check.hpp"

__device__ float decay_rate(int k) { return 95.2f / (0.5f + 0.01f * k); }

__global__ void decay(int K, int T, float dt, float r0, float* r_final) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    const float rate = decay_rate(k);
    float r = r0;                       // a local variable: lives in a register
    for (int t = 0; t < T; ++t) r += dt * (-rate * r);
    r_final[k] = r;                     // the only write to global memory
}

int main() {
    const int K = 1024, T = 30;
    const float dt = 0.02f, r0 = 0.1f;
    mppi::DeviceBuffer<float> d_r(K);
    decay<<<(K + 255) / 256, 256>>>(K, T, dt, r0, d_r.get());
    CUDA_CHECK(cudaGetLastError());
    std::vector<float> r(K);
    d_r.download(r.data());

    int unstable = 0;
    double worst = 0.0;
    for (int k = 0; k < K; ++k) {
        const double rate = 95.2 / (0.5 + 0.01 * k);
        const double exact = r0 * std::pow(1.0 - rate * dt, T);
        worst = std::fmax(worst, std::fabs(r[k] - exact) / std::fmax(1e-12, std::fabs(exact)));
        unstable += std::fabs(1.0 - rate * dt) > 1.0;
    }
    std::printf("max relative error %.1e, %d unstable threads (vx < 0.95 m/s)\n", worst, unstable);
    return 0;
}
```

### `cuda/include/mppi/cuda_check.hpp`

```cpp
// Error checking and a small RAII owner for device memory.
//
// Every CUDA runtime call returns a cudaError_t that is silently ignored unless
// you check it. Kernel launches return nothing: check cudaGetLastError() right
// after the launch (bad configuration), and the result of the next
// synchronizing call (errors raised while the kernel ran).
#pragma once

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <utility>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n  %s\n",                \
                         cudaGetErrorString(err_), __FILE__, __LINE__, #call);    \
            std::exit(EXIT_FAILURE);                                              \
        }                                                                         \
    } while (0)

namespace mppi {

// Owns n elements of T in GPU memory. Freed automatically when it goes out of
// scope (RAII), so no cudaFree to forget. Movable, not copyable: two owners of
// the same pointer would free it twice.
template <class T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t n) : n_(n) { CUDA_CHECK(cudaMalloc(&ptr_, n * sizeof(T))); }
    ~DeviceBuffer() { if (ptr_) cudaFree(ptr_); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& o) noexcept { *this = std::move(o); }
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        std::swap(ptr_, o.ptr_);
        std::swap(n_, o.n_);
        return *this;
    }

    T* get() const { return ptr_; }
    std::size_t size() const { return n_; }

    void upload(const T* host) { CUDA_CHECK(cudaMemcpy(ptr_, host, n_ * sizeof(T), cudaMemcpyHostToDevice)); }
    void download(T* host) const { CUDA_CHECK(cudaMemcpy(host, ptr_, n_ * sizeof(T), cudaMemcpyDeviceToHost)); }

private:
    T* ptr_ = nullptr;
    std::size_t n_ = 0;
};

}  // namespace mppi
```

### `cuda/include/mppi/hd.hpp`

```cpp
// HD marks a function compiled for both the CPU and the GPU.
//
// nvcc understands __host__ __device__. A plain C++ compiler (g++ on a .cpp
// file) does not, so the macro expands to nothing there and the header stays
// includable from host-only code.
#pragma once

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#include <algorithm>
#include <cmath>
using std::isfinite;
using std::max;
using std::min;
#endif
```

### `cuda/include/mppi/config.hpp`

```cpp
// Parses config/mppi.yaml, the same file the Python side reads.
//
// Two levels:
//   - Config: everything, host only (std::string, paths).
//   - Params: the plain numbers the kernel needs, passed BY VALUE to the kernel.
//     Only float, int and enums: no pointer, no std::string, no std::vector,
//     so it can be copied to the GPU as is.
//
// Same validation as python/mppi/config.py: a typo must fail at load time.
#pragma once

#include <string>
#include <vector>

namespace mppi {

constexpr int CONTROL_DIM = 2;   // [a, delta]

enum class ModelKind : int { Kinematic = 0, Dynamic = 1 };
enum class TireModel : int { Linear = 0, Tanh = 1, Pacejka = 2 };
enum class Integrator : int { Euler = 0, Rk4 = 1 };

struct VehicleParams {
    float wheelbase, lf, lr, width, mu, mass, izz;
    float cs_front, cs_rear;           // cornering_stiffness_front / _rear
    TireModel tire_model;
    float pacejka_c, pacejka_e;
    float blend_speed_low, blend_speed_high;
    Integrator integrator;
    int substeps;
};

struct CostParams {
    float w_lateral, w_progress, w_offtrack, w_control, w_adhesion, w_speed;
    float lateral_scale, speed_scale, v_ref, track_half_width;
};

// Everything a rollout needs, in one trivially copyable struct.
struct Params {
    ModelKind model;
    int state_dim;
    int K, T;
    float dt, lambda, gamma;
    float noise_std[CONTROL_DIM];
    float u_min[CONTROL_DIM], u_max[CONTROL_DIM];
    VehicleParams vehicle;
    CostParams cost;
};

struct Config {
    Params p;
    unsigned long long seed;
    std::string track_bin_path;   // absolute
    int max_steps;
};

// path: config/mppi.yaml. Relative paths of the YAML are resolved from the repo
// root (parent of the config/ directory), like config.py.
// overrides: "section.key=value" strings applied to the YAML before parsing,
// e.g. {"vehicle.tire_model=tanh", "mppi.horizon=50"}. Used by the parity
// check and the sweeps, so that the C++ and Python sides see the same change.
Config load_config(const std::string& path, const std::vector<std::string>& overrides = {});

}  // namespace mppi
```

### `cuda/src/config.cpp`

```cpp
#include "mppi/config.hpp"

#include <cmath>
#include <filesystem>
#include <stdexcept>

#include <yaml-cpp/yaml.h>

namespace mppi {
namespace {

std::runtime_error bad(const std::string& msg) { return std::runtime_error("config: " + msg); }

// node["key"] that throws with the key name instead of a cryptic yaml-cpp error
YAML::Node at(const YAML::Node& node, const std::string& key) {
    YAML::Node child = node[key];
    if (!child) throw bad("missing key '" + key + "'");
    return child;
}

float f(const YAML::Node& node, const std::string& key) { return at(node, key).as<float>(); }

// "vehicle.tire_model=tanh" -> raw["vehicle"]["tire_model"] = "tanh"
void apply_override(YAML::Node& raw, const std::string& kv) {
    const auto eq = kv.find('=');
    if (eq == std::string::npos) throw bad("override '" + kv + "' needs key=value");
    std::string path = kv.substr(0, eq);
    YAML::Node node = raw;
    for (auto dot = path.find('.'); dot != std::string::npos; dot = path.find('.')) {
        node.reset(at(node, path.substr(0, dot)));   // reset rebinds; '=' would copy into node
        path = path.substr(dot + 1);
    }
    at(node, path);   // the key must already exist: a typo fails here
    node[path] = YAML::Load(kv.substr(eq + 1));
}

}  // namespace

Config load_config(const std::string& path, const std::vector<std::string>& overrides) {
    YAML::Node raw = YAML::LoadFile(path);
    for (const std::string& kv : overrides) apply_override(raw, kv);
    Config cfg{};
    Params& p = cfg.p;

    const std::string model = at(raw, "model").as<std::string>();
    if (model == "kinematic") p.model = ModelKind::Kinematic;
    else if (model == "dynamic") p.model = ModelKind::Dynamic;
    else throw bad("model='" + model + "', expected kinematic | dynamic");
    p.state_dim = at(raw, "state_dim").as<int>();
    if (p.state_dim != (p.model == ModelKind::Dynamic ? 6 : 4))
        throw bad("state_dim does not match model '" + model + "'");
    if (at(raw, "control_dim").as<int>() != CONTROL_DIM) throw bad("control_dim must be 2");

    const YAML::Node m = at(raw, "mppi");
    p.K = at(m, "num_samples").as<int>();
    p.T = at(m, "horizon").as<int>();
    p.dt = f(m, "dt");
    p.lambda = f(m, "lambda");
    p.gamma = f(m, "gamma");
    const YAML::Node std_ = at(m, "noise_std");
    if (!std_.IsSequence() || std_.size() != CONTROL_DIM) throw bad("noise_std needs 2 entries");
    for (int j = 0; j < CONTROL_DIM; ++j) p.noise_std[j] = std_[j].as<float>();
    cfg.seed = at(m, "seed").as<unsigned long long>();

    const YAML::Node b = at(raw, "control_bounds");
    p.u_min[0] = f(b, "a_min");     p.u_max[0] = f(b, "a_max");
    p.u_min[1] = f(b, "delta_min"); p.u_max[1] = f(b, "delta_max");
    for (int j = 0; j < CONTROL_DIM; ++j)
        if (!(p.u_min[j] < p.u_max[j])) throw bad("control_bounds: min must be < max");

    const YAML::Node v = at(raw, "vehicle");
    VehicleParams& veh = p.vehicle;
    veh.wheelbase = f(v, "wheelbase"); veh.lf = f(v, "lf"); veh.lr = f(v, "lr");
    veh.width = f(v, "width"); veh.mu = f(v, "mu"); veh.mass = f(v, "mass"); veh.izz = f(v, "izz");
    veh.cs_front = f(v, "cornering_stiffness_front");
    veh.cs_rear = f(v, "cornering_stiffness_rear");
    const std::string tire = at(v, "tire_model").as<std::string>();
    if (tire == "linear") veh.tire_model = TireModel::Linear;
    else if (tire == "tanh") veh.tire_model = TireModel::Tanh;
    else if (tire == "pacejka") veh.tire_model = TireModel::Pacejka;
    else throw bad("tire_model='" + tire + "', expected linear | tanh | pacejka");
    veh.pacejka_c = f(v, "pacejka_c"); veh.pacejka_e = f(v, "pacejka_e");
    veh.blend_speed_low = f(v, "blend_speed_low");
    veh.blend_speed_high = f(v, "blend_speed_high");
    if (!(0.0f < veh.blend_speed_low && veh.blend_speed_low < veh.blend_speed_high))
        throw bad("need 0 < blend_speed_low < blend_speed_high");
    const std::string integ = at(v, "integrator").as<std::string>();
    if (integ == "euler") veh.integrator = Integrator::Euler;
    else if (integ == "rk4") veh.integrator = Integrator::Rk4;
    else throw bad("integrator='" + integ + "', expected euler | rk4");
    veh.substeps = at(v, "substeps").as<int>();
    if (veh.substeps < 1) throw bad("substeps must be >= 1");
    if (std::fabs(veh.lf + veh.lr - veh.wheelbase) > 1e-6f) throw bad("lf + lr != wheelbase");

    const YAML::Node c = at(raw, "cost");
    CostParams& cost = p.cost;
    cost.w_lateral = f(c, "w_lateral"); cost.w_progress = f(c, "w_progress");
    cost.w_offtrack = f(c, "w_offtrack"); cost.w_control = f(c, "w_control");
    cost.w_adhesion = f(c, "w_adhesion"); cost.w_speed = f(c, "w_speed");
    cost.lateral_scale = f(c, "lateral_scale"); cost.speed_scale = f(c, "speed_scale");
    cost.v_ref = f(c, "v_ref"); cost.track_half_width = f(c, "track_half_width");

    namespace fs = std::filesystem;
    const fs::path root = fs::absolute(path).parent_path().parent_path();   // config/ -> repo
    cfg.track_bin_path = (root / at(at(raw, "track"), "bin_path").as<std::string>()).string();
    cfg.max_steps = at(at(raw, "sim"), "max_steps").as<int>();
    return cfg;
}

}  // namespace mppi
```

### `cuda/include/mppi/npy.hpp`

```cpp
// Minimal .npy reader/writer: little-endian float32 or float64, C order.
//
// Enough to exchange arrays with NumPy (np.save / np.load) without a library.
// Format: magic "\x93NUMPY", version 1.0, a 2-byte header length, then a
// Python dict literal such as {'descr': '<f4', 'fortran_order': False,
// 'shape': (1024, 30, 2), }, padded with spaces to a multiple of 64 bytes.
#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace mppi::npy {

template <class T>
const char* descr() {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "float or double only");
    return std::is_same_v<T, float> ? "<f4" : "<f8";
}

template <class T>
struct Array {
    std::vector<std::size_t> shape;
    std::vector<T> data;
};

template <class T>
Array<T> load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("npy: cannot open " + path);
    char magic[8];
    in.read(magic, 8);
    if (std::memcmp(magic, "\x93NUMPY", 6) != 0 || magic[6] != 1)
        throw std::runtime_error("npy: " + path + " is not a version 1.0 .npy file");
    std::uint16_t len = 0;
    in.read(reinterpret_cast<char*>(&len), 2);
    std::string header(len, ' ');
    in.read(header.data(), len);

    if (header.find(std::string("'descr': '") + descr<T>() + "'") == std::string::npos)
        throw std::runtime_error("npy: " + path + " has dtype " + header.substr(0, 40) +
                                 ", expected " + descr<T>());
    if (header.find("'fortran_order': False") == std::string::npos)
        throw std::runtime_error("npy: " + path + " is Fortran ordered");

    Array<T> arr;
    const auto open = header.find('(', header.find("'shape'"));
    const auto close = header.find(')', open);
    std::string dims = header.substr(open + 1, close - open - 1);   // "1024, 30, 2" or "6,"
    std::size_t count = 1, pos = 0;
    while (pos < dims.size()) {
        const auto comma = dims.find(',', pos);
        const std::string tok = dims.substr(pos, comma - pos);
        if (tok.find_first_not_of(' ') != std::string::npos) {
            arr.shape.push_back(std::stoul(tok));
            count *= arr.shape.back();
        }
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    arr.data.resize(count);
    in.read(reinterpret_cast<char*>(arr.data.data()), count * sizeof(T));
    if (!in) throw std::runtime_error("npy: " + path + " is truncated");
    return arr;
}

template <class T>
void save(const std::string& path, const T* data, const std::vector<std::size_t>& shape) {
    std::string dims;
    std::size_t count = 1;
    for (std::size_t d : shape) { dims += std::to_string(d) + ", "; count *= d; }
    if (shape.size() > 1) dims.resize(dims.size() - 1);   // "(3, 4)" but "(6,)"
    else if (shape.empty()) dims.clear();
    std::string header = std::string("{'descr': '") + descr<T>() +
                         "', 'fortran_order': False, 'shape': (" + dims + "), }";
    header.append(64 - (10 + header.size() + 1) % 64, ' ');
    header += '\n';
    const std::uint16_t len = static_cast<std::uint16_t>(header.size());

    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("npy: cannot write " + path);
    out.write("\x93NUMPY\x01\x00", 8);
    out.write(reinterpret_cast<const char*>(&len), 2);
    out.write(header.data(), header.size());
    out.write(reinterpret_cast<const char*>(data), count * sizeof(T));
}

template <class T>
void save(const std::string& path, const std::vector<T>& data, const std::vector<std::size_t>& shape) {
    save(path, data.data(), shape);
}

}  // namespace mppi::npy
```

### `cuda/include/mppi/track.cuh`

```cpp
// Cost map: load the binary written by python/mppi/track.py, look up a cell.
//
// TrackHost owns the grid in CPU memory. TrackView is what a rollout reads:
// a raw pointer plus the grid geometry, trivially copyable, passed by value to
// the kernel. The pointer is a host pointer on the CPU path and a device
// pointer on the GPU path; lookup() does not care.
#pragma once

#include <string>
#include <vector>

#include "mppi/hd.hpp"

namespace mppi {

struct TrackView {
    const float* grid;   // (ny, nx, 2) row-major: grid[(iy * nx + ix) * 2 + c], c = 0 d, 1 s
    int nx, ny;
    float x_min, y_min, res, length;
};

struct TrackHost {
    std::vector<float> grid;
    int nx = 0, ny = 0;
    float x_min = 0, y_min = 0, res = 0, length = 0;

    TrackView view() const { return {grid.data(), nx, ny, x_min, y_min, res, length}; }
};

TrackHost load_track(const std::string& bin_path);

// Nearest cell, indices clamped to the grid. Same as track.lookup in Python.
HD inline void lookup(const TrackView& tr, float x, float y, float& d, float& s) {
    int ix = static_cast<int>(floorf((x - tr.x_min) / tr.res));
    int iy = static_cast<int>(floorf((y - tr.y_min) / tr.res));
    ix = min(max(ix, 0), tr.nx - 1);
    iy = min(max(iy, 0), tr.ny - 1);
    const float* cell = tr.grid + (static_cast<long>(iy) * tr.nx + ix) * 2;
    d = cell[0];
    s = cell[1];
}

// Progress difference brought back to [-L/2, L/2). Python's % is a floored
// modulo (result has the sign of the divisor); fmodf truncates (sign of the
// dividend), hence the correction.
HD inline float wrap(float ds, float length) {
    float r = fmodf(ds + 0.5f * length, length);
    if (r < 0.0f) r += length;
    return r - 0.5f * length;
}

}  // namespace mppi
```

### `cuda/src/track.cpp`

```cpp
#include "mppi/track.cuh"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace mppi {

// Header written by track.save: struct "<4sIiiffff", 32 bytes, little-endian.
TrackHost load_track(const std::string& bin_path) {
    std::ifstream in(bin_path, std::ios::binary);
    if (!in) throw std::runtime_error("track: cannot open " + bin_path + " (run `pixi run track`)");

    char magic[4];
    std::uint32_t version = 0;
    TrackHost tr;
    in.read(magic, 4);
    in.read(reinterpret_cast<char*>(&version), 4);
    in.read(reinterpret_cast<char*>(&tr.nx), 4);
    in.read(reinterpret_cast<char*>(&tr.ny), 4);
    in.read(reinterpret_cast<char*>(&tr.x_min), 4);
    in.read(reinterpret_cast<char*>(&tr.y_min), 4);
    in.read(reinterpret_cast<char*>(&tr.res), 4);
    in.read(reinterpret_cast<char*>(&tr.length), 4);
    if (std::memcmp(magic, "MPPG", 4) != 0 || version != 1)
        throw std::runtime_error("track: bad header in " + bin_path);

    tr.grid.resize(static_cast<std::size_t>(tr.nx) * tr.ny * 2);
    in.read(reinterpret_cast<char*>(tr.grid.data()), tr.grid.size() * sizeof(float));
    if (!in) throw std::runtime_error("track: " + bin_path + " is truncated");
    return tr;
}

}  // namespace mppi
```

### `cuda/include/mppi/dynamics.cuh`

```cpp
// step(state, control, dt) -> state, compiled for the CPU and the GPU (HD).
// Mirrors python/mppi/dynamics.py line by line, same frame conventions:
//   kinematic state [x, y, psi, v], dynamic state [x, y, psi, vx, vy, r],
//   control [a, delta], SI units, radians, psi from the x axis, CCW positive.
//
// FP32 everywhere: float literals (0.5f) and float functions (sinf, atanf).
// A bare 0.5 or sin() would silently promote to double, which is slow on a
// GeForce GPU and changes the numbers.
//
// The model is a type (Kinematic or Dynamic) chosen at compile time by a
// template parameter: a static choice, never a branch on the state.
#pragma once

#include "mppi/config.hpp"
#include "mppi/hd.hpp"

namespace mppi {

constexpr float G = 9.81f;

// ---------------------------------------------------------------------------
// Kinematic bicycle at the CG, state [x, y, psi, v] (step 1)
// ---------------------------------------------------------------------------
struct Kinematic {
    static constexpr int DIM = 4;

    HD static void step(const float* s, float a, float delta, float dt,
                        const VehicleParams& v, float* out) {
        const float tan_d = tanf(delta);
        const float beta = atanf(v.lr / v.wheelbase * tan_d);
        out[0] = s[0] + s[3] * cosf(s[2] + beta) * dt;
        out[1] = s[1] + s[3] * sinf(s[2] + beta) * dt;
        out[2] = s[2] + s[3] * cosf(beta) / v.wheelbase * tan_d * dt;
        out[3] = fmaxf(s[3] + a * dt, 0.0f);   // never reverses
    }
};

// ---------------------------------------------------------------------------
// Dynamic single-track model blended with the kinematic one (step 2)
// ---------------------------------------------------------------------------
HD inline float tire_force(float alpha, float fz, float c_s, const VehicleParams& v) {
    // Static choice read from the YAML: the same for every thread, no divergence.
    switch (v.tire_model) {
        case TireModel::Linear:
            return v.mu * c_s * fz * alpha;
        case TireModel::Tanh:
            return v.mu * fz * tanhf(c_s * alpha);
        default: {   // Pacejka, B = c_s / C so that the slope at 0 is mu * c_s * fz
            const float b_alpha = c_s / v.pacejka_c * alpha;
            return v.mu * fz * sinf(v.pacejka_c *
                                    atanf(b_alpha - v.pacejka_e * (b_alpha - atanf(b_alpha))));
        }
    }
}

struct Dynamic {
    static constexpr int DIM = 6;

    // d(state)/dt, theory 6.3
    HD static void derivative(const float* s, float a, float delta,
                              const VehicleParams& v, float* ds) {
        const float psi = s[2], vx = s[3], vy = s[4], r = s[5];
        const float fzf = v.mass * G * v.lr / v.wheelbase;
        const float fzr = v.mass * G * v.lf / v.wheelbase;

        const float vx_safe = fmaxf(vx, v.blend_speed_low);   // 0 * NaN guard (step 2, part D)
        const float alpha_f = delta - atanf((vy + v.lf * r) / vx_safe);
        const float alpha_r = -atanf((vy - v.lr * r) / vx_safe);
        const float fyf = tire_force(alpha_f, fzf, v.cs_front, v);
        const float fyr = tire_force(alpha_r, fzr, v.cs_rear, v);
        const float cos_d = cosf(delta), sin_d = sinf(delta);
        const float cos_p = cosf(psi), sin_p = sinf(psi);

        ds[0] = vx * cos_p - vy * sin_p;
        ds[1] = vx * sin_p + vy * cos_p;
        ds[2] = r;
        ds[3] = a - fyf * sin_d / v.mass + vy * r;
        ds[4] = (fyf * cos_d + fyr) / v.mass - vx * r;
        ds[5] = (v.lf * fyf * cos_d - v.lr * fyr) / v.izz;
    }

    // `substeps` Euler or RK4 sub-steps of the pure dynamic model
    HD static void step_dynamic_only(const float* s0, float a, float delta, float dt,
                                     const VehicleParams& v, float* out) {
        const float h = dt / v.substeps;
        float s[DIM], k1[DIM], k2[DIM], k3[DIM], k4[DIM], tmp[DIM];
        for (int i = 0; i < DIM; ++i) s[i] = s0[i];
        for (int n = 0; n < v.substeps; ++n) {
            if (v.integrator == Integrator::Euler) {
                derivative(s, a, delta, v, k1);
                for (int i = 0; i < DIM; ++i) s[i] += h * k1[i];
            } else {
                derivative(s, a, delta, v, k1);
                for (int i = 0; i < DIM; ++i) tmp[i] = s[i] + 0.5f * h * k1[i];
                derivative(tmp, a, delta, v, k2);
                for (int i = 0; i < DIM; ++i) tmp[i] = s[i] + 0.5f * h * k2[i];
                derivative(tmp, a, delta, v, k3);
                for (int i = 0; i < DIM; ++i) tmp[i] = s[i] + h * k3[i];
                derivative(tmp, a, delta, v, k4);
                for (int i = 0; i < DIM; ++i)
                    s[i] += h / 6.0f * (k1[i] + 2.0f * k2[i] + 2.0f * k3[i] + k4[i]);
            }
        }
        for (int i = 0; i < DIM; ++i) out[i] = s[i];
    }

    // Kinematic model in the 6-state, (vy, r) put back on the manifold
    HD static void step_kinematic6(const float* s, float a, float delta, float dt,
                                   const VehicleParams& v, float* out) {
        const float tan_d = tanf(delta);
        const float k_vy = v.lr / v.wheelbase * tan_d;
        const float k_r = tan_d / v.wheelbase;
        const float vx = s[3], vy = k_vy * vx;
        const float cos_p = cosf(s[2]), sin_p = sinf(s[2]);
        out[0] = s[0] + (vx * cos_p - vy * sin_p) * dt;
        out[1] = s[1] + (vx * sin_p + vy * cos_p) * dt;
        out[2] = s[2] + k_r * vx * dt;
        const float vx_next = fmaxf(vx + a * dt, 0.0f);
        out[3] = vx_next;
        out[4] = k_vy * vx_next;
        out[5] = k_r * vx_next;
    }

    HD static void step(const float* s, float a, float delta, float dt,
                        const VehicleParams& v, float* out) {
        const float lo = v.blend_speed_low, hi = v.blend_speed_high;
        const float kappa = fminf(fmaxf((s[3] - lo) / (hi - lo), 0.0f), 1.0f);
        float dyn[DIM], kin[DIM];
        step_dynamic_only(s, a, delta, dt, v, dyn);   // both always evaluated
        step_kinematic6(s, a, delta, dt, v, kin);
        for (int i = 0; i < DIM; ++i) out[i] = kappa * dyn[i] + (1.0f - kappa) * kin[i];
        out[3] = fmaxf(out[3], 0.0f);                 // never reverses
    }
};

}  // namespace mppi
```

### `cuda/include/mppi/cost.cuh`

```cpp
// Cost of one rollout, compiled for the CPU and the GPU (HD).
// Mirrors python/mppi/cost.py and controller.rollout_costs.
#pragma once

#include "mppi/config.hpp"
#include "mppi/dynamics.cuh"
#include "mppi/hd.hpp"
#include "mppi/track.cuh"

namespace mppi {

constexpr float S_MAX = 1e6f;   // cost of a rollout that produced a NaN or an inf

// Running cost of x_t and of the control u_{t-1} that led to it
template <class Model>
HD float stage_cost(const float* x, const float* u, const TrackView& tr, const Params& p) {
    const CostParams& c = p.cost;
    float d, s;
    lookup(tr, x[0], x[1], d, s);
    const float v = x[3];   // v (kinematic) or vx (dynamic): same index

    const float d0 = c.lateral_scale;
    const float lateral = c.w_lateral * (d / d0) * (d / d0);

    const float d_max = c.track_half_width - 0.5f * p.vehicle.width;
    const float excess = fabsf(d) - d_max;
    const float offtrack = excess > 0.0f ? c.w_offtrack * (1.0f + excess / d0) : 0.0f;

    const float dv = (v - c.v_ref) / c.speed_scale;
    const float speed = c.w_speed * dv * dv;

    float effort = 0.0f;
    for (int j = 0; j < CONTROL_DIM; ++j) {
        const float uj = u[j] / p.u_max[j];
        effort += uj * uj;
    }
    float cost = lateral + offtrack + speed + c.w_control * effort;

    if constexpr (Model::DIM == Kinematic::DIM) {   // resolved at compile time
        const float a_lat = v * v * fabsf(tanf(u[1])) / p.vehicle.wheelbase;
        const float over = fmaxf(0.0f, a_lat / (p.vehicle.mu * G) - 1.0f);
        cost += c.w_adhesion * over * over;
    }
    return cost;
}

// Progress reward on x_T, normalized by the distance covered at v_ref
HD inline float terminal_cost(const float* xT, float s0, const TrackView& tr, const Params& p) {
    float d, sT;
    lookup(tr, xT[0], xT[1], d, sT);
    const float progress = wrap(sT - s0, tr.length);
    return -p.cost.w_progress * progress / (p.cost.v_ref * p.T * p.dt);
}

// Total cost S^k of rollout k. Layout (row-major, as NumPy):
//   x0 (DIM,), U (T, 2), eps (K, T, 2) raw noise, before clipping.
// The CPU path calls it in a for loop, the GPU kernel with k = thread index.
template <class Model>
HD float rollout_cost(int k, const float* x0, const float* U, const float* eps,
                      const TrackView& tr, const Params& p) {
    float x[Model::DIM], x_next[Model::DIM];
    for (int i = 0; i < Model::DIM; ++i) x[i] = x0[i];

    float d, s0;
    lookup(tr, x0[0], x0[1], d, s0);

    float S = 0.0f, correction = 0.0f;
    for (int t = 0; t < p.T; ++t) {
        float vt[CONTROL_DIM];
        for (int j = 0; j < CONTROL_DIM; ++j) {
            const float u = U[t * CONTROL_DIM + j];
            const float e = eps[(static_cast<long>(k) * p.T + t) * CONTROL_DIM + j];
            vt[j] = fminf(fmaxf(u + e, p.u_min[j]), p.u_max[j]);   // clip(U + eps)
            const float sigma = p.noise_std[j];
            correction += u / (sigma * sigma) * (vt[j] - u);      // U Sigma^-1 eps_effective
        }
        Model::step(x, vt[0], vt[1], p.dt, p.vehicle, x_next);
        for (int i = 0; i < Model::DIM; ++i) x[i] = x_next[i];
        S += stage_cost<Model>(x, vt, tr, p);
    }
    S += terminal_cost(x, s0, tr, p);
    S += p.gamma * correction;
    return isfinite(S) ? S : S_MAX;
}

}  // namespace mppi
```

### `cuda/include/mppi/rollouts.hpp`

```cpp
// The K rollout costs, on the CPU or on the GPU. Same function underneath
// (rollout_cost in cost.cuh), compiled twice.
//
// Plain C++ declarations: callable from a .cpp file compiled by g++.
// Layout of every array: row-major, as NumPy.
//   x0 (state_dim,), U (T, 2), eps (K, T, 2) raw noise, S (K,)
#pragma once

#include "mppi/config.hpp"
#include "mppi/track.cuh"

namespace mppi {

// All pointers are host pointers.
void rollouts_cpu(const float* x0, const float* U, const float* eps,
                  const TrackView& track, const Params& p, float* S);

// All pointers are DEVICE pointers (track.grid included). Asynchronous: returns
// as soon as the kernel is queued. The next cudaMemcpy waits for it.
void rollouts_gpu(const float* d_x0, const float* d_U, const float* d_eps,
                  const TrackView& d_track, const Params& p, float* d_S);

}  // namespace mppi
```

### `cuda/src/mppi_kernel.cu`

```cpp
// Step 3: one thread per trajectory, horizon loop inside the kernel, cost
// accumulated on the fly, one float written per thread. Correctness first.
// Step 4 fuses rollout and cost and keeps the state in registers.
#include "mppi/cost.cuh"
#include "mppi/cuda_check.hpp"
#include "mppi/rollouts.hpp"

namespace mppi {
namespace {

constexpr int BLOCK = 256;   // threads per block, a multiple of the warp size (32)

template <class Model>
__global__ void rollout_kernel(const float* x0, const float* U, const float* eps,
                               TrackView track, Params p, float* S) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;   // global thread index = rollout
    if (k >= p.K) return;                                   // last block may overhang
    S[k] = rollout_cost<Model>(k, x0, U, eps, track, p);
}

template <class Model>
void cpu_loop(const float* x0, const float* U, const float* eps,
              const TrackView& track, const Params& p, float* S) {
    for (int k = 0; k < p.K; ++k) S[k] = rollout_cost<Model>(k, x0, U, eps, track, p);
}

}  // namespace

void rollouts_cpu(const float* x0, const float* U, const float* eps,
                  const TrackView& track, const Params& p, float* S) {
    if (p.model == ModelKind::Dynamic) cpu_loop<Dynamic>(x0, U, eps, track, p, S);
    else cpu_loop<Kinematic>(x0, U, eps, track, p, S);
}

void rollouts_gpu(const float* d_x0, const float* d_U, const float* d_eps,
                  const TrackView& d_track, const Params& p, float* d_S) {
    const int grid = (p.K + BLOCK - 1) / BLOCK;   // ceil(K / BLOCK)
    if (p.model == ModelKind::Dynamic)
        rollout_kernel<Dynamic><<<grid, BLOCK>>>(d_x0, d_U, d_eps, d_track, p, d_S);
    else
        rollout_kernel<Kinematic><<<grid, BLOCK>>>(d_x0, d_U, d_eps, d_track, p, d_S);
    CUDA_CHECK(cudaGetLastError());   // launch errors (bad grid, no kernel image for this GPU)
}

}  // namespace mppi
```

### `cuda/include/mppi/controller.hpp`

```cpp
// Host-side MPPI controller: noise, rollouts (CPU or GPU), weights, update.
//
// Same algorithm as python/mppi/controller.py. Step 3 keeps everything but the
// rollouts on the CPU: noise drawn on the host and copied to the GPU each
// iteration, weights and weighted mean on the host (step 4 moves them).
#pragma once

#include <array>
#include <random>
#include <vector>

#include "mppi/config.hpp"
#include "mppi/cuda_check.hpp"
#include "mppi/track.cuh"

namespace mppi {

enum class Backend { Cpu, Gpu };

struct Diagnostics {
    float ess = 0, rho = 0;
    float t_noise_ms = 0, t_rollouts_ms = 0, t_update_ms = 0;    // host clock, every backend
    float t_upload_ms = 0, t_kernel_ms = 0, t_download_ms = 0;   // GPU only, CUDA events
};

class Controller {
public:
    Controller(const Config& cfg, const TrackHost& track, Backend backend);
    ~Controller();
    Controller(const Controller&) = delete;
    Controller& operator=(const Controller&) = delete;

    // One MPPI iteration from x0 (state_dim floats). Returns u0 = [a, delta].
    std::array<float, CONTROL_DIM> command(const float* x0, Diagnostics* diag = nullptr);

    // Deterministic part, as rollout_costs in Python: K costs for given x0, U, eps.
    // Host pointers in and out, whatever the backend. Used by the parity check.
    void rollout_costs(const float* x0, const float* U, const float* eps, float* S,
                       Diagnostics* diag = nullptr);

    const std::vector<float>& nominal() const { return U_; }

private:
    Params p_;
    Backend backend_;
    TrackView track_host_;   // points into the caller's TrackHost, which must outlive us
    TrackView track_dev_;    // same geometry, grid pointer in GPU memory

    std::vector<float> U_, eps_, S_;
    std::vector<double> w_;
    std::mt19937_64 rng_;
    std::normal_distribution<float> normal_{0.0f, 1.0f};

    // GPU memory, allocated once in the constructor, reused at every iteration
    DeviceBuffer<float> d_grid_, d_x0_, d_U_, d_eps_, d_S_;
    cudaEvent_t ev_[4] = {};
};

}  // namespace mppi
```

### `cuda/src/controller.cpp`

```cpp
// C++ control loop driving the kernel. Latency measured with CUDA events on
// the device side only, never a wall clock that includes the network.
#include "mppi/controller.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

#include "mppi/rollouts.hpp"

namespace mppi {
namespace {

using Clock = std::chrono::steady_clock;
float ms_since(Clock::time_point t0) {
    return std::chrono::duration<float, std::milli>(Clock::now() - t0).count();
}

}  // namespace

Controller::Controller(const Config& cfg, const TrackHost& track, Backend backend)
    : p_(cfg.p), backend_(backend), track_host_(track.view()), track_dev_(track.view()),
      U_(static_cast<std::size_t>(p_.T) * CONTROL_DIM, 0.0f),
      eps_(static_cast<std::size_t>(p_.K) * p_.T * CONTROL_DIM),
      S_(p_.K), w_(p_.K), rng_(cfg.seed) {
    if (backend_ != Backend::Gpu) return;   // the CPU path never touches CUDA

    d_grid_ = DeviceBuffer<float>(track.grid.size());
    d_grid_.upload(track.grid.data());       // the map never changes: copied once
    track_dev_.grid = d_grid_.get();
    d_x0_ = DeviceBuffer<float>(p_.state_dim);
    d_U_ = DeviceBuffer<float>(U_.size());
    d_eps_ = DeviceBuffer<float>(eps_.size());
    d_S_ = DeviceBuffer<float>(S_.size());
    for (cudaEvent_t& e : ev_) CUDA_CHECK(cudaEventCreate(&e));
}

Controller::~Controller() {
    for (cudaEvent_t e : ev_)
        if (e) cudaEventDestroy(e);
}

void Controller::rollout_costs(const float* x0, const float* U, const float* eps, float* S,
                               Diagnostics* diag) {
    if (backend_ == Backend::Cpu) {
        rollouts_cpu(x0, U, eps, track_host_, p_, S);
        return;
    }
    CUDA_CHECK(cudaEventRecord(ev_[0]));
    d_x0_.upload(x0);
    d_U_.upload(U);
    d_eps_.upload(eps);
    CUDA_CHECK(cudaEventRecord(ev_[1]));
    rollouts_gpu(d_x0_.get(), d_U_.get(), d_eps_.get(), track_dev_, p_, d_S_.get());
    CUDA_CHECK(cudaEventRecord(ev_[2]));
    d_S_.download(S);                          // waits for the kernel to finish
    CUDA_CHECK(cudaEventRecord(ev_[3]));
    CUDA_CHECK(cudaEventSynchronize(ev_[3]));
    if (diag) {
        CUDA_CHECK(cudaEventElapsedTime(&diag->t_upload_ms, ev_[0], ev_[1]));
        CUDA_CHECK(cudaEventElapsedTime(&diag->t_kernel_ms, ev_[1], ev_[2]));
        CUDA_CHECK(cudaEventElapsedTime(&diag->t_download_ms, ev_[2], ev_[3]));
    }
}

std::array<float, CONTROL_DIM> Controller::command(const float* x0, Diagnostics* diag) {
    const int K = p_.K, T = p_.T;
    Diagnostics local;
    if (!diag) diag = &local;

    // 1. noise (K, T, 2), one sigma per control
    auto t0 = Clock::now();
    for (int k = 0; k < K; ++k)
        for (int t = 0; t < T; ++t)
            for (int j = 0; j < CONTROL_DIM; ++j)
                eps_[(static_cast<std::size_t>(k) * T + t) * CONTROL_DIM + j] =
                    normal_(rng_) * p_.noise_std[j];

    diag->t_noise_ms = ms_since(t0);

    // 2-4. rollouts and costs (clipping done inside, eps_ stays the raw noise)
    t0 = Clock::now();
    rollout_costs(x0, U_.data(), eps_.data(), S_.data(), diag);
    diag->t_rollouts_ms = ms_since(t0);

    // 5. softmin weights, in double on the host
    t0 = Clock::now();
    const double rho = *std::min_element(S_.begin(), S_.end());
    double sum = 0.0;
    for (int k = 0; k < K; ++k) sum += w_[k] = std::exp(-(S_[k] - rho) / p_.lambda);
    double sum_sq = 0.0;
    for (int k = 0; k < K; ++k) { w_[k] /= sum; sum_sq += w_[k] * w_[k]; }

    // 6. U += sum_k w_k * eps_effective_k, eps_effective = clip(U + eps) - U
    std::vector<double> dU(U_.size(), 0.0);
    for (int k = 0; k < K; ++k)
        for (int t = 0; t < T; ++t)
            for (int j = 0; j < CONTROL_DIM; ++j) {
                const std::size_t i = static_cast<std::size_t>(t) * CONTROL_DIM + j;
                const float v = std::clamp(U_[i] + eps_[k * U_.size() + i], p_.u_min[j], p_.u_max[j]);
                dU[i] += w_[k] * (v - U_[i]);
            }
    for (std::size_t i = 0; i < U_.size(); ++i) U_[i] += static_cast<float>(dU[i]);

    // 7-8. apply U[0], shift by one step, repeat the last command
    const std::array<float, CONTROL_DIM> u0 = {U_[0], U_[1]};
    std::copy(U_.begin() + CONTROL_DIM, U_.end(), U_.begin());
    std::copy(U_.end() - 2 * CONTROL_DIM, U_.end() - CONTROL_DIM, U_.end() - CONTROL_DIM);

    diag->t_update_ms = ms_since(t0);
    diag->ess = static_cast<float>(1.0 / sum_sq);
    diag->rho = static_cast<float>(rho);
    return u0;
}

}  // namespace mppi
```

### `cuda/apps/parity.cpp`

```cpp
// Parity check, C++ side: K rollout costs on a noise tensor written by Python.
//
//   mppi_parity --config config/mppi.yaml --dir results/parity/<case> \
//               [--backend cpu|gpu|both] [--set vehicle.tire_model=tanh ...]
//
// Reads <dir>/x0.npy (state_dim,), U.npy (T, 2), eps.npy (K, T, 2), float32.
// Writes <dir>/S_cpu.npy and/or S_gpu.npy (K,), float32.
// bench/check_parity.py drives it and compares with NumPy.
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "mppi/config.hpp"
#include "mppi/controller.hpp"
#include "mppi/npy.hpp"
#include "mppi/track.cuh"

using namespace mppi;

int main(int argc, char** argv) try {
    std::string config = "config/mppi.yaml", dir, backend = "both";
    std::vector<std::string> overrides;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string key = argv[i], value = argv[i + 1];
        if (key == "--config") config = value;
        else if (key == "--dir") dir = value;
        else if (key == "--backend") backend = value;
        else if (key == "--set") overrides.push_back(value);
        else throw std::runtime_error("unknown argument " + key);
    }
    if (dir.empty()) throw std::runtime_error("--dir is required");

    const Config cfg = load_config(config, overrides);
    const TrackHost track = load_track(cfg.track_bin_path);
    const auto x0 = npy::load<float>(dir + "/x0.npy");
    const auto U = npy::load<float>(dir + "/U.npy");
    const auto eps = npy::load<float>(dir + "/eps.npy");

    const Params& p = cfg.p;
    if (x0.data.size() != static_cast<std::size_t>(p.state_dim) ||
        U.data.size() != static_cast<std::size_t>(p.T) * CONTROL_DIM ||
        eps.data.size() != static_cast<std::size_t>(p.K) * p.T * CONTROL_DIM)
        throw std::runtime_error("input shapes do not match K, T, state_dim of the config");

    std::vector<float> S(p.K);
    for (const char* name : {"cpu", "gpu"}) {
        if (backend != "both" && backend != name) continue;
        Controller ctrl(cfg, track, std::string(name) == "gpu" ? Backend::Gpu : Backend::Cpu);
        Diagnostics diag;
        ctrl.rollout_costs(x0.data.data(), U.data.data(), eps.data.data(), S.data(), &diag);
        npy::save(dir + "/S_" + name + ".npy", S, {S.size()});
        if (std::string(name) == "gpu")
            std::printf("gpu: upload %.3f ms, kernel %.3f ms, download %.3f ms\n",
                        diag.t_upload_ms, diag.t_kernel_ms, diag.t_download_ms);
    }
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "mppi_parity: %s\n", e.what());
    return 1;
}
```

### `cuda/apps/sim.cpp`

```cpp
// Closed loop in C++: MPPI controller (CPU or GPU rollouts) driving the dynamic
// plant, until one lap or max_steps. Same loop as simulate() in run_sim.py.
//
//   mppi_sim --config config/mppi.yaml --out results/trajectories/step3_gpu \
//            --x0 <dir>/x0.npy [--backend cpu|gpu] [--set key=value ...]
//
// x0.npy is the plant's start state (6,), written by python/run_sim_cuda.py
// (the C++ side does not know the centerline). Writes to --out: states.npy,
// controls.npy, next_states.npy (N, 6|2), ess.npy, rho.npy, t_iter_ms.npy,
// t_kernel_ms.npy (N,).
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "mppi/config.hpp"
#include "mppi/controller.hpp"
#include "mppi/dynamics.cuh"
#include "mppi/npy.hpp"
#include "mppi/track.cuh"

using namespace mppi;

int main(int argc, char** argv) try {
    std::string config = "config/mppi.yaml", out, x0_path, backend = "gpu";
    std::vector<std::string> overrides;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string key = argv[i], value = argv[i + 1];
        if (key == "--config") config = value;
        else if (key == "--out") out = value;
        else if (key == "--x0") x0_path = value;
        else if (key == "--backend") backend = value;
        else if (key == "--set") overrides.push_back(value);
        else throw std::runtime_error("unknown argument " + key);
    }
    if (out.empty() || x0_path.empty()) throw std::runtime_error("--out and --x0 are required");

    const Config cfg = load_config(config, overrides);
    const TrackHost track = load_track(cfg.track_bin_path);
    const TrackView tv = track.view();
    const Params& p = cfg.p;
    Controller ctrl(cfg, track, backend == "gpu" ? Backend::Gpu : Backend::Cpu);

    float x[6];
    const auto x0 = npy::load<float>(x0_path);
    if (x0.data.size() != 6) throw std::runtime_error("x0 must have 6 components");
    for (int i = 0; i < 6; ++i) x[i] = x0.data[i];

    std::vector<float> states, controls, nexts, ess, rho, t_iter, t_kernel;
    float d, s_prev, s, progress = 0.0f;
    lookup(tv, x[0], x[1], d, s_prev);

    int n = 0;
    for (; n < cfg.max_steps; ++n) {
        // What the controller sees: the full state, or [x, y, psi, |v|] for a kinematic one
        float obs[6] = {x[0], x[1], x[2], x[3], x[4], x[5]};
        if (p.model == ModelKind::Kinematic) obs[3] = std::hypot(x[3], x[4]);

        Diagnostics diag;
        const auto tic = std::chrono::steady_clock::now();
        const auto u = ctrl.command(obs, &diag);
        const auto toc = std::chrono::steady_clock::now();

        float x_next[6];
        Dynamic::step(x, u[0], u[1], p.dt, p.vehicle, x_next);   // the plant is always dynamic

        lookup(tv, x_next[0], x_next[1], d, s);
        progress += wrap(s - s_prev, tv.length);
        s_prev = s;

        states.insert(states.end(), x, x + 6);
        controls.insert(controls.end(), u.begin(), u.end());
        nexts.insert(nexts.end(), x_next, x_next + 6);
        ess.push_back(diag.ess);
        rho.push_back(diag.rho);
        t_iter.push_back(std::chrono::duration<float, std::milli>(toc - tic).count());
        t_kernel.push_back(diag.t_kernel_ms);

        bool finite = true;
        for (int i = 0; i < 6; ++i) { x[i] = x_next[i]; finite = finite && std::isfinite(x[i]); }
        if (progress >= tv.length || !finite) { ++n; break; }
    }

    std::filesystem::create_directories(out);
    const std::size_t N = static_cast<std::size_t>(n);
    npy::save(out + "/states.npy", states, {N, 6});
    npy::save(out + "/controls.npy", controls, {N, 2});
    npy::save(out + "/next_states.npy", nexts, {N, 6});
    npy::save(out + "/ess.npy", ess, {N});
    npy::save(out + "/rho.npy", rho, {N});
    npy::save(out + "/t_iter_ms.npy", t_iter, {N});
    npy::save(out + "/t_kernel_ms.npy", t_kernel, {N});
    std::printf("%d steps, progress %.1f/%.1f m\n", n, progress, tv.length);
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "mppi_sim: %s\n", e.what());
    return 1;
}
```

### `cuda/apps/bench.cpp`

```cpp
// Latency of one MPPI iteration, phase by phase.
//
//   mppi_bench --config config/mppi.yaml [--backend cpu|gpu] [--iters 200]
//              [--warmup 10] [--set mppi.num_samples=8192 ...]
//
// The controller runs from a fixed state (mid-track, 6 m/s): the time of an
// iteration does not depend on the state, every rollout does the same work.
// The first iterations are discarded (warm-up): the first kernel launch loads
// the module on the GPU (lazy loading), caches are cold.
// Prints p50 and p99 of each phase, in ms.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "mppi/config.hpp"
#include "mppi/controller.hpp"
#include "mppi/track.cuh"

using namespace mppi;

namespace {

float percentile(std::vector<float> v, float q) {
    std::sort(v.begin(), v.end());
    return v[static_cast<std::size_t>(q * (v.size() - 1))];
}

}  // namespace

int main(int argc, char** argv) try {
    std::string config = "config/mppi.yaml", backend = "gpu";
    int iters = 200, warmup = 10;
    std::vector<std::string> overrides;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string key = argv[i], value = argv[i + 1];
        if (key == "--config") config = value;
        else if (key == "--backend") backend = value;
        else if (key == "--iters") iters = std::stoi(value);
        else if (key == "--warmup") warmup = std::stoi(value);
        else if (key == "--set") overrides.push_back(value);
        else throw std::runtime_error("unknown argument " + key);
    }
    const Config cfg = load_config(config, overrides);
    const TrackHost track = load_track(cfg.track_bin_path);
    Controller ctrl(cfg, track, backend == "gpu" ? Backend::Gpu : Backend::Cpu);

    // Somewhere on the track, at speed. Only the first state_dim values are read.
    const float x0[6] = {8.0f, 0.2f, 0.1f, 6.0f, 0.0f, 0.5f};
    std::vector<float> total, noise, rollouts, update, upload, kernel, download;
    for (int n = 0; n < warmup + iters; ++n) {
        Diagnostics d;
        const auto tic = std::chrono::steady_clock::now();
        ctrl.command(x0, &d);
        const float t = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - tic).count();
        if (n < warmup) continue;
        total.push_back(t);
        noise.push_back(d.t_noise_ms);
        rollouts.push_back(d.t_rollouts_ms);
        update.push_back(d.t_update_ms);
        upload.push_back(d.t_upload_ms);
        kernel.push_back(d.t_kernel_ms);
        download.push_back(d.t_download_ms);
    }

    std::printf("backend %s, K = %d, T = %d, model %s, %d iterations\n", backend.c_str(), cfg.p.K,
                cfg.p.T, cfg.p.model == ModelKind::Dynamic ? "dynamic" : "kinematic", iters);
    std::printf("%-22s %9s %9s\n", "phase (ms)", "p50", "p99");
    auto row = [](const char* name, const std::vector<float>& v) {
        std::printf("%-22s %9.3f %9.3f\n", name, percentile(v, 0.5f), percentile(v, 0.99f));
    };
    row("noise (host)", noise);
    row("rollouts", rollouts);
    if (backend == "gpu") {
        row("  upload (H2D)", upload);
        row("  kernel", kernel);
        row("  download (D2H)", download);
    }
    row("weights + update", update);
    row("total", total);
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "mppi_bench: %s\n", e.what());
    return 1;
}
```

### `bench/check_parity.py`

```python
"""Non-regression gate: Python costs vs C++/CUDA costs on a fixed noise tensor.

For each case: draws x0, U and the (K, T, 2) noise in NumPy, saves them as
float32 .npy, runs build/mppi_parity on the same files, and compares the K
per-trajectory costs one by one with NumPy's rollout_costs (float64).
A single mismatch points to a bug in the port, far faster than comparing
final trajectories.

    pixi run check                    # CPU and GPU paths
    pixi run check --backend cpu      # machine without a GPU: C++ CPU path only

Error measure, per rollout: |S_cpp - S_py| / max(1, |S_py|). Costs are O(1) to
O(100), and some are close to 0 (progress reward), hence the max(1, .).

The cost is not continuous: lookup() takes the nearest grid cell, so d and s
jump when a state crosses a cell edge. A rollout that passes within a few
micrometres of an edge can land in different cells in float32 and float64.
Such a rollout is "explained" (edge distance < EDGE_EPS along the NumPy
trajectory). The gate fails on any unexplained mismatch, or if more than
MAX_EXPLAINED of the rollouts are explained ones.
"""
import argparse
import dataclasses
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from mppi import track as trk
from mppi.config import STATE_DIM, Config, load_config
from mppi.controller import S_MAX, rollout_costs
import report
from report import cell

TOL = 1e-4
EDGE_EPS = 1e-5        # m, float32 position error after T steps is ~1e-6 m
MAX_EXPLAINED = 0.01   # fraction of K
EXE = ROOT / "build" / "mppi_parity"
OUT = ROOT / "results" / "parity"

# name -> (YAML overrides, start: (centerline index, lateral offset m, heading offset rad, vx, vy, r), U)
CASES = {
    "standstill":  ({}, (0, 0.0, 0.0, 0.0, 0.0, 0.0), (0.0, 0.0)),
    "cornering":   ({}, (250, 0.1, 0.1, 6.0, -0.6, 1.2), (0.5, 0.1)),
    "in the band": ({}, (100, 0.0, 0.0, 1.2, 0.05, 0.2), (1.0, 0.05)),
    "off track":   ({}, (500, 0.6, 0.4, 5.0, 0.0, 0.5), (0.0, 0.0)),
    "start line":  ({}, (-20, 0.0, 0.0, 6.0, 0.0, 0.0), (0.0, 0.0)),   # s wraps from L to 0
    "kinematic":   ({"model": "kinematic", "state_dim": 4}, (250, 0.1, 0.1, 6.0, 0.0, 0.0), (0.5, 0.1)),
    "tanh, euler x2, gamma": ({"vehicle.tire_model": "tanh", "vehicle.integrator": "euler",
                               "vehicle.substeps": 2, "mppi.gamma": 3.0},
                              (250, 0.1, 0.1, 6.0, -0.6, 1.2), (0.5, 0.1)),
    "linear tires": ({"vehicle.tire_model": "linear"}, (250, 0.1, 0.1, 6.0, -0.6, 1.2), (0.5, 0.1)),
    "K=8192, T=50": ({"mppi.num_samples": 8192, "mppi.horizon": 50},
                     (250, 0.1, 0.1, 6.0, -0.6, 1.2), (0.5, 0.1)),
}

PY_FIELD = {"mppi.lambda": "mppi.lam"}   # "lambda" is a Python keyword, see config.py


def apply(cfg: Config, overrides: dict) -> Config:
    """Same overrides as `--set key=value` on the C++ side, on the frozen dataclasses."""
    for key, value in overrides.items():
        path = PY_FIELD.get(key, key).split(".")
        if len(path) == 1:
            cfg = dataclasses.replace(cfg, **{path[0]: value})
        else:
            section = getattr(cfg, path[0])
            cfg = dataclasses.replace(cfg, **{path[0]: dataclasses.replace(section, **{path[1]: value})})
    return cfg


def make_inputs(cfg: Config, track: trk.Track, start, u, seed: int):
    """x0, U, eps in float32: both sides must start from exactly the same numbers."""
    i, offset, dpsi, vx, vy, r = start
    normal = np.array([-np.sin(track.heading[i]), np.cos(track.heading[i])])
    x, y = track.centerline[i] + offset * normal
    full = np.array([x, y, track.heading[i] + dpsi, vx, vy, r])
    x0 = full if cfg.state_dim == STATE_DIM["dynamic"] else np.append(full[:3], np.hypot(vx, vy))
    m = cfg.mppi
    U = np.tile(np.array(u), (m.horizon, 1))
    eps = np.random.default_rng(seed).standard_normal((m.num_samples, m.horizon, 2)) * m.noise_std
    return x0.astype(np.float32), U.astype(np.float32), eps.astype(np.float32)


def edge_distance(X: np.ndarray, track: trk.Track) -> np.ndarray:
    """(K, T+1, n) states -> (K,) smallest distance (m) of x_1..x_T to a grid cell edge."""
    g = (X[:, 1:, :2] - np.array([track.x_min, track.y_min])) / track.res
    return (np.abs(g - np.round(g)).min(axis=(1, 2))) * track.res


def compare(S_py: np.ndarray, S_cpp: np.ndarray, edge: np.ndarray) -> dict:
    err = np.abs(S_cpp.astype(np.float64) - S_py) / np.maximum(1.0, np.abs(S_py))
    over = err > TOL
    explained = over & (edge < EDGE_EPS)
    unexplained = over & ~explained
    shown = np.where(unexplained, err, 0.0) if unexplained.any() else err   # worst unexplained first
    worst = int(np.argmax(shown))
    return {"max": float(err.max()), "p99": float(np.percentile(err, 99)),
            "explained": int(explained.sum()), "unexplained": int(unexplained.sum()),
            "worst": worst, "S_py": float(S_py[worst]), "S_cpp": float(S_cpp[worst]),
            "smax": int((S_py >= S_MAX).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/mppi.yaml")
    ap.add_argument("--backend", choices=("cpu", "gpu", "both"), default="both")
    args = ap.parse_args()
    base = load_config(args.config)
    track = trk.load(base.track.npz_path)
    if not EXE.exists():
        sys.exit(f"{EXE} not found, run `pixi run build` first")

    report.header("check_parity", f"tolerance {TOL:g} · {args.backend}")
    backends = ("cpu", "gpu") if args.backend == "both" else (args.backend,)
    rows, failed = [], False
    for n, (name, (overrides, start, u)) in enumerate(CASES.items()):
        cfg = apply(base, overrides)
        x0, U, eps = make_inputs(cfg, track, start, u, seed=n)
        case_dir = OUT / f"case{n}"
        case_dir.mkdir(parents=True, exist_ok=True)
        for arr, fname in ((x0, "x0"), (U, "U"), (eps, "eps")):
            np.save(case_dir / f"{fname}.npy", arr)

        cmd = [str(EXE), "--config", args.config, "--dir", str(case_dir), "--backend", args.backend]
        for key, value in overrides.items():
            cmd += ["--set", f"{key}={value}"]
        run = subprocess.run(cmd, capture_output=True, text=True)
        if run.returncode != 0:
            sys.exit(f"{name}: mppi_parity failed\n{run.stderr}")

        # NumPy in float64, on the float32 inputs converted exactly
        S_py, _, X = rollout_costs(x0.astype(np.float64), U.astype(np.float64),
                                   eps.astype(np.float64), track, cfg)
        edge = edge_distance(X, track)
        for b in backends:
            r = compare(S_py, np.load(case_dir / f"S_{b}.npy"), edge)
            too_many = r["explained"] > MAX_EXPLAINED * cfg.mppi.num_samples
            failed |= r["unexplained"] > 0 or too_many
            rows.append([cell(name), cell(b), cell(f"{r['max']:.1e}"), cell(f"{r['p99']:.1e}"),
                         cell(str(r["explained"]), report.BAD if too_many else ""),
                         cell(str(r["unexplained"]), report.BAD if r["unexplained"] else report.OK),
                         cell(f"k={r['worst']}: {r['S_py']:.4f} / {r['S_cpp']:.4f}"),
                         cell(str(r["smax"]))])
    report.print_table(["case", "path", "max err", "p99 err", "edge", "unexplained", "worst (NumPy / C++)",
                        "S_MAX"], rows,
                       f"err > {TOL:g}: 'edge' if the rollout passes within {EDGE_EPS:g} m of a cell edge "
                       f"(allowed up to {MAX_EXPLAINED:.0%} of K), 'unexplained' otherwise (must be 0)")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
```

### `python/run_sim_cuda.py`

```python
"""Closed loop with the C++ controller (build/mppi_sim), analysed like run_sim.py.

The C++ executable runs the whole loop (controller + dynamic plant) and writes
its logs as .npy. This script gives it the start state, then rebuilds a SimLog
from the logs and reuses summarize() and the figures of run_sim.py, so that the
NumPy and C++ laps are measured by exactly the same code.

    pixi run sim-cuda                 # GPU rollouts
    pixi run sim-cuda --backend cpu   # same C++ code, rollouts on the CPU
    pixi run sim-cuda --numpy         # also runs the NumPy controller, side by side
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

from mppi import track as trk
from mppi.config import load_config
import report
from run_sim import SimLog, describe, plot_series, plot_trajectory, simulate, summarize, summary_rows

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "build" / "mppi_sim"


def run_cpp(config: str, out: Path, x0: np.ndarray, backend: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "x0.npy", x0.astype(np.float32))
    cmd = [str(EXE), "--config", config, "--out", str(out), "--x0", str(out / "x0.npy"),
           "--backend", backend]
    run = subprocess.run(cmd, capture_output=True, text=True)
    if run.returncode != 0:
        sys.exit(f"mppi_sim failed\n{run.stderr}")


def load_log(out: Path, cfg, track: trk.Track) -> tuple[SimLog, np.ndarray]:
    """C++ logs -> SimLog (same fields as the NumPy run). Also returns the kernel times (ms)."""
    ld = lambda name: np.load(out / f"{name}.npy").astype(np.float64)   # noqa: E731
    states, nexts = ld("states"), ld("next_states")
    d, s = trk.lookup(track, nexts[:, 0], nexts[:, 1])
    _, s0 = trk.lookup(track, states[0, 0], states[0, 1])
    progress = float(trk.wrap(np.diff(np.concatenate([[s0], s])), track.length).sum())
    log = SimLog(states, ld("controls"), nexts, ld("ess"), ld("rho"), d,
                 ld("t_iter_ms") / 1e3, progress, None, cfg.vehicle)
    return log, ld("t_kernel_ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/mppi.yaml")
    ap.add_argument("--backend", choices=("gpu", "cpu"), default="gpu")
    ap.add_argument("--numpy", action="store_true", help="also run the NumPy controller")
    args = ap.parse_args()
    if not EXE.exists():
        sys.exit(f"{EXE} not found, run `pixi run build` first")
    cfg = load_config(args.config)
    track = trk.load(cfg.track.npz_path)
    results = cfg.track.npz_path.parent.parent
    x0 = np.array([*track.centerline[0], track.heading[0], 0.0, 0.0, 0.0])

    report.header("run_sim_cuda", describe(cfg))
    runs = {}
    with report.console.status(f"C++ controller, {args.backend} rollouts"):
        out = results / "trajectories" / f"step3_{args.backend}"
        run_cpp(args.config, out, x0, args.backend)
    log, t_kernel = load_log(out, cfg, track)
    runs[f"C++ {args.backend}"] = summarize(log, cfg, track)
    fig_dir = results / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_trajectory(log, cfg, track, fig_dir / f"step3_{args.backend}_trajectory.png")
    plot_series(log, cfg, fig_dir / f"step3_{args.backend}_series.png")

    if args.numpy:
        with report.console.status("NumPy controller"):
            runs["NumPy"] = summarize(simulate(cfg, track), cfg, track)

    report.print_table(["", *runs], summary_rows(runs))
    t = log.t_iter * 1e3
    report.console.print(f"C++ iteration time (ms): median {np.median(t):.2f}, p99 {np.percentile(t, 99):.2f}")
    if args.backend == "gpu":
        report.console.print(f"kernel alone (ms, CUDA events): median {np.median(t_kernel):.3f}, "
                             f"p99 {np.percentile(t_kernel, 99):.3f}")


if __name__ == "__main__":
    main()
```

### `bench/run_bench.py`

```python
"""Latency sweeps: K from 256 to 32768, T, model, compute budget (power limit, SM share).

Reports p50 and p99, not just the mean. Jitter matters as much as the average
under a real-time budget.

Step 3: sweep over K with the naive kernel, phase by phase (build/mppi_bench).

    pixi run bench                        # GPU, T = 50, K = 256 ... 32768
    pixi run bench --backend cpu --k 1024 8192
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import report
from report import cell

EXE = ROOT / "build" / "mppi_bench"
PHASES = ("noise (host)", "rollouts", "  upload (H2D)", "  kernel", "  download (D2H)", "weights + update", "total")


def run(config: str, backend: str, K: int, T: int, iters: int) -> dict[str, tuple[float, float]]:
    cmd = [str(EXE), "--config", config, "--backend", backend, "--iters", str(iters),
           "--set", f"mppi.num_samples={K}", "--set", f"mppi.horizon={T}"]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(out.stderr)
    times = {}
    for line in out.stdout.splitlines():
        m = re.match(r"(.+?)\s+([\d.]+)\s+([\d.]+)$", line)
        if m:
            times[m.group(1)] = (float(m.group(2)), float(m.group(3)))
    return times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/mppi.yaml")
    ap.add_argument("--backend", choices=("gpu", "cpu"), default="gpu")
    ap.add_argument("--k", type=int, nargs="+", default=[256, 1024, 4096, 8192, 16384, 32768])
    ap.add_argument("--horizon", type=int, default=50)
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()
    if not EXE.exists():
        sys.exit(f"{EXE} not found, run `pixi run build` first")

    report.header("run_bench", f"{args.backend} · T = {args.horizon} · p50 / p99 in ms")
    phases = [p for p in PHASES if args.backend == "gpu" or not p.startswith("  ")]
    rows = []
    with report.console.status("measuring"):
        for K in args.k:
            t = run(args.config, args.backend, K, args.horizon, args.iters)
            budget = report.BAD if t["total"][1] > 20.0 else report.OK   # p99 against the 20 ms budget
            rows.append([cell(str(K))] + [cell(f"{t[p][0]:.2f} / {t[p][1]:.2f}",
                                               budget if p == "total" else "") for p in phases])
    report.print_table(["K", *(p.strip() for p in phases)], rows, "total in red when p99 > 20 ms")


if __name__ == "__main__":
    main()
```
