# Fiche CUDA

Tout ce qu'il faut savoir pour écrire, compiler et déboguer du CUDA dans ce projet, sans avoir jamais touché à un GPU. À lire avant `docs/step_3.md`, à garder ouvert pendant les étapes 3 et 4.

## Sommaire

- [1. Le modèle mental](#1-le-modèle-mental)
- [2. Le matériel, juste ce qu'il faut](#2-le-matériel-juste-ce-quil-faut)
- [3. Ton premier programme, ligne par ligne](#3-ton-premier-programme-ligne-par-ligne)
- [4. Threads, blocs, grille : le calcul d'indice](#4-threads-blocs-grille--le-calcul-dindice)
- [5. La mémoire](#5-la-mémoire)
- [6. Écrire du code device](#6-écrire-du-code-device)
- [7. Compiler](#7-compiler)
- [8. Erreurs et débogage](#8-erreurs-et-débogage)
- [9. Mesurer le temps](#9-mesurer-le-temps)
- [10. Performance : les idées qui comptent](#10-performance--les-idées-qui-comptent)
- [11. Cheat sheet](#11-cheat-sheet)
- [12. Rappels C++ pour un développeur Python](#12-rappels-c-pour-un-développeur-python)
- [13. Pour aller plus loin](#13-pour-aller-plus-loin)

---

## 1. Le modèle mental

**Un GPU est un coprocesseur.** Ton programme est un programme C++ normal qui tourne sur le CPU (on dit l'**hôte**, *host*). De temps en temps, il demande au GPU (le **device**) d'exécuter une fonction sur des milliers de données à la fois. Cette fonction s'appelle un **kernel**.

```
        CPU (host)                                     GPU (device)
  ┌──────────────────────┐                      ┌──────────────────────────┐
  │ main()               │  1. cudaMemcpy H→D   │  mémoire GPU (VRAM)      │
  │   prépare les données├─────────────────────►│                          │
  │                      │  2. kernel<<<...>>>  │  des milliers de threads │
  │   lance le kernel    ├─────────────────────►│  exécutent LE MÊME code  │
  │   (rend la main      │                      │  sur des données         │
  │    tout de suite)    │  3. cudaMemcpy D→H   │  différentes             │
  │   récupère le résultat│◄────────────────────┤                          │
  └──────────────────────┘                      └──────────────────────────┘
       mémoire CPU (RAM)                     deux mémoires séparées (sur PC)
```

Les quatre idées à retenir :

1. **Deux mémoires.** Sur un PC, la RAM du CPU et la VRAM du GPU sont physiquement séparées, reliées par le bus PCIe. Un pointeur CPU n'a aucun sens pour le GPU, et inversement. Il faut **copier** explicitement (`cudaMemcpy`). C'est le cas de la RTX 4000 Ada du cluster : 20 Go de GDDR6 derrière un bus PCIe 4.0 x16. Pour MPPI, les copies par itération sont minuscules (état, commande) : c'est leur latence fixe qui compte, pas le débit (étape 5).
2. **Un kernel = une fonction exécutée par N threads.** Tu écris le code **d'un seul** thread. Chaque thread connaît son numéro et s'en sert pour savoir sur quelle donnée travailler. C'est exactement « un thread par rollout » : le thread `k` calcule le rollout `k`.
3. **Le lancement est asynchrone.** `kernel<<<...>>>(...)` met le travail en file et rend la main immédiatement au CPU. Le CPU doit attendre (`cudaDeviceSynchronize`, ou un `cudaMemcpy` qui attend tout seul) avant de lire le résultat.
4. **Les erreurs sont silencieuses.** Chaque fonction CUDA renvoie un code d'erreur, et si tu ne le vérifies pas, rien ne se passe : tu lis des zéros ou des déchets. Toujours vérifier (section 8).

**Comparaison avec NumPy.** En NumPy, tu écris `X = step(X, V)` sur un tableau `(K, 6)` et NumPy boucle sur K en C. En CUDA, tu écris le code pour **un** `k`, et le GPU lance K exécutions en parallèle. NumPy vectorise sur K et boucle sur t en Python ; CUDA parallélise sur K et boucle sur t dans chaque thread (théorie §2.4).

---

## 2. Le matériel, juste ce qu'il faut

```
GPU
├── SM 0 (Streaming Multiprocessor)          RTX 4000 Ada : 48 SM
│   ├── 128 cœurs FP32
│   ├── registres : 65 536 × 32 bits, partagés entre les threads actifs du SM
│   ├── mémoire partagée / cache L1 : ~100 Ko
│   └── ordonnanceurs de warps
├── SM 1
├── ...
├── cache L2 (quelques Mo, partagé)
└── mémoire globale (VRAM : 20 Go de GDDR6 sur la RTX 4000 Ada)
```

**Le warp.** Le GPU n'exécute pas les threads un par un : il les groupe par paquets de **32**, les warps. Les 32 threads d'un warp exécutent **la même instruction au même moment**, chacun sur ses données (modèle SIMT). Conséquences :

- un `if` dont la condition varie d'un thread à l'autre dans un warp fait exécuter **les deux branches** l'une après l'autre, les threads inactifs attendant : c'est la **divergence**. C'est pourquoi le projet écrit `clip`, `fmaxf`, des mélanges arithmétiques plutôt que des `if` sur l'état ;
- un `if` dont la condition est la même pour tous (lue dans la config : `tire_model`, `integrator`) ne coûte rien : tout le warp prend la même branche ;
- les tailles de bloc sont des multiples de 32.

**Le masquage de latence.** Une lecture en mémoire globale prend plusieurs centaines de cycles. Pendant ce temps, le SM exécute **d'autres warps** prêts. C'est tout le secret du GPU : beaucoup de threads en vol pour toujours avoir du travail prêt. D'où la notion d'**occupation** (*occupancy*) : le nombre de warps actifs par SM, rapporté au maximum. Elle est limitée par les registres utilisés par chaque thread (un kernel à 72 registres par thread laisse tenir moins de threads qu'un kernel à 32).

**Compute capability.** Chaque génération de GPU a un numéro : 8.9 pour la RTX 4000 Ada (architecture Ada Lovelace), 8.6 pour une RTX 30xx (Ampere). Le code machine (SASS) est compilé pour un numéro donné : `sm_89`. C'est le `CMAKE_CUDA_ARCHITECTURES 89` du `CMakeLists.txt`. Un binaire sans code pour ton GPU échoue au lancement avec `no kernel image is available for execution on the device`.

**FP64.** Une carte grand public ou de station de travail, comme la RTX 4000 Ada, calcule en double précision 64 fois moins vite qu'en simple. Le projet est en FP32 côté GPU, toujours.

---

## 3. Ton premier programme, ligne par ligne

Fichier `saxpy.cu` : `y = a·x + y` sur un million de floats.

```cpp
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>                      // l'API runtime : cudaMalloc, cudaMemcpy...

// __global__ : un kernel. Appelé depuis le CPU, exécuté sur le GPU. Renvoie toujours void.
__global__ void saxpy(int n, float a, const float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // mon numéro global de thread
    if (i < n)                                       // le dernier bloc déborde : garde
        y[i] = a * x[i] + y[i];
}

int main() {
    const int n = 1000000;
    std::vector<float> x(n, 1.0f), y(n, 2.0f);       // données sur le CPU

    float *d_x, *d_y;                                // d_ = pointeur device (convention)
    cudaMalloc(&d_x, n * sizeof(float));             // réserve de la mémoire GPU
    cudaMalloc(&d_y, n * sizeof(float));
    cudaMemcpy(d_x, x.data(), n * sizeof(float), cudaMemcpyHostToDevice);   // CPU → GPU
    cudaMemcpy(d_y, y.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    int block = 256;                                 // threads par bloc
    int grid = (n + block - 1) / block;              // nombre de blocs, arrondi au-dessus
    saxpy<<<grid, block>>>(n, 3.0f, d_x, d_y);       // lancement : grid blocs de block threads

    cudaMemcpy(y.data(), d_y, n * sizeof(float), cudaMemcpyDeviceToHost);   // attend le kernel, GPU → CPU
    cudaFree(d_x);
    cudaFree(d_y);
    std::printf("y[0] = %f\n", y[0]);                // 5.0
}
```

Compiler et lancer : `pixi run nvcc -arch=sm_89 -o saxpy saxpy.cu && ./saxpy`.

Les trois mots-clés de fonction :

| Qualificatif | Appelée depuis | Exécutée sur | Usage |
|---|---|---|---|
| `__global__` | le CPU (avec `<<<>>>`) | le GPU | le kernel, point d'entrée |
| `__device__` | le GPU | le GPU | fonctions auxiliaires du kernel (`step`, `stage_cost`) |
| `__host__ __device__` | les deux | les deux | **compilée deux fois** : une version CPU, une version GPU. C'est ce qui permet de tester la même dynamique sur CPU et GPU |
| (rien) ou `__host__` | le CPU | le CPU | du C++ normal |

---

## 4. Threads, blocs, grille : le calcul d'indice

Un lancement `kernel<<<grid, block>>>` crée `grid` **blocs** de `block` **threads**. Variables disponibles dans le kernel :

| Variable | Sens | Dans `<<<4, 256>>>` |
|---|---|---|
| `threadIdx.x` | numéro du thread dans son bloc | 0 … 255 |
| `blockIdx.x` | numéro du bloc dans la grille | 0 … 3 |
| `blockDim.x` | nombre de threads par bloc | 256 |
| `gridDim.x` | nombre de blocs | 4 |

```
blockIdx.x =      0                1                2                3
             [0 1 2 ... 255] [0 1 2 ... 255] [0 1 2 ... 255] [0 1 2 ... 255]   threadIdx.x
indice global :  0 ... 255      256 ... 511     512 ... 767     768 ... 1023

i = blockIdx.x * blockDim.x + threadIdx.x
```

**L'arrondi au-dessus.** Pour N éléments : `grid = (N + block - 1) / block`. Si N n'est pas un multiple de `block`, le dernier bloc contient des threads « en trop » : la garde `if (i >= N) return;` est obligatoire.

**Pourquoi des blocs ?** Un bloc entier tourne sur un seul SM, ses threads peuvent coopérer (mémoire partagée, `__syncthreads()`). Des blocs différents ne peuvent pas se synchroniser entre eux pendant le kernel. Pour l'étape 3, on n'utilise pas cette coopération : chaque thread est indépendant. Elle servira pour les réductions de l'étape 4.

**Choisir la taille de bloc.** 128 ou 256 pour commencer. Multiple de 32, au plus 1024. Il faut aussi assez de blocs pour occuper tous les SM : avec K = 1024 et des blocs de 256, il n'y a que 4 blocs pour 48 SM sur la RTX 4000 Ada. Le GPU est alors vide à plus de 90 %.

Il existe aussi des grilles 2D/3D (`dim3 block(16, 16)`, `threadIdx.y`), utiles pour les images. On n'en a pas besoin.

---

## 5. La mémoire

### 5.1 Les espaces mémoire

| Espace | Déclaration | Portée | Vitesse | Dans le projet |
|---|---|---|---|---|
| **registres** | variable locale scalaire (`float x;`) | un thread | la plus rapide | l'état du véhicule, le coût accumulé |
| **locale** | tableau local indexé dynamiquement, ou trop de registres (*spill*) | un thread | lente (en réalité en mémoire globale, avec cache) | à éviter. `float s[6]` reste en registres si tous les indices sont connus à la compilation (boucles déroulées) |
| **partagée** | `__shared__ float buf[256];` | un bloc | très rapide | réductions, étape 4 |
| **globale** | `cudaMalloc` | tout le GPU, persiste entre kernels | lente (centaines de cycles) | bruit, `U`, grille de coût, coûts S |
| **constante** | `__constant__ Params c;` | tout le GPU, lecture seule | rapide si tous les threads lisent la même adresse | paramètres (option de l'étape 4) |
| **texture** | objet texture | lecture seule, cache 2D | rapide pour des accès 2D voisins | la grille de coût, étape 4 |

**Les arguments du kernel** (passés par valeur : `Params p`, `TrackView tr`) sont copiés par le runtime dans une mémoire constante, jusqu'à 4 Ko (32 Ko depuis CUDA 12.1 sur Volta et plus récent). Passer une struct de paramètres par valeur est donc simple et efficace. **Mais une struct passée au kernel ne doit contenir aucun pointeur CPU, ni `std::vector`, ni `std::string`** : seulement des nombres, et des pointeurs **device**.

### 5.2 L'API mémoire

```cpp
float* d_a;
cudaMalloc(&d_a, n * sizeof(float));                                   // allouer
cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);       // dst, src, octets, sens
cudaMemcpy(h_a, d_a, n * sizeof(float), cudaMemcpyDeviceToHost);
cudaMemset(d_a, 0, n * sizeof(float));                                 // mettre à zéro (octets !)
cudaFree(d_a);                                                         // libérer
```

Pièges classiques :

- **la taille est en octets**, pas en éléments : `n * sizeof(float)` ;
- l'ordre est **destination, source** (comme `memcpy`) ;
- **ne jamais déréférencer un pointeur device sur le CPU** (`d_a[0]` dans `main` = segfault) ;
- `cudaMalloc` est **lent** (des centaines de µs) : on alloue une fois au démarrage, jamais dans la boucle de contrôle ;
- `cudaMemcpy` est **bloquant** : il attend que les kernels précédents aient fini. Pratique (synchronisation implicite), mais à savoir quand on mesure.

En C++ propre, on enveloppe ça dans une classe RAII qui libère dans son destructeur (`DeviceBuffer<T>` de l'étape 3).

### 5.3 Coalescence (en avoir entendu parler)

Quand les 32 threads d'un warp lisent la mémoire globale, le matériel regroupe les lectures en transactions de 32 à 128 octets. Si le thread `k` lit `a[k]` (adresses contiguës), une transaction sert tout le warp : accès **coalescé**. Si le thread `k` lit `a[k * 100]`, chaque thread déclenche sa propre transaction : jusqu'à 32 fois plus de trafic.

Le bruit de l'étape 3 est rangé `(K, T, 2)` comme en NumPy : au pas `t`, le thread `k` lit `eps[(k*T + t)*2]`. Deux threads voisins lisent à `T*2*4 = 240` octets d'écart : **non coalescé**. C'est voulu (correction d'abord). Ranger `(T, 2, K)` (structure de tableaux) rendrait la lecture contiguë : c'est la ligne 5 du tableau d'optimisation de l'étape 4.

### 5.4 Mémoire unifiée et mémoire épinglée (pour plus tard)

- `cudaMallocManaged` : un seul pointeur valable sur CPU et GPU, le pilote migre les pages. Plus simple, mais performances moins prévisibles sur un GPU discret : les migrations de pages passent par le PCIe. À mesurer contre la mémoire épinglée à l'étape 5.
- `cudaMallocHost` (mémoire « épinglée », *pinned*) : mémoire CPU non paginable, transferts plus rapides et seule à permettre des copies asynchrones (`cudaMemcpyAsync`). Étape 4 ou 5.
- `cudaHostAlloc(..., cudaHostAllocMapped)` (*zero-copy*) : mémoire épinglée que le GPU lit directement à travers le PCIe, sans copie. Lent pour de gros volumes, potentiellement intéressant pour quelques octets (état, commande). Étape 5.

---

## 6. Écrire du code device

### 6.1 Ce qui marche, ce qui ne marche pas dans un kernel

| Autorisé | Interdit ou à éviter |
|---|---|
| arithmétique, boucles, `if`, `switch`, fonctions `__device__`, templates, `struct`, `constexpr`, `if constexpr`, lambdas `__device__` | `std::vector`, `std::string`, `std::cout`, exceptions, `new`/`delete` (possible mais lent : à éviter), récursion profonde |
| fonctions mathématiques `sinf`, `cosf`, `atanf`, `tanhf`, `expf`, `sqrtf`, `fabsf`, `fminf`, `fmaxf`, `floorf`, `fmodf`, `isfinite` | les versions de `std::` ne sont pas toutes utilisables côté device ; préfère les fonctions C en `f` |
| `printf` (oui, depuis un kernel : utile pour déboguer un thread : `if (k == 0) printf(...)`) | lire un pointeur CPU |
| tableaux locaux de taille fixe (`float s[6];`) | tableaux locaux de taille variable |

### 6.2 FP32 strict : le piège des doubles

En C++, `0.5` est un `double`, `0.5f` est un `float`. Et `sin(x)` peut appeler la version double. Une seule constante sans `f` dans une expression suffit à faire tout le calcul en double :

```cpp
float y = 0.5 * x;      // x converti en double, multiplication FP64, reconversion : lent sur GeForce
float y = 0.5f * x;     // FP32
float s = sin(x);       // selon les surcharges, risque de version double
float s = sinf(x);      // FP32, toujours
```

Règle du projet : littéraux en `f`, fonctions en `f`. Vérification : `cuobjdump -sass` ne doit montrer aucune instruction `DADD`, `DMUL`, `DFMA` dans le kernel.

### 6.3 Résultats différents du CPU : normal

Le même calcul FP32 ne donne pas exactement le même résultat sur CPU et GPU, pour trois raisons :

1. **FMA.** `nvcc` fusionne par défaut `a*b + c` en une seule instruction FMA (un seul arrondi au lieu de deux). Plus précis, mais différent. `--fmad=false` désactive (diagnostic seulement).
2. **Fonctions mathématiques.** `sinf`, `atanf` du GPU ont une erreur garantie de quelques ULP, différente de celle de la libm du CPU.
3. **Ordre des opérations.** Une somme dans un ordre différent arrondit différemment.

Les écarts sont de l'ordre de 1e-7 relatif par opération et s'accumulent au fil des pas. C'est pour ça que la parité se juge à une tolérance (1e-4), jamais à l'égalité.

`-use_fast_math` remplace `sinf` par `__sinf` (beaucoup plus rapide, moins précis), met les dénormaux à zéro et fait la division approchée. C'est la ligne 6 de l'étape 4, à mesurer, jamais à activer par défaut.

### 6.4 Templates : choisir à la compilation

Quand un choix dépend de la config (modèle cinématique ou dynamique), on peut en faire un **paramètre de template** : le compilateur génère une version du kernel par cas, sans aucun test à l'exécution.

```cpp
struct Kinematic { static constexpr int DIM = 4; /* step() */ };
struct Dynamic   { static constexpr int DIM = 6; /* step() */ };

template <class Model>
__global__ void rollout_kernel(...) {
    float x[Model::DIM];          // taille connue à la compilation : reste en registres
    Model::step(x, ...);
}

// côté hôte, un seul test, avant le lancement :
if (p.model == ModelKind::Dynamic) rollout_kernel<Dynamic><<<g, b>>>(...);
else                               rollout_kernel<Kinematic><<<g, b>>>(...);
```

`if constexpr (Model::DIM == 4) { ... }` dans le kernel : branche éliminée à la compilation.

### 6.5 Une fonction, deux cibles

```cpp
__host__ __device__ inline float wrap(float ds, float L) { ... }
```

Cette fonction est compilée pour le CPU **et** pour le GPU. Le projet l'utilise pour toute la dynamique et le coût : on peut alors calculer les K coûts dans une boucle CPU ordinaire (déboguable avec `gdb`, `printf`, sans GPU) puis sur le GPU, avec **le même code**. Si le CPU est juste et le GPU faux, le bug est dans la mécanique CUDA (indices, copies), pas dans la physique.

Un fichier `.cpp` compilé par `g++` ne connaît pas `__host__ __device__`. Astuce : une macro qui disparaît hors de `nvcc`.

```cpp
#ifdef __CUDACC__                 // défini seulement quand nvcc compile
#define HD __host__ __device__
#else
#define HD
#endif
```

---

## 7. Compiler

### 7.1 Ce que fait nvcc

```
fichier.cu
   │  nvcc sépare le code
   ├──► code hôte  ──► g++ (le compilateur « hôte ») ──► .o
   └──► code device ──► PTX (assembleur virtuel) ──► ptxas ──► SASS (code machine sm_89)
                                       │
                                       └─ embarqué dans le .o (« fatbinary »)
```

- Les `.cu` passent par `nvcc`. Les `.cpp` passent directement par `g++`. Un `.cpp` peut appeler l'API runtime (`cudaMalloc`, inclure `cuda_runtime.h`) mais **ne peut pas** contenir de kernel ni la syntaxe `<<<>>>`.
- Règle simple : tout fichier qui définit ou lance un kernel est un `.cu`. Les en-têtes qui contiennent du code device sont des `.cuh`.
- **PTX vs SASS.** Le SASS est le code machine d'une architecture précise. Le PTX est portable : le pilote peut le compiler à la volée (JIT) pour un GPU plus récent. `CMAKE_CUDA_ARCHITECTURES 89` produit le SASS de la RTX 4000 Ada, plus son PTX (un GPU plus récent le compilera à la volée).

### 7.2 Les options utiles

| Option | Effet |
|---|---|
| `-arch=sm_89` | compile pour la compute capability 8.9 (RTX 4000 Ada) |
| `-std=c++17` | standard C++ |
| `-O3` | optimisations du code hôte (le code device est optimisé par défaut) |
| `-lineinfo` | numéros de ligne dans le code device, pour `compute-sanitizer` et Nsight. Aucun coût en vitesse : à laisser |
| `-G` | mode debug device : désactive les optimisations, très lent. Pour `cuda-gdb` seulement |
| `-Xptxas -v` | affiche registres, mémoire partagée et *spills* de chaque kernel à la compilation |
| `--use_fast_math` | intrinsèques rapides et imprécises (étape 4) |
| `--fmad=false` | pas de FMA (diagnostic de parité) |
| `-Xcompiler -Wall` | passe `-Wall` au compilateur hôte |

### 7.3 CMake

CMake traite CUDA comme un langage de première classe :

```cmake
project(edge_mppi LANGUAGES CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES 89)
find_package(CUDAToolkit REQUIRED)               # cible CUDA::cudart

add_library(mppi_core STATIC src/a.cpp src/kernel.cu)
target_link_libraries(mppi_core PUBLIC CUDA::cudart)
target_compile_options(mppi_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

add_executable(app apps/main.cpp)
target_link_libraries(app PRIVATE mppi_core)
```

`$<$<COMPILE_LANGUAGE:CUDA>:...>` : option appliquée seulement aux fichiers `.cu`.

### 7.4 Avec pixi (important)

Le dépôt fournit `nvcc` 12.9, `cmake`, `ninja` **et** un `gcc` 14 compatible via pixi. Lance toujours la compilation **dans l'environnement pixi** : `pixi run build`, `pixi run nvcc ...`, ou depuis un `pixi shell`. Lancé hors pixi, `nvcc` prend le `gcc` du système avec les en-têtes du `gcc` de pixi, et tu obtiens des centaines d'erreurs incompréhensibles dans `type_traits` ou `c++config.h` (`user-defined literal operator not found`, `__is_array is undefined`). Ce n'est pas ton code : c'est le mauvais compilateur hôte.

### 7.5 Compiler sans GPU, exécuter avec

`nvcc` compile sans carte graphique : seule l'**exécution** a besoin d'un GPU et de son pilote. Sur la machine WSL2 sans GPU, `pixi run build` marche, et un programme CUDA lancé échoue avec `CUDA driver version is insufficient for CUDA runtime version` (pas de pilote). Sur le nœud GPU du cluster (VS Code Remote-SSH), tout marche. Vérifier le GPU disponible : `nvidia-smi` (modèle, pilote, version CUDA maximale supportée par le pilote, mémoire, processus en cours).

**Pilote et toolkit.** La version CUDA affichée par `nvidia-smi` est la version **maximale** que le pilote supporte. Le toolkit (nvcc 12.9) doit être inférieur ou égal, sinon : `CUDA driver version is insufficient`. Remède : mettre à jour le pilote, ou épingler un `cuda-toolkit` plus ancien dans `pixi.toml`.

---

## 8. Erreurs et débogage

### 8.1 Toujours vérifier

```cpp
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), \
                         __FILE__, __LINE__);                                  \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

CUDA_CHECK(cudaMalloc(&d_a, bytes));
kernel<<<g, b>>>(...);
CUDA_CHECK(cudaGetLastError());          // erreurs de LANCEMENT (config invalide, pas de code pour ce GPU)
CUDA_CHECK(cudaDeviceSynchronize());     // erreurs PENDANT l'exécution (accès hors bornes...)
```

Un kernel ne renvoie rien. Deux moments d'erreur :

1. **au lancement** : `cudaGetLastError()` juste après `<<<>>>` ;
2. **pendant l'exécution** : signalée par le prochain appel qui synchronise (`cudaDeviceSynchronize`, `cudaMemcpy`). Elle peut donc apparaître sur une ligne qui n'y est pour rien. En débogage, ajoute un `cudaDeviceSynchronize()` juste après le kernel suspect.

Une erreur d'exécution (accès illégal) **corrompt le contexte** : tous les appels CUDA suivants échouent. Seul un redémarrage du programme répare.

### 8.2 Les erreurs que tu vas rencontrer

| Message | Cause probable |
|---|---|
| `an illegal memory access was encountered` | indice hors bornes, pointeur CPU lu sur le GPU, buffer trop petit. → `compute-sanitizer` |
| `invalid configuration argument` | `block > 1024`, `grid = 0` (K = 0 ?) |
| `no kernel image is available for execution on the device` | pas compilé pour ce GPU : vérifier `CMAKE_CUDA_ARCHITECTURES` et `nvidia-smi --query-gpu=compute_cap --format=csv` |
| `CUDA driver version is insufficient for CUDA runtime version` | pas de GPU/pilote (machine WSL2), ou pilote trop ancien pour le toolkit |
| `out of memory` | allocation trop grande, ou fuite (allocation dans la boucle) |
| `too many resources requested for launch` | trop de registres × threads par bloc : réduire la taille de bloc |
| le résultat est plein de zéros, sans erreur | erreur non vérifiée, kernel jamais exécuté, copie dans le mauvais sens |
| le résultat change d'un lancement à l'autre | lecture de mémoire non initialisée, ou concurrence entre threads (deux threads écrivent la même case) |

### 8.3 Les outils

- **`printf` dans le kernel**, filtré sur un thread : `if (k == 42 && t < 3) printf("t=%d vx=%f\n", t, x[3]);`. La sortie apparaît à la synchronisation suivante.
- **Le chemin CPU** : si la fonction est `__host__ __device__`, appelle-la depuis une boucle CPU et débogue avec `gdb` ou des `printf` normaux. C'est le meilleur outil du projet.
- **`compute-sanitizer`** (fourni par pixi) : le valgrind du GPU. `pixi run compute-sanitizer ./build/mppi_parity ...` indique le kernel, le thread, l'adresse et (avec `-lineinfo`) la ligne de l'accès fautif. Outils : `--tool memcheck` (défaut, accès hors bornes), `--tool initcheck` (lecture de mémoire non initialisée), `--tool racecheck` (concurrence en mémoire partagée). Programme 10 à 100 fois plus lent : utiliser un petit K.
- **`cuda-gdb`** : gdb pour le GPU, avec `-G`. Rarement nécessaire si le chemin CPU existe.
- **`cuobjdump --dump-resource-usage fichier.o`** ou `-Xptxas -v` : registres et pile par kernel.

---

## 9. Mesurer le temps

Le lancement est asynchrone : un chronomètre CPU autour de `kernel<<<>>>()` mesure le temps de **mise en file** (quelques µs), pas le kernel. Deux méthodes correctes :

**Chronomètre CPU + synchronisation** (temps de bout en bout, ce qui compte pour le budget de 20 ms) :

```cpp
auto t0 = std::chrono::steady_clock::now();
kernel<<<g, b>>>(...);
cudaDeviceSynchronize();
float ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t0).count();
```

**Événements CUDA** (temps mesuré par le GPU lui-même, précis à ~0,5 µs) :

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start); cudaEventCreate(&stop);       // une fois
cudaEventRecord(start);
kernel<<<g, b>>>(...);
cudaEventRecord(stop);
cudaEventSynchronize(stop);                             // attend que stop soit atteint
float ms; cudaEventElapsedTime(&ms, start, stop);
```

Règles :

- **échauffement** : jeter les premières itérations. Le premier lancement charge le module sur le GPU (chargement paresseux, `CUDA_MODULE_LOADING=LAZY` par défaut depuis CUDA 12.2), les caches sont froids, le GPU sort de son état d'économie d'énergie ;
- **p50 et p99**, pas la moyenne : en temps réel, c'est la queue qui casse le budget ;
- **le GPU change de fréquence** selon la charge, la température et la limite de puissance. Lire : `nvidia-smi -q -d CLOCK`. Verrouiller (`nvidia-smi -lgc`) demande les droits root : à voir avec les admins du cluster (étape 5) ;
- **le nœud est partagé** : un autre job sur le même GPU ou les mêmes cœurs CPU fausse la p99. `nvidia-smi` liste les processus du GPU ; noter ce qui tournait pendant la mesure.

**Profileurs** (étape 4) :

- **Nsight Systems** (`nsys profile ./app`) : chronologie de tout le programme (CPU, copies, kernels). Répond à « où passe le temps ? ».
- **Nsight Compute** (`ncu ./app`) : analyse détaillée d'un kernel (occupation, bande passante, instructions). Répond à « pourquoi ce kernel est lent ? ».

---

## 10. Performance : les idées qui comptent

Pour l'étape 4. À l'étape 3, on ne fait **que** du correct.

1. **Assez de threads.** Il faut des dizaines de milliers de threads pour remplir un GPU. K = 1024 occupe 4 SM sur 48 ; K = 8192, 32 blocs, donc 32 SM sur 48. Le K de la cible est 8192 pour cette raison.
2. **Le moins de transferts CPU↔GPU possible.** Le PCIe (~10-25 Go/s) est 10 à 30 fois plus lent que la VRAM. Envoyer 3,3 Mo de bruit par itération coûte de l'ordre de la milliseconde : générer le bruit sur le GPU (cuRAND) supprime le transfert et le tirage CPU.
3. **Le moins d'accès mémoire globale possible.** État en registres, coût accumulé en registre, un seul float écrit par thread.
4. **Accès coalescés.** Threads voisins, adresses voisines (§5.3).
5. **Pas de divergence sur les données.** Mélanges arithmétiques plutôt que `if` sur l'état.
6. **Registres.** Plus un thread utilise de registres, moins il y a de threads actifs par SM. Le kernel dynamique de référence en utilise 72 : 65 536 / 72 ≈ 900 threads au plus par SM, contre 1536 possibles sur Ada (8.9) comme sur Ampere (8.6). À regarder avec Nsight Compute avant de conclure quoi que ce soit.
7. **Memory-bound ou compute-bound ?** Un kernel est limité soit par la bande passante mémoire, soit par le calcul. Le modèle *roofline* (Nsight Compute) le dit. Optimiser l'arithmétique d'un kernel limité par la mémoire ne sert à rien, et inversement.

---

## 11. Cheat sheet

```cpp
// ---------- squelette de kernel « un thread par élément » ----------
__global__ void kern(int n, const float* __restrict__ in, float* __restrict__ out, Params p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = f(in[i], p);
}
int block = 256, grid = (n + block - 1) / block;
kern<<<grid, block>>>(n, d_in, d_out, p);
CUDA_CHECK(cudaGetLastError());

// ---------- grid-stride loop : un kernel qui marche pour n'importe quel n ----------
for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) { ... }

// ---------- mémoire ----------
CUDA_CHECK(cudaMalloc(&d, n * sizeof(T)));
CUDA_CHECK(cudaMemcpy(d, h, n * sizeof(T), cudaMemcpyHostToDevice));
CUDA_CHECK(cudaMemcpy(h, d, n * sizeof(T), cudaMemcpyDeviceToHost));
CUDA_CHECK(cudaMemset(d, 0, n * sizeof(T)));
CUDA_CHECK(cudaFree(d));

// ---------- synchronisation ----------
cudaDeviceSynchronize();                  // attend tout le GPU
__syncthreads();                          // dans un kernel : barrière pour le BLOC

// ---------- infos GPU ----------
cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
prop.name, prop.major, prop.minor, prop.multiProcessorCount, prop.totalGlobalMem

// ---------- index 2D d'un tableau (ny, nx, c) row-major, comme NumPy ----------
a[(iy * nx + ix) * C + c]
// (K, T, 2) : eps[(k * T + t) * 2 + j]

// ---------- maths FP32 ----------
sinf cosf tanf atanf atan2f tanhf expf logf sqrtf powf fabsf fminf fmaxf floorf fmodf isfinite
min(int, int) max(int, int)              // versions entières, host et device
__sinf __expf ...                        // intrinsèques rapides et imprécises (étape 4)

// ---------- clamp sans branche ----------
float c = fminf(fmaxf(x, lo), hi);

// ---------- réduction naïve par atomique (lent mais simple, pour tester) ----------
atomicAdd(&d_sum[0], v);                 // float atomicAdd : OK sur tout GPU récent
```

```bash
# ---------- terminal ----------
nvidia-smi                                        # GPU, pilote, mémoire, processus
nvidia-smi --query-gpu=name,compute_cap --format=csv
pixi run build                                    # cmake + ninja dans l'env pixi
pixi run nvcc -std=c++17 -arch=sm_89 -lineinfo -o a a.cu
pixi run nvcc -arch=sm_89 -Xptxas -v -c a.cu      # registres et spills par kernel
pixi run cuobjdump --dump-resource-usage build/libmppi_core.a
pixi run cuobjdump -sass build/libmppi_core.a | grep -c DFMA   # instructions FP64 ?
pixi run compute-sanitizer ./build/app            # accès mémoire illégaux
pixi run compute-sanitizer --tool initcheck ./build/app
pixi run nsys profile -o prof ./build/app         # chronologie (étape 4)
pixi run ncu --set full -o kern ./build/app       # analyse de kernel (étape 4)
```

**Checklist quand ça ne marche pas :**

1. Toutes les fonctions CUDA vérifiées avec `CUDA_CHECK` ? `cudaGetLastError()` après le lancement ?
2. Tailles en **octets** ? Copie dans le bon **sens** ?
3. Garde `if (i >= n) return;` ?
4. Indice `(k * T + t) * 2 + j` cohérent avec la disposition écrite par NumPy (ordre C) ?
5. Le chemin CPU donne-t-il le bon résultat ? Si oui, le bug est dans la mécanique CUDA, sinon dans les maths.
6. `compute-sanitizer` ?
7. Une constante `double` qui traîne (`0.5` au lieu de `0.5f`) ?

---

## 12. Rappels C++ pour un développeur Python

| Python | C++ | Remarque |
|---|---|---|
| `import x` | `#include "x.hpp"` | copie textuelle du fichier ; `#pragma once` évite la double inclusion |
| module | `namespace mppi { ... }` | |
| `x = 3.0` | `float x = 3.0f;` ou `const float x = 3.0f;` | types statiques ; `const` = non modifiable, à mettre partout où c'est possible |
| `list` / `np.ndarray` 1D | `std::vector<float> v(n);` | `v.data()` donne le pointeur brut, `v.size()` la taille |
| `@dataclass(frozen=True)` | `struct Params { float dt; int K; };` | agrégat ; initialisation `Params p{};` (tout à zéro) |
| passage d'objet | `void f(const Params& p)` | `&` = référence (pas de copie), `const` = lecture seule |
| tableau NumPy en argument | `const float* x` + taille | pointeur brut vers le premier élément |
| `raise ValueError(...)` | `throw std::runtime_error("...");` | côté hôte seulement, jamais dans un kernel |
| `try/except` | `try { ... } catch (const std::exception& e) { ... }` | |
| `with open(...)` | RAII : l'objet libère sa ressource dans son destructeur | `std::ifstream`, `DeviceBuffer` |
| générique (`typing.Generic`) | `template <class T>` | résolu à la compilation |
| `@staticmethod` | `static` dans une struct | `Dynamic::step(...)` |
| `print(f"{x:.3f}")` | `std::printf("%.3f\n", x);` | marche aussi dans un kernel |

**Compilation C++.** Chaque `.cpp`/`.cu` est compilé séparément en `.o`, puis l'éditeur de liens les assemble. Un `.hpp` contient des **déclarations** (« cette fonction existe ») ; la **définition** (le corps) est dans un seul `.cpp`. Exceptions : les fonctions `inline` et les templates, définis entièrement dans l'en-tête. Erreur `undefined reference to ...` = déclaré mais jamais défini, ou fichier oublié dans `CMakeLists.txt`.

**Ordre C (row-major).** Un `np.ndarray` de forme `(K, T, 2)` en ordre C est rangé avec le dernier indice qui varie le plus vite : `eps[k, t, j]` est à la position `(k*T + t)*2 + j`. `np.save` et `tobytes()` écrivent cet ordre : le C++ le lit tel quel.

---

## 13. Pour aller plus loin

- *CUDA C++ Programming Guide* (NVIDIA) : la référence. Chapitres 2 (modèle de programmation) et 5 (performance) d'abord.
- *CUDA C++ Best Practices Guide* (NVIDIA) : coalescence, occupation, transferts. Pour l'étape 4.
- Mark Harris, *An Even Easier Introduction to CUDA* (blog NVIDIA) : 20 minutes, très bien pour démarrer.
- Mark Harris, *Optimizing Parallel Reduction in CUDA* (slides) : le classique pour les réductions de l'étape 4.
- Kirk & Hwu, *Programming Massively Parallel Processors* : le livre de cours.
