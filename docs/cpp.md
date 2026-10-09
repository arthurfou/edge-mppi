# Fiche C++

Le C++ nécessaire pour les étapes 3 et 4, pour quelqu'un qui connaît le C et Java. Ce n'est pas un cours complet : seulement ce que le projet utilise, et les pièges qui vont avec. Pour le CUDA lui-même, voir `docs/cuda.md`.

## Sommaire

- [Fiche C++](#fiche-c)
  - [Sommaire](#sommaire)
  - [0. Ce qu'il faut savoir, ce qu'on peut ignorer](#0-ce-quil-faut-savoir-ce-quon-peut-ignorer)
  - [1. Le C++ vu depuis le C et Java](#1-le-c-vu-depuis-le-c-et-java)
    - [Depuis le C : ce qui reste, ce qui change](#depuis-le-c--ce-qui-reste-ce-qui-change)
    - [Depuis Java : les différences qui piègent](#depuis-java--les-différences-qui-piègent)
  - [2. Fichiers, compilation, édition de liens](#2-fichiers-compilation-édition-de-liens)
    - [Les quatre extensions](#les-quatre-extensions)
    - [Déclaration et définition](#déclaration-et-définition)
    - [`namespace`](#namespace)
  - [3. Types, `const`, références](#3-types-const-références)
    - [Types numériques](#types-numériques)
    - [`const`](#const)
    - [Références](#références)
  - [4. `struct` et `class`](#4-struct-et-class)
    - [Une classe complète](#une-classe-complète)
    - [`enum class`](#enum-class)
  - [5. RAII : constructeur, destructeur, copie, déplacement](#5-raii--constructeur-destructeur-copie-déplacement)
    - [Le principe](#le-principe)
    - [`DeviceBuffer<T>` : RAII pour la mémoire GPU](#devicebuffert--raii-pour-la-mémoire-gpu)
    - [Copie et déplacement](#copie-et-déplacement)
  - [6. Templates et `constexpr`](#6-templates-et-constexpr)
    - [Templates](#templates)
    - [Le modèle comme paramètre de template (étape 3)](#le-modèle-comme-paramètre-de-template-étape-3)
    - [`constexpr`](#constexpr)
    - [Lire une erreur de template](#lire-une-erreur-de-template)
  - [7. La bibliothèque standard utile au projet](#7-la-bibliothèque-standard-utile-au-projet)
    - [`std::vector<T>` (`<vector>`)](#stdvectort-vector)
    - [`std::array<T, N>` (`<array>`)](#stdarrayt-n-array)
    - [`std::string` (`<string>`)](#stdstring-string)
    - [Boucles et lambdas](#boucles-et-lambdas)
    - [Nombres aléatoires (`<random>`)](#nombres-aléatoires-random)
    - [Mesure du temps (`<chrono>`)](#mesure-du-temps-chrono)
    - [Fichiers binaires (`<fstream>`)](#fichiers-binaires-fstream)
    - [Affichage](#affichage)
    - [Conversions explicites](#conversions-explicites)
  - [8. Erreurs et exceptions](#8-erreurs-et-exceptions)
  - [9. Bibliothèques externes : yaml-cpp](#9-bibliothèques-externes--yaml-cpp)
  - [10. CMake, juste ce qu'il faut](#10-cmake-juste-ce-quil-faut)
  - [11. Déboguer du C++](#11-déboguer-du-c)
    - [Avertissements du compilateur](#avertissements-du-compilateur)
    - [Sanitizers (sur le chemin CPU)](#sanitizers-sur-le-chemin-cpu)
    - [gdb](#gdb)
    - [`printf` stratégique](#printf-stratégique)
  - [12. Pièges classiques](#12-pièges-classiques)
  - [13. Cheat sheet](#13-cheat-sheet)
  - [14. Pour aller plus loin](#14-pour-aller-plus-loin)

---

## 0. Ce qu'il faut savoir, ce qu'on peut ignorer

Le C++ est immense. Le code de ce projet n'en utilise qu'une petite partie, proche du C : des `struct` de nombres, des pointeurs `float*`, des boucles et des fonctions mathématiques. Côté GPU, c'est encore plus restreint (pas de `std::vector`, pas d'exceptions, pas d'héritage virtuel).

| À maîtriser | À savoir lire | Ignorable ici |
|---|---|---|
| références `&`, `const` | `std::array`, `auto`, boucle `for (x : v)` | héritage, `virtual`, polymorphisme |
| `struct`, `class`, constructeurs | lambdas | `std::unique_ptr`, `std::shared_ptr` |
| RAII, destructeur, `= delete` | déplacement (`&&`, `std::move`) | surcharge d'opérateurs |
| `template <class T>`, `constexpr`, `if constexpr` | `namespace` | `iostream` (`std::cout`) : `printf` suffit |
| `std::vector`, `std::string` | `<random>`, `<chrono>` | algorithmes `<algorithm>`, itérateurs |
| `enum class`, `static_cast` | exceptions | `concepts`, modules, coroutines |
| en-têtes / `.cpp` / édition de liens | yaml-cpp | métaprogrammation avancée |

Ta base en C couvre déjà l'essentiel du code device. Ce qui est vraiment nouveau, c'est la gestion des ressources (§5) et les templates (§6).

---

## 1. Le C++ vu depuis le C et Java

### Depuis le C : ce qui reste, ce qui change

Presque tout le C est du C++ valide : pointeurs, tableaux, `printf`, `struct`, `sinf`, `malloc`. Ce qui change au quotidien :

| C | C++ | Pourquoi |
|---|---|---|
| `#include <stdio.h>` | `#include <cstdio>` puis `std::printf` | versions C++ des en-têtes C, dans le namespace `std` |
| `typedef struct { ... } P;` | `struct P { ... };` | le nom de la struct est directement un type |
| `NULL` | `nullptr` | typé, pas confondu avec l'entier 0 |
| `int` pour un booléen | `bool`, `true`, `false` | |
| `(int)x` | `static_cast<int>(x)` | conversion explicite, facile à chercher |
| `void f(P* p)` pour éviter une copie | `void f(const P& p)` | référence, voir §3 |
| `malloc` / `free` | `std::vector<float> v(n);` | libéré automatiquement (§5) |
| `#define N 6` | `constexpr int N = 6;` | typé, respecte les portées |
| `f_float`, `f_double` | surcharge : deux `f` de signatures différentes | |
| pas de valeur par défaut | `void f(int a, int b = 3);` | défaut dans la **déclaration** seulement |

### Depuis Java : les différences qui piègent

| Java | C++ | Conséquence |
|---|---|---|
| les objets sont des références | les objets sont des **valeurs** | `Foo a = b;` **copie** tout `b` |
| `new Foo()` partout | `Foo f;` ou `Foo f(args);` (sur la pile) | `new` est rare ; ne l'utilise pas ici |
| ramasse-miettes | destructeur appelé à la fin de la portée | déterministe, voir §5 |
| `s1.equals(s2)` | `s1 == s2` compare le **contenu** des `std::string` | |
| `final int x = 3;` | `const int x = 3;` | `const` va aussi sur les paramètres et les méthodes |
| `package` | `namespace mppi { ... }` | |
| `import` | `#include` | copie textuelle, pas un système de modules |
| génériques (effacement de type) | templates (une version compilée par type) | plus rapide, mais erreurs de compilation verbeuses |
| tableaux avec vérification de bornes | `v[i]` **sans** vérification | dépassement = comportement indéfini |
| `int` toujours 32 bits, débordement défini | `int` 32 bits ici, débordement **signé** indéfini | |
| un fichier = une classe | déclaration dans `.hpp`, définition dans `.cpp` | §2 |
| exceptions vérifiées (`throws`) | aucune vérification à la compilation | |

L'idée la plus importante : **en C++, une variable *est* l'objet**, ce n'est pas un pointeur vers lui. Passer un `std::vector` par valeur à une fonction copie tout son contenu.

---

## 2. Fichiers, compilation, édition de liens

### Les quatre extensions

| Extension | Contenu | Compilé par |
|---|---|---|
| `.hpp` | **déclarations** C++ : structs, prototypes, templates, fonctions `inline` | celui qui l'inclut |
| `.cpp` | **définitions** C++ : le corps des fonctions | `g++` |
| `.cuh` | en-tête qui contient du code device | celui qui l'inclut, donc `nvcc` |
| `.cu` | source avec kernels | `nvcc` |

### Déclaration et définition

```cpp
// config.hpp : la déclaration, "cette fonction existe"
#pragma once                                  // n'inclure qu'une fois par fichier compilé
#include <string>

namespace mppi {
struct Config { int K; int T; float lambda; };
Config load_config(const std::string& path);
}

// config.cpp : la définition, le corps
#include "mppi/config.hpp"

namespace mppi {
Config load_config(const std::string& path) {
    Config c{};
    // ...
    return c;
}
}
```

Chaque `.cpp`/`.cu` est compilé **séparément** en un fichier objet `.o`. Le compilateur ne voit que ce fichier et les en-têtes qu'il inclut. L'éditeur de liens (`ld`) assemble ensuite les `.o` et relie chaque appel à sa définition.

```
config.hpp ─┐
            ├─► config.cpp ──g++──► config.o ──┐
            └─► sim.cpp    ──g++──► sim.o    ──┼──ld──► mppi_sim
mppi_kernel.cu ─────────────nvcc──► kernel.o ──┘
```

**Règle de la définition unique (ODR).** Une fonction ordinaire doit être définie dans **un seul** `.cpp`. Si tu mets son corps dans un `.hpp` inclus par deux `.cpp`, l'édition de liens échoue avec `multiple definition of ...`. Il y a deux exceptions, qu'on peut définir entièrement dans l'en-tête :
- les fonctions `inline` (c'est le cas de toutes les fonctions `HD` de `dynamics.cuh` et `cost.cuh`) ;
- les templates, car le compilateur doit voir leur corps pour les instancier.

**Lire les erreurs de liens :**

| Message | Cause |
|---|---|
| `undefined reference to mppi::load_config(...)` | déclarée mais jamais définie, ou `.cpp` absent de `CMakeLists.txt` |
| `multiple definition of ...` | corps d'une fonction non `inline` dans un en-tête |
| `undefined reference to cudaMalloc` | cible non liée à `CUDA::cudart` |
| `undefined reference to YAML::LoadFile` | cible non liée à `yaml-cpp` |

### `namespace`

```cpp
namespace mppi {
    float wrap(float ds, float L);
}

mppi::wrap(1.0f, 48.3f);     // nom complet
using mppi::wrap;            // dans un .cpp, pour raccourcir
```

Ne mets jamais `using namespace std;` dans un en-tête : tous ceux qui l'incluent en héritent.

---

## 3. Types, `const`, références

### Types numériques

| Type | Taille ici | Usage dans le projet |
|---|---|---|
| `float` | 32 bits | **tout** le calcul (CPU et GPU) |
| `double` | 64 bits | accumulations délicates côté hôte (`w_` dans `Controller`) |
| `int` | 32 bits | indices, K, T |
| `long` / `std::int64_t` | 64 bits | indice calculé qui peut dépasser 2³¹ |
| `std::size_t` | 64 bits, **non signé** | tailles (`v.size()`), octets |
| `std::uint32_t`, `std::uint16_t` | exacts | en-têtes binaires (`costmap.bin`, `.npy`) |

Les types de taille exacte sont dans `<cstdint>`.

**Littéraux.** `0.5` est un `double`, `0.5f` est un `float`. Dans du calcul en `float`, un littéral sans `f` fait passer toute l'expression en double, ce qui est très lent sur GPU (voir `cuda.md` §6.2).

**`auto`.** Le compilateur déduit le type. Utile pour les noms longs, à éviter quand le type importe :

```cpp
auto t0 = std::chrono::steady_clock::now();   // bien : le type exact est illisible
auto x = 0.5;                                 // piège : double, pas float
```

### `const`

`const` signifie « je ne modifierai pas ça ». Mets-le partout où c'est possible : le compilateur vérifie, et le lecteur sait ce qui peut changer.

```cpp
const float dt = p.dt;                  // variable non modifiable
void f(const float* x);                 // f ne modifie pas ce que x pointe
float* const p = buf;                   // le pointeur lui-même est fixe (rare)
float size() const;                     // méthode qui ne modifie pas l'objet (§4)
```

Lecture d'un type pointeur : de droite à gauche. `const float* x` se lit « x est un pointeur vers un float constant ».

### Références

Une référence est un **alias** d'une variable existante. Elle s'utilise comme la variable elle-même (pas de `*`, pas de `->`), ne peut pas être nulle et ne peut pas être ré-attachée.

```cpp
void scale(float& x) { x *= 2.0f; }          // modifie la variable de l'appelant
float a = 3.0f;
scale(a);                                    // a == 6.0f, pas de &a à écrire
```

Trois façons de passer un argument :

| Signature | Copie ? | Modifiable ? | Quand |
|---|---|---|---|
| `void f(Params p)` | oui | non (copie locale) | petits types : `int`, `float`, petites structs ; **arguments de kernel** |
| `void f(const Params& p)` | non | non | tout objet plus gros qu'un pointeur côté hôte, `std::string`, `std::vector` |
| `void f(Params& p)` | non | oui | paramètre de sortie |

Dans le projet, `lookup(tr, x, y, d, s)` renvoie `d` et `s` par référence (`float& d, float& s`), là où le C aurait pris des pointeurs.

Côté kernel : un argument de `__global__` est toujours passé **par valeur**. La référence d'une fonction `HD` n'existe que dans la mémoire du processeur qui l'exécute.

---

## 4. `struct` et `class`

`struct` et `class` sont la même chose, à une différence près : les membres sont publics par défaut dans une `struct`, privés dans une `class`. Convention du projet :
- `struct` pour les agrégats de données sans invariant (`Params`, `TrackView`, `VehicleParams`) ;
- `class` pour les objets qui gèrent une ressource ou un état interne (`Controller`, `DeviceBuffer`).

```cpp
struct Params {
    int   K = 8192;            // valeur par défaut du membre
    int   T = 50;
    float dt = 0.05f;
    float noise_std[2] = {0.5f, 0.3f};
};

Params p{};                    // {} : chaque membre à sa valeur par défaut (ou 0)
Params q{1024, 30};            // initialisation dans l'ordre des membres
```

Sans `{}` ni valeur par défaut, un membre `int`/`float` contient **n'importe quoi**, comme en C.

### Une classe complète

```cpp
class Controller {
public:
    explicit Controller(const Config& cfg);                     // constructeur
    std::array<float, 2> command(const float* x0, Diagnostics* diag = nullptr);
    int horizon() const { return p_.T; }                        // const : ne modifie rien

private:
    Params p_;                                                  // suffixe _ = membre privé
    std::vector<float> U_, eps_, S_;
    std::mt19937_64 rng_;
};

// controller.cpp
Controller::Controller(const Config& cfg)
    : p_(cfg.params),                                           // liste d'initialisation
      U_(cfg.params.T * 2, 0.0f),
      rng_(cfg.seed)
{
    // corps : ce qui ne tient pas dans la liste
}
```

- `Controller::` dans le `.cpp` indique à quelle classe appartient la méthode.
- **La liste d'initialisation** (`: p_(...), U_(...)`) construit les membres directement. Elle est obligatoire pour les membres `const`, les références et les objets sans constructeur par défaut. Les membres sont construits dans l'**ordre de leur déclaration** dans la classe, pas dans l'ordre de la liste (`-Wall` prévient si les deux diffèrent).
- **`explicit`** interdit la conversion implicite `Controller c = cfg;`. À mettre sur tout constructeur à un argument.
- **`static`** dans une struct : méthode sans objet, appelée `Dynamic::step(...)`. C'est ainsi que sont écrits les modèles de `dynamics.cuh`.

### `enum class`

```cpp
enum class TireModel : int { Linear = 0, Tanh = 1, Pacejka = 2 };

TireModel m = TireModel::Pacejka;            // nom qualifié obligatoire
if (m == TireModel::Tanh) { ... }
int code = static_cast<int>(m);              // pas de conversion implicite vers int
```

Contrairement à l'`enum` du C, les noms ne fuient pas dans la portée englobante et la conversion vers `int` doit être explicite. Le projet s'en sert pour remplacer les chaînes du YAML (`tire_model: pacejka`) dans `Params`, qui ne peut contenir que des nombres pour être passé au kernel.

---

## 5. RAII : constructeur, destructeur, copie, déplacement

C'est **la** notion du C++ sans équivalent en C ni en Java.

### Le principe

**RAII** (*Resource Acquisition Is Initialization*) : une ressource (mémoire, fichier, buffer GPU) est acquise dans le constructeur d'un objet et libérée dans son **destructeur**. Le destructeur est appelé automatiquement quand l'objet sort de sa portée, y compris si une exception est levée.

```cpp
{
    std::vector<float> v(1000);      // malloc fait par le constructeur
    std::ifstream in("track.bin");   // fopen
    // ...
}                                    // destructeurs : free et fclose, dans l'ordre inverse
```

Conséquence : dans du C++ bien écrit, il n'y a ni `free`, ni `delete`, ni `fclose`, ni `cudaFree` dans le code métier. Ce n'est pas un ramasse-miettes : la libération est immédiate et l'instant où elle a lieu est connu.

### `DeviceBuffer<T>` : RAII pour la mémoire GPU

C'est la classe de `cuda/include/mppi/cuda_check.hpp` à l'étape 3 :

```cpp
template <class T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t n) : n_(n) {
        CUDA_CHECK(cudaMalloc(&ptr_, n * sizeof(T)));
    }
    ~DeviceBuffer() { if (ptr_) cudaFree(ptr_); }                 // destructeur

    // Interdire la copie : deux objets libéreraient le même pointeur.
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    // Autoriser le déplacement : on vole le pointeur de l'autre.
    DeviceBuffer(DeviceBuffer&& o) noexcept : ptr_(o.ptr_), n_(o.n_) {
        o.ptr_ = nullptr; o.n_ = 0;
    }
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        if (this != &o) {
            if (ptr_) cudaFree(ptr_);
            ptr_ = o.ptr_; n_ = o.n_;
            o.ptr_ = nullptr; o.n_ = 0;
        }
        return *this;
    }

    T* get() const { return ptr_; }
    std::size_t size() const { return n_; }

private:
    T* ptr_ = nullptr;
    std::size_t n_ = 0;
};
```

Utilisation :

```cpp
DeviceBuffer<float> d_eps(K * T * 2);                     // cudaMalloc
CUDA_CHECK(cudaMemcpy(d_eps.get(), eps.data(), d_eps.size() * sizeof(float),
                      cudaMemcpyHostToDevice));
kernel<<<grid, block>>>(d_eps.get(), ...);                // le kernel reçoit le pointeur brut
// pas de cudaFree : fait par le destructeur
```

### Copie et déplacement

- **Copie** (`const T&`) : duplique le contenu. `std::vector` sait se copier (il alloue et recopie). `DeviceBuffer` refuse (`= delete`) : copier un pointeur GPU sans recopier les données mènerait à deux `cudaFree` du même pointeur.
- **Déplacement** (`T&&`, « rvalue reference ») : transfère la ressource d'un objet qui va disparaître vers un autre, sans copie. L'objet source reste valide mais vide.

```cpp
DeviceBuffer<float> a(100);
DeviceBuffer<float> b = a;              // erreur de compilation : copie supprimée
DeviceBuffer<float> c = std::move(a);   // OK : c possède le buffer, a est vide
d_S_ = DeviceBuffer<float>(K);          // OK : affectation par déplacement depuis un temporaire
```

`std::move` ne déplace rien lui-même : il autorise le déplacement en traitant `a` comme un objet jetable. N'utilise plus `a` ensuite (sauf pour lui réaffecter une valeur).

**Règle des cinq / règle du zéro.** Si une classe a besoin d'un destructeur personnalisé, elle a aussi besoin de décider de la copie et du déplacement (les cinq fonctions ci-dessus). Mieux : la plupart des classes n'en écrivent **aucune** et se contentent de membres qui gèrent eux-mêmes leurs ressources. `Controller` n'a pas de destructeur : ses `std::vector` et `DeviceBuffer` se libèrent tout seuls.

---

## 6. Templates et `constexpr`

### Templates

Un template est un **patron de code** : le compilateur génère une version distincte pour chaque type ou valeur utilisé. Rien n'est vérifié avant l'instanciation, et tout se passe à la compilation.

```cpp
template <class T>
T clamp(T x, T lo, T hi) { return x < lo ? lo : (x > hi ? hi : x); }

clamp(1.5f, 0.0f, 1.0f);      // génère clamp<float>
clamp(7, 0, 5);               // génère clamp<int>
clamp(1.5f, 0, 1);            // erreur : T ne peut pas être à la fois float et int
```

Différences avec les génériques Java :
- un paramètre peut être une **valeur** connue à la compilation : `template <int N> struct Vec { float v[N]; };` ;
- le code n'est pas partagé : `Vec<4>` et `Vec<6>` sont deux types sans rapport ;
- pas d'interface à déclarer : si `Model::step` existe, ça compile (« duck typing » à la compilation) ;
- le corps doit être visible là où on l'utilise, donc dans l'en-tête.

### Le modèle comme paramètre de template (étape 3)

```cpp
struct Kinematic { static constexpr int DIM = 4; HD static void step(...); };
struct Dynamic   { static constexpr int DIM = 6; HD static void step(...); };

template <class Model>
HD float rollout_cost(int k, const float* x0, /* ... */ const Params& p) {
    float x[Model::DIM];                          // taille connue à la compilation
    for (int i = 0; i < Model::DIM; ++i) x[i] = x0[i];
    for (int t = 0; t < p.T; ++t) {
        Model::step(x, /* ... */);
        if constexpr (Model::DIM == Kinematic::DIM) {
            // compilé uniquement dans rollout_cost<Kinematic>
        }
    }
    // ...
}

// Choix à l'exécution, une fois, côté hôte :
if (p.model == ModelKind::Dynamic) launch<Dynamic>(...);
else                               launch<Kinematic>(...);
```

Pourquoi c'est mieux qu'un `if (state_dim == 6)` dans le kernel : voir la réponse S-C1 de `step_3.md`.

### `constexpr`

- `constexpr int DIM = 6;` : une constante connue à la compilation, utilisable comme taille de tableau ou paramètre de template. Plus sûre qu'un `#define`.
- `static constexpr` dans une struct : constante attachée au type (`Dynamic::DIM`).
- `if constexpr (cond)` : la branche fausse n'est **pas compilée** du tout. Elle n'a même pas besoin d'être valide pour ce type.
- `constexpr` sur une fonction : elle *peut* être évaluée à la compilation si ses arguments sont constants.

### Lire une erreur de template

Les erreurs de template font facilement 50 lignes. Méthode :
1. Cherche la **première** ligne `error:`. Les suivantes en découlent souvent.
2. Remonte les lignes `required from here` jusqu'à **ton** fichier : c'est là qu'est le vrai problème.
3. Le message intéressant est souvent du type `no member named 'step' in 'Foo'` ou `no matching function for call to ...` suivi de `candidate: ...` (les signatures possibles, à comparer avec ton appel).

---

## 7. La bibliothèque standard utile au projet

### `std::vector<T>` (`<vector>`)

Le tableau dynamique. Son contenu est **contigu** en mémoire, comme un tableau C, ce qui permet de le passer à `cudaMemcpy` ou à une fonction qui attend un `float*`.

```cpp
std::vector<float> v(n);              // n zéros
std::vector<float> w(n, 1.0f);        // n fois 1.0f
v.size();                             // nombre d'éléments (std::size_t)
v.data();                             // float* vers le premier élément
v[i];                                 // sans vérification de bornes
v.at(i);                              // avec vérification (exception) : utile en débogage
v.resize(m);                          // change la taille (peut réallouer)
v.push_back(x);                       // ajoute à la fin (peut réallouer)
v.assign(n, 0.0f);                    // remet n zéros sans changer de type
std::fill(v.begin(), v.end(), 0.0f);  // remet à zéro sans réallouer
```

**Attention :** `resize` et `push_back` peuvent réallouer, ce qui invalide tous les pointeurs obtenus avec `data()` avant. Dans `Controller`, tout est dimensionné dans le constructeur et plus jamais redimensionné : aucune allocation pendant `command` (règle de l'étape 3).

### `std::array<T, N>` (`<array>`)

Un tableau C de taille fixe, mais qui se copie et se renvoie comme une valeur. LA ON FIXE VRAIMENT LE NOMBRE D'ELEMENTS

```cpp
std::array<float, 2> command(...);    // renvoie deux floats, sans pointeur de sortie
auto u = ctrl.command(x0);
float a = u[0], delta = u[1];
```

### `std::string` (`<string>`)

```cpp
std::string s = "pacejka";
if (s == "tanh") { ... }              // compare le contenu
s + "_v2";                            // concaténation
s.c_str();                            // const char* pour les fonctions C (printf("%s"))
std::to_string(42);                   // "42"
std::stof("0.05"), std::stoi("50");   // conversions (exception si invalide)
auto pos = s.find('=');               // std::string::npos si absent
s.substr(0, pos);                     // découpage
```

### Boucles et lambdas

```cpp
for (std::size_t i = 0; i < v.size(); ++i) { ... }     // classique
for (float x : v) { ... }                              // copie chaque élément
for (const auto& s : names) { ... }                    // sans copie (objets)
for (auto& x : v) x *= 2.0f;                           // modification en place
```

Une lambda est une fonction anonyme qui peut capturer des variables locales :

```cpp
auto sq = [](float x) { return x * x; };
float m = 0.0f;
auto add = [&m](float x) { m += x; };     // [&m] : capture par référence
std::sort(idx.begin(), idx.end(), [&](int a, int b) { return S[a] < S[b]; });
```

### Nombres aléatoires (`<random>`)

```cpp
std::mt19937_64 rng(seed);                           // générateur (état ~2,5 Ko)
std::normal_distribution<float> normal(0.0f, 1.0f);  // N(0, 1)
for (auto& e : eps) e = normal(rng);
```

Le générateur et la distribution sont des **objets** que l'on garde comme membres (`rng_`, `normal_`) : les recréer à chaque appel réinitialiserait la séquence. Les nombres ne sont pas les mêmes que ceux de NumPy pour une même graine : c'est attendu (voir `step_3.md`).

### Mesure du temps (`<chrono>`)

```cpp
auto t0 = std::chrono::steady_clock::now();
// ...
float ms = std::chrono::duration<float, std::milli>(
               std::chrono::steady_clock::now() - t0).count();
```

`steady_clock` est monotone. N'utilise pas `system_clock` pour mesurer une durée : l'heure système peut sauter. Pour un kernel, utilise plutôt les événements CUDA (`cuda.md` §9).

### Fichiers binaires (`<fstream>`)

```cpp
std::ifstream in(path, std::ios::binary);
if (!in) throw std::runtime_error("cannot open " + path);

char magic[4];
in.read(magic, 4);
std::int32_t nx;
in.read(reinterpret_cast<char*>(&nx), sizeof(nx));

std::vector<float> grid(static_cast<std::size_t>(nx) * ny * 2);
in.read(reinterpret_cast<char*>(grid.data()), grid.size() * sizeof(float));
if (!in) throw std::runtime_error("truncated file " + path);
```

C'est l'équivalent de `fopen` + `fread`. Le fichier est fermé par le destructeur de `in`. `reinterpret_cast` réinterprète les octets sans conversion ; c'est légitime ici parce que Python a écrit des `float32` en little-endian, le format natif de la machine.

### Affichage

```cpp
std::printf("K=%d lambda=%.3f model=%s\n", p.K, p.lambda, name.c_str());
std::fprintf(stderr, "warning: ...\n");
```

`printf` fonctionne aussi dans un kernel. `std::cout << x` existe mais n'apporte rien ici.

### Conversions explicites

| Cast | Usage |
|---|---|
| `static_cast<int>(x)` | conversions numériques, `enum` ↔ `int` : le cas courant |
| `reinterpret_cast<char*>(p)` | voir des octets bruts (lecture/écriture binaire) |
| `const_cast` | retirer un `const` : presque jamais légitime |
| `(int)x` | cast C : marche, mais cache lequel des trois on fait |

`static_cast<long>(k) * T` est important dans le calcul d'indices : la conversion se fait **avant** la multiplication, donc le produit est calculé sur 64 bits.

---

## 8. Erreurs et exceptions

**Côté hôte**, une erreur de configuration ou de fichier lève une exception :

```cpp
#include <stdexcept>

if (!node["mppi"]["lambda"]) throw std::runtime_error("missing key mppi.lambda");

int main(int argc, char** argv) {
    try {
        auto cfg = mppi::load_config(argv[1]);
        // ...
    } catch (const std::exception& e) {        // toujours par référence const
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
```

Une exception non attrapée termine le programme avec un message peu lisible (`terminate called after throwing...`). Le `try/catch` dans `main` donne un message propre.

**Côté device**, les exceptions n'existent pas. On vérifie le code de retour de chaque appel CUDA avec `CUDA_CHECK` (`cuda.md` §8), et un kernel ne signale rien : il écrit un résultat, éventuellement `NaN`.

**Les assertions** (`<cassert>`) vérifient un invariant en Debug et disparaissent en Release (`-DNDEBUG`, ajouté par CMake) :

```cpp
assert(eps.size() == static_cast<std::size_t>(K) * T * 2);
```

---

## 9. Bibliothèques externes : yaml-cpp

Une bibliothèque C++ se compose de deux parties :
- des **en-têtes** (`#include <yaml-cpp/yaml.h>`) : les déclarations, pour le compilateur ;
- une **bibliothèque compilée** (`libyaml-cpp.so`) : les définitions, pour l'éditeur de liens.

pixi installe les deux dans `.pixi/envs/default/include` et `.pixi/envs/default/lib`. CMake les trouve avec `find_package` (§10).

Utilisation dans `config.cpp` :

```cpp
#include <yaml-cpp/yaml.h>

YAML::Node raw = YAML::LoadFile(path);                    // exception si fichier absent
int K = raw["mppi"]["num_samples"].as<int>();
std::string tire = raw["vehicle"]["tire_model"].as<std::string>();
YAML::Node std_ = raw["mppi"]["noise_std"];               // une séquence
for (std::size_t i = 0; i < std_.size(); ++i) p.noise_std[i] = std_[i].as<float>();
if (!raw["mppi"]["lambda"]) throw std::runtime_error("missing key mppi.lambda");
```

`.as<T>()` lève `YAML::BadConversion` si la valeur n'a pas le bon type, et `YAML::InvalidNode` si la clé n'existe pas. Le piège de `node = node["x"]` est décrit dans `step_3.md` (partie B).

---

## 10. CMake, juste ce qu'il faut

CMake ne compile pas : il **génère** les commandes de compilation pour Ninja (ou Make). `pixi run build` fait les deux étapes : `configure` (cmake) puis `build` (ninja).

```cmake
cmake_minimum_required(VERSION 3.24)
project(edge_mppi LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 89)

find_package(yaml-cpp REQUIRED)            # cherche la bibliothèque installée par pixi
find_package(CUDAToolkit REQUIRED)         # fournit la cible CUDA::cudart

# Une bibliothèque statique : le code partagé par tous les exécutables
add_library(mppi_core STATIC
  cuda/src/config.cpp
  cuda/src/track.cpp
  cuda/src/controller.cpp
  cuda/src/mppi_kernel.cu)
target_include_directories(mppi_core PUBLIC ${CMAKE_SOURCE_DIR}/cuda/include)
target_link_libraries(mppi_core PUBLIC yaml-cpp::yaml-cpp CUDA::cudart)
target_compile_options(mppi_core PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>)

# Des exécutables qui utilisent la bibliothèque
add_executable(mppi_sim cuda/apps/sim.cpp)
target_link_libraries(mppi_sim PRIVATE mppi_core)
```

| Commande | Effet |
|---|---|
| `add_library(nom STATIC fichiers...)` | compile des fichiers en une bibliothèque `libnom.a` |
| `add_executable(nom fichiers...)` | compile et lie un exécutable |
| `target_include_directories(cible PUBLIC dir)` | ajoute `-I dir` ; `PUBLIC` le transmet aux cibles qui dépendent de celle-ci |
| `target_link_libraries(cible PUBLIC lib)` | lie `lib` et récupère ses en-têtes |
| `target_compile_options(cible PRIVATE ...)` | ajoute des flags de compilation |
| `find_package(X REQUIRED)` | trouve une bibliothèque installée, échoue sinon |

`PRIVATE` / `PUBLIC` : `PRIVATE` ne concerne que la cible elle-même. `PUBLIC` se propage aux cibles qui la lient. Un en-tête de `mppi_core` inclut `yaml-cpp` ? Alors `PUBLIC`. Seuls ses `.cpp` l'incluent ? `PRIVATE` suffit.

**Type de build :**

```bash
cmake -S . -B build       -G Ninja -DCMAKE_BUILD_TYPE=Release   # -O3 -DNDEBUG : pour mesurer
cmake -S . -B build-debug -G Ninja -DCMAKE_BUILD_TYPE=Debug     # -g -O0 : pour déboguer
cmake --build build-debug
```

Deux dossiers de build distincts permettent d'avoir les deux versions en parallèle. Après une modification de `CMakeLists.txt`, CMake se relance tout seul à la compilation suivante. En cas de comportement étrange, `pixi run clean` puis `pixi run build`.

`CMAKE_EXPORT_COMPILE_COMMANDS ON` (déjà dans ton `CMakeLists.txt`) écrit `build/compile_commands.json`. L'extension C/C++ de VS Code (ou clangd) le lit pour savoir avec quels flags compiler chaque fichier : sans lui, l'éditeur souligne en rouge des `#include` qui sont pourtant corrects.

---

## 11. Déboguer du C++

### Avertissements du compilateur

`-Wall -Wextra` signale une bonne partie des bugs avant l'exécution : variable non initialisée, comparaison signé/non signé, ordre d'initialisation, paramètre inutilisé. Le critère de sortie de l'étape 3 exige un build **sans avertissement** : traite-les comme des erreurs.

### Sanitizers (sur le chemin CPU)

Le chemin CPU du projet (`Backend::Cpu`) tourne sans GPU, donc aussi sur ta machine WSL2. Pour lui, GCC fournit deux détecteurs à l'exécution :

```bash
cmake -S . -B build-asan -G Ninja -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer"
```

- **AddressSanitizer** : dépassement de tableau, utilisation après libération, fuites. Il affiche la ligne exacte de l'accès fautif et celle de l'allocation.
- **UndefinedBehaviorSanitizer** : débordement d'entier signé, décalage invalide, division entière par zéro.

C'est l'équivalent CPU de `compute-sanitizer`. ASan cohabite mal avec le runtime CUDA. Si un exécutable lié à CUDA plante au démarrage sous ASan, lance-le avec `ASAN_OPTIONS=protect_shadow_gap=0`, ou n'utilise ASan que pour les tests du chemin CPU.

### gdb

Build Debug obligatoire. Lancement : `pixi run gdb --args ./build-debug/mppi_parity --dir results/parity/case0 --backend cpu`.

| Commande | Effet |
|---|---|
| `run` (`r`) | lance le programme |
| `break config.cpp:42` (`b`) | point d'arrêt à une ligne |
| `break mppi::load_config` | point d'arrêt à l'entrée d'une fonction |
| `next` (`n`) / `step` (`s`) | ligne suivante / entrer dans la fonction |
| `continue` (`c`) | jusqu'au prochain point d'arrêt |
| `print x` (`p`) | affiche une variable ; `p *ptr@10` affiche 10 éléments |
| `p v` | affiche un `std::vector` lisiblement |
| `backtrace` (`bt`) | pile d'appels : **la** commande après un crash |
| `frame 2` (`f`) | se placer dans un appelant |
| `info locals` | toutes les variables locales |
| `watch x` | s'arrête quand `x` change |

Après un `Segmentation fault`, il suffit de relancer sous gdb et de taper `bt` : tu vois où et par quel chemin le programme est arrivé là. `cuda-gdb` a les mêmes commandes, plus celles du GPU (`cuda.md` §8).

VS Code peut piloter gdb avec une interface graphique (extension C/C++, configuration `launch.json` de type `cppdbg`, en indiquant comme `miDebuggerPath` le `gdb` de `.pixi/envs/default/bin`).

### `printf` stratégique

Souvent plus rapide que gdb pour comparer avec Python : afficher les mêmes valeurs aux mêmes endroits des deux côtés (`%.9g` pour un `float` sans perte de précision) et chercher le premier pas où elles divergent.

---

## 12. Pièges classiques

**Comportement indéfini (UB).** Comme en C, certaines erreurs ne sont pas des erreurs : le programme peut faire n'importe quoi, y compris marcher en Debug et échouer en Release. Les principales :
- lire au-delà d'un tableau (`v[v.size()]`) ;
- lire une variable non initialisée (`float x; x += 1.0f;`) ;
- déborder un entier **signé** (`int i = INT_MAX; i + 1`) ;
- utiliser une référence ou un pointeur vers un objet détruit.

**Copie involontaire.**

```cpp
void f(std::vector<float> v);            // copie tout le vecteur à chaque appel
void f(const std::vector<float>& v);     // correct
for (auto s : names)                     // copie chaque string
for (const auto& s : names)              // correct
```

**Référence pendante.**

```cpp
const float& first(const std::vector<float>& v) { return v[0]; }
const float& x = first(std::vector<float>{1.0f});   // le vecteur temporaire est détruit : x pend
```

Ne renvoie jamais une référence vers une variable locale ou un temporaire.

**`std::size_t` est non signé.**

```cpp
for (std::size_t i = n - 1; i >= 0; --i)    // boucle infinie : i >= 0 toujours vrai
int k = -1; if (k < v.size())               // -1 converti en 2^64 - 1 : faux
```

Utilise `int` pour les indices qui peuvent devenir négatifs, et `static_cast` pour comparer (`-Wall` prévient).

**Débordement dans un calcul d'indice.** `k * T * 2` avec `int` déborde au-delà de 2³¹ ≈ 2,1 milliards. Pas de risque à K = 8192, T = 50, mais d'où `static_cast<long>(k) * T` dans le code du projet.

**Littéraux et fonctions en `double`.** `0.5` au lieu de `0.5f`, `sin` au lieu de `sinf`, `std::sin(x)` sur un `double` : le calcul passe en double précision. Sur CPU c'est seulement plus lent ; sur GPU, ça peut l'être 64 fois plus (`cuda.md` §6.2).

**Modulo négatif.** `-3 % 10 == -3` et `fmodf(-3, 10) == -3` en C/C++, contre `7` en Python. Voir `wrap` dans `step_3.md` (S-B2).

**Un `std::vector` dans une struct passée au kernel.** Ça compile parfois, mais le GPU reçoit des pointeurs CPU. Les structs envoyées au kernel ne contiennent que des nombres, des `enum` et des pointeurs device.

**Ordre d'initialisation des membres.** Si `U_` est déclaré avant `p_` mais initialisé avec `p_.T`, `p_` n'est pas encore construit au moment où `U_` l'est. `-Wall` le signale (`-Wreorder`).

---

## 13. Cheat sheet

```cpp
// ---------- en-tête type ----------
#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace mppi {

enum class Backend { Cpu, Gpu };

struct Params {                    // agrégat, copiable, passable au kernel
    int K = 0, T = 0;
    float dt = 0.0f, lambda = 1.0f;
};

class Controller {
public:
    explicit Controller(const Params& p, Backend b);
    std::array<float, 2> command(const float* x0);
    const Params& params() const { return p_; }
private:
    Params p_;
    Backend backend_;
    std::vector<float> U_;
};

template <class T>
inline T sq(T x) { return x * x; }

}  // namespace mppi
```

| Je veux | J'écris |
|---|---|
| passer un objet sans copie, en lecture | `const T& x` |
| un paramètre de sortie | `T& out` |
| un tableau dynamique | `std::vector<float> v(n);` puis `v.data()` |
| renvoyer deux floats | `std::array<float, 2>` |
| une constante de compilation | `constexpr int N = 6;` |
| choisir un type à la compilation | `template <class Model>` + `if constexpr` |
| convertir | `static_cast<int>(x)` |
| un pointeur nul | `nullptr` |
| interdire la copie | `T(const T&) = delete;` |
| signaler une erreur hôte | `throw std::runtime_error("...");` |
| libérer une ressource | rien : un destructeur s'en charge |

```bash
# ---------- terminal ----------
pixi run build                                              # Release
cmake -S . -B build-debug -G Ninja -DCMAKE_BUILD_TYPE=Debug && cmake --build build-debug
pixi run gdb --args ./build-debug/app args                  # puis r, bt, p x
pixi run g++ -std=c++17 -Wall -Wextra -O2 a.cpp -o a        # petit essai hors CMake
pixi run c++filt _ZN4mppi11load_configERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
                                                            # démêle un nom de symbole d'une erreur de liens
```

---

## 14. Pour aller plus loin

- **cppreference.com** : la référence de la bibliothèque standard. Chaque page a des exemples ; regarde surtout la section « Example » et la colonne de version (C++17 ici).
- **A Tour of C++** (Bjarne Stroustrup, 3ᵉ éd.) : 250 pages, le C++ moderne pour quelqu'un qui programme déjà. Les chapitres 1 à 7 couvrent cette fiche.
- **C++ Core Guidelines** (isocpp.github.io) : les bonnes pratiques officielles. Sections R (ressources) et F (fonctions) en priorité.
- **learncpp.com** : cours complet et progressif, pour approfondir un point précis.
- **Compiler Explorer** (godbolt.org) : montre l'assembleur généré par `g++` ou `nvcc` pour un petit bout de code. Pratique pour vérifier qu'une boucle est déroulée ou qu'il n'y a pas de `double` caché.
