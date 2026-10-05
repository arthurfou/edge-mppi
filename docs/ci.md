# Intégration continue : lancer les tests à chaque push

Mini-TP autonome. But : à chaque `git push` sur GitHub, une machine distante installe l'environnement pixi et lance `pixi run test`. Chaque commit affiche alors ✅ ou ❌ sur GitHub.

**Durée estimée.** 30 minutes à 1 heure.

## Sommaire

- [0. Objectif et critère de sortie](#0-objectif-et-critère-de-sortie)
- [1. Les notions](#1-les-notions)
- [2. Préparer le dépôt : le lockfile](#2-préparer-le-dépôt--le-lockfile)
- [3. Écrire le workflow](#3-écrire-le-workflow)
- [4. Pousser et vérifier](#4-pousser-et-vérifier)
- [5. Le problème CUDA](#5-le-problème-cuda)
- [6. Bonus](#6-bonus)
- [7. Dépannage](#7-dépannage)
- [Solutions](#solutions)

---

## 0. Objectif et critère de sortie

- [ ] un fichier `.github/workflows/tests.yml` commité ;
- [ ] `pixi.lock` versionné ;
- [ ] un run vert dans l'onglet **Actions** du dépôt ;
- [ ] un run rouge provoqué volontairement (un test cassé exprès), puis réparé ;
- [ ] (bonus) un badge de statut dans le `README.md`.

**Pourquoi c'est utile pour un projet perso.** Tu vois tout de suite quand une modification casse quelque chose (par exemple dans `dynamics.py`), au lieu de le découvrir trois étapes plus tard. Et pour quelqu'un qui lit le dépôt (un recruteur, par exemple), un dossier `.github/workflows` et un badge vert montrent que tu travailles avec tests, reproductibilité et CI. Ça coûte peu et ça se voit.

**CI ou CD ?** Ici, on ne fait que de la CI (*continuous integration* : vérifier chaque changement). Le CD (*continuous delivery/deployment* : publier automatiquement un package, une doc, une image Docker) n'a pas d'objet pour ce projet pour l'instant.

---

## 1. Les notions

**GitHub Actions.** Le service de CI intégré à GitHub. Il est gratuit et sans limite de minutes pour les dépôts publics. Tu décris ce qu'il faut faire dans un fichier YAML placé dans `.github/workflows/`. GitHub le détecte tout seul, aucune configuration n'est nécessaire sur le site.

**Vocabulaire :**

| Terme | Sens |
|---|---|
| *workflow* | un fichier YAML dans `.github/workflows/` |
| *trigger* (`on:`) | l'événement qui déclenche le workflow : `push`, `pull_request`, un cron, un clic manuel… |
| *job* | un ensemble d'étapes qui s'exécutent sur une même machine virtuelle |
| *runner* (`runs-on:`) | la machine virtuelle, par exemple `ubuntu-latest` |
| *step* | une étape : soit une commande shell (`run:`), soit une *action* réutilisable (`uses:`) |
| *action* | un bloc tout fait publié sur GitHub, par exemple `actions/checkout` |

**Les deux actions dont tu as besoin :**

- `actions/checkout` clone ton dépôt dans le runner. Le runner part de zéro, ton code n'y est pas.
- `prefix-dev/setup-pixi` installe pixi, puis l'environnement décrit par `pixi.toml` et `pixi.lock`. Avec l'option `cache: true`, il garde l'environnement en cache d'un run à l'autre.

**Q1.** Le runner est une machine neuve à chaque run. Qu'est-ce que ça implique pour les fichiers ignorés par git, comme `build/`, `.pixi/` ou les fichiers de `results/` ?

---

## 2. Préparer le dépôt : le lockfile

Aujourd'hui, `pixi.lock` est dans le `.gitignore`. C'est à changer.

**Q2.** Pourquoi le CI a-t-il besoin du lockfile ? Pense à deux raisons : une sur la reproductibilité, une sur le cache.

À faire :

1. retirer la ligne `pixi.lock` du `.gitignore` ;
2. `pixi install` pour être sûr que le lockfile est à jour avec `pixi.toml` ;
3. `git add pixi.lock .gitignore`.

---

## 3. Écrire le workflow

Crée `.github/workflows/tests.yml`. Squelette à compléter :

```yaml
name: tests            # nom affiché dans l'onglet Actions

on:
  # TODO : déclencher sur push vers main, et sur toute pull request

jobs:
  pytest:              # identifiant du job, libre
    runs-on: # TODO
    steps:
      - uses: # TODO : récupérer le code
      - uses: # TODO : installer pixi + l'environnement, avec cache
        with:
          # TODO
      - run: # TODO : la commande de test
```

**Rappels d'API :**

- triggers : `on: { push: { branches: [main] }, pull_request: {} }`, ou la forme multi-lignes équivalente ;
- les actions se référencent avec une version : `actions/checkout@v4`, `prefix-dev/setup-pixi@v0.9.0`. Vérifie la dernière version de `setup-pixi` sur https://github.com/prefix-dev/setup-pixi/releases ;
- `setup-pixi` accepte entre autres `cache: true` et `environments: <nom>` (voir §5).

**Q3.** Pourquoi déclencher aussi sur `pull_request`, alors que tu travailles seul sur `main` ?

**Q4.** `pixi run test` définit déjà `PYTHONPATH=python` dans `pixi.toml`. Faut-il le redéfinir dans le workflow ?

---

## 4. Pousser et vérifier

1. Commit (`.github/workflows/tests.yml`, `pixi.lock`, `.gitignore`, et `python/tests/` s'il n'est pas encore suivi par git), puis push.
2. Sur GitHub, ouvre l'onglet **Actions**. Le run apparaît en quelques secondes. Clique dessus pour voir les logs de chaque step.
3. **Le test du test.** Casse volontairement une assertion, push, et vérifie que le run devient rouge. Remets l'assertion, push, et vérifie qu'il redevient vert. Un CI qu'on n'a jamais vu échouer ne prouve rien.

**Q5.** Tes tests passent chez toi mais échouent en CI. Cite deux causes typiques.

---

## 5. Le problème CUDA

Dans `pixi.toml`, l'environnement `linux-64` tire `cuda-toolkit = "12.*"`, soit plusieurs Go. Or :

- les tests de `python/tests/` sont en pur Python/NumPy ;
- les runners GitHub gratuits n'ont **pas de GPU**. `pixi run bench` et `pixi run check` ne pourront jamais y tourner.

Le CI marchera quand même, mais le premier run sera lent. Les suivants passeront par le cache, s'il n'a pas été évincé.

**Q6.** Comment faire pour que le CI n'installe pas CUDA, sans rien changer à ton environnement de dev ?

Indice : les *features* et *environments* de pixi. Une feature est un paquet de dépendances nommé, un environnement est une combinaison de features. Doc : https://pixi.sh/latest/workspace/multi_environment/

Ne le fais que si le temps de run te gêne vraiment. Ce n'est pas obligatoire.

---

## 6. Bonus

**Badge dans le README.** Ajoute en haut de `README.md` :

```markdown
![tests](https://github.com/arthurfou/edge-mppi/actions/workflows/tests.yml/badge.svg)
```

**Lancement manuel.** Ajoute `workflow_dispatch:` sous `on:`. Un bouton « Run workflow » apparaît alors dans l'onglet Actions.

**Plus tard (étapes C++/CUDA).** Un second job peut vérifier que le code compile (`pixi run build`), même sans GPU : `nvcc` compile sans carte graphique, seule l'exécution en a besoin. C'est un bon garde-fou contre les erreurs de compilation.

---

## 7. Dépannage

| Symptôme | Piste |
|---|---|
| Le workflow n'apparaît pas dans Actions | mauvais chemin (`.github/workflows/`, avec un **s**), extension autre que `.yml`/`.yaml`, ou YAML invalide |
| `setup-pixi` se plaint du lockfile | `pixi.lock` absent du dépôt, ou désynchronisé de `pixi.toml` : relancer `pixi install` puis commiter |
| `ModuleNotFoundError: mppi` | le test est lancé avec `pytest` au lieu de `pixi run test`, donc sans le `PYTHONPATH` |
| Un test lit un fichier introuvable | ce fichier est généré localement (piste, grille) ou ignoré par git, donc absent du runner (voir Q1) |
| Run très long au premier coup | téléchargement de `cuda-toolkit` (voir §5) |

---

## Solutions

**Q1.** Rien de ce qui est ignoré par git n'existe sur le runner : pas de `build/`, pas d'environnement `.pixi/`, pas de fichiers générés. Si un test a besoin d'un fichier produit par `pixi run track` par exemple, il faut soit le générer dans le test (fixture pytest, `tmp_path`), soit lancer la tâche dans le workflow avant les tests. Des tests autonomes valent mieux.

**Q2.** Reproductibilité : sans lockfile, pixi résout les versions au moment du run. Le CI peut alors tester avec un NumPy différent du tien, et un test peut casser sans que ton code ait changé. Cache : `setup-pixi` calcule la clé de cache à partir du contenu de `pixi.lock`. Sans lockfile, pas de cache fiable.

**Q3.** Si un jour tu travailles sur une branche, ou si quelqu'un propose une contribution, le CI vérifie la PR **avant** le merge. Et ça ne coûte rien.

**Q4.** Non. La variable est définie dans la tâche pixi, donc `pixi run test` se comporte pareil partout. C'est justement l'intérêt de passer par des tâches pixi plutôt que par des commandes brutes.

**Q5.** (a) Un fichier présent chez toi mais pas dans git : non commité, ignoré, ou généré. (b) Une version de dépendance différente (lockfile absent ou périmé). Autres causes possibles : un chemin absolu propre à ta machine, un test non déterministe (graine aléatoire non fixée), une différence de plateforme.

**Q6.** Sortir CUDA dans une feature, puis définir deux environnements :

```toml
[feature.cuda.target.linux-64.dependencies]
cuda-toolkit = "12.*"

[environments]
default = ["cuda"]   # ton environnement de dev, inchangé
test = []            # sans CUDA, pour le CI
```

(et supprimer l'ancienne section `[target.linux-64.dependencies]`). Dans le workflow :

```yaml
      - uses: prefix-dev/setup-pixi@v0.9.0
        with:
          cache: true
          environments: test
      - run: pixi run -e test test
```

**Workflow complet (§3) :**

```yaml
name: tests

on:
  push:
    branches: [main]
  pull_request:

jobs:
  pytest:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: prefix-dev/setup-pixi@v0.9.0
        with:
          cache: true

      - run: pixi run test
```
