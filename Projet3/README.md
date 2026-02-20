# Projet 3 : Classification Bayésienne & Analyse Factorielle Discriminante
### Thèses de doctorat françaises — Catégorisation par domaine d'étude
**ESIEE Paris – 2025-2026 – E4 AP-4209**  
**Auteurs :** TORRES Diego, WU Lucas  
**Encadrant :** Badr TAJINI

---

## Table des matières

1. [Vue d'ensemble](#-vue-densemble)
2. [Dataset](#-dataset)
3. [Pipeline complète](#-pipeline-complète)
4. [Structure du projet](#-structure-du-projet)
5. [Installation et dépendances](#-installation-et-dépendances)
6. [Utilisation](#-utilisation)
7. [Résultats](#-résultats)
8. [Méthodes et fondements théoriques](#-méthodes-et-fondements-théoriques)
9. [Limites et travaux futurs](#-limites-et-travaux-futurs)
10. [Références](#-références)

---

## Vue d'ensemble

Ce projet met en place une **classification automatique de thèses de doctorat françaises** en domaines d'étude, à partir de leurs résumés textuels. Il s'agit d'un problème **multi-classes** (10 domaines) sur données textuelles, présentant les défis caractéristiques de ce type de données : haute dimensionnalité, fort déséquilibre des classes et nature sémantique des features.

**Pipeline en cinq étapes :**

```
Résumés textuels (519 578 thèses brutes)
    ↓
Nettoyage + Stopwords + Stemming
    ↓
Vectorisation TF-IDF  →  15 000 docs × 1 711 termes
    +
Modélisation thématique LDA  →  10 topics (matrice γ)
    ↓
AFD (Analyse Factorielle Discriminante)  →  9 axes discriminants
    ↓
Classifieur Naive Bayes (lissage Laplace)
    ↓
Évaluation : Accuracy 76,71 % | F1 0,768 | AUC-ROC 0,972
```

---

## Dataset

**Source :** [French Doctoral Thesis Semantic Similarity Search — Kaggle](https://www.kaggle.com/code/antoinebourgois2/french-doctoral-thesissemantic-similarity-search)

**Fichier :** `french_thesis_20231021_metadata.csv`

### Structure du dataset brut

| Colonne | Description |
|---|---|
| `URL` | Lien vers la thèse |
| `Title` | Titre de la thèse |
| `Author` | Auteur |
| `Description` | Résumé textuel (variable d'entrée principale) |
| `Domain` | Domaine d'étude **(variable cible)** |
| `Direction` | Directeur de thèse |
| `Statuts` | Statut de la thèse |
| `Date` | Date de soutenance |

### Caractéristiques clés

| Indicateur | Valeur |
|---|---|
| Nombre total de thèses | 519 578 |
| Domaines distincts | 27 248 |
| Valeurs manquantes dans `Description` | 20,21 % |
| Ratio déséquilibre max/min | > 22 000 |
| Longueur médiane des résumés | 240 mots |

### Top 10 domaines sélectionnés (après exploration)

| Rang | Domaine | Effectif brut |
|---|---|---|
| 1 | Physique | 16 234 |
| 2 | Informatique | 15 990 |
| 3 | Chimie | 10 519 |
| 4 | Sciences économiques | 9 375 |
| 5 | Sciences appliquées | 8 705 |
| 6 | Sciences biologiques et fondamentales appliquées. Psychologie | 8 044 |
| 7 | Histoire | 7 449 |
| 8 | Sciences de gestion | 6 635 |
| 9 | Sociologie | 6 105 |
| 10 | Droit public | 5 999 |

> **Sous-échantillonnage :** 1 500 thèses par domaine → **15 000 documents** au total pour l'analyse.

### Placement du fichier

```
projet3/
└── data/
    └── french_thesis_20231021_metadata.csv
```

---

## Pipeline complète

### Étape 1 — Nettoyage du texte

- Conversion en minuscules
- Remplacement des caractères accentués (ASCII)
- Suppression de la ponctuation et des chiffres
- Suppression des stopwords français + anglais (termes non discriminants dans les thèses)
- Stemming en français (`SnowballC`)

### Étape 2 — Vectorisation TF-IDF

```
Corpus : 15 000 documents
DTM TF-IDF : 15 000 × 1 711 termes
(termes présents dans au moins 5 documents, longueur ≥ 3 caractères, sparsité ≤ 99 %)
```

### Étape 3 — Modélisation thématique LDA (Latent Dirichlet Allocation)

- K = 10 topics (correspondant au nombre de domaines)
- Algorithme Gibbs Sampling (burnin = 500, iter = 1000)
- Extraction de la matrice γ (distribution des topics par document)
- Combinaison : features TF-IDF + matrice γ → **15 000 × 1 721 features**

**Top 10 mots par topic (résultats obtenus) :**

| Topic | Mots clés représentatifs |
|---|---|
| T1 | etat, droit, public, intern, juridiqu, publiqu, princip, pouvoir, politiqu, administr |
| T2 | processus, entrepris, organis, gestion, acteur, pratiqu, qualit, projet, relat, action |
| T3 | activit, deux, ete, cellul, montr, protein, chez, specifiqu, gene, effet |
| T4 | plus, etr, comm, peut, autr, grand, tout, certain, fait, meme |
| T5 | economiqu, developp, march, pay, effet, entr, product, impact, economi, term |
| T6 | system, problem, donne, utilis, base, applic, algorithm, proposon, inform, reseaux |
| T7 | mesur, phase, temperatur, champ, effet, surfac, energi, propriet, optiqu, etudi |
| T8 | social, entr, politiqu, societ, siecl, franc, anne, vie, espac, rapport |
| T9 | ete, complex, compos, synthes, utilis, permi, reaction, format, structur, propriet |
| T10 | model, deux, present, premier, differ, structur, imag, different, point, ensuit |

### Étape 4 — AFD (Analyse Factorielle Discriminante)

- Implémentée via `MASS::lda()` (cas linéaire)
- Prior uniforme sur les 10 classes
- Tolérance numérique : `tol = 1e-4`
- **9 axes discriminants** produits (K_classes − 1)

### Étape 5 — Classification Naive Bayes

$$P(D_k \mid \mathbf{x}) \propto P(D_k) \prod_{i=1}^{p} P(x_i \mid D_k)$$

- Lissage de Laplace (λ = 1) pour éviter les probabilités nulles
- Implémenté via `e1071::naiveBayes()`
- Entraînement : 10 500 documents | Test : 4 500 documents (split 70/30 stratifié)

---

## Structure du projet

```
projet3/
│
├── data/
│   └── french_thesis_20231021_metadata.csv     # Dataset brut (à télécharger)
│
├── projet3.Rmd                                  # Document principal (code + rapport)
├── projet3.html                                 # Rapport compilé
│
└── README.md                                    # Ce fichier
```

---

## Installation et dépendances

### Prérequis

- **R** ≥ 4.3.0
- **RStudio** (recommandé) ou tout éditeur compatible RMarkdown
- Mémoire RAM recommandée : **≥ 8 Go** (dataset de 519 578 lignes chargé en mémoire)

### Installation des packages

Les packages manquants sont **installés automatiquement** au lancement du Rmd. Pour une installation manuelle préalable, coller dans la console R :

```r
install.packages(c(
  # Manipulation de données
  "readr", "dplyr", "stringr", "tidyr",

  # Visualisation
  "ggplot2", "scales", "RColorBrewer",

  # NLP / Text mining
  "tm", "SnowballC", "tidytext", "topicmodels",

  # Modélisation
  "e1071",       # Naive Bayes
  "MASS",        # LDA discriminant
  "caret",       # Matrice de confusion, validation croisée
  "pROC",        # Courbes ROC / AUC

  # Optionnel (clustering)
  "naivebayes"
))
```

> **Note :** `MASS` est inclus dans R de base et n'a généralement pas besoin d'être installé séparément.

---

## Utilisation

### Lancement du rapport

```r
rmarkdown::render("projet3.Rmd")
```

Ou depuis le terminal Windows :

```powershell
& "C:\Program Files\R\R-4.5.2\bin\Rscript.exe" -e "rmarkdown::render('projet3.Rmd')"
```

### Paramètres clés modifiables

```r
TOP_N_DOMAINS  <- 10     # Nombre de domaines sélectionnés parmi les plus représentés
MAX_PER_DOMAIN <- 1500   # Nombre max de thèses par domaine (sous-échantillonnage)
K              <- 10     # Nombre de topics LDA thématique
```

> **Ajustement mémoire :** Si la RAM est limitée, réduire `MAX_PER_DOMAIN` à 500 ou 800.

### Note sur la compatibilité caret

Les noms de classes sont automatiquement nettoyés via `make.names()` (suppression des accents et caractères spéciaux) pour satisfaire les contraintes de `caret`. Une table de correspondance `label_map` permet de restaurer les noms originaux dans toutes les visualisations.

---

## Résultats

### Métriques globales (ensemble de test — 4 500 documents)

| Métrique | Valeur |
|---|---|
| **Accuracy** | **76,71 %** |
| **F1-Score moyen (macro)** | **0,7680** |
| **AUC-ROC moyen (One-vs-Rest)** | **0,9724** |
| Kappa de Cohen | 0,7412 |

### Validation croisée 5-fold

| Métrique | Valeur |
|---|---|
| Accuracy CV moyenne | **87,59 % ± 0,78 %** |
| Kappa CV moyen | 0,8621 |

> L'écart entre l'accuracy sur l'ensemble de test (76,71 %) et la validation croisée (87,59 %) s'explique par la différence de taille des sous-corpus utilisés. La CV porte sur un sous-ensemble de 6 000 documents pour des raisons de temps de calcul.

### Métriques par domaine (F1-Score, classement décroissant)

| Domaine | Precision | Rappel | F1-Score | Spécificité |
|---|---|---|---|---|
| Droit public | 0,9218 | 0,8911 | **0,9062** | 0,9916 |
| Sc. biologiques & Psychologie | 0,8929 | 0,8156 | **0,8525** | 0,9891 |
| Histoire | 0,8355 | 0,8578 | **0,8465** | 0,9812 |
| Informatique | 0,8000 | 0,7911 | **0,7955** | 0,9780 |
| Sociologie | 0,7946 | 0,7911 | **0,7929** | 0,9773 |
| Sciences de gestion | 0,7770 | 0,7356 | **0,7557** | 0,9765 |
| Chimie | 0,7358 | 0,7489 | **0,7423** | 0,9701 |
| Sciences économiques | 0,7164 | 0,7578 | **0,7365** | 0,9667 |
| Physique | 0,7004 | 0,7533 | **0,7259** | 0,9642 |
| Sciences appliquées | 0,5231 | 0,5289 | **0,5260** | 0,9464 |

### AUC-ROC par domaine (One-vs-Rest)

| Domaine | AUC |
|---|---|
| Droit public | 0,9936 |
| Histoire | 0,9867 |
| Informatique | 0,9815 |
| Sc. biologiques & Psychologie | 0,9795 |
| Sociologie | 0,9774 |
| Sciences de gestion | 0,9728 |
| Chimie | 0,9717 |
| Physique | 0,9705 |
| Sciences économiques | 0,9704 |
| Sciences appliquées | 0,9203 |
| **Moyenne macro** | **0,9724** |

### Synthèse finale

```
╔════════════════════════════════════════════════════════╗
║         SYNTHESE DES RESULTATS — PROJET 3             ║
╠════════════════════════════════════════════════════════╣
║  Documents traités        : 15 000                    ║
║  Domaines (classes)       : 10                        ║
║  Features TF-IDF          : 1 711 termes              ║
║  Topics LDA               : 10                        ║
║  Axes discriminants (AFD) : 9                         ║
╠════════════════════════════════════════════════════════╣
║  Accuracy (test)          : 76,71 %                   ║
║  F1-Score moyen (macro)   : 0,7680                    ║
║  AUC moyen (One-vs-Rest)  : 0,9724                    ║
║  Kappa de Cohen           : 0,7412                    ║
╚════════════════════════════════════════════════════════╝
```

---

## Méthodes et fondements théoriques

### Latent Dirichlet Allocation (LDA thématique)

Modèle génératif probabiliste qui suppose que chaque document est un mélange de K topics, et chaque topic une distribution sur le vocabulaire. Utilisé ici pour enrichir la représentation des documents avec des features sémantiques latentes (matrice γ).

### Analyse Factorielle Discriminante (AFD / LDA discriminant)

L'AFD cherche la projection **w** maximisant le critère de Fisher :

$$J(\mathbf{w}) = \frac{\mathbf{w}^\top S_B \,\mathbf{w}}{\mathbf{w}^\top S_W \,\mathbf{w}}$$

avec $S_B$ = dispersion inter-classes, $S_W$ = dispersion intra-classes. La solution est le problème aux valeurs propres généralisé $S_W^{-1} S_B \,\mathbf{w} = \lambda\,\mathbf{w}$.

Pour K = 10 classes, l'AFD produit au maximum **K − 1 = 9 axes discriminants**.

### Naive Bayes

Sous l'hypothèse d'indépendance conditionnelle des features :

$$P(D_k \mid \mathbf{x}) \propto P(D_k) \prod_{i=1}^{p} P(x_i \mid D_k)$$

Le **lissage de Laplace** (λ = 1) est appliqué pour éviter les probabilités nulles sur des termes absents de l'ensemble d'entraînement.

### Gestion du déséquilibre des classes

Le dataset original présente un ratio max/min > 22 000. Stratégie retenue :

- **Sous-échantillonnage** à 1 500 thèses par domaine (équilibre parfait)
- **Comparaison de priors** : prior uniforme vs prior reflétant la distribution réelle (résultats identiques dans ce cas, car les données sont parfaitement équilibrées après sous-échantillonnage)

---

## Limites et travaux futurs

### Limites identifiées

- **27 248 domaines originaux** : la réduction à 10 domaines est nécessaire mais exclut la grande majorité du dataset. Une approche hiérarchique permettrait de conserver plus d'information.
- **Sciences appliquées** : F1 = 0,526, le domaine le moins bien classifié — fort chevauchement sémantique avec Physique et Informatique.
- **Stemming agressif** : peut dégrader la précision sémantique (ex. "informatique" et "information" ramenés à la même racine).

### Pistes d'amélioration

| Piste | Bénéfice attendu |
|---|---|
| Word embeddings (Word2Vec, FastText) | Représentation sémantique plus riche que TF-IDF |
| CamemBERT / FlauBERT | Embeddings contextuels pré-entraînés en français |
| Kernel Discriminant Analysis (KDA) | Capturer des non-linéarités dans l'espace des features |
| SMOTE | Génération d'exemples synthétiques pour les classes sous-représentées |
| Classification hiérarchique | Grandes familles → sous-domaines (ex. Sciences → Physique / Chimie) |
| Grid Search / Optimisation bayésienne | Optimisation systématique des hyperparamètres |
| Lemmatisation (spaCy fr) | Plus précise que le stemming de Snowball |

---

## Références

| Référence | Citation |
|---|---|
| **Dataset** | [French Doctoral Thesis Semantic Similarity — Kaggle](https://www.kaggle.com/code/antoinebourgois2/french-doctoral-thesissemantic-similarity-search) |
| **LDA thématique** | Blei, D.M., Ng, A.Y., Jordan, M.I. (2003). *Latent Dirichlet Allocation*. JMLR, 3, 993-1022 |
| **Naive Bayes texte** | Manning, C.D., Raghavan, P., Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press |
| **AFD / LDA discriminant** | Fisher, R.A. (1936). *The use of multiple measurements in taxonomic problems*. Annals of Eugenics, 7(2), 179-188 |
| **Package topicmodels** | Grün, B., Hornik, K. (2011). *topicmodels: An R Package for Fitting Topic Models*. JSS, 40(13) |
| **Package e1071** | Meyer, D. et al. (2023). *e1071: Misc Functions of the Department of Statistics*. CRAN |
| **Package tm** | Feinerer, I., Hornik, K. (2023). *tm: Text Mining Package*. CRAN |

---

## Auteurs

| Nom | Établissement |
|---|---|
| TORRES Diego | ESIEE Paris |
| WU Lucas | ESIEE Paris |

**Encadrant :** Badr TAJINI — ESIEE Paris  
**Cours :** AP-4209 — E4 — 2025-2026  
**Date de rendu :** 19/02/2026
