# Projet 2 : Analyse Factorielle Discriminante (AFD)
### Twitter Entity Sentiment Analysis — Regroupement et visualisation des sentiments
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

Ce projet applique l'**Analyse Factorielle Discriminante (AFD)**, implémentée via l'Analyse Discriminante Linéaire (`MASS::lda()`), sur le dataset **Twitter Entity Sentiment Analysis**. L'objectif est de **réduire la dimensionnalité** des représentations textuelles de tweets et de **visualiser le regroupement des sentiments** dans un espace de dimension réduite.

**Pipeline en six étapes :**

```
Tweets bruts (73 824 entrées d'entraînement)
    ↓
Nettoyage (URLs, mentions, hashtags, caractères spéciaux)
    + Stopwords (anglais) + Stemming (Snowball)
    ↓
TF-IDF  →  72 346 docs × 6 010 termes
    ↓
SVD tronquée (irlba)  →  72 346 × 100 dimensions latentes
    ↓
AFD via LDA (MASS)  →  3 axes discriminants (LD1, LD2, LD3)
    ↓
Visualisation + Évaluation (Accuracy 51,55 % | Silhouette −0,048)
    + Topic Modeling (LDA, k = 6)
```

---

## Dataset

**Source :** [Twitter Entity Sentiment Analysis — Kaggle](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis)

### Fichiers

| Fichier | Rôle | Lignes |
|---|---|---|
| `twitter_training.csv` | Entraînement du modèle AFD | 73 824 |
| `twitter_validation.csv` | Évaluation des performances | 999 |

### Structure des données

| Colonne | Description |
|---|---|
| `id` | Identifiant numérique du tweet |
| `entity` | Entité mentionnée (ex. : Borderlands, Google…) |
| `sentiment` | **Variable cible** : `Irrelevant`, `Negative`, `Neutral`, `Positive` |
| `tweet` | Texte brut du tweet |

### Distribution des sentiments (entraînement)

| Sentiment | Effectif | Proportion |
|---|---|---|
| Negative | 22 312 | 30,2 % |
| Positive | 20 619 | 27,9 % |
| Neutral | 18 051 | 24,5 % |
| Irrelevant | 12 842 | 17,4 % |

**Statistiques sur la longueur des tweets :**

| Indicateur | Valeur |
|---|---|
| Nombre de tweets (train) | 73 824 |
| Longueur moyenne | 109 caractères |
| Longueur médiane | 91 caractères |
| Percentile 90 | 236 caractères |

### Placement des fichiers

```
projet2/
└── data/
    ├── twitter_training.csv
    └── twitter_validation.csv
```

---

## Pipeline complète

### Étape 1 — Nettoyage du texte

Chaque tweet est nettoyé séquentiellement :

```
1. Mise en minuscules
2. Suppression des URLs        (http://, www.)
3. Suppression des mentions    (@username)
4. Suppression des hashtags    (#topic)
5. Suppression des caractères non alphabétiques
6. Normalisation des espaces
7. Suppression des stopwords anglais (tm::stopwords("en") + "rt")
8. Stemming anglais (SnowballC::wordStem)
```

**Exemple :**

| Avant | Après |
|---|---|
| `im getting on borderlands and i will murder you all` | `im get borderland will murder` |
| `I am coming to the borders and I will kill you all` | `come border will kill` |

### Étape 2 — Vectorisation TF-IDF

- Outil : `text2vec`
- Pruning : `term_count_min = 10`, `doc_proportion_max = 0.35`
- **Résultat :** 72 346 documents × 6 010 termes

### Étape 3 — Réduction SVD tronquée (LSA)

- Outil : `irlba::irlba()`
- Paramètre : k = 100 composantes latentes
- **Résultat :** 72 346 × 100 (train) | 999 × 100 (validation)

### Étape 4 — AFD via LDA (MASS)

- `MASS::lda(sentiment ~ ., data = train_lda_df)`
- Priors estimés depuis les fréquences observées :

| Sentiment | Prior |
|---|---|
| Irrelevant | 0,174 |
| Negative | 0,303 |
| Neutral | 0,244 |
| Positive | 0,279 |

- **Résultat :** 3 axes discriminants (K − 1, avec K = 4 classes)

**Proportion de trace (variance discriminante) :**

| Axe | Proportion |
|---|---|
| LD1 | **55,63 %** |
| LD2 | **29,09 %** |
| LD3 | **15,28 %** |
| **Total** | **100 %** |

### Étape 5 — Topic Modeling (LDA thématique)

- k = 6 topics, Gibbs sampling, seed = 42
- Sous-échantillon : 8 000 tweets (pour limiter le temps de calcul)

**Top 10 mots par topic (après stemming) :**

| Topic | Thème probable | Mots clés |
|---|---|---|
| T1 | Politique / Sport | itali, just, johnson, can, make, come, year, will, play |
| T2 | Engagement / Communauté | com, day, player, can, get, go, thank, peopl, love |
| T3 | Jeux vidéo (négatif) | game, like, fuck, play, go, get, m, shit |
| T4 | Jeux vidéo (général) | s, game, just, realli, love, twitter, will, look, call |
| T5 | Réseaux sociaux / Gaming | t, com, play, game, twitter, get, pic, now, realli |
| T6 | Général / Divers | com, t, s, game, can, fuck, one, pic, best, good |

---

## Structure du projet

```
projet2/
│
├── data/
│   ├── twitter_training.csv          # Données d'entraînement
│   └── twitter_validation.csv        # Données de validation
│
├── projet2.Rmd                        # Document principal (code + rapport)
├── projet2.html                       # Rapport compilé
│
└── README.md                          # Ce fichier
```

---

## Installation et dépendances

### Prérequis

- **R** ≥ 4.3.0
- **RStudio** (recommandé) ou tout éditeur compatible RMarkdown
- Mémoire RAM recommandée : **≥ 8 Go** (matrice TF-IDF 72K × 6K en mémoire)

### Installation des packages

```r
install.packages(c(
  # Manipulation de données
  "readr", "dplyr", "stringr",

  # Visualisation
  "ggplot2",

  # NLP / Vectorisation
  "tm",          # Stopwords
  "SnowballC",   # Stemming
  "text2vec",    # TF-IDF vectorizer
  "irlba",       # SVD tronquée

  # Modélisation AFD
  "MASS",        # LDA discriminant

  # Évaluation
  "caret",       # Matrice de confusion
  "cluster",     # Score silhouette

  # Topic Modeling
  "topicmodels", # LDA thématique
  "Matrix"       # Gestion matrices creuses
))
```

---

## Utilisation

### Lancement du rapport

```r
rmarkdown::render("projet2.Rmd")
```

Ou depuis le terminal Windows :

```powershell
& "C:\Program Files\R\R-4.5.2\bin\Rscript.exe" -e "rmarkdown::render('projet2.Rmd')"
```

### Paramètres clés modifiables

```r
k         <- 100    # Nombre de dimensions SVD (compromis vitesse/précision)
n_sil     <- 5000   # Taille du sous-échantillon pour le score silhouette
n_topic   <- 8000   # Taille du sous-échantillon pour le topic modeling
k_topics  <- 6      # Nombre de topics LDA thématique
```

---

## Résultats

### Évaluation de la classification (ensemble de validation — 999 tweets)

| Métrique | Valeur |
|---|---|
| **Accuracy** | **51,55 %** |
| Kappa de Cohen | 0,3352 |
| No Information Rate (baseline) | 28,53 % |
| p-value (Acc > NIR) | < 2,2 × 10⁻¹⁶ |

> L'accuracy de 51,55 % est **significativement supérieure à la baseline** (28,53 %), confirmant que l'AFD capture une structure discriminante réelle dans les données. La nature très courte des tweets et la proximité sémantique des classes limitent les performances absolues.

### Métriques par classe (validation)

| Classe | Sensibilité | Spécificité | Precision | Balanced Acc. |
|---|---|---|---|---|
| Irrelevant | 0,110 | 0,975 | 0,475 | 0,543 |
| Negative | 0,694 | 0,752 | 0,503 | 0,723 |
| Neutral | 0,554 | 0,777 | 0,498 | 0,666 |
| Positive | 0,556 | 0,831 | 0,558 | 0,694 |

> **Observation :** La classe `Irrelevant` est la plus difficile à détecter (sensibilité = 11 %), car ces tweets ne présentent pas de signal sémantique clair lié au sentiment.

### Matrice de confusion (validation)

|  | Irrelevant | Negative | Neutral | Positive |
|---|---|---|---|---|
| **Prédit Irrelevant** | 19 | 6 | 4 | 11 |
| **Prédit Negative** | 62 | 184 | 72 | 48 |
| **Prédit Neutral** | 47 | 48 | 158 | 64 |
| **Prédit Positive** | 44 | 27 | 51 | 154 |

### Score silhouette

| Indicateur | Valeur | Interprétation |
|---|---|---|
| Score silhouette moyen | **−0,048** | Chevauchement fort entre les classes dans l'espace AFD |

> Un score silhouette négatif ou proche de 0 indique que les clusters de sentiments se chevauchent significativement dans l'espace LD1–LD2. Cela est cohérent avec la nature ambiguë du sentiment dans les tweets courts.

### Proportion de variance discriminante

```
LD1 : 55,63 %  ████████████████████████████░░░░░░░░░░░░░░░░
LD2 : 29,09 %  ██████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
LD3 : 15,28 %  ███████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
```

LD1 seul capture plus de la moitié de la variance discriminante entre les 4 classes de sentiment.

---

## Méthodes et fondements théoriques

### TF-IDF (Term Frequency – Inverse Document Frequency)

Pondération qui valorise les termes fréquents dans un document mais rares dans l'ensemble du corpus, capturant ainsi le contenu lexical discriminant de chaque tweet.

### SVD tronquée (Singular Value Decomposition)

La matrice TF-IDF (72K × 6K) est projetée vers un espace dense de dimension k = 100 via décomposition SVD :

$$X \approx U_k \Sigma_k V_k^\top$$

Cette réduction (1) stabilise le calcul de l'AFD, (2) élimine le bruit lexical, (3) capture les relations sémantiques latentes entre termes.

### Analyse Factorielle Discriminante (AFD / LDA discriminant)

L'AFD cherche les projections **w** maximisant le critère de Fisher :

$$J(\mathbf{w}) = \frac{\mathbf{w}^\top S_B \,\mathbf{w}}{\mathbf{w}^\top S_W \,\mathbf{w}}$$

avec $S_B$ = dispersion inter-classes, $S_W$ = dispersion intra-classes. Pour K = 4 classes, on obtient au maximum **K − 1 = 3 axes discriminants**.

### Score silhouette

Pour chaque point $i$ :

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

avec $a(i)$ = distance moyenne intra-cluster et $b(i)$ = distance moyenne au cluster voisin le plus proche. Un score proche de 1 indique une bonne séparation ; proche de 0 ou négatif, un fort chevauchement.

---

## Limites et travaux futurs

### Limites identifiées

- **Tweets très courts** (médiane 91 caractères) : la représentation TF-IDF est creuse et peu informative pour des textes aussi courts, ce qui limite la séparabilité des classes.
- **Classe Irrelevant** : très faible sensibilité (11 %), car ces tweets ne présentent pas de signal sémantique lié au sentiment envers l'entité mentionnée.
- **Score silhouette négatif** (−0,048) : fort chevauchement des classes dans l'espace AFD — les sentiments sont intrinsèquement ambigus dans ce type de données.
- **Stemming trop agressif** : certains tokens stemés (`t`, `s`, `m`) sont des artefacts qui polluent le vocabulaire.

### Pistes d'amélioration

| Piste | Bénéfice attendu |
|---|---|
| Embeddings contextuels (BERT, RoBERTa) | Représentation sémantique bien supérieure à TF-IDF pour les tweets courts |
| Kernel Discriminant Analysis (KDA) | Capturer des frontières non linéaires entre classes de sentiment |
| Lexicons de sentiment (AFINN, VADER, NRC) | Features de sentiment explicites en complément du TF-IDF |
| Filtrage des tokens aberrants post-stemming | Supprimer `t`, `s`, `m` du vocabulaire |
| Oversampling SMOTE sur `Irrelevant` | Rééquilibrer l'entraînement pour améliorer la sensibilité |
| Modèles de classification dédiés (SVM, XGBoost) | Performances supérieures à l'AFD seule pour la classification |
| Analyse par entité | Explorer si certaines entités ont des profils de sentiment distincts |

---

## Références

| Référence | Citation |
|---|---|
| **Dataset** | [Twitter Entity Sentiment Analysis — Kaggle](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis) |
| **AFD / LDA discriminant** | Fisher, R.A. (1936). *The use of multiple measurements in taxonomic problems*. Annals of Eugenics, 7(2), 179–188 |
| **SVD / LSA** | Deerwester et al. (1990). *Indexing by Latent Semantic Analysis*. JASIS, 41(6), 391–407 |
| **text2vec** | Selivanov, D. (2023). *text2vec: Modern Text Mining Framework for R*. CRAN |
| **irlba** | Baglama & Reichel (2005). *Augmented implicitly restarted Lanczos bidiagonalization methods*. SIAM |
| **Score silhouette** | Rousseeuw, P.J. (1987). *Silhouettes: A graphical aid to the interpretation and validation of cluster analysis*. JCAM, 20, 53–65 |
| **Stemming Snowball** | Porter, M.F. (1980). *An algorithm for suffix stripping*. Program, 14(3), 130–137 |
| **Package MASS** | Venables, W.N., Ripley, B.D. (2002). *Modern Applied Statistics with S*. Springer |

---

## Auteurs

| Nom | Établissement |
|---|---|
| TORRES Diego | ESIEE Paris |
| WU Lucas | ESIEE Paris |

**Encadrant :** Badr TAJINI — ESIEE Paris  
**Cours :** AP-4209 — E4 — 2025-2026  
**Date de rendu :** 14/02/2026
