# Détection de Texte Généré par IA
### Classification Bayésienne & Analyse Factorielle Discriminante
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

Ce projet implémente un **système complet de détection de texte généré par IA**, opposant des textes humains (label `0`) à des textes produits par des LLMs (label `1`). Il s'inscrit dans le challenge Kaggle [LLM - Detect AI Generated Text](https://www.kaggle.com/competitions/llm-detect-ai-generated-text).

L'approche repose sur une pipeline à cinq étapes :

```
Texte brut
    ↓
Features stylométriques (17 métriques linguistiques)
    +
TF-IDF bigrammes (332 486 termes)
    ↓
SVD tronquée — k = 150 dimensions latentes
    ↓
AFD (Analyse Factorielle Discriminante)  →  1 axe LD1 (Cohen's d = 5.49)
    ↓
Classification Bayésienne MCMC (NUTS/HMC via rstanarm)
    ↓
Probabilités calibrées P(texte IA)
```

---

## Dataset

### Source principale — LLM-Detect (Kaggle)

| Fichier | Description | Lignes |
|---|---|---|
| `train_essays.csv` | Essais annotés (Kaggle officiel) | 1 378 |
| `test_essays.csv` | Essais de test (soumission Kaggle) | 3 (jeu exemple) |
| `sample_submission.csv` | Format de soumission attendu | — |

### Source complémentaire — DRCAT

| Fichier | Description | Lignes |
|---|---|---|
| `train_drcat_01.csv` | Textes humains & IA (fold 1) | ~40 000 |
| `train_drcat_02.csv` | Textes humains & IA (fold 2) | ~40 000 |
| `train_drcat_03.csv` | Textes humains & IA (fold 3) | ~40 000 |
| `train_drcat_04.csv` | Textes humains & IA (fold 4) | ~40 000 |

**Dataset consolidé après fusion et nettoyage :**

| Classe | Effectif | Proportion |
|---|---|---|
| Humain (0) | 116 747 | 72,6 % |
| IA (1) | 44 087 | 27,4 % |
| **Total** | **160 834** | — |

> **Structure CSV attendue :** colonnes `text` et `generated` (0 ou 1). Les fichiers DRCAT peuvent avoir une colonne `label` qui est automatiquement renommée.

### Placement des fichiers

```
projet_final/
└── data/
    ├── llm-detect-ai-generated-text/
    │   └── llm-detect-ai-generated-text/
    │       ├── train_essays.csv
    │       ├── test_essays.csv
    │       └── sample_submission.csv
    ├── train_drcat_01.csv
    ├── train_drcat_02.csv
    ├── train_drcat_03.csv
    └── train_drcat_04.csv
```

---

## Pipeline complète

### Étape 1 — Prétraitement & Nettoyage

- Fusion des sources Kaggle et DRCAT
- Nettoyage du texte (retrait des doublons, suppression des entrées vides, filtre `nchar > 30`)
- Détection et correction des `NA` via `safe_texts()` (critique pour `itoken`)

### Étape 2 — Extraction de features stylométriques

17 métriques linguistiques extraites en parallèle (21 cœurs) avec mise en cache `.rds` :

| Feature | Description |
|---|---|
| `char_count` | Nombre total de caractères |
| `word_count` | Nombre total de mots |
| `sent_count` | Nombre de phrases |
| `avg_sent_len` | Longueur moyenne des phrases (mots) |
| `sent_len_sd` | **Variabilité** des longueurs de phrases ← discriminant fort |
| `avg_word_len` | Longueur moyenne des mots |
| `ttr` | **Type-Token Ratio** — richesse du vocabulaire |
| `hapax_ratio` | Mots n'apparaissant qu'une seule fois |
| `flesch` | Score de lisibilité de Flesch (approché) |
| `lex_entropy` | Entropie lexicale |
| `punct_rate` | Taux de ponctuation |
| `comma_rate` | Taux de virgules |
| `upper_rate` | Taux de majuscules |
| `discourse_markers` | Connecteurs formels typiques des textes IA |
| `ai_phrases` | Expressions génériques détectées dans les textes IA |
| `long_word_rate` | Taux de mots longs (> 6 caractères) |
| `char_bigram_entropy` | **Proxy de perplexité** via entropie des bigrammes de caractères |

### Étape 3 — Vectorisation TF-IDF + SVD

```
TF-IDF bigrammes :  128 668 documents × 332 486 termes
         ↓ SVD tronquée (irlba, k = 150)
Espace dense :      128 668 documents × 150 dimensions
```

- Vocabulaire unigrammes + bigrammes (`ngram = c(1L, 2L)`)
- Filtres : `term_count_min = 5`, `doc_proportion_max = 0.45`
- SVD : 54,7 % de la variance capturée dès k = 20 ; 100 % pour k = 150 (variance relative aux 150 valeurs singulières calculées)

### Étape 4 — Analyse Factorielle Discriminante (AFD)

L'AFD cherche la projection **w** maximisant le critère de Fisher :

$$J(\mathbf{w}) = \frac{\mathbf{w}^\top S_B \,\mathbf{w}}{\mathbf{w}^\top S_W \,\mathbf{w}}$$

avec $S_B$ = dispersion inter-classes, $S_W$ = dispersion intra-classes.

Pour K = 2 classes, il existe **un unique axe discriminant LD1**.

**Résultat obtenu :**

| Métrique | Valeur | Interprétation |
|---|---|---|
| Moyenne LD1 Humain | −1.715 | Projection négative |
| Moyenne LD1 IA | +4.560 | Projection positive |
| Cohen's d | **5.49** | Effet **Grand** (d > 0.8) |
| Score silhouette | **0.806** | Bonne séparation |

> **Justification de l'AFD linéaire :** La forte séparation linéaire (Cohen's d = 5.49) observée sur LD1 valide le choix de l'AFD classique. Une Kernel Discriminant Analysis (KDA) apporterait une complexité supplémentaire sans gain attendu dans ce contexte.

### Étape 5 — Classification Bayésienne MCMC

Modèle logistique bayésien :

$$\text{logit}(p_i) = \alpha + \beta \cdot \text{LD1}_i$$

| Paramètre | Prior | Justification |
|---|---|---|
| β | N(0, 2.5) autoscalé | Régularisation Ridge bayésienne |
| α | N(0, 5) | Prior diffus sur l'intercept |

**Algorithme :** NUTS (No-U-Turn Sampler), variante adaptative de HMC — 4 chaînes × 2000 itérations (warmup = 1000).

**Diagnostics MCMC :**
- R̂ max = **1.0006** (seuil < 1.01 ✅)
- n_eff ≈ 1 600–3 200 (très satisfaisant)
- Convergence confirmée sur toutes les chaînes

**Estimations a posteriori (IC 95 %) :**

| Paramètre | Moyenne | SD | 2.5% | 97.5% |
|---|---|---|---|---|
| α (Intercept) | −3.064 | 0.048 | −3.161 | −2.975 |
| β (LD1) | +3.033 | 0.044 | +2.951 | +3.123 |

---

## Structure du projet

```
projet_final/
│
├── data/                              # Données brutes (non versionnées)
│   ├── llm-detect-ai-generated-text/
│   └── train_drcat_0[1-4].csv
│
├── projet_final.Rmd                   # Document principal (code + rapport)
├── projet_final.html                  # Rapport rendu (sortie RMarkdown)
├── submission.csv                     # Prédictions pour soumission Kaggle
│
├── cache_stylo_all.rds               # Cache features stylométriques (train)
├── cache_stylo_cv.rds                # Cache features stylométriques (CV)
├── cache_stylo_test.rds              # Cache features stylométriques (test)
│
└── README.md                          # Ce fichier
```

---

## 🔧 Installation et dépendances

### Prérequis

- **R** ≥ 4.3.0
- **RStudio** (recommandé) ou tout éditeur compatible RMarkdown
- Mémoire RAM recommandée : **≥ 16 Go** (dataset de 160 000 textes)
- CPU multi-cœurs recommandé (extraction stylométrique parallélisée sur 21 cœurs)

### Installation des packages

Coller dans la console R **avant** de lancer le render :

```r
install.packages(c(
  # Manipulation de données
  "readr", "dplyr", "stringr", "tidyr", "tibble",
  
  # Visualisation
  "ggplot2", "gridExtra", "scales",
  
  # NLP / Vectorisation
  "text2vec", "irlba",
  
  # Topic modeling
  "topicmodels", "slam", "tidytext",
  
  # Modélisation
  "MASS", "caret",
  
  # Classification Bayésienne MCMC
  "rstanarm",        # Stan — NUTS/HMC (backend principal)
  "arm",             # bayesglm — fallback si Stan non disponible
  "bayesplot",       # Visualisation des posteriors
  
  # Évaluation
  "pROC", "cluster"
))
```

> **Important :** `rstanarm` requiert l'installation de **Stan**. Sur Windows, il peut être nécessaire d'installer [Rtools](https://cran.r-project.org/bin/windows/Rtools/) au préalable.  
> Si `rstanarm` n'est pas disponible, le code bascule automatiquement sur `arm::bayesglm` (estimateur MAP — fallback sans MCMC complet).

---

## Utilisation

### Lancement du rapport complet

```r
rmarkdown::render("projet_final.Rmd")
```

Ou depuis le terminal :

```powershell
& "C:\Program Files\R\R-4.5.2\bin\Rscript.exe" -e "rmarkdown::render('projet_final.Rmd')"
```

### Paramètres clés modifiables dans le Rmd

```r
USE_RSTANARM <- TRUE     # FALSE = utilise arm::bayesglm (plus rapide, moins complet)
K_SVD        <- 150      # Nombre de dimensions SVD (compromis vitesse/précision)
K_TOPICS     <- 6        # Nombre de topics LDA thématique
MAX_CV_N     <- 6000     # Taille du sous-échantillon pour la validation croisée
```

### Cache stylométrique

L'extraction stylométrique est **mise en cache automatiquement** (fichiers `.rds`). Pour forcer un recalcul, supprimez les fichiers `cache_stylo_*.rds` avant de relancer.

---

## Résultats

### Métriques sur l'ensemble de validation (80/20 stratifié)

| Métrique | Valeur |
|---|---|
| **Accuracy** | **99.34 %** |
| Precision | 99.14 % |
| Rappel (Sensibilité) | 98.49 % |
| Spécificité | 99.67 % |
| **F1-score** | **98.81 %** |
| **AUC-ROC** | **0.9993** |
| **Score de Brier** | **0.0053** |
| Kappa de Cohen | 0.9836 |
| Seuil optimal F1 | 0.43 |

### Validation croisée 5-fold stratifiée

| Fold | AUC | Brier | Accuracy |
|---|---|---|---|
| 1 | 0.9979 | 0.0119 | 98.67 % |
| 2 | 0.9968 | 0.0091 | 99.08 % |
| 3 | 0.9991 | 0.0080 | 99.00 % |
| 4 | 0.9990 | 0.0111 | 98.25 % |
| 5 | 0.9973 | 0.0155 | 98.08 % |
| **Moyenne** | **0.9980 ± 0.0010** | **0.0111 ± 0.0029** | **98.62 % ± 0.44 %** |

### Séparation AFD

```
Classe Humain : LD1 = −1.715  (σ = 0.792)
Classe IA     : LD1 = +4.560  (σ = 1.410)
Cohen's d     : 5.49  →  Effet Grand
Score silhouette : 0.806  →  Bonne séparation
```

### Topics LDA (k = 6, Gibbs, sous-échantillon 5 000 docs)

| Topic | Thème identifié | Mots clés |
|---|---|---|
| Topic 1 | Éducation | students, school, learning, classes |
| Topic 2 | Transport / Vote | car, driving, Electoral, vote |
| Topic 3 | Conseil / Aide | how, could, know, someone, better |
| Topic 4 | Opinion personnelle | we, think, want, good, my |
| Topic 5 | Sciences / Espace | Venus, face, author, is_a |
| Topic 6 | Argumentation | may, important, can_be, lead |

---

## Méthodes et fondements théoriques

### Théorème de Bayes

$$P(\boldsymbol{\theta} \mid \mathcal{D}) \propto P(\mathcal{D} \mid \boldsymbol{\theta}) \cdot P(\boldsymbol{\theta})$$

Le modèle logistique bayésien prédit la probabilité qu'un texte soit généré par IA. L'inférence complète via MCMC fournit des **distributions a posteriori** sur les paramètres, pas simplement des estimations ponctuelles.

### Critère de Fisher (AFD)

$$J(\mathbf{w}) = \frac{\mathbf{w}^\top S_B \,\mathbf{w}}{\mathbf{w}^\top S_W \,\mathbf{w}}$$

Résolu comme un problème aux valeurs propres généralisé $S_W^{-1} S_B \,\mathbf{w} = \lambda\,\mathbf{w}$.

### Pourquoi SVD avant AFD ?

La matrice TF-IDF brute est **creuse** et de dimension >> 10 000 colonnes, rendant l'inversion de $S_W$ numériquement instable. La SVD vers k = 150 dimensions denses (1) stabilise le calcul, (2) élimine le bruit lexical, (3) approche la normalité multivariée requise par l'AFD.

### No-U-Turn Sampler (NUTS)

Variante adaptative de HMC (Hamiltonian Monte Carlo) qui explore l'espace a posteriori bien plus efficacement que Metropolis-Hastings classique, évitant les random walks et le réglage manuel du pas d'intégration.

---

## Limites et travaux futurs

### Limites identifiées

- **Performances élevées — interprétation prudente :** Les textes IA du dataset LLM-Detect présentent des patterns stylistiques très distincts. Une validation par groupes (par source ou prompt) permettrait de détecter un éventuel leakage contextuel lié à la structure du dataset.
- **Jeu de test réduit :** Le fichier `test_essays.csv` officiel Kaggle ne contient que 3 exemples — la pipeline de soumission est fonctionnelle quelle que soit la taille réelle du jeu de test.
- **Absence de perplexité réelle :** La feature `char_bigram_entropy` est un proxy ; une vraie perplexité GPT-2 (via `reticulate`) serait plus discriminante.

### Travaux futurs

| Piste | Bénéfice attendu |
|---|---|
| Perplexité GPT-2 via `reticulate` | Feature très discriminante documentée dans la littérature |
| BERT / sentence embeddings | Remplacement du TF-IDF par des représentations contextuelles |
| Kernel Discriminant Analysis (KDA) | Capturer d'éventuelles non-linéarités inter-classes |
| Inférence variationnelle (`algorithm = "meanfield"`) | Passage à l'échelle sur très grands corpus |
| LOO-CV / WAIC via `rstanarm::loo()` | Comparaison formelle de modèles bayésiens |
| Validation par groupe (leave-one-prompt-out) | Évaluation de la robustesse hors-distribution |
| Test de Box's M | Vérification de l'homoscédasticité (hypothèse AFD) |

---

## Références

| Référence | Lien / Citation |
|---|---|
| **Dataset LLM-Detect** | [Kaggle — LLM Detect AI Generated Text](https://www.kaggle.com/competitions/llm-detect-ai-generated-text) |
| **Dataset DRCAT** | Complément d'entraînement avec essais humains et IA |
| **rstanarm** | Goodrich et al. (2023). *rstanarm: Bayesian applied regression modeling via Stan.* CRAN |
| **Stan / NUTS** | Carpenter et al. (2017). *Stan: A probabilistic programming language.* JOSS |
| **Fisher LDA** | Fisher, R.A. (1936). *The use of multiple measurements in taxonomic problems.* Annals of Eugenics |
| **SVD / LSA** | Deerwester et al. (1990). *Indexing by Latent Semantic Analysis.* JASIS |
| **irlba** | Baglama & Reichel (2005). *Augmented implicitly restarted Lanczos bidiagonalization methods.* SIAM |
| **text2vec** | Selivanov, D. (2023). *text2vec: Modern Text Mining Framework for R.* CRAN |
| **Stylométrie** | Stamatatos, E. (2009). *A survey of modern authorship attribution methods.* JASIS&T |
| **Score de Brier** | Brier, G.W. (1950). *Verification of forecasts expressed in terms of probability.* Monthly Weather Review |

---

## Auteurs

| Nom | Email | Établissement |
|---|---|---|
| TORRES Diego | — | ESIEE Paris |
| WU Lucas | — | ESIEE Paris |

**Encadrant :** Badr TAJINI — ESIEE Paris  
**Cours :** AP-4209 — E4 — 2025-2026  
**Rapport généré le :** 19/02/2026

---

*README rédigé en correspondance avec le rapport `projet_final.html` et le sujet `final_project.pdf`.*
