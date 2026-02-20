# Projet 1 – Classification Bayésienne des Émotions
## ESIEE Paris – 2025-2026 – AP-4209
**Auteur : Torres, Wu**

---

## Objectif du projet

Ce projet a pour objectif de développer un **classifieur bayésien naïf (Naive Bayes)** capable de prédire une émotion à partir d’un texte.

Le travail s’appuie sur un **jeu de données Kaggle portant sur les émotions**, et suit une démarche complète :

- Analyse exploratoire des données (EDA)
- Prétraitement du texte
- Vectorisation TF-IDF
- Entraînement d’un modèle Naive Bayes
- Évaluation des performances
- Validation croisée

---

## Dataset

**Source :** Emotion Dataset – Kaggle  

Le dataset contient :

- `Text` : texte exprimant une émotion  
- `Emotion` : étiquette cible (classe)

Exemples d’émotions :
- joy  
- sadness  
- anger  
- fear  
- love  
- surprise  

---

## Méthodologie

### 1️ Analyse exploratoire (EDA)

- Distribution des classes  
- Analyse du déséquilibre éventuel  
- Longueur moyenne des textes  
- Visualisations graphiques  

---

### 2️ Prétraitement du texte

Les étapes appliquées :

- Passage en minuscules  
- Suppression des caractères spéciaux et chiffres  
- Suppression des stopwords  
- Tokenisation  
- Stemming  
- Construction d’un corpus (`tm`)  
- Vectorisation TF-IDF  

---

### 3️ Entraînement du modèle

- Division des données :
  - 70% entraînement  
  - 30% test  
- Modèle : `naiveBayes()` du package `e1071`  
- Lissage de Laplace appliqué  

---

### 4️ Évaluation

Les métriques calculées :

- Accuracy  
- Precision  
- Recall  
- F1-score  
- Matrice de confusion  
- Validation croisée k-fold  

---

## Structure du dossier

```
Projet1/
│
├── data/
│   └── emotion_dataset.csv
│
├── projet1.Rmd
├── Projet1.html
├── project1.pdf
└── README.md
```

---

## Packages utilisés

```r
readr
dplyr
stringr
ggplot2
tm
SnowballC
e1071
caret
pROC
```

---

## Compétences mobilisées

- NLP (Natural Language Processing)  
- TF-IDF  
- Naive Bayes  
- Évaluation de modèles  
- Validation croisée  
- Interprétation statistique  

---

## Conclusion

Ce projet démontre l’efficacité d’un classifieur bayésien naïf pour la classification de texte court.

Malgré l’hypothèse d’indépendance conditionnelle forte, le modèle offre de bonnes performances et constitue une base solide pour des approches plus avancées (SVM, réseaux neuronaux, transformers).
