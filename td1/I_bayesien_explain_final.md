> AP-4209 ESIEE-Paris: 2023 -2024\
> Fouille de données avec R pour la data science et l'intelligence artificielle \
> --TD 2: CONSTRUCTION D'UN CLASSIFIEUR BAYESIEN NAIF EN R--\
> Author : Badr TAJINI -- ESIEE Paris
---


### Fonction #1
```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

### Explication
Cette portion de code est utilisée pour configurer les options globales des blocs de code R dans un document R Markdown. L'option `echo = TRUE` fait en sorte que le code R sera affiché dans le document final, en plus des résultats de ce code.

---

### Fonction #2
```{r}
rm(list = ls())       # initialisation
library(kableExtra)   # chargement du paquet
library('e1071')
library('caret')
library(caret)
```

### Explication
Ce bloc de code effectue plusieurs opérations initiales :
1. `rm(list = ls())` : Supprime tous les objets présents dans l'environnement de travail R pour repartir sur une base propre.
2. `library(kableExtra)` : Charge le package `kableExtra`, utilisé pour créer des tables formatées et améliorées dans R Markdown.
3. `library('e1071')` et `library('caret')` : Charge les packages `e1071` et `caret`. `e1071` est souvent utilisé pour des méthodes statistiques, et `caret` pour le machine learning.

---

### Fonction #3A
```{r, echo = T}
df = data.frame('Salaire'  = c('< 30', '30 - 50', '30 - 50', '30 - 50', '> 50'),
                'Impots'   = c('< 20','< 20','< 20','> 20','> 20'),
                'Etudiant' = c('Oui','Non','O ...
```

### Explication
Ce code crée un dataframe nommé `df` en R. Ce dataframe contient trois colonnes (`Salaire`, `Impots`, `Etudiant`) avec des valeurs données. L'option `echo = T` assure que le code est visible dans le document R Markdown.

---

### Fonction #3B
```{r}
df %>% kbl(digits=3) %>%    
       kable_styling(bootstrap_options = "striped", full_width = F, position = "center", latex_options = 'stripped') %>% 
       scroll_box( height = "250px")
```
### Explication
Cette fonction sert à mettre en forme et à afficher le dataframe `df` dans un format de tableau amélioré :
1. `df %>% kbl(digits=3)` : Utilise la fonction `kbl` du package `kableExtra` pour convertir le dataframe `df` en un tableau, en formatant les nombres avec trois chiffres après la virgule.
2. `kable_styling(...)` : Applique des styles de mise en forme au tableau. Les options incluent `bootstrap_options = "striped"` pour des lignes alternées en couleurs différentes (rayé), `full_width = F` pour ne pas étendre le tableau sur toute la largeur disponible, et `position = "center"` pour centrer le tableau. `latex_options = 'stripped'` s'applique lors de l'exportation en LaTeX.
3. `scroll_box(height = "250px")` : Place le tableau dans une boîte avec un défilement vertical (scroll box) si la hauteur du tableau dépasse 250 pixels. Cela est utile pour les tableaux longs pour les maintenir dans un espace compact.

Cette fonction est typique dans les documents R Markdown pour présenter des données de manière claire et esthétique.

---

### Fonction #4
```{r}
X <- df[,-4] ; Y = df$Controle
```

### Explication
Ce code effectue deux opérations :
1. `X <- df[,-4]` : Crée une nouvelle variable `X` en extrayant toutes les colonnes du dataframe `df` sauf la quatrième.
2. `Y = df$Controle` : Crée une variable `Y` en extrayant la colonne `Controle` du dataframe `df`.

---

### Fonction #5
```{r}
X %>% kbl(digits=3) %>%    
       kable_styling(bootstrap_options = "striped", full_width = F, position = "center", latex_options = 'stripped') %>% 
       scroll_box( height = "250px")
```

### Explication
Ce bloc de code utilise le "pipe" `%>%` pour enchaîner plusieurs fonctions afin de formater et afficher le contenu de `X` :
1. `kbl(digits=3)` : Utilise la fonction `kbl` pour convertir `X` en un tableau avec trois chiffres après la virgule.
2. `kable_styling` : Applique des styles CSS (comme `striped`, `full_width = F`, etc.) pour améliorer la présentation du tableau.
3. `scroll_box(height = "250px")` : Encadre le tableau dans une boîte défilante de 250 pixels de haut.

---

### Fonction #6
```{r}
data.frame(Y = Y) %>% kbl(digits=3) %>%    
       kable_styling(bootstrap_options = "striped", full_width = F, position = "center", latex_options = 'stripped') %>% 
       scroll_box( height = "250px")
```

### Explication
Cette fonction suit la même logique que la précédente mais s'applique à la variable `Y`. Elle crée d'abord un dataframe à partir de `Y`, puis le convertit en un tableau formaté et stylisé similairement.

---


### Fonction #7
```{r}
contingence<- lapply(X, function(x){  
  temp_1   <- table(Y,x)         # tableau initiale 
}) 
contingence
```

### Explication
Ce code crée une liste nommée `contingence`. La fonction `lapply` est utilisée pour appliquer une fonction anonyme à chaque colonne de `X`. Cette fonction crée un tableau de contingence (`table`) entre la variable `Y` et chaque colonne de `X`. Le résultat est une liste de tableaux de contingence pour chaque variable explicative dans `X`.

---

### Fonction #8
```{r}
k <- length(unique(Y))
m <- 1

prob_cond <- lapply(X, function(x){  
                temp_1 <- table(Y,x)         # tableau initiale 
                s      <- apply(temp_1,1, sum) +  m * k  # calcul des sommes + pondération
                mat    <- matrix(s, ncol = ncol(temp_1), nrow = nrow(temp_1))  # matrice des sommes
                out <- (temp_1 + m)/mat  # calcul des nk = pondération de Laplace
                }) # somme des controles

prob_cond 
```

### Explication
Ce script calcule les probabilités conditionnelles :
1. `k` est le nombre de valeurs uniques dans `Y`.
2. `m` est une constante (ici, 1) utilisée pour la pondération.
3. La fonction `lapply` applique une fonction à chaque colonne de `X` qui :
   - Crée un tableau de contingence entre `Y` et la colonne courante de `X`.
   - Calcule des sommes pondérées pour chaque ligne.
   - Crée une matrice de ces sommes.
   - Calcule les probabilités conditionnelles avec une correction de Laplace (pondération `m`).

---

### Fonction #9
```{r}
data_test <- data.frame( 'Salaire' = c('30 - 50', '> 50','< 30'),
                         'Impots'  = c('< 20', '< 20' ,'< 20')  ,
                         'Etudiant' = c('Oui','Non' ,'Oui')
                            )
```

### Explication
Ce code crée un nouveau dataframe `data_test` contenant des données de test. Il comprend trois colonnes (`Salaire`, `Impots`, `Etudiant`) avec des valeurs spécifiques.

---


### Fonction #10
```{r}
data_test %>% kbl(digits=3) %>%    
       kable_styling(bootstrap_options = "striped", full_width = F, position = "center", latex_options = 'stripped')
```

### Explication
Ce code utilise le dataframe `data_test` et l'affiche sous forme de tableau formaté à l'aide de la suite de fonctions de `kableExtra`. Les options de style sont similaires à celles utilisées précédemment, y compris le formatage à trois chiffres après la virgule, l'ajout de rayures pour améliorer la lisibilité et le positionnement central du tableau.

---

### Fonction #11
```{r}
priori <- prop.table(table(Y))
priori_labels  <- names(priori)
```

### Explication
1. `priori <- prop.table(table(Y))` : Cette ligne crée une table de proportions (`prop.table`) à partir du tableau de fréquences de `Y`. Cela permet de calculer les probabilités a priori pour chaque catégorie unique dans `Y`.
2. `priori_labels <- names(priori)` : Enregistre les noms des catégories uniques de `Y` dans une variable `priori_labels`.

---

### Fonction #12
```{r}
priori
```

### Explication
Ceci est un simple appel à la variable `priori` pour afficher ses valeurs. Rappelons que `priori` contient les probabilités a priori de chaque catégorie unique dans `Y`.

---

### Fonction #13
```{r}
priori_labels
```

### Explication
Cette ligne affiche simplement le contenu de la variable `priori_labels`, qui contient les noms des catégories uniques de `Y` déterminées précédemment. C'est utile pour vérifier les différentes catégories utilisées dans les calculs de probabilités.

---

### Fonction #14
```{r}
Test    <- NULL
LnTest  <- NULL
for (i_test in 1:nrow(data_test))    # sélection des individus du jeux test
{ V    <- NULL
  lnV  <- NULL
  for(ic in 1:length(priori))        # sélection des Ck Oui
    {   somme <- 0 ; prod <- 1
        for(i in 1:ncol(X))          # sélection des tableau de proba conditionnelles
          {
           ar    <-  prob_cond[[i]]  # on extrait les tableau des probailiyés conditionnelles de la liste 
           somme <- somme + log( ar[priori_labels[ic],data_test[i_test,i]])
           prod  <- prod *   ar[priori_labels[ic],data_test[i_test,i]]
      
        }
     V     <- c(V, priori[ic] * prod )
    lnV   <- c(lnV, log(priori[ic]) + somme ) 
  }  

Test  <- rbind(Test, V)    
LnTest <- rbind(LnTest, lnV)    

```

### Explication
Ce script complexe effectue les calculs suivants pour chaque ligne dans `data_test` :
1. Itère sur chaque catégorie dans `priori`.
2. Calcule la somme des logarithmes et le produit des probabilités conditionnelles pour chaque variable dans `X` par rapport à la catégorie actuelle.
3. Multiplie la probabilité a priori de la catégorie par le produit des probabilités conditionnelles.
4. Ajoute ces résultats dans les tableaux `V` et `lnV`.
5. Les résultats pour chaque ligne de `data_test` sont stockés dans `Test` et `LnTest`.

---

### Fonction #15
```{r}
data.frame(LnTest)%>% kbl(digits=4) %>%    
       kable_styling(bootstrap_options = "striped", full_width = F, position = "center", latex_options = 'stripped')
```

### Explication
Cette fonction formate et affiche `LnTest` (logarithmes des probabilités calculées précédemment) sous forme de tableau. Le tableau est stylisé de manière similaire aux précédents, avec un formatage à quatre chiffres après la virgule et un style rayé.

---

### Fonction #16
```{r}
resulTest <- data.frame(data_test, 'Pred' = colnames(LnTest)[apply(LnTest,1,which.max)] ) 

resulTest %>% kbl(digits=4) %>%    
       kable_styling(bootstrap_options = "striped", full_width = F, position = "center", latex_options = 'stripped')
```

### Explication
Ce code combine les données de test (`data_test`) avec les prédictions effectuées (`Pred`), où `Pred` est calculé en prenant l'indice de la valeur maximale dans chaque ligne de `LnTest`. Les résultats sont ensuite formatés et affichés sous forme de tableau.

---

### Fonction #17
```{r}
Naive_Bayes <- function(X, Y, m = 1)
{
  # ...
  return(out)
}
```

### Explication
Cette fonction définit un classificateur Naïve Bayes personnalisé. Elle prend en entrée les variables explicatives `X`, la variable à prédire `Y` et un paramètre `m` pour la correction de Laplace. La fonction calcule les probabilités conditionnelles et les probabilités a priori, puis retourne ces valeurs.

---

### Fonction #18
```{r}
model_1 <- Naive_Bayes(X, Y, m=1)
print(model_1)
```

### Explication
Ceci crée un modèle Naïve Bayes en utilisant les données `X` et `Y`, avec `m = 1` pour la correction de Laplace. Le modèle résultant (`model_1`) est ensuite affiché.

---

### Fonction #19
```{r}
Predict_Bayes <- function(model, X_pred)
{
  # ...
  return(out)
}
```

### Explication
Cette fonction définit une méthode de prédiction pour le modèle Naïve Bayes. Elle prend un modèle Naïve Bayes préalablement ajusté et un ensemble de données de test (`X_pred`), puis calcule les prédictions en utilisant le modèle donné.

### Note :
Oui, la fonction #19, qui définit `Predict_Bayes`, est effectivement similaire à une autre fonction dans mes réponses précédentes. Elle est semblable à la fonction #14, où un processus similaire de prédiction est appliqué. Dans les deux cas, les fonctions exécutent des étapes de prédiction basées sur le modèle Naïve Bayes :

1. **Itération sur les données de test** : Les deux fonctions itèrent sur chaque ligne de l'ensemble de données de test.
2. **Calcul des probabilités** : Elles calculent les probabilités en utilisant les probabilités conditionnelles et les probabilités a priori du modèle Naïve Bayes.
3. **Logarithme des probabilités** : Les probabilités sont traitées en utilisant le logarithme pour éviter des problèmes de sous-débordement numérique (lorsque les probabilités sont très petites).
4. **Combinaison des probabilités** : Les probabilités de chaque classe sont combinées (par exemple, en utilisant la somme des logarithmes) pour obtenir la probabilité finale.
5. **Détermination de la classe prédictive** : La classe avec la plus haute probabilité (ou son logarithme) est choisie comme la prédiction pour chaque observation.

La principale différence est que la fonction #19 est définie comme une fonction générique à part entière, prête à être utilisée sur différents ensembles de données ou modèles, tandis que la fonction #14 représente un script spécifique pour une tâche de prédiction donnée.

---

### Fonction #20
```{r}
result_pred <- Predict_Bayes(model_1, data_test)
```

### Explication
Ce code utilise la fonction `Predict_Bayes` pour faire des prédictions sur `data_test` en utilisant le modèle `model_1`.

---

### Fonction #21
```{r}
result_pred %>% kbl(digits=3) %>%    
       kable_styling(bootstrap_options = "striped", full_width = F, position = "center", latex_options = 'stripped') %>% 
       scroll_box( height = "250px")
```

### Explication
Ici, les résultats des prédictions (`result_pred`) sont formatés et affichés sous forme de tableau avec des options de style similaires aux précédentes.

---

### Fonction #22
```{r}
load('data_test.Rda')
```

### Explication
Cette commande charge les données sauvegardées dans le fichier `data_test.Rda`. Le format `.Rda` est utilisé pour stocker des objets R de manière efficace. En chargeant ce fichier, on récupère les données ou les objets qu'il contient dans l'environnement de travail actuel.

---

### Fonction #23
```{r}
X <- data_test[,1:11]
obs <- data_test[,12] 
model <- Naive_Bayes(X, obs, m = 1)
print(model)
```

### Explication
1. `X <- data_test[,1:11]` : Cette ligne sélectionne les 11 premières colonnes de `data_test` pour les utiliser comme variables explicatives (`X`).
2. `obs <- data_test[,12]` : Ici, on sélectionne la 12ème colonne de `data_test` comme variable cible (`obs`).
3. `model <- Naive_Bayes(X, obs, m = 1)` : Crée un modèle Naïve Bayes avec ces données.
4. `print(model)` : Affiche le modèle créé.

---

### Fonction #24
```{r}
pred_1 <- Predict_Bayes(model, X)
table(pred_1$pred, obs)
```

### Explication
1. `pred_1 <- Predict_Bayes(model, X)` : Utilise la fonction `Predict_Bayes` pour faire des prédictions sur `X` en utilisant le modèle `model`.
2. `table(pred_1$pred, obs)` : Crée un tableau de contingence entre les prédictions (`pred_1$pred`) et les observations réelles (`obs`). C'est utile pour évaluer la performance du modèle.

---

### Fonction #25
```{r}
model_2 <- naiveBayes(event ~ ., data = data_test)
pred_2 <- predict(model_2, newdata = X) 
confusionMatrix(pred_2, obs)
```

### Explication
1. `model_2 <- naiveBayes(event ~ ., data = data_test)` : Utilise la fonction `naiveBayes` (probablement du package `e1071` ou similaire) pour créer un modèle Naïve Bayes. `event ~ .` indique que `event` est la variable à prédire, et le `.` représente toutes les autres variables de `data_test` comme variables explicatives.
2. `pred_2 <- predict(model_2, newdata = X)` : Prédit les valeurs de `X` en utilisant le modèle `model_2`.
3. `confusionMatrix(pred_2, obs)` : Affiche la matrice de confusion pour les prédictions `pred_2` par rapport aux observations réelles `obs`. La matrice de confusion est un outil important pour évaluer la performance des modèles de classification.

---





