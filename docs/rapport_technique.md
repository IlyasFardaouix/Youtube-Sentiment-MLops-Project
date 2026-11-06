# Rapport technique - YouTube Sentiment Lens

## 1. Problème et solution

Le projet automatise l'analyse de commentaires YouTube en trois classes : négatif (`-1`), neutre (`0`) et positif (`1`). Les commentaires visibles sont extraits par une extension Chrome, envoyés par batch à une API FastAPI, puis classés par un pipeline TF-IDF et Logistic Regression.

## 2. Données et préparation

La source officielle Reddit contient 37 249 commentaires (répartition brute : négatif 8 277, neutre 13 142, positif 15 830). Après mise en minuscules, nettoyage, suppression des doublons et des textes de moins de trois caractères, 36 562 observations restent disponibles (négatif 8 233, neutre 12 606, positif 15 723), soit bien plus de 1 000 par classe. Le split stratifié reproductible (`random_state=42`) utilise 80 % pour l'entraînement (29 249) et 20 % pour le test (7 313).

Le nettoyage remplace les URLs, mentions et hashtags par des tokens (`URL`, `USER`, `HASHTAG`) afin de retirer le bruit sans perdre totalement leur signal. Les caractères non alphanumériques sont normalisés et les emojis Unicode sont conservés lorsqu'ils sont compatibles avec le texte.

## 3. Modélisation

Deux modèles ont été comparés : Logistic Regression et Linear SVM, tous deux avec TF-IDF et pondération des classes. Le Linear SVM non optimisé obtient une accuracy brute légèrement supérieure (89,9 % contre 86,2 %), mais la Logistic Regression a été retenue car elle fournit des probabilités (`predict_proba`) indispensables au score de confiance affiché par l'extension. Elle a ensuite été optimisée avec GridSearchCV (`cv=3`, score `f1_macro`) sur `C`, `min_df` et `ngram_range`; le meilleur réglage est `C=2.0`, `min_df=2`, `ngram_range=(1,1)`, portant son accuracy à 87,8 %. La comparaison des candidats est faite sur le jeu de test : c'est un choix pragmatique, une sélection sur un jeu de validation dédié serait plus rigoureuse.

## 4. Résultats

Les résultats sont issus du test set de 7 313 commentaires :

| Mesure | Résultat | Seuil | Statut |
|---|---:|---:|---|
| Accuracy | 87,79 % | 80 % | Atteint |
| F1 négatif | 81,18 % | 75 % | Atteint |
| F1 neutre | 90,79 % | 75 % | Atteint |
| F1 positif | 88,69 % | 75 % | Atteint |
| Inférence, 50 textes | 3,57 ms | 100 ms | Atteint |

La matrice de confusion est disponible dans `models/metrics.json`. Les erreurs principales opposent les classes négative et positive aux commentaires neutres ou ambigus.

## 5. API et extension

`GET /health` vérifie le chargement du modèle. `POST /predict_batch` valide un batch de 1 à 500 commentaires et renvoie le label, le sentiment, la confiance et les statistiques. L'extension Chrome Manifest V3 extrait les commentaires visibles, affiche les pourcentages, filtre les résultats, propose un mode sombre et copie les résultats.

## 6. Déploiement

Le `Dockerfile` utilise `python:3.10-slim`, installe uniquement `requirements-prod.txt`, copie le code et les modèles, et expose un port configurable via la variable `PORT` (défaut 7860). Le déploiement s'effectue dans un Codespace GitHub : `docker build` puis `docker run`, port 7860 rendu public. L'URL Codespaces est éphémère (arrêt automatique après environ 30 min d'inactivité) et doit être relancée avant la démonstration. La vidéo de démonstration relève de l'étudiant.

## 7. Difficultés et solutions

Le premier GridSearchCV a échoué avec `BrokenProcessPool` sous Windows. Le parallélisme a été fixé à `n_jobs=1`, ce qui a permis l'entraînement complet. L'interpréteur Python de VS Code ne correspondait pas au Python contenant les dépendances; il a été configuré sur Python 3.12. Les tests API et le smoke test réel passent après ces corrections.

## 8. Améliorations futures

Ajouter une collecte continue des nouveaux commentaires, une authentification API, un monitoring de dérive, une mise à jour versionnée du modèle et une stratégie de cache côté extension.
