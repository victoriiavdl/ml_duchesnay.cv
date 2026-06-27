# Prosper Loan Default Prediction - Prédiction du risque de défaut

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://mlduchesnaycv-f2rtatwlkasrapghqsgg72.streamlit.app/)

## Présentation

**Prosper Loan Default Prediction** est un projet de machine learning visant à prédire si un emprunteur va faire **défaut sur un prêt** issu de la plateforme de prêt entre particuliers [Prosper](https://www.prosper.com/).

Le projet permet de :
- **Explorer** un jeu de données financier complexe (113 937 prêts, 81 variables) via une analyse univariée et bivariée approfondie
- **Construire des features** pertinentes pour la détection du risque de crédit (ratio d'endettement, flag d'historique de défaut, stress revolving)
- **Comparer** quatre algorithmes de classification (Régression Logistique, Random Forest, XGBoost, SVM) tunés par GridSearchCV
- **Visualiser** les résultats et l'importance des variables dans un dashboard Streamlit interactif

## Données utilisées

| Source | Description |
|--------|-------------|
| **Prosper Marketplace** | 113 937 prêts entre particuliers (2005–2014), 81 variables : taux d'intérêt, score de crédit, statut d'emploi, historique de paiement, etc. |
| **Kaggle** | [Dataset disponible ici](https://www.kaggle.com/datasets/henryokam/prosper-loan-data/data) |

> Le fichier `prosperLoanData.csv` (86 Mo) n'est pas inclus dans ce dépôt. À télécharger depuis Kaggle et à placer à la racine du projet pour activer l'EDA en direct dans l'application.

## Structure du projet

```
projet_ml_duchesnay/
├── app.py                          # Dashboard Streamlit (4 pages)
├── eda_cleaning.ipynb              # Nettoyage, valeurs manquantes, EDA, feature engineering
├── loan_default_prediction.ipynb   # Modélisation, tuning et évaluation des modèles
├── ProsperLoanDataDoc.pdf          # Dictionnaire officiel des variables Prosper
├── requirements.txt                # Dépendances Python
└── pyproject.toml                  # Configuration du projet
```

## Stack technique

- **Python 3.12+**
- **Streamlit** — framework web pour le dashboard
- **Pandas / NumPy** — manipulation et analyse des données
- **Matplotlib / Seaborn** — visualisations
- **Scikit-learn** — preprocessing, pipelines, GridSearchCV, métriques
- **XGBoost** — meilleur modèle final (ROC-AUC 0.759)

## Installation locale

```bash
git clone <url-du-repo>
cd projet_ml_duchesnay

# Avec pip
pip install -r requirements.txt
streamlit run app.py
```

> Placer `prosperLoanData.csv` à la racine pour débloquer l'EDA en direct. Sans le CSV, l'application affiche les résultats pré-calculés.

## Contexte

Ce projet a été réalisé dans le cadre du cours de **Machine Learning** (Master MOSEF, Université Paris 1 Panthéon-Sorbonne). L'objectif était d'appliquer une démarche complète de data science sur des données financières réelles : exploration, nettoyage, feature engineering, modélisation et évaluation.
