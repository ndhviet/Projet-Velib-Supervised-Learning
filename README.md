<div align="center">

<h1>Projet Vélib' — Supervised Learning</h1>

<p><em>Prédire, 30 et 60 minutes à l'avance, les stations Vélib' qui n'auront presque plus de vélos</em></p>

<a href="#"><img src="https://img.shields.io/badge/Python-3-3776AB?logo=python&logoColor=white&style=flat-square"></a>
<a href="#"><img src="https://img.shields.io/badge/scikit--learn-Supervised%20Learning-F7931E?logo=scikitlearn&logoColor=white&style=flat-square"></a>
<a href="#"><img src="https://img.shields.io/badge/Quarto-Website-75AADB?logo=quarto&logoColor=white&style=flat-square"></a>
<a href="#"><img src="https://img.shields.io/badge/Spatio--Temporal-Forecasting-success?style=flat-square"></a>
<a href="#"><img src="https://img.shields.io/badge/Status-In%20progress-yellow?style=flat-square"></a>
<a href="#"><img src="https://img.shields.io/badge/License-MIT-blue?style=flat-square"></a>

<a href="https://ndhviet.github.io/Projet-Velib-Supervised-Learning/">
  <img src="https://img.shields.io/badge/Demo-Live%20Website-brightgreen?style=for-the-badge&logo=googlechrome&logoColor=white">
</a>

</div>
<br>



<h2 align="center">Réalisé par</h2>

<div align="center">

| Membre | Contact |
|---|---|
| **Duc Hoang Viet NGUYEN** | [ndhviet.github.io](https://ndhviet.github.io/) |
| **Zhenxuan SUN** | [LIEN-A-COMPLETER](#) |

</div>

<br>

## À propos

Pour chaque station et chaque relevé (pas de 15 minutes), on estime la probabilité que la station compte **moins de deux vélos** dans 30 ou 60 minutes. Les données couvrent 80 stations du 25 août au 8 octobre 2026.

La démarche suit un protocole strict : découpage **chronologique** (train, validation, test), choix des modèles et des seuils **sur la validation seulement**, test noté **une seule fois**, et comparaison systématique à une règle de **persistance** (« dans 30 minutes, ce sera comme maintenant »).

**Modèles comparés :** Bayes naïf gaussien (écrit à la main), régression logistique, k plus proches voisins (usuel et pondéré), arbre de décision, forêt aléatoire, gradient boosting.

## Contenu du site

| Partie | Contenu |
|---|---|
| **1. Exploration** | Carte des stations, usage par heure et par jour, taux d'occupation |
| **2. Qualité et préparation** | Valeurs manquantes, coupures de collecte, grille de 15 minutes, cible, découpage, variables |
| **3. Modèles supervisés** | Construction du problème, références, Bayes naïf, entraînement et réglage |
| **4. Comparaison des modèles** | Validation, évaluation sur la semaine de test, analyses par groupe, bootstrap, sensibilité, ablation |
| **5. Conclusion** | Résultats et limites |

## Reproduire le projet

```bash
git clone https://github.com/ndhviet/NOM-DU-DEPOT.git
cd NOM-DU-DEPOT
bash setup.sh          # crée .venv, installe les paquets, enregistre le noyau Jupyter
quarto render          # génère le site dans docs/
```

Les données (`velib-2026-10-08.csv.gz`) sont téléchargées automatiquement, et leur empreinte MD5 est vérifiée à chaque chargement. Le code commun est dans `velib_utils.py`.
