
<div align="center">

<h1>Projet Vélib' — Supervised Learning</h1>

<p>
  <em>Prédiction des stations Vélib' à risque de pénurie de vélos à horizons de 30 et 60 minutes</em>
</p>

<p>
  <img src="https://img.shields.io/badge/Python-3-3776AB?logo=python&logoColor=white&style=flat-square" alt="Python">
  <img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white&style=flat-square" alt="scikit-learn">
</p>

<a href="https://ndhviet.github.io/Projet-Velib-Supervised-Learning/">
  <img src="https://img.shields.io/badge/View-Live%20Demo-2ea44f?style=for-the-badge&logo=githubpages&logoColor=white" alt="Live Demo">
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


# Vélib' sous tension — Question A: *departure impossible*

M2 Data Science project (Université Paris-Saclay). Can we anticipate, **30 or 60 minutes ahead**, that a Vélib'
station will have **fewer than 2 bikes available**?

## Exact target

For station `s` and time `t` (15-minute UTC grid), with `h` ∈ {30, 60} minutes:

```
y(s, t) = 1  if  bikes_available(s, t + h) < 2   else 0
```

* Population: all 80 stations, one common model. A sample is kept when the station is **observed and open**
  (`is_renting = 1`) at `t` and at `t + h`, and `t + h` falls in the same block (train / validation / test) as `t`.
  Every excluded slot is counted by reason and by block (`reports/results/population_{30,60}min.csv`). Counting a
  closure at `t + h` as "departure impossible" instead is tested as a sensitivity analysis.
* References: the majority class, and **persistence**: `ŷ(s, t) = 1{bikes_available(s, t) < 2}`. For PR curves and
  PR-AUC, persistence ranks samples by the current number of bikes, a graded score that is generous to the reference.

## Data

| | |
|---|---|
| Source | Vélib' Métropole — real-time station availability, [Paris Data](https://opendata.paris.fr) |
| Licence | Open Database License (ODbL). Attribution: *Vélib' Métropole / Ville de Paris, Paris Data* |
| File | `velib-2026-10-08.csv.gz` — 329,680 rows, 80 stations (50 Paris, 30 suburbs), 25 Aug → 8 Oct 2026 |
| URL | https://docs.ambroise.cloud/velib/velib-2026-10-08.csv.gz |
| md5 | `02ab1e3fdac10579e4ce93d1540a5dc7` |
