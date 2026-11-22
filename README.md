
<div align="center">

<h1>Projet Vélib' — Supervised Learning</h1>

<a href="https://ndhviet.github.io/Projet-Velib-Supervised-Learning/">
  <img src="https://img.shields.io/badge/View-Live%20Demo-2ea44f?style=for-the-badge&logo=githubpages&logoColor=white" alt="Live Demo">
</a>

<h3>Réalisé par</h3>

<p>
  <a href="https://ndhviet.github.io/">Duc Hoang Viet NGUYEN</a>
  ·
  Zhenxuan SUN
</p>

</div>



# Vélib' sous tension — Question A: *departure impossible*
Projet de M2 Data Science (Université Paris-Saclay). Peut-on anticiper, 30 ou 60 minutes à l'avance, qu'une station Vélib' disposera de moins de 2 vélos disponibles ?


## À propos

Pour chaque station et chaque relevé (pas de 15 minutes), on estime la probabilité que la station compte **moins de deux vélos** dans 30 ou 60 minutes. Les données couvrent 80 stations du 25 août au 8 octobre 2026.

La démarche suit un protocole strict : découpage **chronologique** (train, validation, test), choix des modèles et des seuils **sur la validation seulement**, test noté **une seule fois**, et comparaison systématique à une règle de **persistance** (« dans 30 minutes, ce sera comme maintenant »).

**Modèles comparés :** Bayes naïf gaussien (écrit à la main), régression logistique, k plus proches voisins (usuel et pondéré), arbre de décision, forêt aléatoire, gradient boosting.

## Reproduire le projet

```bash
git clone https://github.com/ndhviet/NOM-DU-DEPOT.git
cd NOM-DU-DEPOT
bash setup.sh          # crée .venv, installe les paquets, enregistre le noyau Jupyter
quarto render          # génère le site dans docs/
```

Les données (`velib-2026-10-08.csv.gz`) sont téléchargées automatiquement, et leur empreinte MD5 est vérifiée à chaque chargement. Le code commun est dans `velib_utils.py`.




## Définition exacte de la variable cible

Pour une station `s` et un instant `t` (grille temporelle UTC à pas de 15 minutes), avec `h` ∈ {30, 60} minutes:

```
y(s, t) = 1  if  bikes_available(s, t + h) < 2   else 0
```

* Population étudiée : les 80 stations, avec un modèle commun. Une observation est conservée lorsque la station est observée et ouverte (is_renting = 1) à t et à t + h, et que t + h appartient au même bloc (entraînement / validation / test) que t. Chaque créneau exclu est comptabilisé selon son motif d'exclusion et son bloc (reports/results/population_{30,60}min.csv). Une analyse de sensibilité examine également le cas où une fermeture à t + h est considérée comme un « départ impossible ».

* Modèles de référence : la classe majoritaire et la persistance : ŷ(s, t) = 1{bikes_available(s, t) < 2}. Pour les courbes précision-rappel et l'aire sous ces courbes (PR-AUC), la persistance classe les observations selon le nombre de vélos actuellement disponibles, à l'aide d'un score gradué qui favorise cette référence.

## Données

| | |
|---|---|
| Source | Vélib' Métropole — real-time station availability, [Paris Data](https://opendata.paris.fr) |
| Licence | Open Database License (ODbL). Attribution: *Vélib' Métropole / Ville de Paris, Paris Data* |
| File | `velib-2026-10-08.csv.gz` — 329,680 rows, 80 stations (50 Paris, 30 suburbs), 25 Aug → 8 Oct 2026 |
| URL | https://docs.ambroise.cloud/velib/velib-2026-10-08.csv.gz |
| md5 | `02ab1e3fdac10579e4ce93d1540a5dc7` |





