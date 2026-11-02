---
title: " Partie 2 Qualité et préparation des données"
author: "NGUYEN Duc Hoang Viet  & Sun Zhenxuan "
date: "2026-10-27"

jupyter: projet-velib
number-sections: true 
code-fold: false
execute:
  warning: false
  message: false
---

```{python}
import numpy as np
import pandas as pd
from velib_utils import (load_raw_dt, coverage, qualite, prepare_sets, population_table,
                         split_summary, n_unaligned, FEATURE_GROUPS, TZ)
```

# Extraction et vérification des données

Les données sont téléchargées depuis `docs.ambroise.cloud` ; leur empreinte MD5 est vérifiée à chaque chargement (le calcul s'arrête si elle ne correspond pas).

```{python}
raw = load_raw_dt(verbose=True)
raw.head(3)
```

# Résumé des données

## Aperçu des données

```{python}
print("Nombre de lignes :", raw.shape[0])
print("Nombre de colonnes :", raw.shape[1])
```

```{python}
raw.info()
```

```{python}
coverage(raw)
```

## Contrôle de la qualité des données

Avant toute analyse, on vérifie la qualité du jeu de données : présence de doublons et de valeurs manquantes.

### Lignes dupliquées

```{python}
print("Nombre de lignes dupliquées :", raw.duplicated().sum())
```

### Valeurs manquantes

```{python}
na = raw.isna().sum()
na[na > 0].to_frame("manquantes").assign(part=lambda d: (d["manquantes"] / len(raw)).round(4))
```

Seule la colonne `capacity` contient des valeurs manquantes (1 920 lignes, soit environ 0,6 % des données). Toutes les autres colonnes sont complètes.

#### Où manque la capacité ?

```{python}
missing_capacity = raw[raw["capacity"].isna()]
print("Stations concernées :", missing_capacity["station_code"].nunique())
print("Observations par station :", missing_capacity["station_code"].value_counts().unique())
g = missing_capacity.groupby("collected_at_utc")["station_code"].nunique()
print("Nombre de stations concernées, par horodatage :")
print(g.value_counts())
```

Les valeurs manquantes ne sont pas dispersées au hasard : elles concernent **les 80 stations**, chacune ayant exactement **24 observations** sans capacité. Il s'agit donc d'un problème lié à une **période de collecte** (24 relevés par station, sur 26 horodatages : 22 communs aux 80 stations et 4 où seul un des deux groupes, Paris ou banlieue, a été collecté), et non à certaines stations ni à des erreurs ponctuelles de mesure.

La capacité est constante pour une station donnée : c'est un attribut statique. On la complète donc par station à partir de ses autres relevés (maximum observé). Ce n'est ni une interpolation ni une fuite d'information du futur. La variable n'intervient pas dans la définition de la cible, qui repose sur `mechanical_bikes + electric_bikes`.

### Cohérence interne et statut des stations {#sec-coherence}

```{python}
Q = qualite(raw)
Q["coherence"]
```

- **`vélos + bornettes > capacité`** (2 256 lignes) est concentré sur quelques stations : la station 45504 en regroupe à elle seule 1 641 (environ 73 %), suivie des stations 9104 (400) et 40007 (145). La capacité déclarée y est peu fiable. C'est pourquoi le taux de remplissage est calculé par `vélos / (vélos + bornettes)` plutôt que par rapport à `capacity`.
- **`vélos + bornettes < capacité`** (168 340 lignes, environ la moitié du jeu) reflète simplement des bornettes hors service.
- **`vélos > capacité`** (86 lignes) et **`capacité manquante`** (1 920 lignes, soit 80 stations × 24 relevés) sont marginaux. Aucune ligne n'a `capacité == 0`, ni `vélos + bornettes == 0`.

#### Stations quasi figées et incohérences de capacité

```{python}
S_ = Q["static_stations"]
top = S_.nlargest(5, "unchanged_share").index.union(S_.nlargest(5, "over_capacity_rows").index)
S_.loc[top].sort_values("unchanged_share", ascending=False).round(3)
```

Une station typique est inchangée d'un relevé à l'autre dans 38 % des cas (médiane sur les 80 stations) ; la station **45504** l'est dans 89 % des cas et concentre l'essentiel des incohérences de capacité. Elle est traitée comme suspecte et écartée dans l'analyse de sensibilité (Partie 3).

#### Statut des stations

```{python}
closed = raw["is_renting"] == 0
print(raw[["is_installed", "is_renting", "is_returning"]].apply(lambda s: s.value_counts()))
print()
print(f"Lignes hors service : {closed.sum()} ({closed.mean():.2%})")
print("is_renting == 0 et is_returning == 0 sur les mêmes lignes :",
      (closed == (raw["is_returning"] == 0)).all())
```

Les stations hors service représentent 812 lignes (0,25 %). Elles ont `is_renting = is_returning = 0` sur exactement les mêmes lignes ; une seule des deux variables suffit donc.

### Fiabilité de la collecte {#sec-collecte}

#### Interruptions de collecte

```{python}
Q["gaps"]
```

#### Relevés incomplets

```{python}
print(f"{Q['n_incomplets']} relevés incomplets ; ensembles de stations disjoints et couvrant les 80 stations : {Q['complementaires']}")
Q["ensembles_incomplets"]
```

#### Stations fermées (`is_renting = 0`)

```{python}
Q["fermetures"]
```

#### Fiabilité de `source_updated_at`

```{python}
print("collected_at_utc - source_updated_at (minutes):")
display(Q["retard_source"].round(1).to_frame("minutes").T)
print("rows whose source timestamp did not change while the bike count did:",
      Q["source_figee_mais_velos_changent"])
```

`source_updated_at` ne permet pas de dater l'état des stations de façon fiable :

- l'horodatage de la source est en retard sur la collecte : de 62 min en médiane, 98 min pour 90 % des lignes et 150 min pour 99 % d'entre elles, avec un maximum d'environ 20 h ;
- dans 91 373 lignes (environ 28 %), il ne change pas alors que le nombre de vélos change.

Nous utilisons donc `collected_at_utc` comme seule référence temporelle et n'employons pas `source_updated_at` dans les variables du modèle.

# Grille régulière, cible et population

```{python}
HORIZONS = (30, 60)
S = {h: prepare_sets(raw, h) for h in HORIZONS}

print("Lignes brutes écartées car hors grille de 15 minutes :", n_unaligned(raw))
for h in HORIZONS:
    print(f"horizon {h} min")
    display(population_table(S[h]["reason"], h))
```

## Distribution de la cible

```{python}
tr = S[30]["train"]
hour = tr["t"].dt.tz_convert(TZ).dt.hour

display(tr.groupby(hour)["y"].mean().rename("taux de positifs par heure (Paris), train").to_frame().T.round(3))
display(tr.groupby(np.where(tr["paris"] == 1, "Paris", "banlieue"))["y"].mean()
          .rename("taux de positifs, train").to_frame().round(3))
display(tr.groupby("station_code")["y"].mean().describe()
          .rename("taux de positifs par station, train").to_frame().T.round(3))

trval = pd.concat([S[30]["train"], S[30]["val"]])
week = trval["t"].dt.tz_convert(TZ).dt.isocalendar().week.to_numpy()
trval.groupby(week)["y"].mean().rename("taux de positifs par semaine ISO (train + val)").to_frame().T.round(3)
```

## Découpage chronologique

```{python}
for h in HORIZONS:
    print(f"horizon {h} min")
    display(split_summary(S[h]))
```

## Variables

```{python}
pd.Series({g: ", ".join(f) for g, f in FEATURE_GROUPS.items()}, name="variables").to_frame()
```

```{python}
FEATS = S[30]["FEATS"]

display(tr[FEATS].isna().mean().rename("part de valeurs manquantes (train)").to_frame().T.round(3))
display(tr.groupby(pd.cut(tr["N"], [-1, 0, 1, 2, 4, 8, 100]), observed=True)["y"].mean()
          .rename("P(y = 1 | vélos maintenant), train").to_frame().T.round(3))
```
