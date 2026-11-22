import hashlib
import os
import urllib.request

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, fbeta_score, log_loss,
                             precision_recall_curve, precision_score, recall_score)
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

URL = "https://docs.ambroise.cloud/velib/velib-2026-10-08.csv.gz"
FILE = "data/velib-2026-10-08.csv.gz"
MD5 = "02ab1e3fdac10579e4ce93d1540a5dc7"
TZ = "Europe/Paris"

# Frontières chronologiques (UTC)
T_VAL = pd.Timestamp("2026-09-24 13:45", tz="UTC")                    # 1er créneau après la panne de 28 h
T_TEST = pd.Timestamp("2026-10-01 00:00", tz=TZ).tz_convert("UTC")    # début de la semaine de test

TAU_PERS = 0.4   # score de persistance > 0.4  <=>  N < 2
HOUR_BANDS = {"nuit 0-7h": (0, 7), "pointe matin 7-10h": (7, 10), "journée 10-17h": (10, 17),
              "pointe soir 17-20h": (17, 20), "soirée 20-24h": (20, 24)}
SUSPECT_STATION = 45504   # station peut-être figée dans le flux (voir Partie 2)
SITUATIONS = ["reste disponible", "devient vide", "reste vide", "redevient disponible"]

FEATURE_GROUPS = {
    "état": ["bikes_available", "docks_available", "fill"],
    "tendance": ["d_bikes_15", "d_bikes_30", "d_bikes_60", "bikes_24h"],
    "calendrier": ["hour_sin", "hour_cos", "dow", "weekend"],
    "station": ["capacity", "paris", "latitude", "longitude", "st_empty_rate"],
}
FEATURES = [f for fs in FEATURE_GROUPS.values() for f in fs]
# k-NN (distance) et Bayes naïf (gaussien) liraient `dow` (0..6) comme une grandeur ordonnée
# (dimanche « loin » de lundi) : ils utilisent `weekend` + l'heure à la place.
FEATURES_NO_DOW = [f for f in FEATURES if f != "dow"]


# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------
def load_raw(verbose=False):
    """Télécharge (une seule fois), vérifie le MD5 et lit le CSV brut.
    verbose=True : affiche le téléchargement et les deux empreintes MD5."""
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(FILE):
        urllib.request.urlretrieve(URL, FILE)
        if verbose:
            print("Download successful:", FILE)
    elif verbose:
        print("File already downloaded:", FILE)
    with open(FILE, "rb") as f:
        calculated = hashlib.md5(f.read()).hexdigest()
    if verbose:
        print("-------------------------------------------------")
        print("Calculated MD5:", calculated)
        print("Expected MD5:  ", MD5)
    assert calculated == MD5, f"MD5 mismatch: {calculated}"
    if verbose:
        print("MD5 OK!")
    return pd.read_csv(FILE)


def prepare(raw):
    """Nettoyage + variables dérivées pour l'EDA. Renvoie (df, stations)."""
    df = raw.copy()
    df["ts"] = pd.to_datetime(df["collected_at_utc"], errors="coerce", utc=True) \
                 .dt.tz_convert(TZ)
    df = df.dropna(subset=["ts", "latitude", "longitude", "capacity"])
    df = df[(df["capacity"] > 0) & (df["is_installed"] == 1)].copy()

    df["hour"] = df["ts"].dt.hour
    df["weekday"] = df["ts"].dt.dayofweek
    df["periode"] = np.where(df["weekday"] >= 5, "Week-end", "Semaine")
    df["occupancy"] = df["bikes_available"] / df["capacity"]

    stations = (
        df.groupby("station_code", as_index=False)
          .agg(name=("name", "first"), commune=("commune", "first"),
               capacity=("capacity", "first"),
               latitude=("latitude", "first"), longitude=("longitude", "first"),
               bikes_available=("bikes_available", "mean"),
               occupancy=("occupancy", "mean"))
    )
    return df, stations


def load_raw_dt(verbose=False):
    """Comme load_raw(), mais convertit les deux colonnes temporelles en datetime UTC."""
    raw = load_raw(verbose)
    for c in ["collected_at_utc", "source_updated_at"]:
        raw[c] = pd.to_datetime(raw[c], utc=True)
    return raw


# ---------------------------------------------------------------------------
# Qualité des données (description seulement : rien n'est modifié)
# ---------------------------------------------------------------------------
def coverage(df: pd.DataFrame) -> pd.Series:
    """Couverture de la collecte : lignes, stations, relevés, bornes temporelles, doublons."""
    per_station = df.groupby("station_code").size()
    return pd.Series({
        "rows": len(df),
        "stations": df["station_code"].nunique(),
        "snapshots": df["collected_at_utc"].nunique(),
        "first_snapshot_utc": df["collected_at_utc"].min(),
        "last_snapshot_utc": df["collected_at_utc"].max(),
        "rows_per_station_min": per_station.min(),
        "rows_per_station_max": per_station.max(),
        "duplicated_station_snapshot": int(df.duplicated(["station_code", "collected_at_utc"]).sum()),
    })


def qualite(raw):
    """Tables de description de la collecte. Renvoie un dictionnaire de tables."""
    r = raw.sort_values(["station_code", "collected_at_utc"]).reset_index(drop=True)
    g = r.groupby("station_code")
    n_st = r["station_code"].nunique()

    # Interruptions entre relevés consécutifs
    snaps = pd.Series(np.sort(r["collected_at_utc"].unique()))
    minutes = snaps.diff().dt.total_seconds() / 60
    gaps = pd.DataFrame({"début": snaps.shift(1), "fin": snaps, "minutes": minutes})
    gaps = gaps[minutes > 20].sort_values("minutes", ascending=False)

    # Relevés incomplets : quels ensembles de stations ?
    par_releve = r.groupby("collected_at_utc")["station_code"].apply(frozenset)
    partiels = par_releve[par_releve.map(len) < n_st]
    ensembles = sorted(set(partiels), key=len)
    paris = set(r.loc[r["commune"].eq("Paris"), "station_code"])
    sets_table = pd.DataFrame([{"stations dans l'ensemble": len(s),
                                "relevés": int(partiels.map(lambda x: x == s).sum()),
                                "dont Paris": len(s & paris)} for s in ensembles])
    union = frozenset().union(*ensembles) if ensembles else frozenset()
    complementaires = bool(ensembles) and sum(map(len, ensembles)) == len(union) == n_st

    # Épisodes de fermeture (is_renting == 0)
    closed = r["is_renting"].eq(0)
    prev = closed.groupby(r["station_code"]).shift(1, fill_value=False)
    episode = (closed & ~prev).groupby(r["station_code"]).cumsum()
    episodes = (r[closed].assign(episode=episode[closed])
                .groupby(["station_code", "episode"])["collected_at_utc"]
                .agg(début="min", fin="max", relevés="size").reset_index())

    # Stations quasi figées
    same = (g["bikes_available"].diff() == 0) & (g["docks_available"].diff() == 0)
    figees = (same.groupby(r["station_code"]).mean().sort_values(ascending=False)
              .rename("part de relevés inchangés"))

    # Cohérence avec la capacité
    total = r["bikes_available"] + r["docks_available"]
    over = r[total > r["capacity"]].groupby("station_code").size().sort_values(ascending=False)
    coherence = pd.Series({
        "bikes != mécaniques + électriques": int((r["bikes_available"] != r["mechanical_bikes"] + r["electric_bikes"]).sum()),
        "vélos + bornettes > capacité": int((total > r["capacity"]).sum()),
        "vélos + bornettes < capacité": int((total < r["capacity"]).sum()),
        "vélos + bornettes == 0": int((total == 0).sum()),
        "vélos > capacité": int(((r["mechanical_bikes"] + r["electric_bikes"]) > r["capacity"]).sum()),
        "capacité == 0": int((r["capacity"] == 0).sum()),
        "capacité manquante": int(r["capacity"].isna().sum()),
    })

    # Fiabilité de source_updated_at
    retard = (r["collected_at_utc"] - r["source_updated_at"]).dt.total_seconds() / 60
    src_same = g["source_updated_at"].diff() == pd.Timedelta(0)
    bikes_moved = g["bikes_available"].diff().fillna(0) != 0

    # Couverture globale
    par_station = g.size()
    slot = r["collected_at_utc"].dt.round("15min")
    couverture = pd.Series({
        "lignes": len(r),
        "stations": n_st,
        "relevés (horodatages distincts)": r["collected_at_utc"].nunique(),
        "premier relevé (UTC)": r["collected_at_utc"].min(),
        "dernier relevé (UTC)": r["collected_at_utc"].max(),
        "lignes par station (min)": int(par_station.min()),
        "lignes par station (max)": int(par_station.max()),
        "doublons (station, relevé)": int(r.duplicated(["station_code", "collected_at_utc"]).sum()),
        "lignes hors grille de 15 min (> 1 min)": int(((r["collected_at_utc"] - slot).abs() > pd.Timedelta("1min")).sum()),
    })

    # Stations figées / hors capacité : une seule table par station
    na = r.isna().sum()
    statiques = pd.concat([figees.rename("unchanged_share"), over.rename("over_capacity_rows")], axis=1)
    statiques["over_capacity_rows"] = statiques["over_capacity_rows"].fillna(0).astype(int)
    statiques = statiques.sort_values("unchanged_share", ascending=False)

    return {"missing_values": na[na > 0], "static_stations": statiques, "coverage": couverture, "gaps": gaps, "ensembles_incomplets": sets_table, "complementaires": complementaires,
            "n_incomplets": len(partiels), "fermetures": episodes, "figees": figees,
            "over_capacity": over, "coherence": coherence,
            "retard_source": retard.describe(percentiles=[.5, .9, .99]),
            "source_figee_mais_velos_changent": int((src_same & bikes_moved).sum())}


# ---------------------------------------------------------------------------
# Grille, cible et variables
# ---------------------------------------------------------------------------
def build_features(raw, h=30, closed_as_positive=False, live=False):
    """Grille de 15 minutes, cible y (à t + h), variables construites avec le passé seul.

    Renvoie (feat, FEATS, reason). `reason` donne, pour chaque créneau (station, t) de la grille,
    la raison d'exclusion prioritaire (ou « conservée »).
    Une ligne est conservée si la station est observée et ouverte à t, et observée et ouverte à t+h.
    Les retards sont lus sur les créneaux OUVERTS seulement : un retard qui tombe dans une panne ou
    une fermeture vaut NaN (rien n'est interpolé ni rempli ; les modèles imputent la médiane du train).
    closed_as_positive=True : une fermeture à t+h compte comme « départ impossible » (y = 1).
    """
    steps = h // 15
    df = raw[raw["collected_at_utc"] != raw["collected_at_utc"].min()].copy()   # relevé 11:57, hors grille
    df["t"] = df["collected_at_utc"].dt.round("15min")
    assert not df.duplicated(["station_code", "t"]).any()

    info = df.groupby("station_code").agg(insee_code=("insee_code", "first"))
    grid = pd.date_range(df["t"].min(), df["t"].max(), freq="15min")
    idx = pd.MultiIndex.from_product([sorted(df["station_code"].unique()), grid],
                                     names=["station_code", "t"])
    g = df.set_index(["station_code", "t"]).reindex(idx)

    g["N"] = g["mechanical_bikes"] + g["electric_bikes"]       # sert à la cible et à la persistance
    # capacity est constante par station : la remplir n'est ni une interpolation ni une fuite
    g["capacity"] = g.groupby(level="station_code")["capacity"].transform("max")
    # fill utilise vélos + bornettes plutôt que la capacité déclarée (incohérente pour quelques stations)
    total = g["bikes_available"] + g["docks_available"]
    g["fill"] = g["bikes_available"] / total.where(total > 0)
    st = g.index.get_level_values("station_code")
    g["paris"] = (st.map(info["insee_code"]) == 75056).astype(int)

    by_st = lambda s: s.groupby(level="station_code", sort=False)

    closed_raw = (g["is_renting"].eq(0) | g["is_returning"].eq(0) | g["is_installed"].eq(0))
    prev = by_st(closed_raw).shift(1, fill_value=False)
    nxt = by_st(closed_raw).shift(-1, fill_value=False)
    g["closed"] = closed_raw | (prev & nxt)

    # Cible : N à t + h, pour la même station
    g["N_future"] = by_st(g["N"]).shift(-steps)
    fut_closed = by_st(g["closed"]).shift(-steps, fill_value=True)

    # Tendance : décalages par station = exactement k x 15 min plus tôt, sur créneaux ouverts seulement
    bikes_open = by_st(g["bikes_available"].where(~g["closed"]))
    for k in (1, 2, 4):
        g[f"d_bikes_{15 * k}"] = g["bikes_available"] - bikes_open.shift(k)
    g["bikes_24h"] = bikes_open.shift(24 * 60 // 15)

    valid = g["N"].notna() & g["N_future"].notna() & ~g["closed"]
    if not closed_as_positive:
        valid &= ~fut_closed
    if live:    # dernier relevé de la grille : l'avenir n'est pas encore connu, pas de cible
         valid = g["N"].notna() & ~g["closed"] & (g.index.get_level_values("t") == grid[-1])

    # Raison d'exclusion : la première condition vraie l'emporte
    r = np.select([g["N"].isna().to_numpy(), g["closed"].to_numpy(),
                   g["N_future"].isna().to_numpy(), fut_closed.to_numpy() & (not closed_as_positive)],
                  ["non observée à t", "fermée à t", "non observée à t+h", "fermée à t+h"],
                  default="conservée")
    reason = pd.Series(r, index=g.index)

    d = g[valid].copy()
    y = d["N_future"] < 2
    if closed_as_positive:
        y = y | fut_closed[valid]
    d["y"] = y.astype(int)

    feat = d.reset_index()
    local = feat["t"].dt.tz_convert(TZ)
    hour = local.dt.hour + local.dt.minute / 60
    feat["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    feat["dow"] = local.dt.dayofweek
    feat["weekend"] = (feat["dow"] >= 5).astype(int)
    # source_updated_at n'est pas utilisé : retard ~1 h et souvent figé (voir Partie 1)

    FEATS = [f for f in FEATURES if f != "st_empty_rate"]      # st_empty_rate est ajouté après le découpage
    return feat, FEATS, reason


def _bloc(t):
    """Bloc chronologique (train / val / test) de chaque instant t."""
    t = pd.DatetimeIndex(t)
    return np.select([t < T_VAL, t < T_TEST], ["train", "val"], default="test")


def split_time(feat, h=30):
    """Découpage chronologique. Une ligne est gardée seulement si t et t + h sont dans le même
    bloc : aucune étiquette d'un bloc n'est observée dans le bloc suivant (pas besoin d'embargo)."""
    t = feat["t"]
    b = _bloc(t)
    ok = b == _bloc(t + pd.Timedelta(minutes=h))
    return feat[ok & (b == "train")], feat[ok & (b == "val")], feat[ok & (b == "test")]


def population_table(reason, h=30):
    """Nombre de créneaux de la grille par raison d'exclusion et par bloc de t."""
    t = reason.index.get_level_values("t")
    b = _bloc(t)
    crosses = b != _bloc(t + pd.Timedelta(minutes=h))
    r = reason.to_numpy().copy()
    r[(r == "conservée") & crosses] = "t+h dans le bloc suivant"
    tab = pd.crosstab(pd.Series(r, name="raison"), pd.Series(b, name="bloc de t"))
    tab["total"] = tab.sum(axis=1)
    return tab[[c for c in ["train", "val", "test", "total"] if c in tab.columns]]


def n_unaligned(raw):
    """Lignes brutes écartées car hors de la grille de 15 minutes (le tout premier relevé)."""
    return int((raw["collected_at_utc"] == raw["collected_at_utc"].min()).sum())


def split_summary(S):
    """Effectif, taux de positifs et bornes (UTC) de chaque bloc. Le taux du test reste caché."""
    rows = {}
    for b in ["train", "val", "test"]:
        d = S[b]
        rows[b] = {"samples": len(d),
                   "positive rate": "(hidden)" if b == "test" else round(d["y"].mean(), 3),
                   "from (UTC)": d["t"].min(), "to (UTC)": d["t"].max()}
    return pd.DataFrame(rows).T


def add_station_rate(train, val, test, FEATS, rate=None):
    """st_empty_rate : part de créneaux « N < 2 » de chaque station, calculée sur le TRAIN seul
    puis appliquée telle quelle à val et test. `rate` permet de réutiliser un taux déjà figé."""
    if rate is None:
        rate = (train["N"] < 2).groupby(train["station_code"]).mean()
    out = [d.assign(st_empty_rate=d["station_code"].map(rate)) for d in (train, val, test)]
    return out[0], out[1], out[2], FEATS + ["st_empty_rate"], rate


def prepare_sets(raw, h=30, closed_as_positive=False, rate=None):
    feat, FEATS, reason = build_features(raw, h=h, closed_as_positive=closed_as_positive)
    train, val, test = split_time(feat, h)
    train, val, test, FEATS, rate = add_station_rate(train, val, test, FEATS, rate)
    return {"train": train, "val": val, "test": test, "FEATS": FEATS, "reason": reason, "rate": rate}


# ---------------------------------------------------------------------------
# Modèles
# ---------------------------------------------------------------------------
class BayesNaifGaussien:
    """Classifieur de Bayes naïf gaussien, écrit à la main."""

    def fit(self, X, y, eps=1e-9):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self.classes_ = np.unique(y)
        # pi_ : proportion de chaque classe (partie Bernoulli de Y)
        self.pi_ = np.array([np.mean(y == k) for k in self.classes_])
        # Hypothèse 1 : une moyenne et un écart-type par variable et par classe
        self.mu_ = np.array([X[y == k].mean(axis=0) for k in self.classes_])
        self.sigma_ = np.array([X[y == k].std(axis=0) for k in self.classes_]) + eps
        return self

    def _log_densite(self, X, k):
        mu, sigma = self.mu_[k], self.sigma_[k]
        # log-densité gaussienne de chaque variable (hypothèse 1)
        log_f_1d = (-0.5 * np.log(2 * np.pi) - np.log(sigma)
                    - (X - mu) ** 2 / (2 * sigma ** 2))        # (n, nb_variables)
        # Hypothèse 2 : indépendance conditionnelle -> somme des log-densités
        return np.sum(log_f_1d, axis=1)                         # (n,)

    def _log_posterieurs(self, X):
        return np.array([np.log(self.pi_[k]) + self._log_densite(X, k)
                         for k in range(len(self.classes_))])    # (2, n)

    def predict_proba(self, X):
        lp = self._log_posterieurs(np.asarray(X, dtype=float))
        lp = lp - lp.max(axis=0)          # stabilité numérique
        p = np.exp(lp)
        return (p / p.sum(axis=0)).T      # colonne 1 = P(Y = 1 | x)

    def predict(self, X, tau=None):
        if tau is None:
            return self.classes_[np.argmax(self._log_posterieurs(np.asarray(X, dtype=float)), axis=0)]
        return (self.predict_proba(X)[:, 1] > tau).astype(int)


def _imp():
    """Médiane du TRAIN pour les rares NaN de tendance (créneaux de retard fermés ou en panne)."""
    return SimpleImputer(strategy="median")


class _BayesImpute:
    """Médiane du train pour les NaN, puis Bayes naïf gaussien écrit à la main."""

    def fit(self, X, y):
        self.imp_ = _imp().fit(X)
        self.nb_ = BayesNaifGaussien().fit(self.imp_.transform(X), y)
        return self

    def predict_proba(self, X):
        return self.nb_.predict_proba(self.imp_.transform(X))


def _bayes():
    return _BayesImpute()


def _knn(k):
    return make_pipeline(_imp(), StandardScaler(), KNeighborsClassifier(n_neighbors=k, n_jobs=-1))


def _arbre(depth):
    return make_pipeline(_imp(), DecisionTreeClassifier(max_depth=depth, min_samples_leaf=50, random_state=0))


def foret(seed=0):
    return make_pipeline(_imp(), RandomForestClassifier(n_estimators=200, min_samples_leaf=20,
                                                        n_jobs=-1, random_state=seed))


def boosting(seed=0):
    # HistGradientBoosting gère les NaN nativement
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                          early_stopping=False, random_state=seed)


def regression_logistique():
    return make_pipeline(_imp(), StandardScaler(), LogisticRegression(max_iter=1000))


class FeatureWeighter(TransformerMixin, BaseEstimator):
    """Multiplie les colonnes (déjà standardisées) par des poids fixes : distance euclidienne pondérée pour le k-NN."""

    def __init__(self, weights=None):
        self.weights = weights

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float) * np.asarray(self.weights, dtype=float)


def _knn_pondere(k, weights):
    return make_pipeline(_imp(), StandardScaler(), FeatureWeighter(tuple(weights)),
                         KNeighborsClassifier(n_neighbors=k, n_jobs=-1))


def knn_weights(train, feats, h=30, seed=0, n_repeats=3):
    """Poids des variables pour le k-NN pondéré, appris DANS LE TRAIN seulement.

    La dernière semaine du train est mise de côté ; une forêt est ajustée sur le reste (avec un embargo de h minutes,
    pour qu'aucune étiquette de la forêt ne recouvre la semaine mise de côté) ; le poids d'une variable est son
    importance par permutation (baisse de l'AP quand on la mélange) sur la semaine mise de côté.
    Poids négatifs ramenés à 0, puis normalisés pour que le poids moyen soit 1. Renvoie une Series indexée par variable."""
    cut = train["t"].max() - pd.Timedelta(days=7)
    fit = train[train["t"] < cut - pd.Timedelta(minutes=h)]
    hold = train[train["t"] >= cut]
    m = foret(seed).fit(fit[feats], fit["y"])
    imp = permutation_importance(m, hold[feats], hold["y"], scoring="average_precision",
                                 n_repeats=n_repeats, random_state=seed, n_jobs=-1)
    w = pd.Series(np.clip(imp.importances_mean, 0, None), index=feats)
    if w.sum() == 0:
        w[:] = 1.0
    return w / w.sum() * len(feats)


def tune(make, grille, train, val, feats):
    """Hyperparamètre qui maximise l'AP sur la validation. Renvoie (meilleur, scores, modèle ajusté)."""
    scores, fitted = {}, {}
    for v in grille:
        m = make(v).fit(train[feats], train["y"])
        scores[v] = average_precision_score(val["y"], m.predict_proba(val[feats])[:, 1])
        fitted[v] = m
    best = max(scores, key=scores.get)
    return best, scores, fitted[best]


# ---------------------------------------------------------------------------
# Mesures
# ---------------------------------------------------------------------------
def persistence_score(N):
    """Score de classement de la persistance : plus il y a de vélos, plus le risque est faible.
    Ce n'est pas une probabilité calibrée ; il sert au calcul de l'AP. Score > 0.4 <=> N < 2."""
    return 1.0 / (1.0 + np.clip(np.asarray(N, dtype=float), 0, None))


def report(name, y, p, tau=0.5, c_fn=5, c_fp=1):
    """Indicateurs d'un modèle à un seuil tau donné (coût : c_fn/c_fp = 5, hypothèse provisoire)."""
    y = np.asarray(y)
    p = np.asarray(p)
    pred = (p > tau).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"model": name, "tau": tau, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred),
            "F1": f1_score(y, pred), "F2": fbeta_score(y, pred, beta=2),
            "AP": average_precision_score(y, p),
            "AP/pi": average_precision_score(y, p) / y.mean(),
            "Brier": brier_score_loss(y, p),
            "logloss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)),
            "cost": (c_fp * fp + c_fn * fn) / len(y)}


def precision_at_recall(y, p, target_recall):
    """Précision maximale atteignable avec un rappel >= target_recall."""
    precision, recall, _ = precision_recall_curve(np.asarray(y), np.asarray(p))
    ok = recall >= target_recall
    return float(precision[ok].max()) if ok.any() else float("nan")


GRILLE_TAU = np.concatenate([np.linspace(0.02, 0.98, 97), [0.99, 0.995, 0.999]])


def select_threshold(y, p, criterion="f1", c_fn=5, c_fp=1):
    """Seuil choisi sur la validation : F1 maximal (criterion='f1') ou coût moyen minimal ('cout')."""
    y = np.asarray(y).astype(bool)
    p = np.asarray(p)
    best, best_tau = -np.inf, None
    for tau in GRILLE_TAU:
        pred = p > tau
        tp = (pred & y).sum()
        fp = (pred & ~y).sum()
        fn = (~pred & y).sum()
        score = (-(c_fp * fp + c_fn * fn) / len(y) if criterion == "cout"
                 else 2 * tp / max(2 * tp + fp + fn, 1))
        if score > best:
            best, best_tau = score, tau
    return float(best_tau)


def evaluate_models(res, d, names=None, scores=None):
    """Mesures des modèles figés de `res` sur le bloc d. Renvoie (tableau, scores, seuils)."""
    y = d["y"].to_numpy()
    sc = dict(scores or {})
    taus = {"majoritaire": 0.5, "persistance": TAU_PERS, **res["taus"]}
    names = names or ["majoritaire", "persistance"] + list(res["models"])
    for n in names:
        if n in sc:
            continue
        if n == "majoritaire":
            sc[n] = np.zeros(len(d))
        elif n == "persistance":
            sc[n] = persistence_score(d["N"])
        else:
            sc[n] = res["models"][n].predict_proba(d[res["feats"].get(n, res["FEATS"])])[:, 1]
    table = pd.DataFrame([report(n, y, sc[n], tau=taus[n]) for n in names]).set_index("model")
    return table, sc, taus


def pipeline_horizon(raw, h=30, criterion="f1", ks=(15, 50, 150), depths=(4, 8, 12, 16)):
    """Étapes 1 à 3 du protocole, sur train et validation uniquement : réglage de k et de la profondeur,
    ajustement de tous les modèles sur train, seuils sur la validation, choix du modèle (AP maximale sur la
    validation). Le test est seulement découpé ici ; il est noté en Partie 4."""
    S = prepare_sets(raw, h)
    train, val, FEATS = S["train"], S["val"], S["FEATS"]
    nodow = [f for f in FEATS if f != "dow"]
    feats = {"bayes_naif": nodow, "knn": nodow}           # les autres modèles utilisent toutes les variables
    F = lambda n: feats.get(n, FEATS)

    models = {"bayes_naif": _bayes().fit(train[F("bayes_naif")], train["y"])}
    k, k_scores, models["knn"] = tune(_knn, ks, train, val, F("knn"))
    depth, d_scores, models["arbre"] = tune(_arbre, depths, train, val, FEATS)
    w = knn_weights(train, nodow, h)                      # poids appris dans le train seul
    feats["knn_pondere"] = nodow
    models["knn_pondere"] = _knn_pondere(k, w[nodow]).fit(train[nodow], train["y"])
    models["foret"] = foret().fit(train[FEATS], train["y"])
    models["boosting"] = boosting().fit(train[FEATS], train["y"])

    p_val = {n: m.predict_proba(val[F(n)])[:, 1] for n, m in models.items()}
    taus = {n: select_threshold(val["y"], p, criterion) for n, p in p_val.items()}
    res = {"h": h, "criterion": criterion, "models": models, "taus": taus, "FEATS": FEATS, "feats": feats, "p_val": p_val, **S,
           "selection": {"k": k, "profondeur": depth, "poids_knn": w, "ap_val_knn": k_scores, "ap_val_arbre": d_scores}}
    res["val_table"], _, _ = evaluate_models(res, val, scores=p_val)
    ml = list(models)
    res["chosen"] = max(ml, key=lambda n: res["val_table"].loc[n, "AP"])
    return res


# ---------------------------------------------------------------------------
# Analyse des erreurs, incertitude, ablation
# ---------------------------------------------------------------------------
def scored_frame(d, score, pred):
    """y / score / prédiction d'un modèle, avec les variables de regroupement."""
    t_loc = d["t"].dt.tz_convert(TZ)
    hour = t_loc.dt.hour.to_numpy()
    band = np.select([(hour >= lo) & (hour < hi) for lo, hi in HOUR_BANDS.values()],
                     list(HOUR_BANDS), default="")
    now_empty = d["N"].to_numpy() < 2
    y = d["y"].to_numpy()
    situation = np.select([~now_empty & (y == 0), ~now_empty & (y == 1),
                           now_empty & (y == 1), now_empty & (y == 0)], SITUATIONS, default="")
    return pd.DataFrame({
        "y": y, "score": np.asarray(score), "pred": np.asarray(pred).astype(int),
        "station_code": d["station_code"].to_numpy(),
        "zone": np.where(d["paris"].to_numpy() == 1, "Paris", "banlieue"),
        "heure": hour,
        "tranche": pd.Categorical(band, categories=list(HOUR_BANDS), ordered=True),
        "jour_type": np.where(t_loc.dt.dayofweek.to_numpy() >= 5, "week-end", "semaine"),
        "jour": t_loc.dt.date.to_numpy(),
        "situation": pd.Categorical(situation, categories=SITUATIONS),
    }, index=d.index)


def class_metrics(y, score, pred):
    y, pred = np.asarray(y).astype(int), np.asarray(pred).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return {"n": len(y), "positifs": int(y.sum()), "prévalence": y.mean(),
            "pr_auc": average_precision_score(y, score) if 0 < y.sum() < len(y) else np.nan,
            "précision": tp / (tp + fp) if tp + fp else np.nan,
            "rappel": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0}


def metrics_by(scored, by):
    return pd.DataFrame({k: class_metrics(g["y"], g["score"], g["pred"])
                         for k, g in scored.groupby(by, observed=True)}).T


def daily_dispersion(scored):
    """Mesures jour par jour (jours calendaires de Paris) : mesure de dispersion qui respecte le temps."""
    cols = ["n", "prévalence", "pr_auc", "précision", "rappel", "f1"]
    daily = metrics_by(scored, "jour")[cols].astype(float)
    return pd.concat([daily, daily.agg(["mean", "std", "min", "max"])])


def station_comparison(sm, sr):
    """F1 et AP par station du modèle et de la référence, de la plus grande perte au plus grand gain."""
    m = metrics_by(sm, "station_code")[["n", "positifs", "pr_auc", "f1"]].astype(float)
    r = metrics_by(sr, "station_code")[["pr_auc", "f1"]].astype(float)
    out = m.join(r, rsuffix="_réf")
    out["gain_f1"] = out["f1"] - out["f1_réf"]
    return out.sort_values("gain_f1")

def group_comparison(SCORED, R, groupes=("zone", "tranche", "jour_type"), metric="f1"):
    """Un seul tableau : `metric` du modèle retenu et de la persistance, par groupe et par horizon.
    Lignes : (groupe, valeur) ; colonnes : horizon, puis n / modèle / persistance / écart."""
    out = {}
    for h, sc in SCORED.items():
        mod, ref = sc[R[h]["chosen"]], sc["persistance"]
        rows = {}
        for by in groupes:
            m, p = metrics_by(mod, by), metrics_by(ref, by)
            for g in m.index:
                rows[(by, g)] = {"n": int(m.loc[g, "n"]), f"{metric} modèle": float(m.loc[g, metric]),
                                 f"{metric} persistance": float(p.loc[g, metric])}
        d = pd.DataFrame(rows).T
        d["écart"] = d[f"{metric} modèle"] - d[f"{metric} persistance"]
        out[h] = d
    return pd.concat(out, axis=1).round(3)

def situation_table(scored_by_model):
    """Part prédite « vide » dans chaque situation (état maintenant -> état à t+h)."""
    first = next(iter(scored_by_model.values()))
    out = first.groupby("situation", observed=True).size().rename("n").to_frame()
    out["part des lignes"] = out["n"] / out["n"].sum()
    for name, s in scored_by_model.items():
        out[f"prédit vide : {name}"] = s.groupby("situation", observed=True)["pred"].mean()
    return out


def station_bootstrap(sa, sb, labels=("modèle", "persistance"), B=500, seed=2026):
    """Bootstrap par STATION (les lignes d'une même station sont dépendantes) : IC à 95 % de l'AP et du F1
    de a, de b et de leur différence, calculée sur les mêmes stations tirées."""
    assert np.array_equal(sa["y"].to_numpy(), sb["y"].to_numpy()), "mêmes lignes requises"
    st = sa["station_code"].to_numpy()
    codes = np.unique(st)
    rows = {c: np.flatnonzero(st == c) for c in codes}
    rng = np.random.default_rng(seed)
    y = sa["y"].to_numpy()
    arr = {l: (s["score"].to_numpy(), s["pred"].to_numpy()) for l, s in zip(labels, (sa, sb))}

    def f1(yy, pp):
        tp = ((pp == 1) & (yy == 1)).sum()
        return 2 * tp / max(2 * tp + ((pp == 1) & (yy == 0)).sum() + ((pp == 0) & (yy == 1)).sum(), 1)

    out = []
    for _ in range(B):
        idx = np.concatenate([rows[c] for c in rng.choice(codes, len(codes), replace=True)])
        d = {}
        for l, (s, p) in arr.items():
            d[f"AP ({l})"] = average_precision_score(y[idx], s[idx])
            d[f"F1 ({l})"] = f1(y[idx], p[idx])
        d[f"AP ({labels[0]} - {labels[1]})"] = d[f"AP ({labels[0]})"] - d[f"AP ({labels[1]})"]
        d[f"F1 ({labels[0]} - {labels[1]})"] = d[f"F1 ({labels[0]})"] - d[f"F1 ({labels[1]})"]
        out.append(d)
    out = pd.DataFrame(out)
    return pd.DataFrame({"moyenne": out.mean(), "ic_bas": out.quantile(0.025), "ic_haut": out.quantile(0.975)})


def ablation(res, seed=0):
    """AP sur la validation d'une forêt réajustée sans chaque groupe de variables."""
    train, val = res["train"], res["val"]
    allf = [f for fs in FEATURE_GROUPS.values() for f in fs]
    assert set(allf) == set(res["FEATS"]), "les groupes doivent couvrir exactement FEATS"

    def score(feats):
        m = foret(seed).fit(train[feats], train["y"])
        return average_precision_score(val["y"], m.predict_proba(val[feats])[:, 1])

    base = score(allf)
    rows = [("toutes les variables", len(allf), base, 0.0)]
    for g_, fs in FEATURE_GROUPS.items():
        rest = [f for f in allf if f not in fs]
        s = score(rest)
        rows.append((f"sans {g_}", len(rest), s, s - base))
    s = score(FEATURE_GROUPS["état"])
    rows.append(("état seul", len(FEATURE_GROUPS["état"]), s, s - base))
    return pd.DataFrame(rows, columns=["configuration", "n_variables", "pr_auc_val", "écart"]).set_index("configuration")


def pr_curves(panels):
    """panels : {h: {"y": ..., "modèle": (nom, score, pred), "persistance": (score, pred)}}
    Affiche les courbes précision-rappel (aucun fichier n'est écrit)."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 3.8), sharey=True, squeeze=False)
    for ax, (h, p) in zip(axes[0], panels.items()):
        y = np.asarray(p["y"])
        nom, ms, mp = p["modèle"]
        for label, s, pr, c in [(nom, ms, mp, "#2a78d6"), ("persistance (score)", *p["persistance"], "#eb6834")]:
            prec, rec, _ = precision_recall_curve(y, s)
            ax.plot(rec, prec, color=c, label=label, drawstyle="steps-post")
            tp = np.sum((pr == 1) & (y == 1))
            ax.plot(tp / y.sum(), tp / max(pr.sum(), 1), "o", color=c)
        ax.axhline(y.mean(), color="grey", ls="--", label=f"prévalence {y.mean():.2f}")
        ax.set(title=f"Horizon {h} min, semaine de test", xlabel="rappel", xlim=(0, 1), ylim=(0, 1.02))
        ax.legend(loc="lower left", fontsize=8)
    axes[0][0].set_ylabel("précision")
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Annexe : expériences exploratoires (validation seulement)
# ---------------------------------------------------------------------------
def correlated_pairs(train, feats, thr=0.9):
    """Paires de variables dont la corrélation absolue (train) dépasse thr."""
    c = train[feats].corr().abs().to_numpy()
    i, j = np.triu_indices(len(feats), 1)
    f = np.array(feats)
    s = pd.Series(c[i, j], index=pd.MultiIndex.from_arrays([f[i], f[j]]))
    return s[s > thr].sort_values(ascending=False)


def seed_sensitivity(res, seeds=range(5)):
    """AP validation de la forêt et du boosting pour plusieurs graines : si deux modèles sont à quelques millièmes
    d'AP l'un de l'autre, une seule graine ne permet pas de les classer."""
    train, val, F = res["train"], res["val"], res["FEATS"]
    out = {}
    for name, make in [("foret", foret), ("boosting", boosting)]:
        out[name] = [average_precision_score(val["y"], make(s).fit(train[F], train["y"]).predict_proba(val[F])[:, 1])
                     for s in seeds]
    t = pd.DataFrame(out, index=[f"graine {s}" for s in seeds])
    return pd.concat([t, t.agg(["mean", "std"])])


def pca_val(res, var=0.95):
    """ACP (variance expliquée >= var, ajustée sur le train) puis k-NN et Bayes naïf : AP validation."""
    train, val = res["train"], res["val"]
    F = res["feats"]["knn"]
    k = res["selection"]["k"]
    pre = make_pipeline(_imp(), StandardScaler(), PCA(n_components=var, svd_solver="full")).fit(train[F])
    Ztr, Zva = pre.transform(train[F]), pre.transform(val[F])
    out = {}
    for name, m in [(f"k-NN (k={k})", KNeighborsClassifier(n_neighbors=k, n_jobs=-1)), ("Bayes naïf (sklearn)", GaussianNB())]:
        m.fit(Ztr, train["y"])
        out[name] = average_precision_score(val["y"], m.predict_proba(Zva)[:, 1])
    t = pd.Series(out, name="AP validation (ACP)").to_frame()
    t["composantes"] = Ztr.shape[1]
    t["AP sans ACP"] = [res["val_table"].loc["knn", "AP"], res["val_table"].loc["bayes_naif", "AP"]]
    return t


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_validation_summary(res, h, colors=None):
    """Quatre diagrammes en barres (AP, précision au rappel de la persistance, Brier, coût 5:1)
    pour tous les modèles de `res` sur la validation. La persistance n'a pas de probabilité
    calibrée : son Brier est omis."""
    import matplotlib.pyplot as plt
    colors = colors or dict(zip(res["models"], plt.cm.tab10.colors))
    val = res["val"]
    yv = val["y"].to_numpy()
    _, sc, _ = evaluate_models(res, val, scores=res["p_val"])
    pers = (val["N"].to_numpy() < 2).astype(int)
    r_p, p_p = recall_score(yv, pers), precision_score(yv, pers)

    t = res["val_table"].drop(index="majoritaire").copy()
    t["P@R_pers"] = [p_p if n == "persistance" else precision_at_recall(yv, sc[n], r_p) for n in t.index]
    t.loc["persistance", "Brier"] = np.nan
    metrics = [("AP", "AP (haut = bon)"),
               ("P@R_pers", "Précision (haut = bon)"),
               ("Brier", "Brier (bas = bon)"),
               ("cost", "Coût 5:1 (bas = bon)")]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (m, titre) in zip(axes, metrics):
        vals = t[m].dropna()
        bars = ax.bar(range(len(vals)), vals.values,
                      color=["gray" if i == "persistance" else colors[i] for i in vals.index])
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(vals.index, rotation=60, ha="right")
        ax.set_title(titre, fontsize=10)
        for b, v in zip(bars, vals.values):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle(f"Horizon {h} min")
    fig.tight_layout()
    plt.show()
    
def plot_test_summary(res, h, table, scores, colors=None):
    """Comme plot_validation_summary, mais sur la semaine de test : quatre diagrammes en barres (AP, Brier,
    coût 5:1, précision au rappel de la persistance). `table` et `scores` viennent de evaluate_models(res, test)."""
    import matplotlib.pyplot as plt
    colors = colors or dict(zip(res["models"], plt.cm.tab10.colors))
    te = res["test"]
    yt = te["y"].to_numpy()
    pers = (te["N"].to_numpy() < 2).astype(int)
    r_p, p_p = recall_score(yt, pers), precision_score(yt, pers)

    t = table.drop(index="majoritaire").copy()
    t["P@R_pers"] = [p_p if n == "persistance" else precision_at_recall(yt, scores[n], r_p) for n in t.index]
    t.loc["persistance", "Brier"] = np.nan
    metrics = [("AP", "AP (haut = bon)"), ("Brier", "Brier (bas = bon)"),
               ("cost", "Coût 5:1 (bas = bon)"), ("P@R_pers", "Précision au rappel de la persistance")]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (m, titre) in zip(axes, metrics):
        vals = t[m].dropna()
        bars = ax.bar(range(len(vals)), vals.values,
                      color=["gray" if i == "persistance" else colors[i] for i in vals.index])
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(vals.index, rotation=60, ha="right")
        ax.set_title(titre, fontsize=10)
        for b, v in zip(bars, vals.values):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle(f"Horizon {h} min, semaine de test")
    fig.tight_layout()
    plt.show()


# Resultat simule 
def predict_now(res, raw):
    """Probabilité que chaque station ait moins de 2 vélos dans h minutes, à partir du DERNIER relevé.

    Utilise les modèles déjà ajustés dans `res` (sortie de pipeline_horizon) : mêmes variables,
    même st_empty_rate (figé sur le train), seuil de chaque modèle. Renvoie une ligne par station.
    """
    h = res["h"]
    feat, _, _ = build_features(raw, h=h, live=True)
    feat = feat.assign(st_empty_rate=feat["station_code"].map(res["rate"]))
    out = feat[["station_code", "t", "N"]].copy()
    #out["zone"] = np.where(feat["paris"].to_numpy() == 1, "Paris", "banlieue")
    out = feat[["station_code", "name", "commune", "t", "N"]].copy()
    out = feat[["station_code", "name", "commune", "latitude", "longitude", "t", "N"]].copy()
    out["zone"] = np.where(feat["paris"].to_numpy() == 1, "Paris", "banlieue")
    for n, m in res["models"].items():
        out[f"p_{n}"] = m.predict_proba(feat[res["feats"].get(n, res["FEATS"])])[:, 1]
    c = res["chosen"]
    out["p"] = out[f"p_{c}"]
    out["alerte"] = (out["p"] > res["taus"][c]).astype(int)
    out["horizon"] = h
    return out.sort_values("p", ascending=False).reset_index(drop=True)
  
  
# Plot des resultats 
def summary_predictions(preds, by="zone"):
    """Résumé des prédictions du dernier relevé : par zone (ou autre colonne) et par horizon.
    n_stations, part de stations en alerte (p > tau), p moyen."""
    g = preds.groupby(["horizon", by])
    return g.agg(n_stations=("station_code", "size"),
                 en_alerte=("alerte", "sum"),
                 part_alerte=("alerte", "mean"),
                 p_moyen=("p", "mean")).round(3)


def plot_predictions_zone(preds, colors=("#4C72B0", "#DD8452")):
    """Barres : part de stations en alerte et p moyen, par zone, pour chaque horizon."""
    import matplotlib.pyplot as plt
    s = summary_predictions(preds)
    hs = sorted(preds["horizon"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col, titre in zip(axes, ["part_alerte", "p_moyen"],
                              ["Part de stations en alerte", "Probabilité moyenne de pénurie"]):
        w = s[col].unstack("horizon").loc[["Paris", "banlieue"]]
        w.plot.bar(ax=ax, color=list(colors), rot=0)
        ax.set(title=titre, xlabel="", ylabel="", ylim=(0, 1))
        ax.legend([f"{h} min" for h in hs], title="Horizon")
        for c in ax.containers:
            ax.bar_label(c, labels=[f"{v:.0%}" if col == "part_alerte" else f"{v:.2f}"
                                    for v in c.datavalues])
    fig.tight_layout()
    return fig
  
  
def summary_zone_test(SCORED, R, by="zone"):
    """Semaine de test, par zone et horizon : taux de pénurie réel, taux d'alerte du modèle retenu
    et de la persistance, rappel et F1 du modèle retenu."""
    rows = []
    for h, sc in SCORED.items():
        c = R[h]["chosen"]
        for z, g in sc[c].groupby(by, observed=True):
            gp = sc["persistance"].loc[g.index]
            m = class_metrics(g["y"], g["score"], g["pred"])
            rows.append({"horizon": h, by: z, "n": m["n"], "pénurie réelle": g["y"].mean(),
                         "alerte modèle": g["pred"].mean(), "alerte persistance": gp["pred"].mean(),
                         "rappel modèle": m["rappel"], "F1 modèle": m["f1"]})
    return pd.DataFrame(rows).set_index(["horizon", by]).round(3)


def plot_zone_test(SCORED, R, order=("Paris", "banlieue"), colors=("#555555", "#4C72B0", "#C9C9C9")):
    """Barres par zone : pénurie réelle, alerte du modèle retenu, alerte de la persistance (test)."""
    import matplotlib.pyplot as plt
    s = summary_zone_test(SCORED, R)
    cols = ["pénurie réelle", "alerte modèle", "alerte persistance"]
    hs = list(SCORED)
    fig, axes = plt.subplots(1, len(hs), figsize=(5.5 * len(hs), 4), sharey=True)
    for ax, h in zip(np.atleast_1d(axes), hs):
        w = s.loc[h, cols].loc[list(order)]
        w.plot.bar(ax=ax, color=list(colors), rot=0)
        ax.set(title=f"Horizon {h} min", xlabel="")
        for c in ax.containers:
            ax.bar_label(c, labels=[f"{v:.1%}" for v in c.datavalues], fontsize=8)
        ax.legend(fontsize=8)
    np.atleast_1d(axes)[0].set_ylabel("Part des lignes (semaine de test)")
    fig.tight_layout()
    return fig
  
def plot_predictions_commune(preds, top=None, min_stations=1):
    """Barres horizontales par commune : probabilité moyenne de pénurie à 30 et 60 min (dernier relevé).
    L'étiquette indique le nombre de stations en alerte / le nombre de stations de la commune.
    top : ne garder que les `top` communes les plus à risque ; min_stations : taille minimale."""
    import matplotlib.pyplot as plt
    hs = sorted(preds["horizon"].unique())
    g = preds.groupby(["commune", "horizon"]).agg(p=("p", "mean"), alerte=("alerte", "sum"),
                                                 n=("station_code", "size")).reset_index()
    n_com = g.groupby("commune")["n"].first()
    order = (g[g["horizon"] == hs[-1]].set_index("commune")["p"]
             .loc[n_com[n_com >= min_stations].index].sort_values(ascending=False))
    if top:
        order = order.head(top)
    order = order.index[::-1]                       # la plus à risque en haut
    fig, ax = plt.subplots(figsize=(9, max(3, 0.32 * len(order) * len(hs) / 1.6)))
    hgt = 0.8 / len(hs)
    for i, h in enumerate(hs):
        d = g[g["horizon"] == h].set_index("commune").loc[order]
        y = np.arange(len(order)) + (i - (len(hs) - 1) / 2) * hgt
        ax.barh(y, d["p"], height=hgt, label=f"{h} min")
        for yy, p, a, n in zip(y, d["p"], d["alerte"], d["n"]):
            ax.text(p + 0.005, yy, f"{a}/{n}", va="center", fontsize=7)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set(xlabel="Probabilité moyenne de pénurie (étiquette : stations en alerte / stations)",
           title="Prévision par commune, dernier relevé")
    ax.legend(title="Horizon")
    fig.tight_layout()
    return fig
  
  
def top_alertes(preds, n=15):
    """Stations les plus à risque au dernier relevé : N actuel, p à 30 et 60 min, alerte à chaque horizon.
    Triées par la probabilité la plus élevée sur les deux horizons."""
    base = preds.drop_duplicates("station_code").set_index("station_code")[["name", "commune", "N"]]
    p = preds.pivot(index="station_code", columns="horizon", values="p")
    a = preds.pivot(index="station_code", columns="horizon", values="alerte")
    out = base.join(p.add_prefix("p_").add_suffix(" min")).join(a.add_prefix("alerte_").add_suffix(" min"))
    pcols = [c for c in out if c.startswith("p_")]
    out = out.assign(_max=out[pcols].max(axis=1)).sort_values("_max", ascending=False).drop(columns="_max")
    return out.head(n).round(3)


def map_predictions(preds):
    """Carte des stations coloriées par la probabilité de pénurie (un panneau par horizon).
    Les stations en alerte (p > tau) sont cerclées de noir."""
    import matplotlib.pyplot as plt
    hs = sorted(preds["horizon"].unique())
    fig, axes = plt.subplots(1, len(hs), figsize=(6.5 * len(hs), 6), sharex=True, sharey=True)
    for ax, h in zip(np.atleast_1d(axes), hs):
        d = preds[preds["horizon"] == h].sort_values("p")
        sc = ax.scatter(d["longitude"], d["latitude"], c=d["p"], cmap="YlOrRd", vmin=0, vmax=1, s=45,
                        edgecolor=np.where(d["alerte"] == 1, "k", "none"), linewidth=1.2)
        ax.set(title=f"Horizon {h} min", xlabel="longitude")
        ax.set_aspect(1 / np.cos(np.radians(d["latitude"].mean())))
    np.atleast_1d(axes)[0].set_ylabel("latitude")
    fig.colorbar(sc, ax=axes, label="Probabilité de pénurie (cerclé : station en alerte)", shrink=0.8)
    return fig
  
def map_predictions_folium(preds, zoom=12):
    """Carte folium des stations, coloriées par la probabilité de pénurie.
    Un calque par horizon (30 / 60 min), à choisir avec le sélecteur en haut à droite.
    Les stations en alerte (p > tau) ont un contour noir. Clic sur un point : détails de la station."""
    import folium
    from branca.colormap import LinearColormap

    hs = sorted(preds["horizon"].unique())
    cmap = LinearColormap(
    colors=["#1D4ED8","#38BDF8","#7DD3FC","#FDE047","#FB923C","#DC2626" ],vmin=0, vmax=1, 
          caption="Probabilité de pénurie")
    wide = preds.pivot(index="station_code", columns="horizon", values="p")
    info = preds.drop_duplicates("station_code").set_index("station_code")

    m = folium.Map(location=[info["latitude"].mean(), info["longitude"].mean()],
                   zoom_start=zoom, tiles="OpenStreetMap")
    for i, h in enumerate(hs):
        fg = folium.FeatureGroup(name=f"Horizon {h} min", show=(i == 0))
        d = preds[preds["horizon"] == h].sort_values("p")          # les plus risquées dessinées en dernier
        for _, r in d.iterrows():
            lignes = "".join(f"<br>p ({k} min) : {wide.loc[r['station_code'], k]:.2f}" for k in hs)
            html = (f"<b>{r['name']}</b><br>{r['commune']}<br>Vélos actuels : {int(r['N'])}{lignes}"
                    f"<br>{'⚠ en alerte' if r['alerte'] == 1 else 'pas d alerte'} à {h} min")
            folium.CircleMarker(
                location=[r["latitude"], r["longitude"]],
                radius=8 if r["alerte"] == 1 else 6,
                color="black" if r["alerte"] == 1 else "white",
                weight=2 if r["alerte"] == 1 else 1,
                fill=True, fill_color=cmap(r["p"]), fill_opacity=0.85,
                tooltip=f"{r['name']} : p = {r['p']:.2f}",
                popup=folium.Popup(html, max_width=260)).add_to(fg)
        fg.add_to(m)
    cmap.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m

def ap_ci_table(res, scores, B=200, seed=2026):
    """AP sur le test de chaque modèle (et de la persistance) avec IC à 95 % par bootstrap de STATIONS.
    Les mêmes stations sont tirées pour tous les modèles : les intervalles sont comparables."""
    d = res["test"]
    y, st = d["y"].to_numpy(), d["station_code"].to_numpy()
    codes = np.unique(st)
    rows = {c: np.flatnonzero(st == c) for c in codes}
    names = list(res["models"]) + ["persistance"]
    rng = np.random.default_rng(seed)
    boots = {n: [] for n in names}
    for _ in range(B):
        idx = np.concatenate([rows[c] for c in rng.choice(codes, len(codes), replace=True)])
        for n in names:
            boots[n].append(average_precision_score(y[idx], scores[n][idx]))
    return pd.DataFrame({n: {"AP": average_precision_score(y, scores[n]),
                             "ic_bas": np.quantile(b, 0.025), "ic_haut": np.quantile(b, 0.975)}
                         for n, b in boots.items()}).T


def plot_ap_ci(R, TE, B=200):
    """AP de chaque modèle sur la semaine de test, avec IC bootstrap par station.
    Le modèle retenu est en couleur, la persistance est la ligne verticale (avec son IC en bande).
    Renvoie (figure, {horizon: tableau})."""
    import matplotlib.pyplot as plt
    hs = list(R)
    fig, axes = plt.subplots(1, len(hs), figsize=(6 * len(hs), 4), sharex=True)
    tables = {}
    for ax, h in zip(np.atleast_1d(axes), hs):
        t = ap_ci_table(R[h], TE[h][1], B=B)
        tables[h] = t.round(3)
        pers = t.loc["persistance"]
        m = t.drop(index="persistance").sort_values("AP")
        y = np.arange(len(m))
        ax.axvspan(pers["ic_bas"], pers["ic_haut"], color="#bbbbbb", alpha=0.4)
        ax.axvline(pers["AP"], color="k", ls="--", label=f"persistance ({pers['AP']:.3f})")
        for yy, (n, r) in zip(y, m.iterrows()):
            chosen = n == R[h]["chosen"]
            ax.errorbar(r["AP"], yy, xerr=[[r["AP"] - r["ic_bas"]], [r["ic_haut"] - r["AP"]]], fmt="o",
                        color="#C44E52" if chosen else "#4C72B0", capsize=3,
                        label="modèle retenu" if chosen else None)
        ax.set_yticks(y)
        ax.set_yticklabels(m.index)
        ax.set(title=f"Horizon {h} min", xlabel="AP sur la semaine de test (IC 95 % bootstrap)")
        ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig, tables
  
  

