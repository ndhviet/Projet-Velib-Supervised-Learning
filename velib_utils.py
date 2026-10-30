import hashlib
import os
import urllib.request

import numpy as np
import pandas as pd

from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                             f1_score, fbeta_score, average_precision_score,
                             brier_score_loss, log_loss)

URL = "https://docs.ambroise.cloud/velib/velib-2026-10-08.csv.gz"
FILE = "data/velib-2026-10-08.csv.gz"
MD5 = "02ab1e3fdac10579e4ce93d1540a5dc7"


def load_raw():
    """Télécharge (une seule fois), vérifie le MD5 et lit le CSV brut."""
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(FILE):
        urllib.request.urlretrieve(URL, FILE)
    with open(FILE, "rb") as f:
        assert hashlib.md5(f.read()).hexdigest() == MD5, "md5 mismatch"
    return pd.read_csv(FILE)


def prepare(raw):
    """Nettoyage + variables dérivées. Renvoie (df, stations)."""
    df = raw.copy()
    df["ts"] = pd.to_datetime(df["collected_at_utc"], errors="coerce", utc=True) \
                 .dt.tz_convert("Europe/Paris")
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
  
def load_raw_dt():
    """Comme load_raw(), mais convertit les deux colonnes temporelles en datetime UTC."""
    raw = load_raw()
    for c in ["collected_at_utc", "source_updated_at"]:
        raw[c] = pd.to_datetime(raw[c], utc=True)
    return raw


def build_features(raw, K=4, H_STEPS=2):
    """Grille de 15 minutes, cible y (à t + 30 min), variables construites avec le passé seul.

    Renvoie (feat, FEATS). Les stations fermées sont exclues ; aucune valeur manquante
    n'est interpolée ni remplie.
    """
    # On retire le premier relevé (11:57), hors grille
    df = raw[raw["collected_at_utc"] != raw["collected_at_utc"].min()].copy()
    df["t"] = df["collected_at_utc"].dt.round("15min")
    assert not df.duplicated(["station_code", "t"]).any()

    info = df.groupby("station_code").agg(insee_code=("insee_code", "first"))

    # Grille régulière station x créneau : les créneaux manquants deviennent des NaN
    grid = pd.date_range(df["t"].min(), df["t"].max(), freq="15min")
    idx = pd.MultiIndex.from_product([sorted(df["station_code"].unique()), grid],
                                     names=["station_code", "t"])
    g = df.set_index(["station_code", "t"]).reindex(idx)

    g["N"] = g["mechanical_bikes"] + g["electric_bikes"]
    st = g.index.get_level_values("station_code")
    g["is_paris"] = (st.map(info["insee_code"]) == 75056).astype(int)

    by_st = lambda s: s.groupby(level="station_code")

    # Station fermée (et créneau « ouvert » isolé entre deux créneaux fermés)
    closed_raw = (g["is_renting"].eq(0) | g["is_returning"].eq(0) | g["is_installed"].eq(0))
    prev = by_st(closed_raw).shift(1, fill_value=False)
    nxt = by_st(closed_raw).shift(-1, fill_value=False)
    g["closed"] = closed_raw | (prev & nxt)

    # Cible (futur) et retards (passé), toujours calculés station par station
    sN = by_st(g["N"])
    g["N_future"] = sN.shift(-H_STEPS)
    lag_cols = [f"N_lag{k}" for k in range(1, K + 1)]
    for k in range(1, K + 1):
        g[f"N_lag{k}"] = sN.shift(k)

    # Une ligne est valide si la station est ouverte à t, à t+30 min et sur toute la fenêtre de retards
    fut_closed = by_st(g["closed"]).shift(-H_STEPS, fill_value=True)
    bad_win = pd.Series(False, index=g.index)
    for k in range(0, K + 1):
        bad_win |= by_st(g["closed"]).shift(k, fill_value=True)

    valid = (g["N"].notna() & g["N_future"].notna() & g[lag_cols].notna().all(axis=1)
             & ~fut_closed & ~bad_win)
    d = g[valid].copy()
    d["y"] = (d["N_future"] < 2).astype(int)

    # Variables dérivées (passé uniquement)
    d["diff1"] = d["N"] - d["N_lag1"]
    d["diff4"] = d["N"] - d["N_lag4"]
    w = ["N"] + lag_cols
    d["N_mean_1h"] = d[w].mean(axis=1)
    d["N_min_1h"] = d[w].min(axis=1)

    # Variables calendaires en heure de Paris (codage cyclique de l'heure)
    feat = d.reset_index()
    local = feat["t"].dt.tz_convert("Europe/Paris")
    hour = local.dt.hour + local.dt.minute / 60
    feat["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    feat["dow"] = local.dt.dayofweek
    feat["weekend"] = (feat["dow"] >= 5).astype(int)
    feat["src_lag_min"] = (feat["collected_at_utc"]
                           - feat["source_updated_at"]).dt.total_seconds() / 60

    FEATS = (["N"] + lag_cols +
             ["diff1", "diff4", "N_mean_1h", "N_min_1h",
              "mechanical_bikes", "electric_bikes",
              "is_paris", "weekend", "hour_sin", "hour_cos", "src_lag_min"])
    return feat, FEATS


def split_time(feat, tz="Europe/Paris"):
    """Découpage chronologique train / val / test, avec un embargo d'1 h avant le test."""
    local = feat["t"].dt.tz_convert(tz)
    train_end = pd.Timestamp("2026-09-23 00:00", tz=tz)
    val_start = pd.Timestamp("2026-09-25 00:00", tz=tz)
    test_start = pd.Timestamp("2026-10-02 00:00", tz=tz)
    emb = pd.Timedelta("1h")
    return (feat[local < train_end],
            feat[(local >= val_start) & (local < test_start - emb)],
            feat[local >= test_start])


def report(name, y, p, tau=0.5, c_fn=5, c_fp=1):
    """Indicateurs d'un modèle à un seuil tau donné.

    c_fn / c_fp = 5 : hypothèse provisoire, à justifier dans le rapport.
    """
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
