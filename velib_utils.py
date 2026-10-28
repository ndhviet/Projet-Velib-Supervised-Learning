import hashlib
import os
import urllib.request

import numpy as np
import pandas as pd

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
