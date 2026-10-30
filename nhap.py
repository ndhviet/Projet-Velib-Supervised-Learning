import hashlib, urllib.request
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                             f1_score, fbeta_score, average_precision_score,
                             brier_score_loss, log_loss)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)

URL  = "https://docs.ambroise.cloud/velib/velib-2026-10-08.csv.gz"
FILE = "velib-2026-10-08.csv.gz"
MD5  = "02ab1e3fdac10579e4ce93d1540a5dc7"

path, _ = urllib.request.urlretrieve(URL, FILE)
with open(path, "rb") as f:
    assert hashlib.md5(f.read()).hexdigest() == MD5, "md5 mismatch"

raw = pd.read_csv(path)
for c in ["collected_at_utc", "source_updated_at"]:
    raw[c] = pd.to_datetime(raw[c], utc=True)
df = raw.copy()                      # raw không bao giờ bị sửa

N_tmp = df["mechanical_bikes"] + df["electric_bikes"]
cap_sum = df["bikes_available"] + df["docks_available"]
print("bikes != mech+elec    :", (df["bikes_available"] != N_tmp).sum())
print("bikes+docks > capacity:", (cap_sum > df["capacity"]).sum())
print("bikes+docks < capacity:", (cap_sum < df["capacity"]).sum())
print("NaN mỗi cột:\n", df[["capacity", "bikes_available", "mechanical_bikes",
                            "electric_bikes", "docks_available"]].isna().sum())

print("\nTổ hợp trạng thái trạm:")
print(df.groupby(["is_installed", "is_renting", "is_returning"], dropna=False)
        .size().rename("rows").reset_index())

# capacity có đổi theo thời gian không? (nếu có thì KHÔNG được dùng max)
n_multi = (df.groupby("station_code")["capacity"].nunique() > 1).sum()
print("\nSố trạm có >1 giá trị capacity:", n_multi)

# snapshot không đủ 80 trạm
per_snap = df.groupby("collected_at_utc").size()
print("\nSố dòng mỗi snapshot:\n", per_snap.value_counts().sort_index())

# khoảng cách giữa các snapshot
snaps = pd.Series(sorted(df["collected_at_utc"].unique()))
gaps = snaps.diff().dt.total_seconds() / 60
print("Số khoảng trống > 20 phút:", (gaps > 20).sum())

# độ trễ của nguồn
src_lag = (df["collected_at_utc"] - df["source_updated_at"]).dt.total_seconds() / 60
print(src_lag.describe(percentiles=[.5, .9, .99]))

first = df["collected_at_utc"].min()
print("Bỏ snapshot đầu (chưa căn):", first)
df = df[df["collected_at_utc"] != first].copy()
df["t"] = df["collected_at_utc"].dt.round("15min")
assert not df.duplicated(["station_code", "t"]).any()

# thông tin tĩnh của trạm (chỉ an toàn nếu n_multi == 0)
station_info = df.groupby("station_code").agg(capacity=("capacity", "max"),
                                              insee_code=("insee_code", "first"))

grid = pd.date_range(df["t"].min(), df["t"].max(), freq="15min")
idx = pd.MultiIndex.from_product([sorted(df["station_code"].unique()), grid],
                                 names=["station_code", "t"])
g = df.set_index(["station_code", "t"]).reindex(idx)

g["N"] = g["mechanical_bikes"] + g["electric_bikes"]
st = g.index.get_level_values("station_code")
g["capacity"] = st.map(station_info["capacity"])
g["is_paris"] = (st.map(station_info["insee_code"]) == 75056).astype(int)

print("Lưới:", len(g), "dòng | có dữ liệu:", g["N"].notna().sum(),
      "| ô trống (lỗ hổng):", g["N"].isna().sum())

by_st = lambda s: s.groupby(level="station_code")
closed_raw = (g["is_renting"].eq(0) | g["is_returning"].eq(0) | g["is_installed"].eq(0))
prev = by_st(closed_raw).shift(1, fill_value=False)
nxt  = by_st(closed_raw).shift(-1, fill_value=False)
g["closed"] = closed_raw | (prev & nxt)      # lấp ô "mở lẻ" giữa 2 ô đóng
print("Số ô đóng cửa:", g["closed"].sum())

H_STEPS = 2          # 2 x 15 = 30 phút
K = 8                # số lag (8 x 15 = 2 giờ). Thử 4 / 8 / 12, so AP trên validation

sN = by_st(g["N"])
g["N_future"] = sN.shift(-H_STEPS)
for k in range(1, K + 1):
    g[f"N_lag{k}"] = sN.shift(k)

# trạm phải "bình thường" ở t, ở t+30', và trong cả cửa sổ lag
fut_closed = by_st(g["closed"]).shift(-H_STEPS, fill_value=True)
bad_win = pd.Series(False, index=g.index)
for k in range(0, K + 1):
    bad_win |= by_st(g["closed"]).shift(k, fill_value=True)

lag_cols = [f"N_lag{k}" for k in range(1, K + 1)]
valid = (g["N"].notna() & g["N_future"].notna() & g[lag_cols].notna().all(axis=1)
         & ~fut_closed & ~bad_win)
print(f"Dòng hợp lệ: {valid.sum():,} / {len(g):,} ({valid.mean():.1%})")

d = g[valid].copy()
d["y"] = (d["N_future"] < 2).astype(int)
print("positive rate:", round(d["y"].mean(), 4))

d["diff1"] = d["N"] - d["N_lag1"]
d["diff4"] = d["N"] - d["N_lag4"]
w1h = ["N"] + lag_cols[:4]
w2h = ["N"] + lag_cols
d["N_mean_1h"] = d[w1h].mean(axis=1)
d["N_min_1h"]  = d[w1h].min(axis=1)
d["N_mean_2h"] = d[w2h].mean(axis=1)
d["N_min_2h"]  = d[w2h].min(axis=1)
d["N_max_2h"]  = d[w2h].max(axis=1)
d["N_std_2h"]  = d[w2h].std(axis=1)

feat = d.reset_index()
local = feat["t"].dt.tz_convert("Europe/Paris")
hour = local.dt.hour + local.dt.minute / 60
feat["hour_sin"] = np.sin(2 * np.pi * hour / 24)
feat["hour_cos"] = np.cos(2 * np.pi * hour / 24)
feat["dow"] = local.dt.dayofweek
feat["weekend"] = (feat["dow"] >= 5).astype(int)
feat["src_lag_min"] = (feat["collected_at_utc"] - feat["source_updated_at"]).dt.total_seconds() / 60

FEATS = (["N"] + lag_cols +
         ["diff1", "diff4", "N_mean_1h", "N_min_1h",
          "N_mean_2h", "N_min_2h", "N_max_2h", "N_std_2h",
          "mechanical_bikes", "electric_bikes",
          "is_paris", "weekend", "hour_sin", "hour_cos", "src_lag_min"])

# phân bố nhãn: trạm nào gần như luôn thiếu xe / không bao giờ thiếu
rate = feat.groupby("station_code")["y"].mean()
print("Tỷ lệ dương theo trạm:\n", rate.describe().round(3))
print("trạm > 50%:", (rate > 0.5).sum(), "| trạm = 0:", (rate == 0).sum())
print("\nTheo giờ (Paris):\n", feat.groupby(local.dt.hour)["y"].mean().round(3))

tz = "Europe/Paris"
train_end  = pd.Timestamp("2026-09-23 00:00", tz=tz)
val_start  = pd.Timestamp("2026-09-25 00:00", tz=tz)
test_start = pd.Timestamp("2026-10-02 00:00", tz=tz)
embargo = pd.Timedelta("1h")

train = feat[local < train_end]
val   = feat[(local >= val_start) & (local < test_start - embargo)]
test  = feat[local >= test_start]
for name, part in [("train", train), ("val", val), ("test", test)]:
    print(name, "| rows:", len(part), "| từ", part["t"].min(), "đến", part["t"].max(),
          "| positive rate:", round(part["y"].mean(), 4))

for name, part in [("val", val), ("test", test)]:
    pers = (part["N"] < 2).astype(int)
    print(name, "| persistence P:", round(precision_score(part["y"], pers), 3),
          "R:", round(recall_score(part["y"], pers), 3),
          "F1:", round(f1_score(part["y"], pers), 3))


NB_FEATS = ["N", "N_lag1", "N_lag2", "diff1", "hour_sin", "hour_cos", "dow"]
nb = GaussianNB().fit(train[NB_FEATS], train["y"])
p_val_nb = nb.predict_proba(val[NB_FEATS])[:, 1]
print("NB val AP:", round(average_precision_score(val["y"], p_val_nb), 3),
      "| baseline AP:", round(val["y"].mean(), 3))

def report(name, y, p, tau=0.5, c_fn=5, c_fp=1):
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
            "cost": (c_fp * fp + c_fn * fn) / len(y)}   # c_fn/c_fp = 5 chỉ là giả định tạm

y = val["y"]
table = pd.DataFrame([
    report("majority",    y, np.zeros(len(val))),
    report("persistence", y, (val["N"] < 2).astype(float)),
    report("naive_bayes", y, p_val_nb, tau=0.7),
]).set_index("model").round(3)
print(table.T)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
lr.fit(train[FEATS], train["y"])
p_val_lr = lr.predict_proba(val[FEATS])[:, 1]

hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                     max_leaf_nodes=31, random_state=0)
hgb.fit(train[FEATS], train["y"])
p_val_hgb = hgb.predict_proba(val[FEATS])[:, 1]

# chọn ngưỡng tối thiểu hóa chi phí trên VALIDATION
taus = np.linspace(0.05, 0.95, 91)
y = val["y"]
rows = [report("persistence", y, (val["N"] < 2).astype(float))]
for name, p in [("logreg", p_val_lr), ("hgb", p_val_hgb)]:
    costs = [report(name, y, p, tau=t)["cost"] for t in taus]
    rows.append(report(name, y, p, tau=taus[int(np.argmin(costs))]))

table2 = pd.DataFrame(rows).set_index("model").round(3)
print(table2.T)

out = val[["station_code", "t", "N", "N_lag1", "N_lag2", "y"]].copy()
out["p_logreg"] = p_val_lr
out["p_hgb"]    = p_val_hgb
print(out.head(10).round(3))

# vài trường hợp cụ thể
print(out[out["N"] == 0].head(5).round(3))     # trạm đang hết xe
print(out[out["N"] >= 15].head(5).round(3))    # trạm còn nhiều xe