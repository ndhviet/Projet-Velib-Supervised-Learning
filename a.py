# 4.  Consistency
print("\nbikes ≠ mech+elec:",
      (df["bikes_available"] != df["mechanical_bikes"] + df["electric_bikes"]).sum())
print("bikes+docks > capacity:",
      (df["bikes_available"] + df["docks_available"] > df["capacity"]).sum())
print("bikes+docks < capacity:",
      (df["bikes_available"] + df["docks_available"] < df["capacity"]).sum())

#>capacity: Impossible, needs inspection.
#<capacity: Maybe there are broken docks.
#Comparisons with NaN return False, so rows with missing capacity are counted in neither line.

# 5.  Status flags
for c in ["is_installed", "is_renting", "is_returning"]:
    print(f"\n{c}:\n", df[c].value_counts())

# 6. Collection gaps
snaps = pd.Series(sorted(df["collected_at_utc"].unique()))
#Difference with previous timestamp; first is NaN
gaps = snaps.diff().dt.total_seconds() / 60 
print("\n snapshot gap(minutes):\n", gaps.describe())
print("times of gap > 20 min:", (gaps > 20).sum())
print(pd.DataFrame({"start": snaps.shift(1), "end": snaps, "minute": gaps})
        [gaps > 20].sort_values("minute", ascending=False).head(10))

# 7.  Source staleness
lag = (df["collected_at_utc"] - df["source_updated_at"]).dt.total_seconds() / 60
print("\nSource staleness(minute):\n", lag.describe(percentiles=[.5, .9, .99]))

# ---------- 疑点 1：哪些快照不满 80 行 ----------
per_snap = df.groupby("collected_at_utc").size()          # 每次快照有几行
print(per_snap.value_counts().sort_index())               # 行数 → 出现几次
print(per_snap[per_snap < 80])                            # 列出不满 80 行的快照

# ---------- 第一次快照是否是"未对齐"的那次 ----------
print(per_snap.index[:3])                                 # 看前 3 个时间

# ---------- capacity：缺失分布 + 是否随时间变化 ----------
miss = df[df["capacity"].isna()]
print("缺 capacity 的快照数:", miss["collected_at_utc"].nunique(),
      " 涉及站点数:", miss["station_code"].nunique())
print("容量取值不止一个的站点数:",
      (df.groupby("station_code")["capacity"].nunique() > 1).sum())

# ---------- bikes+docks > capacity 集中在哪 ----------
over = df[df["bikes_available"] + df["docks_available"] > df["capacity"]]
print(over.groupby("station_code").size().sort_values(ascending=False).head())

# ---------- is_renting = 0：和 is_returning 是否同一批？集中在哪？ ----------
print("两列 0 的位置完全相同:",
      ((df["is_renting"] == 0) == (df["is_returning"] == 0)).all())
closed = df[df["is_renting"] == 0]
print(closed.groupby("station_code")["collected_at_utc"].agg(["size", "min", "max"]))

# ---------- 疑点 2：源延迟的含义 ----------
g = df.groupby("station_code")
src_same   = g["source_updated_at"].diff() == pd.Timedelta(0)   # 时间戳没变
bikes_diff = g["bikes_available"].diff() != 0                   # 车数变了
print("时间戳没变，但车数变了的行数:", (src_same & bikes_diff).sum())

lag = (df["collected_at_utc"] - df["source_updated_at"]).dt.total_seconds() / 60
hour_paris = df["collected_at_utc"].dt.tz_convert("Europe/Paris").dt.hour
print(lag.groupby(hour_paris).median().round(1))                 # 按巴黎时间每小时的延迟中位数

# 30 行的快照是否全是郊区、50 行的是否全是巴黎
per_snap = df.groupby("collected_at_utc").size()
partial = df[df["collected_at_utc"].isin(per_snap[per_snap < 80].index)]
partial = partial.assign(is_paris=partial["commune"].eq("Paris"),
                         n=partial["collected_at_utc"].map(per_snap))
print(partial.groupby("n")["is_paris"].agg(["mean", "size"]))
# 预期：n=30 → mean=0（全郊区）；n=50 → mean=1（全巴黎）

# 站点 4101 是否每晚关闭（按巴黎时间的小时统计）
c4101 = df[(df["station_code"] == 4101) & (df["is_renting"] == 0)]
print(c4101["collected_at_utc"].dt.tz_convert("Europe/Paris").dt.hour.value_counts().sort_index())

#  A. 每对不完整快照里的站点是否互补
ts = sorted(per_snap[per_snap < 80].index)
for a, b in zip(ts[0::2], ts[1::2]):                       # 按时间两两配对
    sa = set(df.loc[df["collected_at_utc"] == a, "station_code"])
    sb = set(df.loc[df["collected_at_utc"] == b, "station_code"])
    print(a, b, "重叠:", len(sa & sb), "并集:", len(sa | sb))
# 预期每对：重叠 0，并集 80

# B. 4 个站点的关闭分成几段
closed = df[df["is_renting"] == 0].sort_values(["station_code", "collected_at_utc"])
new_ep = closed.groupby("station_code")["collected_at_utc"].diff() > pd.Timedelta("20min")
closed = closed.assign(ep=new_ep.groupby(closed["station_code"]).cumsum())
print(closed.groupby(["station_code", "ep"])["collected_at_utc"]
            .agg(开始="min", 结束="max", 快照数="size"))


# ---------- 1. 时间对齐到 15 分钟网格 ----------
first = df["collected_at_utc"].min()
df = df[df["collected_at_utc"] != first].copy()            # 丢掉第一次未对齐的快照
df["t"] = df["collected_at_utc"].dt.round("15min")         # 取整到最近的 15 分钟
assert not df.duplicated(["station_code", "t"]).any()      # 取整后不能出现重复

df["capacity"] = df.groupby("station_code")["capacity"].transform("max")  # 按站点填补容量

# ---------- 2. 建规则网格 ----------
grid = pd.date_range(df["t"].min(), df["t"].max(), freq="15min")
idx = pd.MultiIndex.from_product([df["station_code"].unique(), grid],
                                 names=["station_code", "t"])
g = df.set_index(["station_code", "t"]).reindex(idx)
print("网格行数:", len(g), " 有数据的行:", g["bikes_available"].notna().sum())

# ---------- 3. 关闭标记（含"短暂开放"） ----------
s = g.groupby(level="station_code")
closed = g["is_renting"].eq(0)
prev = closed.groupby(level="station_code").shift(1, fill_value=False)
nxt  = closed.groupby(level="station_code").shift(-1, fill_value=False)
g["closed"] = closed | (prev & nxt)

# ---------- 4. 构造目标 ----------
def make_target(g, H):
    s = g.groupby(level="station_code")
    fut_bikes  = s["bikes_available"].shift(-H)
    fut_closed = s["closed"].shift(-H, fill_value=True)
    valid = (g["bikes_available"].notna() & ~g["closed"]
             & fut_bikes.notna() & ~fut_closed)
    y = (fut_bikes < 2).astype(int)
    return y[valid], valid

for H in [2, 4]:
    y, valid = make_target(g, H)
    print(f"horizon {15*H} min: 样本 {valid.sum():,}  正例比例 {y.mean():.3f}")

# ---------- 5. 正例比例的分布（30 分钟） ----------
y, valid = make_target(g, 2)
d = g[valid].assign(y=y)
d["hour"]  = d.index.get_level_values("t").tz_convert("Europe/Paris").hour
d["paris"] = d["commune"].eq("Paris")

print(d.groupby("hour")["y"].mean().round(3))
print(d.groupby("paris")["y"].mean().round(3))
rate = d.groupby(level="station_code")["y"].mean()
print(rate.describe().round(3))
print("正例比例 > 50% 的站点数:", (rate > 0.5).sum(), " = 0 的站点数:", (rate == 0).sum())