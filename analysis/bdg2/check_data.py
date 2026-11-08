"""BDG2 data check (G3 review §5, 2026-11-08). Read-only; prints counts and writes <dir>/bdg2_5p_fits.csv. Usage: PYTHONSAFEPATH=1 python3 analysis/bdg2/check_data.py data/bdg2"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

d = Path(sys.argv[1])
w = pd.read_csv(d / "weather.csv", parse_dates=["timestamp"])
meta = pd.read_csv(d / "metadata.csv")
print("=== weather: per site airTemperature ===")
g = w.groupby("site_id").airTemperature
print(pd.DataFrame({"n": g.size(), "missing": g.apply(lambda s: s.isna().mean()).round(3),
                    "min": g.min(), "p5": g.quantile(.05).round(1), "p95": g.quantile(.95).round(1), "max": g.max()}).to_string())
# daily mean temperature per site
w["date"] = w.timestamp.dt.floor("D")
tday = w.groupby(["site_id", "date"]).airTemperature.mean().unstack(0)


def fit_5p(t, y):
    """5P change-point: y = b0 + bh*(th - t)+ + bc*(t - tc)+, grid over (th, tc). Returns best params, CV(RMSE), R2."""
    best = None
    qs = np.quantile(t, np.linspace(0.05, 0.95, 37))
    for th in qs:
        for tc in qs:
            if tc < th:
                continue
            X = np.column_stack([np.ones_like(t), np.clip(th - t, 0, None), np.clip(t - tc, 0, None)])
            b, *_ = np.linalg.lstsq(X, y, rcond=None)
            r = y - X @ b
            sse = float(r @ r)
            if best is None or sse < best[0]:
                best = (sse, th, tc, b)
    sse, th, tc, b = best
    cv = np.sqrt(sse / len(y)) / y.mean() if y.mean() > 0 else np.nan
    r2 = 1 - sse / float(((y - y.mean()) ** 2).sum())
    return th, tc, b, cv, r2


rows = []
for mtype in ["chilledwater", "steam", "hotwater", "gas", "electricity"]:
    m = pd.read_csv(d / f"{mtype}_cleaned.csv", parse_dates=["timestamp"])
    m["date"] = m.timestamp.dt.floor("D")
    cols = [c for c in m.columns if c not in ("timestamp", "date")]
    daily = m.groupby("date")[cols].mean()          # daily mean of hourly readings (NaN-aware)
    cnt = m.groupby("date")[cols].count()
    print(f"\n=== {mtype}: {len(cols)} meters, hourly missing {m[cols].isna().mean().mean():.3f} ===")
    ok = 0
    for c in cols:
        site = c.split("_")[0]
        if site not in tday.columns:
            continue
        y = daily[c].where(cnt[c] >= 20)
        df = pd.DataFrame({"t": tday[site], "y": y}).dropna()
        df = df[df.y > 0]
        if len(df) < 300:
            continue
        th, tc, b, cv, r2 = fit_5p(df.t.values, df.y.values)
        rows.append(dict(meter=c, type=mtype, site=site, n_days=len(df), tmin=df.t.min(), tmax=df.t.max(),
                         th=th, tc=tc, slope_h=b[1], slope_c=b[2], base=b[0], cv=cv, r2=r2))
        ok += 1
    print(f"meters with >=300 valid days: {ok}")

r = pd.DataFrame(rows)
out = d / "bdg2_5p_fits.csv"
r.to_csv(out, index=False)
print(f"\nwrote {out} ({len(r)} fits)")
# stability: R2 >= 0.5 and both balance points >= 2C inside observed range and separated
r["inside"] = (r.th >= r.tmin + 2) & (r.tc <= r.tmax - 2)
r["good"] = r.inside & (r.r2 >= 0.5)
print("\n=== fits by type: n, R2>=0.5, R2>=0.5 & balance points >=2C inside range ===")
print(r.groupby("type").agg(n=("meter", "size"), r2_ok=("r2", lambda s: int((s >= 0.5).sum())), good=("good", "sum")).to_string())
print("\n=== good fits: balance points by site (median th, tc; count) ===")
gg = r[r.good].groupby(["type", "site"]).agg(n=("meter", "size"), th_med=("th", "median"), tc_med=("tc", "median"),
                                              th_iqr=("th", lambda s: s.quantile(.75) - s.quantile(.25)),
                                              tc_iqr=("tc", lambda s: s.quantile(.75) - s.quantile(.25))).round(1)
print(gg.to_string())
print("\n=== good fits: overall spread of tc (cooling balance) and th (heating balance) ===")
print(r[r.good][["th", "tc"]].describe().round(1).to_string())
