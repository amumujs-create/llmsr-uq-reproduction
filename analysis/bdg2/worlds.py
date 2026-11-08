"""BDG2 worlds for the chilled-water extrapolation study (docs/research/bdg2-regime-extrapolation.md §2, §5).

A world is one chilled-water meter whose whole-range 5P change-point fit is stable (R2 >= 0.5, both balance points
at least 2 degC inside the observed daily-temperature range). The reference breakpoint is the cooling balance point
`tc`; the reference shape is the cold-side / warm-side slope ratio `rho`.

Usage: PYTHONSAFEPATH=1 python3 analysis/bdg2/worlds.py data/bdg2 [--out analysis/bdg2/worlds.csv]
Also writes analysis/bdg2/descriptions.json (unique L1/L2 descriptions and their member meters) and the daily cache
data/bdg2/daily_chilledwater.csv (gitignored).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from check_data import fit_5p  # noqa: E402

METER = "chilledwater"
MIN_DAYS, MIN_HOURS = 300, 20
R2_MIN, INSIDE_C = 0.5, 2.0
SQM_BINS = [0, 2000, 5000, 10000, 20000, 1e9]
SQM_LABELS = ["under 2,000", "2,000 to 5,000", "5,000 to 10,000", "10,000 to 20,000", "over 20,000"]


def daily_temperature(data_dir):
    """{site: Series(date -> daily mean air temperature)}"""
    w = pd.read_csv(Path(data_dir) / "weather.csv", parse_dates=["timestamp"])
    w["date"] = w.timestamp.dt.floor("D")
    return {s: g.groupby("date").airTemperature.mean().dropna() for s, g in w.groupby("site_id")}


def daily_load(data_dir, meter=METER):
    """DataFrame(date x meter) of daily mean load, NaN where fewer than MIN_HOURS hourly readings. Cached."""
    cache = Path(data_dir) / f"daily_{meter}.csv"
    if cache.exists():
        return pd.read_csv(cache, index_col=0, parse_dates=True)
    m = pd.read_csv(Path(data_dir) / f"{meter}_cleaned.csv", parse_dates=["timestamp"])
    m["date"] = m.timestamp.dt.floor("D")
    cols = [c for c in m.columns if c not in ("timestamp", "date")]
    daily = m.groupby("date")[cols].mean().where(m.groupby("date")[cols].count() >= MIN_HOURS)
    daily.to_csv(cache)
    return daily


def climate(temps):
    """Per-site climate summary given to the LLM (numbers only, no site name)."""
    return {s: {"mean": round(float(t.mean()), 1), "p5": round(float(t.quantile(0.05)), 1),
                "p95": round(float(t.quantile(0.95)), 1)} for s, t in temps.items()}


def series(daily, temps, meter):
    """Aligned daily (t, y) for one meter, y > 0 only."""
    site = meter.split("_")[0]
    df = pd.DataFrame({"t": temps[site], "y": daily[meter]}).dropna()
    return df[df.y > 0]


def build_worlds(data_dir, meta=None):
    data_dir = Path(data_dir)
    meta = meta if meta is not None else pd.read_csv(data_dir / "metadata.csv").set_index("building_id")
    temps = daily_temperature(data_dir)
    daily = daily_load(data_dir)
    rows = []
    for meter in daily.columns:
        site = meter.split("_")[0]
        if site not in temps or meter not in meta.index:
            continue
        df = series(daily, temps, meter)
        if len(df) < MIN_DAYS:
            continue
        th, tc, b, cv, r2 = fit_5p(df.t.values, df.y.values)
        tmin, tmax = float(df.t.min()), float(df.t.max())
        stable = r2 >= R2_MIN and th >= tmin + INSIDE_C and tc <= tmax - INSIDE_C
        if not stable:
            continue
        rho = float(-b[1] / b[2]) if b[2] != 0 else np.nan   # cold-side slope / warm-side slope (dy/dt)
        md = meta.loc[meter]
        rows.append(dict(meter=meter, site=site, n_days=len(df), tmin=tmin, tmax=tmax, tc=float(tc), th=float(th),
                         rho=rho, r2=float(r2), cv=float(cv), use=md.primaryspaceusage,
                         sub_use=md.sub_primaryspaceusage if isinstance(md.sub_primaryspaceusage, str) else "",
                         sqm=float(md.sqm)))
    w = pd.DataFrame(rows)
    w["sqm_bin"] = pd.cut(w.sqm, SQM_BINS, labels=SQM_LABELS).astype(str)
    w["desc_L1"] = w.site + "|" + w.use
    w["desc_L2"] = w.desc_L1 + "|" + w.sub_use + "|" + w.sqm_bin
    return w, climate(temps)


def descriptions(worlds, clim):
    """{level: {desc_id: {"fields": {...}, "meters": [...]}}}; desc_id is a stable short key (no site names)."""
    out = {}
    for level in ("L1", "L2"):
        d = {}
        for i, (key, g) in enumerate(sorted(worlds.groupby(f"desc_{level}"))):
            r = g.iloc[0]
            fields = {"use": r.use, "climate": clim[r.site]}
            if level == "L2":
                fields.update({"sub_use": r.sub_use, "sqm_bin": r.sqm_bin})
            d[f"{level}-{i:03d}"] = {"fields": fields, "meters": sorted(g.meter), "key": key}
        out[level] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--out", default=str(HERE / "worlds.csv"))
    a = ap.parse_args()
    w, clim = build_worlds(a.data_dir)
    w.to_csv(a.out, index=False)
    desc = descriptions(w, clim)
    json.dump(desc, open(HERE / "descriptions.json", "w"), indent=1)
    print(f"{len(w)} worlds -> {a.out}; L1 descriptions {len(desc['L1'])}, L2 {len(desc['L2'])}")
    print(w.groupby("site").size().to_string())


if __name__ == "__main__":
    main()
