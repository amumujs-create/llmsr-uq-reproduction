"""Summarize grid_cells.csv / grid_windows.csv per the pre-registration (docs/research/ccpp-twin-grid.md).

Writes grid_summary.json and prints the tables used in the results section.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
cells = pd.read_csv(HERE / "grid_cells.csv")
wins = pd.read_csv(HERE / "grid_windows.csv")
KEY = ["module", "vmode", "noise", "setting", "w", "d", "rep"]

wide = cells.pivot_table(index=KEY + ["n_test"], columns="model", values="rmse_clean").reset_index()
wide["diff"] = wide["proportional"] - wide["raw_linear"]
wide["physics_wins"] = (wide["diff"] < 0).astype(int)
data = wide.merge(wins, on=KEY + ["n_test"])
prim = data[data.noise > 0]


def ci(x):
    return [float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))]


# 1. Win map (primary worlds).
grp = ["module", "vmode", "setting", "w", "d"]
models = [m for m in ("raw_linear", "density_linear", "proportional", "ridge", "mlp", "xgboost", "persistence")
          if m in wide.columns]
winmap = prim.groupby(grp).agg(n_test=("n_test", "first"), physics_win_rate=("physics_wins", "mean"),
                               diff_mean=("diff", "mean"),
                               **{m: (m, "mean") for m in models}).reset_index()
winmap["diff_ci"] = prim.groupby(grp)["diff"].apply(ci).values
pd.set_option("display.width", 250, "display.max_rows", 500)
print("=== win map (noise 4.3, clean RMSE MW; diff = proportional - raw_linear) ===")
print(winmap[grp + ["n_test", "physics_win_rate", "diff_mean"] + models].round(2).to_string(index=False))


def cell(module, vmode, setting, w, d=None):
    m = (prim.module == module) & (prim.vmode == vmode) & (prim.setting == setting) & (prim.w == w)
    if d is not None:
        m &= prim.d == d
    return prim[m]


# 2. Pre-registered predictions.
pred = {}
b = prim[(prim.module == "base")]
pred["P1_base_physics_win_rate"] = float(b.physics_wins.mean())
pred["P1_base_physics_win_rate_by_setting"] = b.groupby("setting").physics_wins.mean().round(3).to_dict()
for mod in ("evap", "evap+cap_mild"):
    c = cell(mod, "realV", "hot->cold", 2.5)
    pred[f"P2_{mod}_hot->cold_w2.5_by_d"] = c.groupby("d").agg(win=("physics_wins", "mean"), diff=("diff", "mean")).round(2).to_dict("index")
for mod in ("cap_mild", "cap_strong"):
    c = cell(mod, "realV", "cold->hot", 2.5)
    pred[f"P3_{mod}_cold->hot_w2.5_by_d"] = c.groupby("d").agg(win=("physics_wins", "mean"), diff=("diff", "mean")).round(2).to_dict("index")
p4 = {}
for mod in sorted(prim.module.unique()):
    r = cell(mod, "realV", "cold->hot", 2.5).groupby("d").raw_linear.mean()
    f = cell(mod, "fixedV", "cold->hot", 2.5).groupby("d").raw_linear.mean()
    p4[mod] = {float(k): [round(float(r[k]), 2), round(float(f[k]), 2)] for k in r.index}
pred["P4_raw_linear_cold->hot_w2.5_realV_vs_fixedV"] = p4
if "xgboost" in prim.columns:
    far = prim[prim.d >= 10]
    pred["P6_xgb_minus_persistence_d>=10"] = {"mean": float((far.xgboost - far.persistence).mean()),
                                              "mean_abs": float((far.xgboost - far.persistence).abs().mean()),
                                              "xgb": float(far.xgboost.mean()), "persistence": float(far.persistence.mean())}


# 3. Indicator AUROC (A6).
def auroc(score, y):
    score, y = np.asarray(score, float), np.asarray(y, int)
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    ranks = pd.Series(np.concatenate([pos, neg])).rank().values
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


INDICATORS = {"slope_se": 1, "slope_gap": 1, "gap_z": -1, "cov_departure": 1, "v_departure": 1,
              "curvature_t": 1, "d": 1}
PREREG_SIGN = {"slope_se": True, "gap_z": True, "cov_departure": True, "curvature_t": True,
               "slope_gap": False, "v_departure": False, "d": False}


def stratified(df, col, sign):
    vals = [auroc(sign * g[col], g.physics_wins) for _, g in df.groupby("d")]
    return float(np.nanmean(vals))


rng = np.random.default_rng(0)
clusters = prim.groupby(["module", "vmode", "setting", "w"]).indices
ckeys = list(clusters)
boot_idx = [np.concatenate([clusters[ckeys[i]] for i in rng.integers(0, len(ckeys), len(ckeys))]) for _ in range(1000)]
ind = {}
for col, sign in INDICATORS.items():
    a = auroc(sign * prim[col], prim.physics_wins)
    s = stratified(prim, col, sign)
    bs = [auroc(sign * prim[col].values[i], prim.physics_wins.values[i]) for i in boot_idx]
    bss = [stratified(prim.iloc[i], col, sign) for i in boot_idx[:200]]
    ind[col] = {"direction": "higher->physics" if sign > 0 else "higher->data", "preregistered_direction": PREREG_SIGN[col],
                "auroc": round(a, 3), "auroc_ci": [round(v, 3) for v in ci(bs)],
                "auroc_distance_stratified": round(s, 3), "stratified_ci": [round(v, 3) for v in ci(bss)]}
print("\n=== indicator AUROC for physics_wins (noise 4.3 worlds) ===")
for k, v in ind.items():
    print(k, v)

# Leave-one-module-out: single-indicator threshold chosen on other modules.
lomo = {}
for col, sign in INDICATORS.items():
    accs = {}
    for mod in sorted(prim.module.unique()):
        tr, te = prim[prim.module != mod], prim[prim.module == mod]
        s_tr = sign * tr[col].values
        cand = np.quantile(s_tr, np.linspace(0.01, 0.99, 99))
        best = max(cand, key=lambda t: ((s_tr > t) == tr.physics_wins.values).mean())
        accs[mod] = round(float(((sign * te[col].values > best) == te.physics_wins.values).mean()), 3)
    lomo[col] = accs
base_rate = {mod: round(float(max(g.physics_wins.mean(), 1 - g.physics_wins.mean())), 3)
             for mod, g in prim.groupby("module")}
print("\n=== leave-one-module-out accuracy (majority-class baseline in 'majority') ===")
print("majority", base_rate)
for k, v in lomo.items():
    print(k, v)

# 4. Noise-free diagnostic: does the win map change without noise?
clean = data[data.noise == 0]
noise_cmp = {"physics_win_rate_noise4.3": float(prim.physics_wins.mean()),
             "physics_win_rate_noise0": float(clean.physics_wins.mean())}

out = {"winmap": json.loads(winmap.to_json(orient="records")), "predictions": pred, "indicators": ind,
       "lomo_accuracy": lomo, "lomo_majority_baseline": base_rate, "noise_comparison": noise_cmp,
       "n_windows_primary": int(len(prim)), "physics_win_rate_primary": float(prim.physics_wins.mean())}
json.dump(out, open(HERE / "grid_summary.json", "w"), indent=1, default=float)
print("\n=== predictions ===")
print(json.dumps(pred, indent=1, default=float))
print(noise_cmp)
