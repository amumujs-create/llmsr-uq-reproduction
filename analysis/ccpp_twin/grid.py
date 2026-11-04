"""Pre-registered extrapolation grid on the CCPP twin (docs/research/ccpp-twin-grid.md).

Usage: python grid.py <Folds5x2_pp.xlsx> [--reps 20] [--quick]
Writes grid_cells.csv (one row per world x setting x distance x rep x model) and
grid_windows.csv (label-free indicators per training window). Real PE is never read.
"""
import argparse
import csv
import sys
import warnings
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ccpp_twin import TwinConfig, load_ccpp_inputs, simulate, with_modules  # noqa: E402

WIDTHS = {2.5: 146, 5.0: 300, 10.0: 1000}
DISTANCES = (5.0, 10.0, 15.0, 20.0)
SHELL = 2.5
MIN_TEST = 30
SETTINGS = ("cold->hot", "hot->cold", "mid->hot", "mid->cold")
FLEX = ("ridge", "mlp", "xgboost")


def build_worlds(x):
    gt = simulate(x["AT"], x["AP"], x["RH"], x["V"])["p_gt"]
    cap = {"mild": float(np.quantile(gt, 0.97)), "strong": float(np.quantile(gt, 0.90))}
    modules = {
        "base": with_modules(),
        "evap": with_modules(evap_cooling=True),
        "cap_mild": with_modules(cold_cap=True, gt_cap_mw=cap["mild"]),
        "cap_strong": with_modules(cold_cap=True, gt_cap_mw=cap["strong"]),
        "evap+cap_mild": with_modules(cold_cap=True, evap_cooling=True, gt_cap_mw=cap["mild"]),
    }
    worlds = []
    for mod, cfg in modules.items():
        for vmode in ("realV", "fixedV"):
            for noise in (4.3, 0.0):
                worlds.append({"module": mod, "vmode": vmode, "noise": noise, "cfg": cfg})
    return worlds, cap


def windows(at):
    lo, hi = np.quantile(at, [0.005, 0.995])
    med = float(np.median(at))
    out = []
    for w, n in WIDTHS.items():
        for s in SETTINGS:
            if s == "cold->hot":
                a, b, up = lo, lo + w, True
            elif s == "hot->cold":
                a, b, up = hi - w, hi, False
            else:
                a, b, up = med - w / 2, med + w / 2, s == "mid->hot"
            pool = np.flatnonzero((at >= a) & (at <= b))
            tests = {}
            for d in DISTANCES:
                t = (at >= b + d) & (at < b + d + SHELL) if up else (at <= a - d) & (at > a - d - SHELL)
                tests[d] = t
            out.append({"setting": s, "w": w, "n": n, "pool": pool, "tests": tests})
    return out


def _seed(name):
    return zlib.crc32(name.encode())


def ols(X, y):
    X1 = np.column_stack([np.ones(len(X)), X])
    b, *_ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1 @ b
    dof = max(len(y) - X1.shape[1], 1)
    s2 = resid @ resid / dof
    cov = s2 * np.linalg.pinv(X1.T @ X1)
    return b, np.sqrt(np.diag(cov))


def run_world(args):
    wi, world, x, reps, quick = args
    warnings.filterwarnings("ignore")
    cfg = world["cfg"]
    at, ap, rh = x["AT"], x["AP"], x["RH"]
    v = x["V"] if world["vmode"] == "realV" else np.full_like(x["V"], np.median(x["V"]))
    sim_clean = simulate(at, ap, rh, v, cfg)["pe_clean"]
    feats = np.column_stack([at, v, ap, rh])
    r = ap / (at + 273.15)
    flex = world["noise"] > 0 and not quick
    if flex:
        from sklearn.linear_model import RidgeCV
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler
        from xgboost import XGBRegressor
    cells, wins = [], []
    for wd in windows(at):
        for rep in range(reps):
            wname = f'{world["module"]}|{world["vmode"]}|{world["noise"]}'
            noise_rng = np.random.default_rng(_seed(f"noise|{wname}|{rep}"))
            noise = noise_rng.normal(0.0, world["noise"], size=len(at)) if world["noise"] > 0 else np.zeros(len(at))
            rng = np.random.default_rng(_seed(f'sample|{wd["setting"]}|{wd["w"]}|{rep}'))
            pe_obs = sim_clean + noise
            idx = rng.choice(wd["pool"], wd["n"], replace=False)
            Xtr, ytr = feats[idx], pe_obs[idx]
            b_raw, se_raw = ols(Xtr, ytr)
            b_den, _ = ols(r[idx, None], ytr)
            tk = at[idx].mean() + 273.15
            slope_prop = -ytr.mean() / tk
            q_b, q_se = ols(np.column_stack([at[idx], at[idx] ** 2]), ytr)
            models = {}
            if flex:
                sc = StandardScaler().fit(Xtr)
                ridge = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(sc.transform(Xtr), ytr)
                ym, ys = ytr.mean(), ytr.std()
                mlp = MLPRegressor((32, 32), max_iter=2000, random_state=rep, learning_rate_init=1e-3)
                mlp.fit(sc.transform(Xtr), (ytr - ym) / ys)
                xgb = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, n_jobs=1,
                                   random_state=rep).fit(Xtr, ytr)
                models = {"ridge": lambda X: ridge.predict(sc.transform(X)),
                          "mlp": lambda X: mlp.predict(sc.transform(X)) * ys + ym,
                          "xgboost": lambda X: xgb.predict(X)}
            lo_tr, hi_tr = Xtr[:, 1:].min(0), Xtr[:, 1:].max(0)
            for d, tmask in wd["tests"].items():
                nt = int(tmask.sum())
                key = dict(world=wi, module=world["module"], vmode=world["vmode"], noise=world["noise"],
                           setting=wd["setting"], w=wd["w"], n=wd["n"], d=d, rep=rep, n_test=nt)
                if nt < MIN_TEST:
                    continue
                Xte = feats[tmask]
                outside = ((Xte[:, 1:] < lo_tr) | (Xte[:, 1:] > hi_tr))
                wins.append({**key,
                             "slope_fit": b_raw[1], "slope_se": se_raw[1], "slope_prop": slope_prop,
                             "slope_gap": abs(b_raw[1] - slope_prop),
                             "gap_z": abs(b_raw[1] - slope_prop) / se_raw[1],
                             "cov_departure": float(outside.any(1).mean()),
                             "v_departure": float(outside[:, 0].mean()),
                             "curvature_t": abs(q_b[2] / q_se[2])})
                preds = {
                    "raw_linear": np.column_stack([np.ones(nt), Xte]) @ b_raw,
                    "density_linear": b_den[0] + b_den[1] * r[tmask],
                    "proportional": ytr.mean() * r[tmask] / r[idx].mean(),
                    "persistence": np.full(nt, ytr.mean()),
                }
                for name, f in models.items():
                    preds[name] = f(Xte)
                for name, p in preds.items():
                    cells.append({**key, "model": name,
                                  "rmse_clean": float(np.sqrt(np.mean((p - sim_clean[tmask]) ** 2))),
                                  "rmse_obs": float(np.sqrt(np.mean((p - pe_obs[tmask]) ** 2)))})
    return cells, wins


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--quick", action="store_true", help="closed-form models only")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    x = load_ccpp_inputs(a.xlsx)
    worlds, cap = build_worlds(x)
    print(f"cap levels (twin GT quantiles): {cap}")
    jobs = [(i, w, x, a.reps, a.quick) for i, w in enumerate(worlds)]
    cells, wins = [], []
    with ProcessPoolExecutor(a.workers) as ex:
        for c, w in ex.map(run_world, jobs):
            cells += c
            wins += w
    write_csv(cells, HERE / "grid_cells.csv")
    write_csv(wins, HERE / "grid_windows.csv")
    print(f"cells={len(cells)} windows={len(wins)}")


if __name__ == "__main__":
    main()
