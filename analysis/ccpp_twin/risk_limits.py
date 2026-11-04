"""Risk limits for untestable regime statements (docs/research/ccpp-risk-limit.md).

Usage: python risk_limits.py <Folds5x2_pp.xlsx> [--reps 20]
Writes risk_cells.csv (gitignored), risk_summary.json, risk_summary.txt, risk_frontier.png. Real PE is never read.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ccpp_twin import load_ccpp_inputs, simulate  # noqa: E402
from grid import _seed, build_worlds, ols, windows  # noqa: E402
from regime_rules import GUARD_MARGIN, compose, is_flat, statements  # noqa: E402

LAMBDAS = (0.25, 0.5, 0.75)
EPSILONS = (2.0, 5.0, 10.0)
HARM_MW = 1.0
CP_GRID = np.logspace(-2, 6, 17)


def coef_prior_fit(X, y, slope_target, rng):
    """OLS with penalty lam*(b_AT - slope_target)^2; lam by 5-fold CV inside the window."""
    def fit(Xa, ya, lam):
        X1 = np.column_stack([np.ones(len(Xa)), Xa])
        pen = np.zeros((1, X1.shape[1]))
        pen[0, 1] = np.sqrt(lam)
        b, *_ = np.linalg.lstsq(np.vstack([X1, pen]), np.append(ya, np.sqrt(lam) * slope_target), rcond=None)
        return b
    folds = rng.permutation(len(y)) % 5
    cv = []
    for lam in CP_GRID:
        err = 0.0
        for k in range(5):
            tr, te = folds != k, folds == k
            b = fit(X[tr], y[tr], lam)
            err += np.sum((np.column_stack([np.ones(te.sum()), X[te]]) @ b - y[te]) ** 2)
        cv.append(err)
    return fit(X, y, CP_GRID[int(np.argmin(cv))])


def run(x, reps):
    worlds = [w for w in build_worlds(x)[0] if w["noise"] > 0]
    at, ap, rh = x["AT"], x["AP"], x["RH"]
    wins = windows(at)
    rows = []
    for world in worlds:
        v = x["V"] if world["vmode"] == "realV" else np.full_like(x["V"], np.median(x["V"]))
        clean = simulate(at, ap, rh, v, world["cfg"])["pe_clean"]
        feats = np.column_stack([at, v, ap, rh])
        wname = f'{world["module"]}|{world["vmode"]}|{world["noise"]}'
        stmts = statements(world["module"])
        # Exploratory (not pre-registered): a fully wrong statement that names only the opposite side.
        swap = {"cap_mild": [("above", 24.0)], "cap_strong": [("above", 24.0)], "evap": [("below", 9.0)]}
        if world["module"] in swap:
            stmts["swap"] = swap[world["module"]]
        for wd in wins:
            up = wd["setting"].endswith("->hot")
            for rep in range(reps):
                pe = clean + np.random.default_rng(_seed(f"noise|{wname}|{rep}")).normal(0.0, world["noise"], len(at))
                idx = np.random.default_rng(_seed(f'sample|{wd["setting"]}|{wd["w"]}|{rep}')).choice(
                    wd["pool"], wd["n"], replace=False)
                Xtr, ytr = feats[idx], pe[idx]
                b, se = ols(Xtr, ytr)
                slope_phys = -ytr.mean() / (at[idx].mean() + 273.15)
                wlo, whi = at[wd["pool"]].min(), at[wd["pool"]].max()
                center, edge = 0.5 * (wlo + whi), (whi if up else wlo)
                z_normal, z_flat = abs(b[1] - slope_phys) / se[1], abs(b[1]) / se[1]
                b_cp = coef_prior_fit(Xtr, ytr, slope_phys, np.random.default_rng(_seed(f"cp|{wname}|{wd['setting']}|{wd['w']}|{rep}")))
                Xq = np.column_stack([at[idx], at[idx] ** 2, Xtr[:, 1:]])
                bq, _ = ols(Xq, ytr)
                s_ends = bq[1] + 2 * bq[2] * np.array([at[idx].min(), at[idx].max()])
                for d, tmask in wd["tests"].items():
                    nt = int(tmask.sum())
                    if nt < 30:
                        continue
                    Xte, yt = feats[tmask], clean[tmask]
                    X1 = np.column_stack([np.ones(nt), Xte])
                    p0 = X1 @ b
                    err0 = np.abs(p0 - yt)
                    rmse = lambda p: float(np.sqrt(np.mean((p - yt) ** 2)))  # noqa: E731
                    disp = Xte[:, 0] - edge
                    q_edge = bq[0] + bq[1] * edge + bq[2] * edge ** 2 + Xte[:, 1:] @ bq[3:]
                    pb_a, pb_b = q_edge + disp * s_ends[0], q_edge + disp * s_ends[1]
                    pb_lo, pb_hi = np.minimum(pb_a, pb_b), np.maximum(pb_a, pb_b)
                    key = dict(module=world["module"], vmode=world["vmode"], setting=wd["setting"], w=wd["w"],
                               d=d, rep=rep, n_test=nt, R0_data=rmse(p0), coef_prior=rmse(X1 @ b_cp),
                               pb_mid=rmse(0.5 * (pb_lo + pb_hi)),
                               pb_coverage=float(np.mean((yt >= pb_lo) & (yt <= pb_hi))))
                    base_at_edge = p0 - b[1] * disp
                    for sname, stmt in stmts.items():
                        wflat = bool(is_flat([center], stmt)[0])
                        z_stated, z_other = (z_flat, z_normal) if wflat else (z_normal, z_flat)
                        rejected = (z_stated - z_other) >= GUARD_MARGIN
                        p2 = p0 if rejected else base_at_edge + compose(stmt, edge, Xte[:, 0], b[1], slope_phys, 0.0, wflat)
                        delta = p2 - p0
                        rec = {**key, "statement": sname, "rejected": rejected, "R2g": rmse(p2),
                               "delta_abs_mean": float(np.abs(delta).mean())}
                        for lam in LAMBDAS:
                            rec[f"blend_{lam}"] = rmse(p0 + lam * delta)
                        for eps in EPSILONS:
                            pe_b = p0 + np.clip(delta, -eps, eps)
                            rec[f"budget_{eps:g}"] = rmse(pe_b)
                            rec[f"maxextra_budget_{eps:g}"] = float(np.max(np.abs(pe_b - yt) - err0))
                        rows.append(rec)
    return pd.DataFrame(rows)


RULES = ["R0_data", "R2g", "blend_0.25", "blend_0.5", "blend_0.75", "budget_2", "budget_5", "budget_10",
         "coef_prior", "pb_mid"]


def summarize(df):
    out, lines = {}, []
    tab = {}
    for (mod, st), g in df.groupby(["module", "statement"]):
        rec = {k: round(float(g[k].mean()), 2) for k in RULES}
        for k in ("R2g", "blend_0.5", "budget_5"):
            rec[f"harm_{k}"] = round(float((g[k] > g.R0_data + HARM_MW).mean()), 3)
        tab[f"{mod}|{st}"] = rec
    out["by_module_statement"] = tab
    lines.append("=== mean clean RMSE (MW) and harm rate (>1 MW worse than R0) ===")
    lines.append(pd.DataFrame(tab).T.to_string())

    maxextra = {f"budget_{e:g}": round(float(df[f"maxextra_budget_{e:g}"].max()), 4) for e in EPSILONS}
    out["max_row_extra_error"] = maxextra
    lines.append(f"\n=== max row-level extra error vs R0 (must be <= eps) === {maxextra}")

    corr = df[df.statement == "correct"]
    cov = corr.groupby(["module", "setting"]).pb_coverage.mean().unstack().round(3)
    out["pb_coverage"] = cov.to_dict()
    lines.append("\n=== pb interval coverage (clean truth), by module x setting ===")
    lines.append(cov.to_string())

    # Expected RMSE under statement reliability p.
    ps = np.round(np.linspace(0, 1, 21), 2)
    curves = {}
    wrong_names = lambda m, names: [n for n in names if n not in ("correct", "swap") and not (m == "base" and n == "miss")  # noqa: E731
                                    and not (m == "evap+cap_mild" and n == "false_alarm")]
    per_module = {}
    for mod, g in df.groupby("module"):
        names = sorted(g.statement.unique())
        Lc = g[g.statement == "correct"][RULES].mean()
        Lw = g[g.statement.isin(wrong_names(mod, names))].groupby("statement")[RULES].mean().mean()
        per_module[mod] = (Lc, Lw)
    for r in RULES:
        curves[r] = [round(float(np.mean([p * Lc[r] + (1 - p) * Lw[r] for Lc, Lw in per_module.values()])), 3) for p in ps]
    out["expected_rmse_by_p"] = {"p": ps.tolist(), **curves}
    breakeven = {}
    for r in RULES:
        if r == "R0_data":
            continue
        better = [p for p, a, b0 in zip(ps, curves[r], curves["R0_data"]) if a < b0]
        breakeven[r] = float(min(better)) if better else None
    out["breakeven_p"] = breakeven
    lines.append("\n=== expected RMSE (world mean) by statement reliability p ===")
    lines.append(pd.DataFrame(curves, index=ps).round(2).to_string())
    lines.append(f"\n=== break-even reliability p* (smallest p beating R0) === {breakeven}")
    best = {float(p): min(RULES, key=lambda r: curves[r][i]) for i, p in enumerate(ps)}
    out["best_rule_by_p"] = best
    lines.append(f"best rule by p: {best}")
    return out, "\n".join(lines)


def plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    e = out["expected_rmse_by_p"]
    fig, ax = plt.subplots(figsize=(8, 5))
    style = {"R0_data": ("#888", "-"), "R2g": ("#4a2", "-"), "blend_0.5": ("#27a", "-"), "blend_0.75": ("#27a", "--"),
             "budget_5": ("#c60", "-"), "budget_10": ("#c60", "--"), "coef_prior": ("#a3a", ":"), "pb_mid": ("#c44", ":")}
    for r, (c, ls) in style.items():
        ax.plot(e["p"], e[r], color=c, ls=ls, label=r)
    ax.set_xlabel("statement reliability p (probability the regime statement is correct)")
    ax.set_ylabel("expected clean RMSE, world mean (MW)")
    ax.set_title("Risk limits for untestable regime statements (CCPP twin)")
    ax.legend(fontsize=8)
    fig.savefig(HERE / "risk_frontier.png", dpi=130, bbox_inches="tight")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("xlsx")
    p.add_argument("--reps", type=int, default=20)
    a = p.parse_args()
    df = run(load_ccpp_inputs(a.xlsx), a.reps)
    df.to_csv(HERE / "risk_cells.csv", index=False)
    out, text = summarize(df)
    json.dump(out, open(HERE / "risk_summary.json", "w"), indent=1)
    (HERE / "risk_summary.txt").write_text(text)
    plot(out)
    print(text)


if __name__ == "__main__":
    main()
