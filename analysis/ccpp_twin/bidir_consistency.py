"""Bidirectional fit-consistency pre-experiment, no LLM (docs/research/ccpp-bidirectional-consistency.md).

Usage: python bidir_consistency.py <Folds5x2_pp.xlsx> [--reps 20] [--out runs/bidir]
Writes bidir_cells.csv (one row per world x window x rep x distance) and bidir_summary.txt.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ccpp_twin import load_ccpp_inputs, simulate  # noqa: E402
from grid import MIN_TEST, _seed, build_worlds, windows  # noqa: E402
from regime_rules import statements  # noqa: E402

BELOW_K = (4.0, 6.0, 8.0, 10.0, 12.0)
ABOVE_K = (22.0, 24.0, 26.0, 28.0, 30.0)
FORMS = (["lin", "quad", "cubic", "density"] + [f"below_{k:g}" for k in BELOW_K]
         + [f"above_{k:g}" for k in ABOVE_K])
K_TOL = 1.05
RULES = ("bi", "fwd", "rcv", "bic", "lin", "best")


def design(form, at, v, ap, rh):
    if form == "density":
        return np.column_stack([np.ones(len(at)), ap / (at + 273.15), v, rh])
    if form.startswith("below_"):
        t = np.maximum(at, float(form[6:]))
    elif form.startswith("above_"):
        t = np.minimum(at, float(form[6:]))
    else:
        t = at
    cols = [np.ones(len(at)), t, v, ap, rh]
    if form in ("quad", "cubic"):
        c = at - 20.0  # centred to keep the normal equations well conditioned
        cols.append(c ** 2)
        if form == "cubic":
            cols.append(c ** 3)
    return np.column_stack(cols)


def fit(X, y):
    """OLS; returns None when the coefficients are not identified on these rows (e.g. a clip column that is
    constant inside the window). A min-norm solution would then extrapolate with an arbitrary slope."""
    if np.linalg.matrix_rank(X) < X.shape[1]:
        return None
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b


def err_or_inf(X, b, y):
    return np.inf if b is None else rmse(X @ b, y)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def classify(stmt, at_tr, up, test_at):
    """visible / hidden / none, from the true boundaries (ccpp-bidirectional-consistency.md §3)."""
    lo, hi = at_tr.min(), at_tr.max()
    far = test_at.max() if up else test_at.min()
    kind = "none"
    for b in stmt:
        a = b[1]
        if lo < a < hi:
            return "visible"
        if (up and hi <= a <= far) or (not up and far <= a <= lo):
            kind = "hidden"
    return kind


def run(x, reps):
    worlds, _ = build_worlds(x, include_ice=True)
    worlds = [w for w in worlds if w["noise"] > 0 and w["vmode"] == "realV"]
    at, ap, rh, v = x["AT"], x["AP"], x["RH"], x["V"]
    rows = []
    for world in worlds:
        clean = simulate(at, ap, rh, v, world["cfg"])["pe_clean"]
        wname = f'{world["module"]}|{world["vmode"]}|{world["noise"]}'
        stmt = statements(world["module"])["correct"]
        Xall = {f: design(f, at, v, ap, rh) for f in FORMS}
        for wd in windows(at):
            up = wd["setting"].endswith("->hot")
            for rep in range(reps):
                pe = clean + np.random.default_rng(_seed(f"noise|{wname}|{rep}")).normal(0.0, world["noise"], len(at))
                idx = np.random.default_rng(_seed(f'sample|{wd["setting"]}|{wd["w"]}|{rep}')).choice(
                    wd["pool"], wd["n"], replace=False)
                med = np.median(at[idx])
                outer = idx[at[idx] > med] if up else idx[at[idx] < med]
                inner = np.setdiff1d(idx, outer)
                folds = np.array_split(np.random.default_rng(_seed(f"cv|{wd['setting']}|{wd['w']}|{rep}"))
                                       .permutation(idx), 5)
                sc = {r: {} for r in ("bi", "fwd", "rcv", "bic")}
                full = {}
                forms = []
                for f in FORMS:
                    X = Xall[f]
                    b = fit(X[idx], pe[idx])
                    if b is None:  # not estimable on this window: not a candidate
                        continue
                    forms.append(f)
                    full[f] = b
                    e_fwd = err_or_inf(X[outer], fit(X[inner], pe[inner]), pe[outer])
                    e_bwd = err_or_inf(X[inner], fit(X[outer], pe[outer]), pe[inner])
                    sc["fwd"][f], sc["bi"][f] = e_fwd, max(e_fwd, e_bwd)
                    cv = []
                    for k in range(5):
                        te = folds[k]
                        tr = np.concatenate([folds[j] for j in range(5) if j != k])
                        cv.append(err_or_inf(X[te], fit(X[tr], pe[tr]), pe[te]) ** 2)
                    sc["rcv"][f] = float(np.sqrt(np.mean(cv)))
                    rss = float(np.sum((X[idx] @ b - pe[idx]) ** 2))
                    sc["bic"][f] = len(idx) * np.log(rss / len(idx)) + X.shape[1] * np.log(len(idx))
                pick = {r: min(sc[r], key=sc[r].get) for r in sc}
                bmin = min(sc["bi"].values())
                K = [f for f in forms if sc["bi"][f] <= K_TOL * bmin]
                Ks = {t: [f for f in forms if sc["bi"][f] <= t * bmin] for t in (1.02, 1.10)}
                for d, tmask in wd["tests"].items():
                    nt = int(tmask.sum())
                    if nt < MIN_TEST:
                        continue
                    yt = clean[tmask]
                    pred = {f: Xall[f][tmask] @ full[f] for f in forms}
                    err = {f: rmse(pred[f], yt) for f in forms}
                    P = np.array([pred[f] for f in K])
                    rec = dict(module=world["module"], setting=wd["setting"], w=wd["w"], d=d, rep=rep, n_test=nt,
                               kind=classify(stmt, at[idx], up, at[tmask]), K_size=len(K), n_forms=len(forms),
                               K_spread=float((P.max(0) - P.min(0)).mean()),
                               pick_bi=pick["bi"], pick_rcv=pick["rcv"], pick_fwd=pick["fwd"], pick_bic=pick["bic"])
                    for r in ("bi", "fwd", "rcv", "bic"):
                        rec[r] = err[pick[r]]
                    rec["lin"], rec["best"] = err["lin"], min(err.values())
                    for t, ks in Ks.items():
                        Pt = np.array([pred[f] for f in ks])
                        rec[f"K_spread_{t:g}"] = float((Pt.max(0) - Pt.min(0)).mean())
                    rows.append(rec)
    return pd.DataFrame(rows)


def summarize(df):
    lines = []
    m = df.groupby("kind")[list(RULES)].mean().round(2)
    n = df.groupby("kind").size()
    lines.append("mean clean RMSE by cell kind (MW)")
    lines.append(m.assign(cells=n).to_string())
    g = {k: df[df.kind == k] for k in ("none", "hidden", "visible")}
    b1 = g["none"]["bi"].mean() < g["none"]["rcv"].mean()
    b2 = (g["hidden"]["bi"].mean() - g["hidden"]["lin"].mean()) > -0.2
    b3 = (g["visible"]["lin"].mean() - g["visible"]["bi"].mean()) >= 0.5
    hn = df[df.kind.isin(["none", "hidden"])]
    rho = float(hn["K_spread"].rank().corr(hn["bi"].rank()))
    b4 = rho > 0.3 and g["hidden"]["K_spread"].mean() > g["none"]["K_spread"].mean()
    lines.append("")
    lines.append(f"B1 none: bi {g['none']['bi'].mean():.2f} < rcv {g['none']['rcv'].mean():.2f} -> {'PASS' if b1 else 'FAIL'}")
    lines.append(f"B2 hidden: bi - lin = {g['hidden']['bi'].mean() - g['hidden']['lin'].mean():+.2f} > -0.2 -> "
                 f"{'PASS' if b2 else 'FAIL'}")
    lines.append(f"B3 visible: lin - bi = {g['visible']['lin'].mean() - g['visible']['bi'].mean():+.2f} >= 0.5 -> "
                 f"{'PASS' if b3 else 'FAIL'}")
    lines.append(f"B4 hidden+none: spearman(K_spread, bi err) = {rho:.2f}; spread hidden "
                 f"{g['hidden']['K_spread'].mean():.2f} vs none {g['none']['K_spread'].mean():.2f} -> "
                 f"{'PASS' if b4 else 'FAIL'}")
    lines.append("")
    lines.append("by world x kind (bi, rcv, lin, best, K_spread)")
    lines.append(df.groupby(["module", "kind"])[["bi", "rcv", "lin", "best", "K_spread"]].mean().round(2).to_string())
    lines.append("")
    lines.append("by width x kind (bi - lin, bi - rcv)")
    t = df.assign(bi_lin=df.bi - df.lin, bi_rcv=df.bi - df.rcv).groupby(["w", "kind"])[["bi_lin", "bi_rcv"]].mean()
    lines.append(t.round(2).to_string())
    lines.append("")
    lines.append("K tolerance sensitivity (exploratory): spearman(K_spread_t, bi err) on hidden+none")
    for t in ("1.02", "1.1"):
        lines.append(f"  tol {t}: {hn[f'K_spread_{t}'].rank().corr(hn['bi'].rank()):.2f}")
    lines.append("")
    lines.append("forms picked by bi, by kind (share)")
    lines.append(df.groupby("kind")["pick_bi"].value_counts(normalize=True).round(2).groupby(level=0).head(5).to_string())
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("xlsx")
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--out", default=str(HERE / "runs" / "bidir"))
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = run(load_ccpp_inputs(a.xlsx), a.reps)
    df.to_csv(out / "bidir_cells.csv", index=False)
    s = summarize(df)
    (out / "bidir_summary.txt").write_text(s + "\n")
    print(s)


if __name__ == "__main__":
    main()
