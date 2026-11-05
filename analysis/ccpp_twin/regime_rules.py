"""Regime-statement extrapolation rules on the CCPP twin (docs/research/ccpp-regime-statement.md).

Usage: python regime_rules.py <Folds5x2_pp.xlsx> [--reps 20]
Writes regime_cells.csv (gitignored) and regime_summary.json / regime_summary.txt. Real PE is never read.
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

COLD = {"cap_mild": 6.0, "cap_strong": 9.0, "evap+cap_mild": 6.0, "ice+cap_mild": 6.0}
HOT = {"evap": 24.0, "evap+cap_mild": 24.0}
# Anti-icing worlds (ccpp-anti-icing-world.md §4): flat below ICE[module] only on rows with RH > ICE_RH.
ICE = {"ice": 7.0, "ice+cap_mild": 7.0}
ICE_RH = 70.0
FALSE_COLD, FALSE_HOT = 9.0, 24.0

# A statement is a list of boundaries. A boundary is (side, at_c) or (side, at_c, rh_above); with rh_above the
# boundary applies only to rows whose relative humidity is above it (row-level assembly, anti-icing §5).
MECH_SIDE = {"power_cap": "below", "anti_icing": "below", "evap_cooling": "above", "exhaust_limit": "above"}


def from_mechanisms(items):
    """Mechanism-tagged boundaries (anti-icing §5) -> statement. Unknown mechanisms keep their own side."""
    out = []
    for it in items:
        side = it.get("side") or MECH_SIDE.get(it.get("mechanism"))
        if side not in ("below", "above") or it.get("at_c") is None:
            continue
        rh = it.get("rh_above") if it.get("mechanism") == "anti_icing" else None
        out.append((side, float(it["at_c"])) if rh is None else (side, float(it["at_c"]), float(rh)))
    return out


def _shift(b, s):
    return (b[0], b[1] + s) + tuple(b[2:])
GUARD_MARGIN = 2.0
HARM_MW = 1.0


def statements(module):
    truth = []
    if module in ICE:
        truth.append(("below", ICE[module], ICE_RH))
    if module in COLD:
        truth.append(("below", COLD[module]))
    if module in HOT:
        truth.append(("above", HOT[module]))
    out = {"correct": truth, "miss": []}
    if module in ICE and module in COLD:
        out["miss_ice"] = [b for b in truth if len(b) == 2]
        out["miss_cap"] = [b for b in truth if len(b) == 3]
    elif len(truth) == 2:
        out["miss_cold"] = [b for b in truth if b[0] == "above"]
        out["miss_hot"] = [b for b in truth if b[0] == "below"]
    fa = list(truth)
    if module not in COLD and module not in ICE:
        fa.append(("below", FALSE_COLD))
    if module not in HOT:
        fa.append(("above", FALSE_HOT))
    out["false_alarm"] = fa
    if truth:
        for s in (-6, -3, 3, 6):
            out[f"shift{s:+d}"] = [_shift(b, s) for b in truth]
    return out


def is_flat(t, stmt):
    """Flat region along AT. Humidity conditions are ignored here (used for the window-level guard)."""
    t = np.asarray(t, float)
    flat = np.zeros(t.shape, bool)
    for side, at, *_ in stmt:
        flat |= (t < at) if side == "below" else (t > at)
    return flat


def compose(stmt, e, at_test, slope_data, slope_phys, flat_slope_factor, window_flat, rh_test=None):
    """Integral of the piecewise slope from window edge e to each test AT.

    Boundaries with an rh_above condition apply only to test rows with RH above it, so rows are grouped by the
    set of boundaries active for them. Statements without conditions take the original single-curve path."""
    if any(len(b) > 2 for b in stmt):
        if rh_test is None:
            raise ValueError("statement has humidity conditions; pass rh_test")
        rh_test = np.asarray(rh_test, float)
        active = np.array([[len(b) == 2 or r > b[2] for b in stmt] for r in rh_test])
        out = np.empty(len(at_test))
        for key in {tuple(a) for a in active}:
            rows = np.all(active == key, axis=1)
            sub = [b[:2] for b, k in zip(stmt, key) if k]
            out[rows] = compose(sub, e, at_test[rows], slope_data, slope_phys, flat_slope_factor, window_flat)
        return out
    lo, hi = min(e, at_test.min()), max(e, at_test.max())
    grid = np.linspace(lo, hi, 2001)
    mid = 0.5 * (grid[1:] + grid[:-1])
    flat = is_flat(mid, stmt)
    s = np.where(flat == window_flat, slope_data, np.where(flat, flat_slope_factor * slope_phys, slope_phys))
    cum = np.concatenate([[0.0], np.cumsum(s * np.diff(grid))])
    return np.interp(at_test, grid, cum) - np.interp(e, grid, cum)


def run(x, reps, include_ice=False):
    worlds, _ = build_worlds(x, include_ice=include_ice)
    worlds = [w for w in worlds if w["noise"] > 0]
    at, ap, rh = x["AT"], x["AP"], x["RH"]
    r = ap / (at + 273.15)
    wins = windows(at)
    rows = []
    for world in worlds:
        v = x["V"] if world["vmode"] == "realV" else np.full_like(x["V"], np.median(x["V"]))
        clean = simulate(at, ap, rh, v, world["cfg"])["pe_clean"]
        feats = np.column_stack([at, v, ap, rh])
        wname = f'{world["module"]}|{world["vmode"]}|{world["noise"]}'
        stmts = statements(world["module"])
        for wd in wins:
            up = wd["setting"].endswith("->hot")
            for rep in range(reps):
                noise = np.random.default_rng(_seed(f"noise|{wname}|{rep}")).normal(0.0, world["noise"], len(at))
                pe = clean + noise
                idx = np.random.default_rng(_seed(f'sample|{wd["setting"]}|{wd["w"]}|{rep}')).choice(
                    wd["pool"], wd["n"], replace=False)
                b, se = ols(feats[idx], pe[idx])
                slope_phys = -pe[idx].mean() / (at[idx].mean() + 273.15)
                wlo, whi = at[wd["pool"]].min(), at[wd["pool"]].max()
                center, edge = 0.5 * (wlo + whi), (whi if up else wlo)
                z_normal, z_flat = abs(b[1] - slope_phys) / se[1], abs(b[1]) / se[1]
                for d, tmask in wd["tests"].items():
                    nt = int(tmask.sum())
                    if nt < 30:
                        continue
                    Xte, yt = feats[tmask], clean[tmask]
                    rmse = lambda p: float(np.sqrt(np.mean((p - yt) ** 2)))  # noqa: E731
                    p_raw = np.column_stack([np.ones(nt), Xte]) @ b
                    p_prop = pe[idx].mean() * r[tmask] / r[idx].mean()
                    e_raw, e_prop = rmse(p_raw), rmse(p_prop)
                    tcenter = float(np.median(Xte[:, 0]))
                    base_at_edge = p_raw - b[1] * (Xte[:, 0] - edge)
                    key = dict(module=world["module"], vmode=world["vmode"], setting=wd["setting"], w=wd["w"],
                               d=d, rep=rep, n_test=nt, R0_data=e_raw, Rp_physics=e_prop, oracle=min(e_raw, e_prop))
                    for sname, stmt in stmts.items():
                        wflat = bool(is_flat([center], stmt)[0])
                        crosses = any(min(center, tcenter) < a < max(center, tcenter) for _, a, *_ in stmt)
                        z_stated, z_other = (z_flat, z_normal) if wflat else (z_normal, z_flat)
                        rejected = (z_stated - z_other) >= GUARD_MARGIN
                        e1 = e_prop if crosses else e_raw
                        e2 = rmse(base_at_edge + compose(stmt, edge, Xte[:, 0], b[1], slope_phys, 0.0, wflat, Xte[:, 3]))
                        e2h = rmse(base_at_edge + compose(stmt, edge, Xte[:, 0], b[1], slope_phys, 0.5, wflat, Xte[:, 3]))
                        rows.append({**key, "statement": sname, "window_flat": wflat, "crosses": crosses,
                                     "rejected": rejected, "R1_switch": e1, "R2_compose": e2,
                                     "R2_compose_half": e2h, "R1g": e_raw if rejected else e1,
                                     "R2g": e_raw if rejected else e2})
    return pd.DataFrame(rows)


RULES = ["R0_data", "Rp_physics", "R1_switch", "R1g", "R2_compose", "R2g", "R2_compose_half", "oracle"]


def summarize(df):
    out, lines = {}, []
    agg = {}
    for (mod, st), g in df.groupby(["module", "statement"]):
        rec = {k: round(float(g[k].mean()), 2) for k in RULES}
        for k in ("R1_switch", "R1g", "R2_compose", "R2g"):
            rec[f"harm_{k}"] = round(float((g[k] > g.R0_data + HARM_MW).mean()), 3)
        rec["reject_rate"] = round(float(g.rejected.mean()), 3)
        agg[f"{mod}|{st}"] = rec
    out["by_module_statement"] = agg
    t = pd.DataFrame(agg).T
    lines.append("=== mean clean RMSE (MW), harm rate vs R0 (>1 MW worse), guard rejection rate ===")
    lines.append(t.to_string())
    corr = df[df.statement == "correct"]
    by_d = corr.groupby(["module", "d"])[["R0_data", "Rp_physics", "R1_switch", "R2_compose", "R2_compose_half"]].mean().round(2)
    lines.append("\n=== correct statement, by distance ===")
    lines.append(by_d.to_string())
    out["correct_by_distance"] = {f"{m}|{d}": v for (m, d), v in by_d.to_dict("index").items()}
    fa = df[df.statement == "false_alarm"]
    fa_set = fa.groupby(["module", "setting"]).agg(reject=("rejected", "mean"))
    fa_set["harm_R1"] = fa.assign(h=fa.R1_switch > fa.R0_data + HARM_MW).groupby(["module", "setting"]).h.mean()
    fa_set["harm_R1g"] = fa.assign(h=fa.R1g > fa.R0_data + HARM_MW).groupby(["module", "setting"]).h.mean()
    lines.append("\n=== false alarm by setting: guard rejection and harm ===")
    lines.append(fa_set.round(3).to_string())
    out["false_alarm_by_setting"] = {f"{m}|{s}": v for (m, s), v in fa_set.round(3).to_dict("index").items()}
    rej = corr.groupby(["module", "setting", "w"]).rejected.mean().round(3)
    lines.append("\n=== correct statement: guard false rejection by window ===")
    lines.append(rej[rej > 0].to_string())
    out["correct_rejection_by_window"] = {f"{m}|{s}|{w}": v for (m, s, w), v in rej.to_dict().items()}
    return out, "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("xlsx")
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--include-ice", action="store_true", help="add the anti-icing worlds (not in the saved outputs)")
    a = p.parse_args()
    df = run(load_ccpp_inputs(a.xlsx), a.reps, include_ice=a.include_ice)
    df.to_csv(HERE / "regime_cells.csv", index=False)
    out, text = summarize(df)
    json.dump(out, open(HERE / "regime_summary.json", "w"), indent=1)
    (HERE / "regime_summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
