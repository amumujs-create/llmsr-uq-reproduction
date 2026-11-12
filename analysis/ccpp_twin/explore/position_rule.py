"""EXPLORATION (post-hoc, not a verdict): rules that keep the stated position instead of "flat from the window edge".

docs/research/ccpp-twin-explore-position.md. Development models only (models.tsv), clean-world answers already seen,
new seeds ("explore_pos1"). Tasks and windows are those of clean_diag.py. Nothing here feeds a registered verdict.

Every rule is scored with the window guard and at two blends: 0.5 (the registered blend) and 1.0 (the statement taken
in full). Positions outside 2..37 degC are dropped, as in the assembly code.
  ORACLE     true position on kink tasks, nothing on no-kink tasks
  EXIST      true side, flat from the window edge (upper bound of position-free rules; uses the answer)
  CALC       no LLM. The description's numbers through one formula: generator limit at 15 - h / c_t degC with
             h = 100 * (limit / 343 - 1) and c_t = |window AT slope| / mean output * 100; coolers at the stated degC
  m|AND_EDGE the rule that failed the confirmation: both questions agree on the side, flat from the window edge
  m|S0POS    sides the direct question puts a boundary on (majority), at the median stated position
  m|S2POS    devices the decomposition claims (majority): generator limit at 15 - h / c_t with the model's own h,
             other devices at the median stated temperature; innermost per side
  m|BOTH     S2POS positions, kept only on sides S0POS also has

Usage: PYTHONSAFEPATH=1 python3 analysis/ccpp_twin/explore/position_rule.py <Folds5x2_pp.xlsx>
"""
import argparse
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent / "llm_statements"))
import clean_diag as cd  # noqa: E402
import info_axis as ia  # noqa: E402
import outside_fallback as fb  # noqa: E402
import outside_regime as orr  # noqa: E402
from assemble import L2_GT_15C_MW  # noqa: E402
from ccpp_twin import load_ccpp_inputs, simulate  # noqa: E402
from elicit import RANGE, _num  # noqa: E402
from grid import ols  # noqa: E402

SEED = "explore_pos1"
LAMS = (0.5, 1.0)
LIMIT_MW = {"cap353": 353.0, "cap350": 350.0, "cap346": 346.0}
STATED_C = {"evap22": 22.0, "evap26": 26.0, "chill24": 24.0}
FIELD = {"inlet_anti_icing_heating": "below_c", "inlet_evaporative_cooling": "above_c",
         "exhaust_or_steam_side_limit": "above_c"}
RULES = ("AND_EDGE", "S0POS", "S2POS", "BOTH")


def in_range(at):
    return at is not None and np.isfinite(at) and RANGE[0] < at < RANGE[1]


def innermost(cands):
    out = []
    for side in ("below", "above"):
        vals = [a for s, a in cands if s == side]
        if vals:
            out.append((side, float(max(vals) if side == "below" else min(vals))))
    return out


def s0_positions(stmts):
    """[(side, median at)] for sides a majority of S0 answers put a boundary on."""
    out = []
    for side in ("below", "above"):
        vals = [a for st in stmts for s, a in st if s == side]
        share = sum(any(s == side for s, _ in st) for st in stmts) / len(stmts)
        if share > 0.5 and in_range(float(np.median(vals))):
            out.append((side, float(np.median(vals))))
    return out


def s2_spec(objs):
    """Majority-claimed devices of the S2 answers: {"headroom": % or None, "fixed": [(side, median degC)]}."""
    share = fb.claim_share(objs)
    spec = {"headroom": None, "fixed": []}
    if share["gt_limit"] > 0.5:
        hs = []
        for o in objs:
            g = o.get("gt_generator_limit") or {}
            lim = _num(g.get("limit_mw"))
            if g.get("reached_in_range") is True and lim is not None:
                hs.append(100.0 * (lim / L2_GT_15C_MW - 1.0))
        spec["headroom"] = float(np.median(hs)) if hs else None
    for mech, field in FIELD.items():
        if share[mech] > 0.5:
            vals = [_num((o.get(mech) or {}).get(field)) for o in objs if (o.get(mech) or {}).get("active") is True]
            vals = [v for v in vals if v is not None]
            if vals and in_range(float(np.median(vals))):
                spec["fixed"].append((fb.SIDE[mech], float(np.median(vals))))
    return spec


def limit_position(headroom, c_t):
    if headroom is None or not np.isfinite(c_t) or c_t <= 0:
        return None
    at = 15.0 - headroom / c_t
    return at if in_range(at) else None


def s2_positions(spec, c_t):
    cands = list(spec["fixed"])
    at = limit_position(spec["headroom"], c_t)
    if at is not None:
        cands.append(("below", at))
    return innermost(cands)


def calc_statement(world, c_t):
    if world in LIMIT_MW:
        at = limit_position(100.0 * (LIMIT_MW[world] / L2_GT_15C_MW - 1.0), c_t)
        return [] if at is None else [("below", at)]
    return [("above", STATED_C[world])]


def run_task(args):
    task, sigma, x, reps, cells, per_tag, cfg = args
    warnings.filterwarnings("ignore")
    world, side, k, kind = cd.TASKS[task]
    word = cd.WORD[side]
    at, v, ap, rh = x["AT"], x["V"], x["AP"], x["RH"]
    clean = simulate(at, ap, rh, v, cfg)["pe_clean"]
    feats = np.column_stack([at, v, ap, rh])
    rows = []
    for delta, w in cells:
        lo, hi = ia.window_bounds(k, side, delta, w)
        pool = np.flatnonzero((at >= lo) & (at <= hi))
        tmask = ia.test_mask(at, k, side, lo, hi)
        nt = int(tmask.sum())
        Xte, yt = feats[tmask], clean[tmask]
        for n in cd.NS:
            for rep in range(reps):
                pe = clean + np.random.default_rng(orr.noise_seed(world, sigma, rep, SEED)).normal(0.0, sigma, len(at))
                idx = np.random.default_rng(orr.sample_seed(task, delta, w, n, rep, SEED)).choice(pool, n,
                                                                                                  replace=False)
                Xtr, ytr = feats[idx], pe[idx]
                b, se = ols(Xtr, ytr)
                slope_phys = -ytr.mean() / (at[idx].mean() + 273.15)
                wlo, whi = at[idx].min(), at[idx].max()
                center, edge = 0.5 * (wlo + whi), (whi if side == "hot" else wlo)
                p0 = np.column_stack([np.ones(nt), Xte]) @ b
                r0 = cd.rmse(p0, yt)
                c_t = abs(np.polyfit(at[idx], ytr, 1)[0]) / ytr.mean() * 100.0
                rec = dict(task=task, kind=kind, world=world, delta=delta, w=w, n=n, sigma=sigma, rep=rep, R0_abs=r0,
                           c_t=c_t)

                def put(name, stmt):
                    half, rejected = ia.predict_statement(stmt, p0, b, se, slope_phys, edge, center, Xte[:, 0],
                                                          Xte[:, 3])
                    full = p0 + (half - p0) / ia.LAMBDA
                    rec[f"{name}@0.5"], rec[f"{name}@1.0"] = cd.rmse(half, yt) / r0, cd.rmse(full, yt) / r0
                    rec[f"{name}|rej"] = bool(rejected)

                edge_stmt = [(word, float(edge))]
                put("ORACLE", cd.ORACLE[world] if kind == "kink" else [])
                put("EXIST", edge_stmt if kind == "kink" else [])
                put("CALC", calc_statement(world, c_t))
                for tag, (and_on, s0pos, spec) in per_tag.items():
                    s2pos = s2_positions(spec, c_t)
                    sides0 = {s for s, _ in s0pos}
                    put(f"{tag}|AND_EDGE", edge_stmt if and_on else [])
                    put(f"{tag}|S0POS", s0pos)
                    put(f"{tag}|S2POS", s2pos)
                    put(f"{tag}|BOTH", [(s, a) for s, a in s2pos if s in sides0])
                rows.append(rec)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx")
    ap.add_argument("--runs", default=str(HERE.parent.parent / "llm_statements" / "runs"))
    ap.add_argument("--models", default=str(HERE.parent.parent / "llm_statements" / "models.tsv"))
    ap.add_argument("--out", default=str(HERE.parent / "runs" / "explore_position"))
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    tags = cd.model_tags(a.models)
    s0, _, raw, _, _ = cd.load_clean(a.runs, tags)
    sh = cd.shares(s0, raw, tags)
    x = load_ccpp_inputs(a.xlsx)
    cfg = cd.world_configs(x)
    cells = cd.task_cells(x, cfg)
    stated = {(t, w): (s0_positions(s0[(t, w)]), s2_spec(raw[(t, w)])) for t in tags for w in cd.PLANTS}
    jobs = []
    for task, (world, side, _, _) in cd.TASKS.items():
        per_tag = {t: (cd.judge("AND", *sh[(t, world, cd.WORD[side])]), *stated[(t, world)]) for t in tags}
        jobs += [(task, s, x, a.reps, cells[task], per_tag, cfg[world]) for s in cd.SIGMAS]
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(run_task, jobs):
            rows += r
    df = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "cells.csv.gz", index=False)

    names = ["ORACLE", "EXIST", "CALC"] + [f"{t}|{r}" for t in tags for r in RULES]
    recs = []
    for name in names:
        for lam in LAMS:
            col = f"{name}@{lam}"
            mw = (df[col] * df["R0_abs"]).groupby([df["kind"], df["task"]]).mean()
            rel = df[col].groupby([df["kind"], df["task"]]).mean()
            harm = (df[df["kind"] == "nokink"][col] > cd.HARM_REL).mean()
            recs.append(dict(rule=name, blend=lam, kink_rel=rel["kink"].mean(), nokink_rel=rel["nokink"].mean(),
                             nokink_harm=harm, kink_MW=mw["kink"].mean(), nokink_MW=mw["nokink"].mean(),
                             balanced_MW=0.5 * (mw["kink"].mean() + mw["nokink"].mean()),
                             rejected=df[f"{name}|rej"].mean()))
    tab = pd.DataFrame(recs).set_index(["rule", "blend"])
    r0 = (df["R0_abs"]).groupby([df["kind"], df["task"]]).mean()
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 200)
    L = ["EXPLORATION (post-hoc, not a verdict). Position-keeping rules on development models. "
         f"rows={len(df)} seeds={SEED}",
         f"R0 absolute RMSE (MW): kink {r0['kink'].mean():.3f}  no-kink {r0['nokink'].mean():.3f}  "
         f"balanced {0.5 * (r0['kink'].mean() + r0['nokink'].mean()):.3f}", "",
         "=== stated positions by model and plant: S0 majority sides | S2 headroom % and fixed devices ==="]
    for t in tags:
        for w in cd.PLANTS:
            s0pos, spec = stated[(t, w)]
            hr = None if spec["headroom"] is None else round(spec["headroom"], 2)
            L.append(f"{t:24s} {w:8s} S0 {[(s, round(p, 1)) for s, p in s0pos]} | S2 h={hr} "
                     f"fixed {[(s, round(p, 1)) for s, p in spec['fixed']]}")
    L += ["", f"window sensitivity c_t (%/degC): mean {df['c_t'].mean():.3f}  p10 {df['c_t'].quantile(.1):.3f}  "
              f"p90 {df['c_t'].quantile(.9):.3f}", "",
          "=== rules (relative RMSE macro over tasks; harm share on no-kink rows; absolute MW) ===",
          tab.round(3).to_string(), "",
          "=== kink tasks, relative RMSE by task, blend 1.0 ===",
          df[df["kind"] == "kink"].groupby("task")[[f"{n}@1.0" for n in names]].mean().T.round(3).to_string(), "",
          "=== no-kink tasks, relative RMSE by task, blend 1.0 ===",
          df[df["kind"] == "nokink"].groupby("task")[[f"{n}@1.0" for n in names]].mean().T.round(3).to_string()]
    text = "\n".join(L) + "\n"
    (out / "summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
