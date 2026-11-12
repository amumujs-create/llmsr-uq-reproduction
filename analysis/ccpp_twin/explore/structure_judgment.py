"""EXPLORATION (post-hoc, not a verdict): does the LLM know WHETHER the response weakens on a side, and is that enough?

docs/research/ccpp-twin-explore-structure.md. Reads the main, strategy and new-world answers of all nine models. The
four new models' answers on these worlds were already scored in the confirmation (ccpp-twin-outside-confirmation.md
section 11), so nothing here is a confirmation. The clean-world answers are NOT read. New seeds ("explore_struct1").

Per (model, level, world, side in below/above):
  pS0 = share of direct answers (S0) that put a boundary on that side
  pS2 = share of decomposition answers (S2) that claim a device acting on that side
  judges: S0 = pS0 > 0.5, S2 = pS2 > 0.5, AND, OR
Rules scored on the confirmation tasks (no position is taken from the LLM):
  <judge>   : "flat beyond the window edge", blend 0.5, when the judge says yes for the extrapolation side; else R0
  CONF      : the same with blend 0.5 * min(pS0, pS2)
References: ORACLE, EDGE_h (always yes), P_side, D_aware, D_read, the frozen rule HA_fb_gt.

Usage: PYTHONSAFEPATH=1 python3 analysis/ccpp_twin/explore/structure_judgment.py <Folds5x2_pp.xlsx>
"""
import argparse
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
LLM = HERE.parent.parent / "llm_statements"
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(LLM))
import info_axis as ia  # noqa: E402
import outside_confirm as oc  # noqa: E402
import outside_fallback as fb  # noqa: E402
import outside_regime as orr  # noqa: E402
from assemble import load  # noqa: E402
from ccpp_twin import load_ccpp_inputs, simulate  # noqa: E402
from grid import ols  # noqa: E402

SEED_PREFIX = "explore_struct1"
TASKS = oc.TASKS
WORLDS = tuple(oc.TRUTH_ALL)
SIDES = ("below", "above")
JUDGES = ("S0", "S2", "AND", "OR")
RULES = JUDGES + ("CONF",)
LAM = ia.LAMBDA


def rmse(p, y):
    return float(np.sqrt(np.mean((p - y) ** 2)))


def shares(runs, tags):
    """{(tag, level, world, side): (pS0, pS2, nS0, nS2)}"""
    s0, _ = load(Path(runs))
    raw = fb.load_s2_raw(runs)
    n0, _, nraw, _ = oc.load_newworld(runs)
    s0.update(n0)
    raw.update(nraw)
    out = {}
    for tag in tags:
        for level in ("L1", "L2"):
            for world in WORLDS:
                st, ob = s0.get((tag, level, world), []), raw.get((tag, level, world), [])
                if not st or not ob:
                    raise RuntimeError(f"no answers for {tag} {level} {world}: S0 {len(st)}, S2 {len(ob)}")
                for side in SIDES:
                    p0 = sum(any(s == side for s, _ in x) for x in st) / len(st)
                    p2 = sum(any(fb.SIDE[m] == side for m in fb.claims(o)) for o in ob) / len(ob)
                    out[(tag, level, world, side)] = (p0, p2, len(st), len(ob))
    return out


def judge(name, p0, p2):
    a, b = p0 > 0.5, p2 > 0.5
    return {"S0": a, "S2": b, "AND": a and b, "OR": a or b}[name]


def run_task(args):
    task, sigma, x, reps, lam, stm, cfg = args
    warnings.filterwarnings("ignore")
    world, side, k, kind = TASKS[task]
    word = "above" if side == "hot" else "below"
    at, v, ap, rh = x["AT"], x["V"], x["AP"], x["RH"]
    clean = simulate(at, ap, rh, v, cfg)["pe_clean"]
    feats = np.column_stack([at, v, ap, rh])
    refs = {"ORACLE": oc.TRUTH_ALL[world], "D_aware": oc.D_AWARE[world], "D_read": oc.D_READ[world], **stm}
    fakes = orr.side_placebos(side)
    lam_values = sorted(set(lam.values()) | {LAM})
    rows = []
    for delta in orr.DELTAS:
        for w in orr.WIDTHS:
            lo, hi = ia.window_bounds(k, side, delta, w)
            pool = np.flatnonzero((at >= lo) & (at <= hi))
            tmask = ia.test_mask(at, k, side, lo, hi)
            nt = int(tmask.sum())
            if nt < ia.MIN_TEST:
                raise RuntimeError(f"{task} delta={delta} w={w}: only {nt} test rows")
            Xte, yt = feats[tmask], clean[tmask]
            for n in orr.NS:
                for rep in range(reps):
                    pe = clean + np.random.default_rng(orr.noise_seed(world, sigma, rep, SEED_PREFIX)).normal(
                        0.0, sigma, len(at))
                    idx = np.random.default_rng(orr.sample_seed(task, delta, w, n, rep, SEED_PREFIX)).choice(
                        pool, n, replace=False)
                    Xtr, ytr = feats[idx], pe[idx]
                    b, se = ols(Xtr, ytr)
                    slope_phys = -ytr.mean() / (at[idx].mean() + 273.15)
                    wlo, whi = at[idx].min(), at[idx].max()
                    center, edge = 0.5 * (wlo + whi), (whi if side == "hot" else wlo)
                    p0 = np.column_stack([np.ones(nt), Xte]) @ b
                    r0 = rmse(p0, yt)

                    def pred(stmt):
                        return ia.predict_statement(stmt, p0, b, se, slope_phys, edge, center, Xte[:, 0], Xte[:, 3])[0]

                    rec = dict(task=task, kind=kind, world=world, delta=delta, w=w, n=n, sigma=sigma, rep=rep, R0_abs=r0)
                    for name, st in refs.items():
                        rec[name] = rmse(pred(st), yt) / r0
                    rec["P_side"] = float(np.mean([rmse(pred(st), yt) / r0 for st in fakes.values()]))
                    p2 = p0 + (pred([(word, float(edge))]) - p0) / LAM          # un-blended "flat beyond the edge"
                    by_lam = {lv: rmse(p0 + lv * (p2 - p0), yt) / r0 for lv in lam_values}
                    rec["EDGE_h"] = by_lam[LAM]
                    for key, lv in lam.items():
                        rec[key] = by_lam[lv]
                    rows.append(rec)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx")
    ap.add_argument("--runs", default=str(LLM / "runs"))
    ap.add_argument("--out", default=str(HERE.parent / "runs" / "explore_structure"))
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    dev, new = oc.model_tags(LLM / "models.tsv"), oc.model_tags(LLM / "models_new.tsv")
    tags = dev + new
    cells = [f"{t}|{lv}" for t in tags for lv in ("L1", "L2")]
    sh = shares(a.runs, tags)
    truth = {(w, s): any(x == s for x, _ in oc.TRUTH_ALL[w]) for w in WORLDS for s in SIDES}

    pd.set_option("display.width", 250)
    L = ["EXPLORATION (post-hoc). Structure judgment: does the response weaken on a side? Clean-world answers not read. "
         f"seeds={SEED_PREFIX}", "", "=== Part A: side judgment against truth (6 true sides, 8 false sides per cell) ==="]
    recs = []
    for tag in tags:
        for level in ("L1", "L2"):
            r = {"group": "dev" if tag in dev else "new", "cell": f"{tag}|{level}"}
            for j in JUDGES:
                yes = {ws: judge(j, *sh[(tag, level, *ws)][:2]) for ws in truth}
                r[f"{j}_TPR"] = np.mean([yes[ws] for ws in truth if truth[ws]])
                r[f"{j}_FPR"] = np.mean([yes[ws] for ws in truth if not truth[ws]])
            recs.append(r)
    ja = pd.DataFrame(recs).set_index("cell")
    L += [ja.round(2).to_string(), "", "group means:", ja.groupby("group").mean(numeric_only=True).round(2).to_string(),
          "", "by level:", ja.assign(level=ja.index.str[-2:]).groupby(["group", "level"]).mean(numeric_only=True)
          .round(2).to_string()]
    L += ["", "=== Part A2: mean share of answers saying 'yes' by world and side (pS0 / pS2), * = true side ==="]
    for grp, gt in (("dev", dev), ("new", new)):
        for level in ("L1", "L2"):
            parts = []
            for w in WORLDS:
                for s in SIDES:
                    a0 = np.mean([sh[(t, level, w, s)][0] for t in gt])
                    a2 = np.mean([sh[(t, level, w, s)][1] for t in gt])
                    parts.append(f"{w}/{s}{'*' if truth[(w, s)] else ''} {a0:.2f}/{a2:.2f}")
            L.append(f"{grp} {level}: " + "  ".join(parts))

    x = load_ccpp_inputs(a.xlsx)
    cfg = oc.world_configs(x)
    stm, _, _ = oc.statements(a.runs, tags)
    jobs = []
    for t in TASKS:
        world, side, _, _ = TASKS[t]
        word = "above" if side == "hot" else "below"
        lam = {}
        for c in cells:
            tag, level = c.split("|")
            p0, p2 = sh[(tag, level, world, word)][:2]
            for j in JUDGES:
                lam[f"{c}|{j}"] = LAM if judge(j, p0, p2) else 0.0
            lam[f"{c}|CONF"] = round(LAM * min(p0, p2), 6)
        frozen = {f"{c}|HA_fb_gt": stm[world][f"{c}|HA_fb_gt"] for c in cells}
        jobs += [(t, s, x, a.reps, lam, frozen, cfg[world]) for s in orr.SIGMAS]
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(run_task, jobs):
            rows += r
    df = pd.DataFrame(rows)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "cells.csv.gz", index=False)

    kink, nokink, prim = list(oc.KINK), list(oc.NOKINK), list(oc.PRIMARY)
    tm = df.groupby("task").mean(numeric_only=True)
    prim_rows = df[df.task.isin(prim)]
    refs = ["ORACLE", "D_read", "D_aware", "P_side", "EDGE_h"]
    ref_tab = pd.DataFrame({r: {"kink": tm.loc[kink, r].mean(), "primary": tm.loc[prim, r].mean(),
                                "harm": float((prim_rows[r] > orr.HARM_REL).mean())} for r in refs}).T
    L += ["", f"=== Part B: rules on the confirmation tasks. rows={len(df)} ===", "reference rules:",
          ref_tab.round(3).to_string()]
    allr = ("HA_fb_gt",) + RULES
    per = []
    for c in cells:
        tag, level = c.split("|")
        r = {"group": "dev" if tag in dev else "new", "cell": c}
        for rule in allr:
            col = f"{c}|{rule}"
            r[f"{rule}_kink"] = tm.loc[kink, col].mean()
            r[f"{rule}_prim"] = tm.loc[prim, col].mean()
            r[f"{rule}_harm"] = float((prim_rows[col] > orr.HARM_REL).mean())
            r[f"{rule}_scope"] = float((df.loc[df.task.isin(oc.SCOPE), col] > orr.HARM_REL).mean())
        per.append(r)
    pc = pd.DataFrame(per).set_index("cell")
    for suffix, title in (("kink", "kink-task mean"), ("prim", "primary-task mean"), ("harm", "harm share, primary"),
                          ("scope", "harm share, gen_hot|hot")):
        cols = [f"{r}_{suffix}" for r in allr]
        L += ["", f"--- {title} ---", pc[["group"] + cols].rename(columns=lambda s: s.replace(f"_{suffix}", ""))
              .round(3).to_string(),
              "group means: " + pc.groupby("group")[cols].mean().rename(columns=lambda s: s.replace(f"_{suffix}", ""))
              .round(3).to_string().replace("\n", "\n             ")]
    L += ["", "--- cells meeting each exploratory threshold (same thresholds as the confirmation; NOT a verdict) ---"]
    p_side_k, d_aware_k, d_read_k = ref_tab.loc["P_side", "kink"], ref_tab.loc["D_aware", "kink"], ref_tab.loc["D_read", "kink"]
    for grp in ("dev", "new"):
        g = pc[pc.group == grp]
        for rule in allr:
            k_, h_ = g[f"{rule}_kink"], g[f"{rule}_harm"]
            l1, l2 = g.index.str.endswith("L1"), g.index.str.endswith("L2")
            L.append(f"{grp} {rule:9s} harm<=0.10: {(h_ <= 0.10).sum()}/{len(g)}  kink<=0.95: {(k_ <= 0.95).sum()}/{len(g)}  "
                     f"kink<P_side: {(k_ < p_side_k).sum()}/{len(g)}  L1 kink<D_aware: {(k_[l1] < d_aware_k).sum()}/{l1.sum()}  "
                     f"L2 kink<=D_read+0.03: {(k_[l2] <= d_read_k + 0.03).sum()}/{l2.sum()}")
    L += ["", "--- task mean, group average over cells ---"]
    for grp, gt in (("dev", dev), ("new", new)):
        cc = [f"{t}|{lv}" for t in gt for lv in ("L1", "L2")]
        tab = pd.DataFrame({rule: tm[[f"{c}|{rule}" for c in cc]].mean(axis=1) for rule in allr}).reindex(list(TASKS))
        L += [grp, tab.T.round(3).to_string()]
    text = "\n".join(L) + "\n"
    (out_dir / "summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
