"""Clean-window diagnostic confirmation (docs/research/ccpp-twin-clean-diagnostic.md).

Six sharp-switch plants whose answers were collected in the clean-world stage (runs/cleanworld/, L2 only). For each
plant: one kink task on the side of its regime and one no-kink task on the opposite side. Only windows that carry no
trace of the regime are used (noise-free criteria, task_cells). Diagnostic claims C1-C7 and the positive claim P1-P3 are scored once on the four
new models; the five old models are development (--models models.tsv --dev).

Usage: PYTHONSAFEPATH=1 python3 analysis/ccpp_twin/clean_diag.py <Folds5x2_pp.xlsx>
"""
import argparse
import glob
import json
import sys
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
LLM = HERE.parent / "llm_statements"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "explore"))
sys.path.insert(0, str(LLM))
import clean_worlds as cw  # noqa: E402
import info_axis as ia  # noqa: E402
import outside_fallback as fb  # noqa: E402
import outside_regime as orr  # noqa: E402
from assemble import _json, assemble, s2_parts  # noqa: E402
from ccpp_twin import load_ccpp_inputs, simulate  # noqa: E402
from grid import build_worlds, ols  # noqa: E402
from parse_score import parse, to_statement  # noqa: E402
from robust_bound import hinge_fit, quad_fit  # noqa: E402

# The development run (old models) used "cleandiag1" and its LLM-free numbers were seen before registration, so the
# confirmation uses a prefix that has never been computed.
SEEDS = {"dev": "cleandiag1", "confirm": "cleandiag2"}
LEVEL = "L2"
MIN_LINES = 192
# world -> (regime side, nominal k on that side, oracle position, k of the no-kink task on the opposite side)
PLANTS = {
    "cap353": ("cold", 11.0, 11.0, 28.0),
    "cap350": ("cold", 12.5, 12.5, 28.0),
    "cap346": ("cold", 14.5, 14.5, 28.0),
    "evap22": ("hot", 21.5, 22.0, 9.0),
    "evap26": ("hot", 25.5, 26.0, 9.0),
    "chill24": ("hot", 24.0, 24.0, 9.0),
}
WORD = {"cold": "below", "hot": "above"}
OPP = {"cold": "hot", "hot": "cold"}
ORACLE = {w: [(WORD[s], pos)] for w, (s, _, pos, _) in PLANTS.items()}
# A careful reader of the L2 description: the stated switch temperature, or the default 8 degC for a generator limit.
READ = {"cap353": [("below", 8.0)], "cap350": [("below", 8.0)], "cap346": [("below", 8.0)],
        "evap22": [("above", 22.0)], "evap26": [("above", 26.0)], "chill24": [("above", 24.0)]}
TASKS = {}
for _w, (_s, _k, _, _ko) in PLANTS.items():
    TASKS[f"{_w}|{_s}"] = (_w, _s, _k, "kink")
    TASKS[f"{_w}|{OPP[_s]}"] = (_w, OPP[_s], _ko, "nokink")
KINK = tuple(t for t, v in TASKS.items() if v[3] == "kink")
NOKINK = tuple(t for t, v in TASKS.items() if v[3] == "nokink")
DELTAS, WIDTHS, NS, SIGMAS = orr.DELTAS, orr.WIDTHS, orr.NS, orr.SIGMAS
LEAK_MAX, GAIN_MIN, EFFECT_MAX, MIN_POOL = 0.5, 3.0, 0.5, 500
N_CELLS = {"kink": 30, "nokink": 33}                      # fixed at registration; checked at run time
LAM, HARM_REL = ia.LAMBDA, orr.HARM_REL
JUDGES = ("S0", "S2", "AND", "OR")
RULES = JUDGES + ("CONF",)
# thresholds of the claims (section 6 of the registration)
C1_REL, C1_TASKS, C1_KEPT = 0.95, 5, 0.10
C2_REL, C2_TASKS = 0.80, 5
C3_HARM, C3_TASKS, C3_BOUND_TOL, C3_BOUND_SHARE = 0.80, 5, 1.001, 0.99
C4_CELLS, C5_CELLS = 18, 8                       # of 24 model x plant cells
SAFE, GAIN = 0.10, 0.80                           # harm share on no-kink tasks; kink-task mean
C6_MODELS, C7_MODELS, C7_REL = 2, 3, 0.70
P_MODELS, P2_MARGIN = 2, 0.10
DIAG, POSITIVE = ("C1", "C2", "C3", "C4", "C5", "C6", "C7"), ("P1", "P2", "P3")


def rmse(p, y):
    return float(np.sqrt(np.mean((p - y) ** 2)))


def model_tags(path):
    return [line.split("\t")[0] for line in open(path) if line.strip() and not line.startswith("#")]


def world_configs(x):
    _, cap = build_worlds(x)
    cfg = cw.configs(cap["strong"])
    return {w: cfg[w] for w in ("base", *PLANTS)}


def task_cells(x, cfg):
    """{task: [(delta, w), ...]} kept by noise-free criteria only (no LLM, no rule score).
    kink: window leak <= LEAK_MAX and R0(world) - R0(base) >= GAIN_MIN. no-kink: window leak and test-band effect
    <= LEAK_MAX / EFFECT_MAX. Both: enough test and pool rows."""
    at = x["AT"]
    feats = np.column_stack([at, x["V"], x["AP"], x["RH"]])
    sim = lambda c: simulate(at, x["AP"], x["RH"], x["V"], c)["pe_clean"]  # noqa: E731
    base = sim(cfg["base"])
    out = {}
    for task, (world, side, k, kind) in TASKS.items():
        clean = sim(cfg[world])
        d = np.abs(clean - base)
        keep = []
        for delta in DELTAS:
            for w in WIDTHS:
                lo, hi = ia.window_bounds(k, side, delta, w)
                pool = (at >= lo) & (at <= hi)
                tm = ia.test_mask(at, k, side, lo, hi)
                if tm.sum() < ia.MIN_TEST or pool.sum() < MIN_POOL or d[pool].max() > LEAK_MAX:
                    continue
                if kind == "kink":
                    Xt = np.column_stack([np.ones(tm.sum()), feats[tm]])
                    r = [rmse(Xt @ ols(feats[pool], y[pool])[0], y[tm]) for y in (clean, base)]
                    ok = r[0] - r[1] >= GAIN_MIN
                else:
                    ok = d[tm].mean() <= EFFECT_MAX
                if ok:
                    keep.append((delta, w))
        out[task] = keep
    return out


def load_clean(runs, tags):
    """S0 statements, S2 parts and raw S2 objects per (tag, world) from runs/cleanworld/<tag>.jsonl."""
    s0, s2, raw, fail, lines = defaultdict(list), defaultdict(list), defaultdict(list), {}, {}
    for tag in tags:
        path = Path(runs) / "cleanworld" / f"{tag}.jsonl"
        recs = [json.loads(line) for line in open(path)] if path.exists() else []
        lines[tag] = len(recs)
        f = {"S0": [0, 0], "S2": [0, 0]}
        for r in recs:
            if r["world"] not in PLANTS or r["level"] != LEVEL or r["temperature"] == 0 or r["error"]:
                continue
            key = (tag, r["world"])
            f[r["strategy"]][1] += 1
            if r["strategy"] == "S0":
                p = parse(r["response"])
                if p["ok"]:
                    s0[key].append(to_statement(p["boundaries"]))
                else:
                    f["S0"][0] += 1
            else:
                obj = _json(r["response"])
                if obj is not None:
                    s2[key].append(s2_parts(obj, LEVEL))
                    raw[key].append(obj)
                else:
                    f["S2"][0] += 1
        fail[tag] = f
    return s0, s2, raw, fail, lines


def shares(s0, raw, tags):
    """{(tag, world, side word): (pS0, pS2)}: share of answers that put a boundary / a device on that side."""
    out = {}
    for tag in tags:
        for world in PLANTS:
            st, ob = s0.get((tag, world), []), raw.get((tag, world), [])
            if not st or not ob:
                raise RuntimeError(f"no usable answers for {tag} {world}: S0 {len(st)}, S2 {len(ob)}")
            for side in ("below", "above"):
                out[(tag, world, side)] = (
                    sum(any(s == side for s, _ in a) for a in st) / len(st),
                    sum(any(fb.SIDE[m] == side for m in fb.claims(o)) for o in ob) / len(ob))
    return out


def judge(name, p0, p2):
    a, b = p0 > 0.5, p2 > 0.5
    return {"S0": a, "S2": b, "AND": a and b, "OR": a or b}[name]


def frozen(s0, s2, raw, tags):
    """The frozen rule of the outside-regime confirmation (HA + generator-limit fallback) on the clean-world answers."""
    return {(t, w): fb.fallback(assemble(s0.get((t, w), []), s2.get((t, w), [])), raw.get((t, w), []), ("gt_limit",))
            for t in tags for w in PLANTS}


def run_task(args):
    task, sigma, x, reps, cells, lam, stm, cfg, seed = args
    warnings.filterwarnings("ignore")
    world, side, k, kind = TASKS[task]
    at, v, ap, rh = x["AT"], x["V"], x["AP"], x["RH"]
    clean = simulate(at, ap, rh, v, cfg)["pe_clean"]
    feats = np.column_stack([at, v, ap, rh])
    refs = {"ORACLE": ORACLE[world] if kind == "kink" else [], "READ": READ[world], **stm}
    lam_values = sorted(set(lam.values()) | {LAM})
    rows = []
    for delta, w in cells:
        lo, hi = ia.window_bounds(k, side, delta, w)
        pool = np.flatnonzero((at >= lo) & (at <= hi))
        tmask = ia.test_mask(at, k, side, lo, hi)
        nt = int(tmask.sum())
        Xte, yt = feats[tmask], clean[tmask]
        for n in NS:
            for rep in range(reps):
                pe = clean + np.random.default_rng(orr.noise_seed(world, sigma, rep, seed)).normal(
                    0.0, sigma, len(at))
                idx = np.random.default_rng(orr.sample_seed(task, delta, w, n, rep, seed)).choice(
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

                rec = dict(task=task, kind=kind, world=world, side=side, delta=delta, w=w, n=n, sigma=sigma, rep=rep,
                           R0_abs=r0)
                for name, st in refs.items():
                    rec[name] = rmse(pred(st), yt) / r0
                p2 = p0 + (pred([(WORD[side], float(edge))]) - p0) / LAM       # un-blended "flat beyond the edge"
                by_lam = {lv: rmse(p0 + lv * (p2 - p0), yt) / r0 for lv in lam_values}
                rec["EDGE_h"] = by_lam[LAM]
                bound = LAM * abs(b[1]) * float(np.sqrt(np.mean((Xte[:, 0] - edge) ** 2)))
                rec["bound_ok"] = bool((by_lam[LAM] - 1.0) * r0 <= C3_BOUND_TOL * bound + 1e-9)
                ph, kept = hinge_fit(Xtr, ytr, Xte, side)
                rec["HINGE"], rec["HINGE_kept"] = rmse(ph, yt) / r0, kept
                rec["QUAD"] = rmse(quad_fit(Xtr, ytr, Xte), yt) / r0
                for key, lv in lam.items():
                    rec[key] = by_lam[lv]
                rows.append(rec)
    return rows


def run(x, tags, sh, frz, seed, reps=20, workers=8):
    cfg = world_configs(x)
    cells = task_cells(x, cfg)
    got = {kind: sum(len(cells[t]) for t in TASKS if TASKS[t][3] == kind) for kind in N_CELLS}
    if got != N_CELLS:
        raise RuntimeError(f"task cells {got} differ from the registered {N_CELLS}")
    jobs = []
    for t, (world, side, _, _) in TASKS.items():
        lam = {}
        for tag in tags:
            p0, p2 = sh[(tag, world, WORD[side])]
            for j in JUDGES:
                lam[f"{tag}|{j}"] = LAM if judge(j, p0, p2) else 0.0
            lam[f"{tag}|CONF"] = round(LAM * min(p0, p2), 6)
        stm = {f"{tag}|FROZEN": frz[(tag, world)] for tag in tags}
        jobs += [(t, s, x, reps, cells[t], lam, stm, cfg[world], seed) for s in SIGMAS]
    rows = []
    with ProcessPoolExecutor(workers) as ex:
        for r in ex.map(run_task, jobs):
            rows += r
    return pd.DataFrame(rows), cells


def verdicts(df, tags, sh):
    g = df.groupby("task")
    tm = g.mean(numeric_only=True)
    kink, nokink = list(KINK), list(NOKINK)
    harm = lambda col, tasks: float((df.loc[df.task.isin(tasks), col] > HARM_REL).mean())  # noqa: E731
    v = {}
    kept = float(df.loc[df.task.isin(kink), "HINGE_kept"].mean())
    v["C1"] = {"hinge_tasks": int((tm.loc[kink, "HINGE"] >= C1_REL).sum()),
               "quad_tasks": int((tm.loc[kink, "QUAD"] >= C1_REL).sum()), "hinge_kept": kept}
    v["C1"]["pass"] = bool(v["C1"]["hinge_tasks"] >= C1_TASKS and v["C1"]["quad_tasks"] >= C1_TASKS and kept <= C1_KEPT)
    v["C2"] = {"tasks": int((tm.loc[kink, "EDGE_h"] <= C2_REL).sum())}
    v["C2"]["pass"] = bool(v["C2"]["tasks"] >= C2_TASKS)
    hs = (df["EDGE_h"] > HARM_REL).groupby(df["task"]).mean()
    v["C3"] = {"tasks": int((hs[nokink] >= C3_HARM).sum()),
               "bound_share": float(df.loc[df.task.isin(nokink), "bound_ok"].mean())}
    v["C3"]["pass"] = bool(v["C3"]["tasks"] >= C3_TASKS and v["C3"]["bound_share"] >= C3_BOUND_SHARE)
    true_side = {w: WORD[s] for w, (s, *_) in PLANTS.items()}
    false_side = {w: WORD[OPP[s]] for w, (s, *_) in PLANTS.items()}
    count = lambda j, sides: sum(judge(j, *sh[(t, w, sides[w])]) for t in tags for w in PLANTS)  # noqa: E731
    n = len(tags) * len(PLANTS)
    v["C4"] = {"cells": count("S2", true_side), "of": n}
    v["C4"]["pass"] = bool(v["C4"]["cells"] >= C4_CELLS)
    v["C5"] = {"cells": count("S2", false_side), "of": n}
    v["C5"]["pass"] = bool(v["C5"]["cells"] >= C5_CELLS)
    read_k = float(tm.loc[kink, "READ"].mean())
    per = {}
    for t in tags:
        per[t] = {r: {"kink": float(tm.loc[kink, f"{t}|{r}"].mean()), "harm_nokink": harm(f"{t}|{r}", nokink)}
                  for r in ("FROZEN",) + RULES}
        for r in per[t]:
            per[t][r]["safe"] = bool(per[t][r]["harm_nokink"] <= SAFE)
            per[t][r]["gainful"] = bool(per[t][r]["kink"] <= GAIN)
    m = len(tags)
    v["C6"] = {"models": sum(not per[t]["S2"]["safe"] for t in tags), "of": m}
    v["C6"]["pass"] = bool(v["C6"]["models"] >= C6_MODELS)
    v["C7"] = {"safe_models": sum(per[t]["FROZEN"]["safe"] for t in tags),
               "small_gain_models": sum(per[t]["FROZEN"]["kink"] >= C7_REL for t in tags), "of": m}
    v["C7"]["pass"] = bool(v["C7"]["safe_models"] >= C7_MODELS and v["C7"]["small_gain_models"] >= C7_MODELS)
    v["P1"] = {"models": sum(per[t]["AND"]["safe"] and per[t]["AND"]["gainful"] for t in tags), "of": m}
    v["P2"] = {"models": sum(per[t]["AND"]["safe"] and per[t]["AND"]["kink"] <= per[t]["FROZEN"]["kink"] - P2_MARGIN
                             for t in tags), "of": m}
    v["P3"] = {"models": sum(per[t]["AND"]["safe"] and per[t]["AND"]["kink"] < read_k for t in tags), "of": m,
               "read_kink": read_k}
    for c in POSITIVE:
        v[c]["pass"] = bool(v[c]["models"] >= P_MODELS)
    v["per_model"] = per
    v["diagnostic"] = bool(all(v[c]["pass"] for c in DIAG))
    v["positive"] = bool(all(v[c]["pass"] for c in POSITIVE))
    return v


def summarize(df, cells, tags, sh, frz, fail, lines, label, seed):
    v = verdicts(df, tags, sh)
    pd.set_option("display.width", 250)
    order = list(TASKS)
    tm = df.groupby("task").mean(numeric_only=True).reindex(order)
    kink, nokink = list(KINK), list(NOKINK)
    L = [f"Clean-window diagnostic confirmation [{label}]. relative RMSE = rule / R0. rows={len(df)} seeds={seed}",
         f"models: {tags}", f"collection lines (need {MIN_LINES}): {lines}", f"unparsed [failed, total]: {fail}",
         "task cells (delta, w): " + "; ".join(f"{t} {len(c)}" for t, c in cells.items()), "", "=== verdicts ===",
         f"C1 data-only bending models do not help: HINGE >= {C1_REL} on {v['C1']['hinge_tasks']}/6, QUAD on "
         f"{v['C1']['quad_tasks']}/6 kink tasks (need {C1_TASKS}), hinge kept {v['C1']['hinge_kept']:.3f} (<= {C1_KEPT}) "
         f"-> {'PASS' if v['C1']['pass'] else 'FAIL'}",
         f"C2 existence alone is enough: EDGE_h <= {C2_REL} on {v['C2']['tasks']}/6 kink tasks (need {C2_TASKS}) -> "
         f"{'PASS' if v['C2']['pass'] else 'FAIL'}",
         f"C3 a false alarm is costly and bounded: EDGE_h harm >= {C3_HARM} on {v['C3']['tasks']}/6 no-kink tasks "
         f"(need {C3_TASKS}), bound holds in {v['C3']['bound_share']:.4f} of rows (>= {C3_BOUND_SHARE}) -> "
         f"{'PASS' if v['C3']['pass'] else 'FAIL'}",
         f"C4 presence is known: S2 says yes on the true side in {v['C4']['cells']}/{v['C4']['of']} cells "
         f"(need {C4_CELLS}) -> {'PASS' if v['C4']['pass'] else 'FAIL'}",
         f"C5 the decomposition question raises false alarms: S2 yes on the false side in {v['C5']['cells']}/"
         f"{v['C5']['of']} cells (need {C5_CELLS}) -> {'PASS' if v['C5']['pass'] else 'FAIL'}",
         f"C6 the S2-only rule is unsafe: no-kink harm > {SAFE} in {v['C6']['models']}/{v['C6']['of']} models "
         f"(need {C6_MODELS}) -> {'PASS' if v['C6']['pass'] else 'FAIL'}",
         f"C7 the frozen rule is safe but gains little: safe {v['C7']['safe_models']}/{v['C7']['of']}, kink mean >= "
         f"{C7_REL} in {v['C7']['small_gain_models']}/{v['C7']['of']} (need {C7_MODELS} each) -> "
         f"{'PASS' if v['C7']['pass'] else 'FAIL'}",
         f"DIAGNOSTIC CLAIMS (C1-C7): {'ALL HOLD' if v['diagnostic'] else 'DO NOT ALL HOLD'}", "",
         f"P1 the AND rule is safe (no-kink harm <= {SAFE}) and gainful (kink mean <= {GAIN}) in {v['P1']['models']}/"
         f"{v['P1']['of']} models (need {P_MODELS}) -> {'PASS' if v['P1']['pass'] else 'FAIL'}",
         f"P2 safe and at least {P2_MARGIN} below the frozen rule on kink tasks in {v['P2']['models']}/{v['P2']['of']} "
         f"models (need {P_MODELS}) -> {'PASS' if v['P2']['pass'] else 'FAIL'}",
         f"P3 safe and below reading the description (READ kink {v['P3']['read_kink']:.3f}) in {v['P3']['models']}/"
         f"{v['P3']['of']} models (need {P_MODELS}) -> {'PASS' if v['P3']['pass'] else 'FAIL'}",
         f"POSITIVE CLAIM (P1-P3): {'HOLDS' if v['positive'] else 'DOES NOT HOLD'}", ""]
    refs = ["ORACLE", "READ", "EDGE_h", "HINGE", "QUAD"]
    L += ["=== reference rules, task mean ===", tm[refs].T.round(3).to_string(), "",
          "harm share (> 1.10):", (df[refs] > HARM_REL).groupby(df["task"]).mean().reindex(order).T.round(3).to_string(),
          "", "R0 absolute RMSE (MW): " + "  ".join(f"{t} {tm.loc[t, 'R0_abs']:.2f}" for t in order), "",
          "=== side judgment: share of answers saying yes (pS0 / pS2); true side first, then false side ==="]
    for t in tags:
        L.append(f"{t:26s} " + "  ".join(
            f"{w} {sh[(t, w, WORD[PLANTS[w][0]])][0]:.2f}/{sh[(t, w, WORD[PLANTS[w][0]])][1]:.2f} | "
            f"{sh[(t, w, WORD[OPP[PLANTS[w][0]]])][0]:.2f}/{sh[(t, w, WORD[OPP[PLANTS[w][0]]])][1]:.2f}" for w in PLANTS))
    rows = []
    for t in tags:
        for r in ("FROZEN",) + RULES:
            col = f"{t}|{r}"
            sub = df[df.task.isin(nokink)]
            rows.append({"model": t, "rule": r, "kink": tm.loc[kink, col].mean(), "nokink": tm.loc[nokink, col].mean(),
                         "harm_kink": float((df.loc[df.task.isin(kink), col] > HARM_REL).mean()),
                         "harm_nokink": float((sub[col] > HARM_REL).mean()),
                         "abs_kink_MW": float((df.loc[df.task.isin(kink), col] * df.loc[df.task.isin(kink), "R0_abs"]).mean()),
                         "abs_nokink_MW": float((sub[col] * sub["R0_abs"]).mean())})
    L += ["", "=== rules by model (macro over tasks; harm and MW pooled over rows) ===",
          pd.DataFrame(rows).set_index(["model", "rule"]).round(3).to_string(),
          "", "=== frozen-rule statements by model and plant ==="]
    L += [f"{t:26s} " + "  ".join(f"{w} {frz[(t, w)]}" for w in PLANTS) for t in tags]
    L += ["", "=== task mean of every model rule ===",
          tm[[c for c in df.columns if "|" in c and c.split("|")[0] in tags]].T.round(3).to_string()]
    return v, "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx")
    ap.add_argument("--runs", default=str(LLM / "runs"))
    ap.add_argument("--models", default=str(LLM / "models_new.tsv"))
    ap.add_argument("--dev", action="store_true", help="development run (old models); writes to clean_diag_dev")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    tags = model_tags(a.models)
    new = set(model_tags(LLM / "models_new.tsv"))
    if a.dev and new & set(tags):
        raise SystemExit("--dev must not read the confirmation models")
    if not a.dev and set(tags) != new:
        raise SystemExit("the confirmation run uses exactly models_new.tsv")
    s0, s2, raw, fail, lines = load_clean(a.runs, tags)
    short = [t for t in tags if lines[t] < MIN_LINES]
    if short:
        raise SystemExit(f"incomplete collection: {short}")
    sh = shares(s0, raw, tags)
    frz = frozen(s0, s2, raw, tags)
    seed = SEEDS["dev" if a.dev else "confirm"]
    df, cells = run(load_ccpp_inputs(a.xlsx), tags, sh, frz, seed, reps=a.reps, workers=a.workers)
    out_dir = HERE / "runs" / ("clean_diag_dev" if a.dev else "clean_diag")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "cells.csv.gz", index=False)
    v, text = summarize(df, cells, tags, sh, frz, fail, lines, "DEVELOPMENT, not a verdict" if a.dev else "CONFIRMATION", seed)
    json.dump({"verdicts": v, "cells": {t: c for t, c in cells.items()}, "lines": lines, "unparsed": fail,
               "shares": {"|".join(k): val for k, val in sh.items()},
               "frozen": {"|".join(k): val for k, val in frz.items()}}, open(out_dir / "summary.json", "w"), indent=1,
              default=str)
    (out_dir / "summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
