"""Outside-regime confirmation (docs/research/ccpp-twin-outside-confirmation.md): the frozen rule HA_fb_gt scored on
answers it was not designed on (new models, two new worlds). Score once, after the collection is back.

Usage: PYTHONSAFEPATH=1 python3 analysis/ccpp_twin/outside_confirm.py <Folds5x2_pp.xlsx>
"""
import argparse
import glob
import json
import math
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
sys.path.insert(0, str(LLM))
import info_axis as ia  # noqa: E402
import outside_fallback as fb  # noqa: E402
import outside_regime as orr  # noqa: E402
from ccpp_twin import load_ccpp_inputs, simulate, with_modules  # noqa: E402
from grid import build_worlds, ols  # noqa: E402
from assemble import _json, assemble, load, s2_parts  # noqa: E402
from parse_score import TRUTH, parse, to_statement  # noqa: E402

SEED_PREFIX = "outside3"
NEW_TRUTH = {"chill": [("above", 20.0)], "gen_hot": []}       # in the flat-statement language (§4)
TRUTH_ALL = {**TRUTH, **NEW_TRUTH}
# task -> (world, side, k, kind); kind: "kink" / "nokink" are the primary tasks, "scope" is reported only (§5)
TASKS = {
    "evap|hot": ("evap", "hot", 24.0, "kink"),
    "evap+cap_mild|hot": ("evap+cap_mild", "hot", 24.0, "kink"),
    "cap_strong|cold": ("cap_strong", "cold", 9.0, "kink"),
    "chill|hot": ("chill", "hot", 20.0, "kink"),
    "base|hot": ("base", "hot", 24.0, "nokink"),
    "base|cold": ("base", "cold", 9.0, "nokink"),
    "cap_strong|hot": ("cap_strong", "hot", 24.0, "nokink"),
    "cap_mild|hot": ("cap_mild", "hot", 24.0, "nokink"),
    "gen_hot|hot": ("gen_hot", "hot", 28.0, "scope"),
}
KINK = tuple(t for t, v in TASKS.items() if v[3] == "kink")
NOKINK = tuple(t for t, v in TASKS.items() if v[3] == "nokink")
PRIMARY = KINK + NOKINK
SCOPE = tuple(t for t, v in TASKS.items() if v[3] == "scope")
D_REAL = orr.D_REAL
D_AWARE = {**orr.D_AWARE, "chill": [("above", 27.0)], "gen_hot": [("above", 27.0)]}
# D_aware with the numbers the L2 description states (evap 24, chiller 20, hot generator limit 28 degC).
D_READ = {"base": [], "cap_mild": [("below", 8.0)], "cap_strong": [("below", 8.0)], "evap": [("above", 24.0)],
          "evap+cap_mild": [("below", 8.0), ("above", 24.0)], "chill": [("above", 20.0)], "gen_hot": [("above", 28.0)]}
REFS = ("ORACLE", "D_real", "D_aware", "D_read", "P_side")
VARIANTS = ("HA_fb_gt", "HA", "HA_fb")                        # the first is the frozen rule; the others are reported
RULE = VARIANTS[0]
MIN_LINES = {"main": 1596, "strategy": 220, "newworld": 128}
H0_ORACLE, H0_DREAL_HARM, H1_HARM, H2_REL, H6_MARGIN, HARM_REL = 0.75, 0.30, 0.10, 0.95, 0.03, 1.10
MIN_MODELS = 2


def two_thirds(n):
    return math.ceil(2 * n / 3)


def model_tags(path):
    return [line.split("\t")[0] for line in open(path) if line.strip() and not line.startswith("#")]


def n_lines(path):
    return sum(1 for _ in open(path)) if Path(path).exists() else 0


def complete_tags(runs, tags):
    """Tags whose three collections are all complete (§3), and the line counts of every tag."""
    runs = Path(runs)
    counts = {t: {"main": n_lines(runs / f"{t}.jsonl"), "strategy": n_lines(runs / "strategy" / f"{t}.jsonl"),
                  "newworld": n_lines(runs / "newworld" / f"{t}.jsonl")} for t in tags}
    return [t for t in tags if all(counts[t][k] >= MIN_LINES[k] for k in MIN_LINES)], counts


def load_newworld(runs):
    """S0 statements, S2 parts and raw S2 objects of the new-world stage, with the filters of assemble.load."""
    s0, s2, raw, fail = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(lambda: [0, 0])
    for path in sorted(glob.glob(str(Path(runs) / "newworld" / "*.jsonl"))):
        for line in open(path):
            r = json.loads(line)
            if r["world"] not in NEW_TRUTH or r["level"] not in ("L1", "L2") or r["temperature"] == 0 or r["error"]:
                continue
            key = (r["tag"], r["level"], r["world"])
            fail[(r["tag"], r["strategy"])][1] += 1
            if r["strategy"] == "S0":
                p = parse(r["response"])
                if p["ok"]:
                    s0[key].append(to_statement(p["boundaries"]))
                else:
                    fail[(r["tag"], "S0")][0] += 1
            elif r["strategy"] == "S2":
                obj = _json(r["response"])
                if obj is not None:
                    s2[key].append(s2_parts(obj, r["level"]))
                    raw[key].append(obj)
                else:
                    fail[(r["tag"], "S2")][0] += 1
    return s0, s2, raw, {f"{t}|{k}": v for (t, k), v in fail.items()}


def statements(runs, tags):
    """{world: {"<tag>|<level>|<variant>": statement}} for every tag, level and assembly variant."""
    s0, s2 = load(Path(runs))
    raw = fb.load_s2_raw(runs)
    n0, n2, nraw, fail = load_newworld(runs)
    for d_old, d_new in ((s0, n0), (s2, n2), (raw, nraw)):
        d_old.update(d_new)
    out = {w: {} for w in TRUTH_ALL}
    for world in TRUTH_ALL:
        for tag in tags:
            for level in ("L1", "L2"):
                key = (tag, level, world)
                ha = assemble(s0.get(key, []), s2.get(key, []))
                objs = raw.get(key, [])
                cell = f"{tag}|{level}"
                out[world][f"{cell}|HA"] = ha
                out[world][f"{cell}|HA_fb_gt"] = fb.fallback(ha, objs, ("gt_limit",))
                out[world][f"{cell}|HA_fb"] = fb.fallback(ha, objs)
    claims = {f"{tag}|{level}|{world}": {"n": len(raw.get((tag, level, world), [])),
                                         **fb.claim_share(raw.get((tag, level, world), []))}
              for tag in tags for level in ("L1", "L2") for world in TRUTH_ALL}
    return out, claims, fail


def world_configs(x):
    worlds, _ = build_worlds(x)
    cfg = {w["module"]: w["cfg"] for w in worlds}
    cfg["chill"] = with_modules(inlet_chill=True)
    cfg["gen_hot"] = with_modules(hot_gen_limit=True)
    return cfg


def run_task(args):
    task, sigma, x, reps, stm, cfg = args
    warnings.filterwarnings("ignore")
    world, side, k, kind = TASKS[task]
    at, v, ap, rh = x["AT"], x["V"], x["AP"], x["RH"]
    clean = simulate(at, ap, rh, v, cfg)["pe_clean"]
    feats = np.column_stack([at, v, ap, rh])
    stmts = {"ORACLE": TRUTH_ALL[world], "D_real": D_REAL, "D_aware": D_AWARE[world], "D_read": D_READ[world], **stm}
    fakes = orr.side_placebos(side)
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
                if len(pool) < n:
                    raise RuntimeError(f"{task} delta={delta} w={w}: pool {len(pool)} < n {n}")
                for rep in range(reps):
                    pe = clean + np.random.default_rng(orr.noise_seed(world, sigma, rep, SEED_PREFIX)
                                                       ).normal(0.0, sigma, len(at))
                    idx = np.random.default_rng(orr.sample_seed(task, delta, w, n, rep, SEED_PREFIX)
                                                ).choice(pool, n, replace=False)
                    Xtr, ytr = feats[idx], pe[idx]
                    b, se = ols(Xtr, ytr)
                    slope_phys = -ytr.mean() / (at[idx].mean() + 273.15)
                    wlo, whi = at[idx].min(), at[idx].max()
                    center, edge = 0.5 * (wlo + whi), (whi if side == "hot" else wlo)
                    p0 = np.column_stack([np.ones(nt), Xte]) @ b
                    r0 = float(np.sqrt(np.mean((p0 - yt) ** 2)))

                    def rel(stmt):
                        p, rej = ia.predict_statement(stmt, p0, b, se, slope_phys, edge, center, Xte[:, 0], Xte[:, 3])
                        return float(np.sqrt(np.mean((p - yt) ** 2))) / r0, rej

                    rec = dict(task=task, kind=kind, world=world, side=side, k=k, delta=delta, w=w, n=n, sigma=sigma,
                               rep=rep, n_test=nt, R0_abs=r0)
                    for name, stmt in stmts.items():
                        rec[name], rec[f"rej|{name}"] = rel(stmt)
                    rec["P_side"] = float(np.mean([rel(st)[0] for st in fakes.values()]))
                    rows.append(rec)
    return rows


def run(x, stm_by_world, reps=20, workers=8):
    cfg = world_configs(x)
    jobs = [(t, s, x, reps, stm_by_world[TASKS[t][0]], cfg[TASKS[t][0]]) for t in TASKS for s in orr.SIGMAS]
    rows = []
    with ProcessPoolExecutor(workers) as ex:
        for r in ex.map(run_task, jobs):
            rows += r
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- aggregation and verdicts (§5-§6)
def rule_cols(df):
    return [c for c in df.columns if c in REFS or c.split("|")[-1] in VARIANTS and not c.startswith("rej|")]


def task_means(df):
    return df.groupby("task")[rule_cols(df)].mean().reindex(list(TASKS))


def macro(tm, tasks):
    return tm.loc[list(tasks)].mean()


def harm(df, tasks):
    return (df.loc[df["task"].isin(tasks), rule_cols(df)] > HARM_REL).mean()


def verdicts(df, cells, variant=RULE):
    """cells: ["<tag>|<level>", ...] of the completed new models. Verdicts of §6 for one assembly variant."""
    tags = sorted({c.split("|")[0] for c in cells})
    if len(tags) < MIN_MODELS:
        return {"no_verdict": f"only {len(tags)} completed model(s); at least {MIN_MODELS} are needed", "cells": cells}
    tm = task_means(df)
    kink, allm = macro(tm, KINK), macro(tm, PRIMARY)
    h_all, h_nokink, h_scope = harm(df, PRIMARY), harm(df, NOKINK), harm(df, SCOPE)
    per = {}
    for c in cells:
        col, level = f"{c}|{variant}", c.split("|")[1]
        ref = "D_aware" if level == "L1" else "D_read"
        per[c] = {"level": level, "kink": float(kink[col]), "all": float(allm[col]), "harm": float(h_all[col]),
                  "scope_harm": float(h_scope[col]), "ref": ref, "ref_kink": float(kink[ref]),
                  "H1": bool(h_all[col] <= H1_HARM), "H2": bool(kink[col] <= H2_REL),
                  "H3": bool(allm[col] < allm["D_real"]), "H4": bool(kink[col] < kink["P_side"]),
                  "H5": bool(kink[col] < kink["D_aware"]) if level == "L1" else None,
                  "H6": bool(kink[col] <= kink["D_read"] + H6_MARGIN) if level == "L2" else None,
                  "S": bool(h_scope[col] > H1_HARM)}
    n = len(cells)
    l1, l2 = [c for c in cells if per[c]["level"] == "L1"], [c for c in cells if per[c]["level"] == "L2"]
    out = {"variant": variant, "n_cells": n,
           "H0": {"oracle_kink": float(kink["ORACLE"]), "dreal_nokink_harm": float(h_nokink["D_real"]),
                  "pass": bool(kink["ORACLE"] < H0_ORACLE and h_nokink["D_real"] >= H0_DREAL_HARM)}}
    for h, pool, need in (("H1", cells, n - 1), ("H2", cells, two_thirds(n)), ("H3", cells, two_thirds(n)),
                          ("H4", cells, two_thirds(n)), ("H5", l1, two_thirds(len(l1))),
                          ("H6", l2, two_thirds(len(l2)))):
        ok = sum(per[c][h] for c in pool)
        out[h] = {"cells": ok, "of": len(pool), "need": need, "pass": bool(ok >= need)}
    out["stage1"] = bool(all(out[h]["pass"] for h in ("H0", "H1", "H2", "H3", "H4")))
    out["strong"] = bool(out["stage1"] and out["H5"]["pass"] and out["H6"]["pass"])
    s_ok = sum(per[c]["S"] for c in cells)
    out["S"] = {"cells_with_harm": s_ok, "of": n, "predicted_harm_in_half_or_more": bool(2 * s_ok >= n)}
    out["per_cell"] = per
    out["reference"] = {r: {"kink": float(kink[r]), "all": float(allm[r]), "harm_primary": float(h_all[r]),
                            "harm_nokink": float(h_nokink[r]), "harm_scope": float(h_scope[r])} for r in REFS}
    return out


def summarize(df, cells, stm, claims, fail, counts):
    v = verdicts(df, cells)
    pd.set_option("display.width", 260)
    L = [f"Outside-regime confirmation. Frozen rule {RULE}. relative RMSE = rule / R0. rows={len(df)}",
         f"collection line counts (need {MIN_LINES}): {counts}", f"verdict cells: {cells}", "", "=== verdicts ==="]
    if "no_verdict" in v:
        return v, "\n".join(L + [v["no_verdict"]]) + "\n"
    L.append(f"H0 premise: ORACLE kink mean {v['H0']['oracle_kink']:.3f} (< {H0_ORACLE}), D_real no-kink harm "
             f"{v['H0']['dreal_nokink_harm']:.3f} (>= {H0_DREAL_HARM}) -> {'PASS' if v['H0']['pass'] else 'FAIL'}")
    for h in ("H1", "H2", "H3", "H4", "H5", "H6"):
        L.append(f"{h}: {v[h]['cells']}/{v[h]['of']} cells (need {v[h]['need']}) -> "
                 f"{'PASS' if v[h]['pass'] else 'FAIL'}")
    L.append(f"stage 1 (H0-H4): {'HOLDS' if v['stage1'] else 'DOES NOT HOLD'}; strong (+H5, H6): "
             f"{'HOLDS' if v['strong'] else 'DOES NOT HOLD'}")
    L.append(f"S scope task gen_hot|hot: harm > {H1_HARM} in {v['S']['cells_with_harm']}/{v['S']['of']} cells "
             f"(prediction 'half or more': {v['S']['predicted_harm_in_half_or_more']})")
    names = rule_cols(df)
    tm = task_means(df)
    L += ["", "=== per verdict cell ===", pd.DataFrame(v["per_cell"]).T.to_string(),
          "", "=== reference rules ===", pd.DataFrame(v["reference"]).T.round(3).to_string(),
          "", "=== task mean (rows = rules) ===", tm.T.round(3).to_string(),
          "", "=== task median ===", df.groupby("task")[names].median().reindex(list(TASKS)).T.round(3).to_string(),
          "", "=== harm share (> 1.10) by task ===",
          (df[names] > HARM_REL).groupby(df["task"]).mean().reindex(list(TASKS)).T.round(3).to_string(),
          "", "=== macro means: kink tasks / primary tasks ===",
          pd.DataFrame({"kink": macro(tm, KINK), "primary": macro(tm, PRIMARY)}).round(3).to_string(),
          "", "=== gap closed (1 - rule) / (1 - ORACLE), kink tasks ===",
          ((1 - tm.loc[list(KINK)]).div(1 - tm.loc[list(KINK), "ORACLE"], axis=0)).T.round(2).to_string(),
          "", "=== task mean by delta ===", df.groupby(["task", "delta"])[names].mean().round(3).T.to_string(),
          "", "=== absolute RMSE (MW), task mean ==="]
    ab = df[names].mul(df["R0_abs"], axis=0)
    ab["R0"] = df["R0_abs"]
    L.append(ab.groupby(df["task"]).mean().reindex(list(TASKS)).T.round(2).to_string())
    L += ["", "=== guard rejection share by task ===",
          df.groupby("task")[[c for c in df.columns if c.startswith("rej|")]].mean().reindex(list(TASKS)).T.round(2)
          .to_string(), "", "=== report only: verdicts of the other assemblies on the same seeds ==="]
    for var in VARIANTS[1:]:
        o = verdicts(df, cells, var)
        L.append(f"{var}: " + ", ".join(f"{h} {o[h]['cells']}/{o[h]['of']}" for h in ("H1", "H2", "H3", "H4", "H5", "H6")))
    L += ["", "=== assembled statements by world (HA_fb_gt | HA) ==="]
    for world in TRUTH_ALL:
        for c in sorted({k.rsplit("|", 1)[0] for k in stm[world]}):
            L.append(f"{world:15s} {c:28s} {stm[world][f'{c}|{RULE}']} | {stm[world][f'{c}|HA']}")
    L += ["", "=== share of S2 answers claiming each device (cell|world) ==="]
    L += [f"{k:45s} " + " ".join(f"{m}={val:.2f}" if m != "n" else f"n={val}" for m, val in d.items())
          for k, d in claims.items()]
    L += ["", f"=== unparsed new-world answers [failed, total] === {fail}"]
    return v, "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx")
    ap.add_argument("--runs", default=str(LLM / "runs"))
    ap.add_argument("--models", default=str(LLM / "models_new.tsv"))
    ap.add_argument("--also", default="", help="comma-separated extra tags to report (never verdict cells)")
    ap.add_argument("--out", default=str(HERE / "runs" / "outside_confirm"))
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    new = model_tags(a.models)
    done, counts = complete_tags(a.runs, new)
    cells = [f"{t}|{lv}" for t in done for lv in ("L1", "L2")]
    extra = [t for t in a.also.split(",") if t]
    stm, claims, fail = statements(a.runs, done + extra)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = run(load_ccpp_inputs(a.xlsx), stm, reps=a.reps, workers=a.workers)
    df.to_csv(out_dir / "cells.csv.gz", index=False)
    v, text = summarize(df, cells, stm, claims, fail, counts)
    json.dump({"verdicts": v, "counts": counts, "cells": cells, "statements": stm, "claims": claims,
               "unparsed": fail}, open(out_dir / "summary.json", "w"), indent=1, default=str)
    (out_dir / "summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
