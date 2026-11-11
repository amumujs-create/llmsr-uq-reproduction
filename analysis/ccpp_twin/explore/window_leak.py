"""EXPLORATION (post-hoc check): do the delta > 0 training windows already contain the regime?

Compares each kink world with the base world row by row (noise-free). Not a verdict; changes no registered rule.
Usage: PYTHONSAFEPATH=1 python3 analysis/ccpp_twin/explore/window_leak.py <Folds5x2_pp.xlsx>
"""
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent / "llm_statements"))
import info_axis as ia  # noqa: E402
import outside_regime as orr  # noqa: E402
from ccpp_twin import load_ccpp_inputs, simulate  # noqa: E402
from grid import build_worlds, ols  # noqa: E402


def main(xlsx):
    warnings.filterwarnings("ignore")
    x = load_ccpp_inputs(xlsx)
    cfg = {w["module"]: w["cfg"] for w in build_worlds(x)[0]}
    at = x["AT"]
    feats = np.column_stack([at, x["V"], x["AP"], x["RH"]])
    clean = {w: simulate(at, x["AP"], x["RH"], x["V"], cfg[w])["pe_clean"] for w in cfg}
    base = clean["base"]
    L = [f"Post-hoc check. mean base output {base.mean():.1f} MW. diff = |world - base|, noise-free.", "",
         "task delta w | window: share of rows with diff > 0.5 MW, mean diff, max diff | test: mean diff | "
         "R0 test RMSE fitted on the whole window (noise-free): world / base world"]
    for t, (world, side, k, kink) in orr.TASKS.items():
        if not kink:
            continue
        d = np.abs(clean[world] - base)
        for delta in orr.DELTAS:
            for w in orr.WIDTHS:
                lo, hi = ia.window_bounds(k, side, delta, w)
                pool = (at >= lo) & (at <= hi)
                tm = ia.test_mask(at, k, side, lo, hi)
                Xt = np.column_stack([np.ones(tm.sum()), feats[tm]])
                r = [float(np.sqrt(np.mean((Xt @ ols(feats[pool], y[pool])[0] - y[tm]) ** 2)))
                     for y in (clean[world], base)]
                L.append(f"{t:18s} {delta:2.0f} {w:3.0f} | {(d[pool] > 0.5).mean():.2f} {d[pool].mean():5.2f} "
                         f"{d[pool].max():5.2f} | {d[tm].mean():5.2f} | {r[0]:5.2f} / {r[1]:4.2f}")
    for world, step, rng in (("evap", 2, range(10, 34, 2)), ("cap_strong", 1, range(2, 18))):
        L += ["", f"{world}: diff from base by AT bin (mean, max)"]
        d = np.abs(clean[world] - base)
        for a in rng:
            m = (at >= a) & (at < a + step)
            if m.sum():
                L.append(f"  {a:2d}-{a + step:2d} degC n={m.sum():4d} mean {d[m].mean():6.2f} max {d[m].max():6.2f}")
    text = "\n".join(L) + "\n"
    out = HERE.parent / "runs" / "explore_window_leak.txt"
    out.write_text(text)
    print(text)


if __name__ == "__main__":
    main(sys.argv[1])
