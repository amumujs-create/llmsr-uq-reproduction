"""EXPLORATION (design aid, no LLM, no rule scored): which twin tasks have a training window without the regime?

For candidate worlds (sharper switch-on, different generator limits) and windows, compares the world with the base
world row by row, noise-free:
  leak   = largest |world - base| inside the training window (MW)
  effect = mean |world - base| in the test band (MW)
  R0     = test RMSE of the window OLS fit, in the world and (same window) in the base world
A task is "clean" when leak <= LEAK_MAX and "matters" when R0(world) - R0(base) >= GAIN_MIN.
Selection uses noise-free world properties only, never a rule's score.

Usage: PYTHONSAFEPATH=1 python3 analysis/ccpp_twin/explore/clean_tasks.py <Folds5x2_pp.xlsx>
"""
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent / "llm_statements"))
import info_axis as ia  # noqa: E402
from ccpp_twin import load_ccpp_inputs, simulate, with_modules  # noqa: E402
from grid import ols  # noqa: E402

LEAK_MAX, GAIN_MIN = 0.5, 3.0
DELTAS, WIDTHS = (1.0, 2.0, 4.0), (5.0, 10.0)
# name -> (side, config). Limits 357.3 / 353 / 350 MW; sharp = small softness or ramp.
WORLDS = {
    "cap357 soft2.0 (registered cap_strong)": ("cold", dict(cold_cap=True, gt_cap_mw=357.3357)),
    "cap357 soft0.5": ("cold", dict(cold_cap=True, gt_cap_mw=357.3357, cap_softness_mw=0.5)),
    "cap353 soft0.5": ("cold", dict(cold_cap=True, gt_cap_mw=353.0, cap_softness_mw=0.5)),
    "cap350 soft0.5": ("cold", dict(cold_cap=True, gt_cap_mw=350.0, cap_softness_mw=0.5)),
    "cap350 soft2.0": ("cold", dict(cold_cap=True, gt_cap_mw=350.0)),
    "cap346 soft0.5": ("cold", dict(cold_cap=True, gt_cap_mw=346.0, cap_softness_mw=0.5)),
    "evap24 ramp3.0 (registered evap)": ("hot", dict(evap_cooling=True)),
    "evap24 ramp0.3": ("hot", dict(evap_cooling=True, evap_ramp_c=0.3)),
    "evap22 ramp0.3": ("hot", dict(evap_cooling=True, evap_ramp_c=0.3, evap_on_c=22.0)),
    "evap26 ramp0.3": ("hot", dict(evap_cooling=True, evap_ramp_c=0.3, evap_on_c=26.0)),
    "chill20 (registered chill)": ("hot", dict(inlet_chill=True)),
    "chill24": ("hot", dict(inlet_chill=True, chill_set_c=24.0)),
    "gen_hot28 (registered gen_hot)": ("hot", dict(hot_gen_limit=True)),
}


def nominal_k(at, d, side):
    """First 0.5 degC step, moving toward the extrapolation side, where the mean |world - base| reaches 1 MW."""
    steps = np.arange(30.0, 2.0, -0.5) if side == "cold" else np.arange(8.0, 36.0, 0.5)
    for a in steps:
        m = (at >= a - 0.5) & (at < a) if side == "cold" else (at >= a) & (at < a + 0.5)
        if m.sum() >= 5 and d[m].mean() >= 1.0:
            return float(a)
    return None


def main(xlsx):
    warnings.filterwarnings("ignore")
    x = load_ccpp_inputs(xlsx)
    at = x["AT"]
    feats = np.column_stack([at, x["V"], x["AP"], x["RH"]])
    sim = lambda cfg: simulate(at, x["AP"], x["RH"], x["V"], cfg)["pe_clean"]  # noqa: E731
    base = sim(with_modules())
    L = [f"Design aid (no LLM, noise-free). clean: leak <= {LEAK_MAX} MW; matters: R0(world) - R0(base) >= {GAIN_MIN} MW.",
         "", "world | k | delta w | pool test_n | leak max / mean | test effect | R0 world / base | flags"]
    for name, (side, kw) in WORLDS.items():
        clean = sim(with_modules(**{k: v for k, v in kw.items() if k in ("cold_cap", "evap_cooling")},
                                 **{k: v for k, v in kw.items() if k not in ("cold_cap", "evap_cooling")}))
        d = np.abs(clean - base)
        k = nominal_k(at, d, side)
        if k is None:
            L.append(f"{name}: no nominal k found")
            continue
        for delta in DELTAS:
            for w in WIDTHS:
                lo, hi = ia.window_bounds(k, side, delta, w)
                pool = (at >= lo) & (at <= hi)
                tm = ia.test_mask(at, k, side, lo, hi)
                if tm.sum() < ia.MIN_TEST or pool.sum() < 500:
                    L.append(f"{name:40s} k={k:4.1f} d={delta:.0f} w={w:2.0f} | pool {pool.sum():4d} test {tm.sum():4d} "
                             "| too few rows")
                    continue
                Xt = np.column_stack([np.ones(tm.sum()), feats[tm]])
                r = [float(np.sqrt(np.mean((Xt @ ols(feats[pool], y[pool])[0] - y[tm]) ** 2))) for y in (clean, base)]
                flags = ("clean " if d[pool].max() <= LEAK_MAX else "") + ("matters" if r[0] - r[1] >= GAIN_MIN else "")
                L.append(f"{name:40s} k={k:4.1f} d={delta:.0f} w={w:2.0f} | {pool.sum():4d} {tm.sum():4d} | "
                         f"{d[pool].max():5.2f} / {d[pool].mean():4.2f} | {d[tm].mean():5.2f} | "
                         f"{r[0]:5.2f} / {r[1]:4.2f} | {flags}")
        L.append("")
    text = "\n".join(L) + "\n"
    (HERE.parent / "runs" / "explore_clean_tasks.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main(sys.argv[1])
