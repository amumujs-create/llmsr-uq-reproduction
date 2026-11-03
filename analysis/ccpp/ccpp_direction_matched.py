"""Matched comparison of CCPP extrapolation direction (analysis; uses test PE).

For each direction, train on an AT window of width w at one end, with n rows
subsampled, and predict a band at distance d beyond the window. Same w, n, d in
both directions. Models are closed-form least squares so the comparison is not
about optimizer behaviour.
"""
import sys

import numpy as np
import openpyxl

rows = list(openpyxl.load_workbook(sys.argv[1], read_only=True, data_only=True)["Sheet1"].iter_rows(values_only=True))
d = np.array([[float(v) for v in r[:5]] for r in rows[1:] if r[0] is not None])
at, v, ap, rh, pe = d.T
tk = at + 273.15


def design(kind, m):
    cols = {
        "raw": [at[m], v[m], ap[m], rh[m]],
        "density": [ap[m] / tk[m], v[m], rh[m]],
        "raw_AT_only": [at[m]],
        "density_only": [ap[m] / tk[m]],
    }[kind]
    return np.column_stack([np.ones(m.sum())] + cols)


rng = np.random.default_rng(0)
lo_end, hi_end = np.quantile(at, [0.005, 0.995])
print(f"AT range used: {lo_end:.1f} ~ {hi_end:.1f} C")
print("direction  w   n   d   | RMSE raw  density | rawAT  densOnly | test rows")
for w, n, dist in ((2.5, 146, 10), (2.5, 146, 15), (5, 300, 10), (5, 300, 15), (10, 1000, 10)):
    for direction in ("cold->hot", "hot->cold"):
        if direction == "cold->hot":
            tr_lo, tr_hi = lo_end, lo_end + w
            te = (at >= tr_hi + dist) & (at < tr_hi + dist + 2.5)
        else:
            tr_lo, tr_hi = hi_end - w, hi_end
            te = (at <= tr_lo - dist) & (at > tr_lo - dist - 2.5)
        pool = np.flatnonzero((at >= tr_lo) & (at <= tr_hi))
        res = {k: [] for k in ("raw", "density", "raw_AT_only", "density_only")}
        for _ in range(20):
            m = np.zeros(len(at), bool)
            m[rng.choice(pool, min(n, len(pool)), replace=False)] = True
            for k in res:
                beta, *_ = np.linalg.lstsq(design(k, m), pe[m], rcond=None)
                res[k].append(np.sqrt(np.mean((design(k, te) @ beta - pe[te]) ** 2)))
        r = {k: np.mean(x) for k, x in res.items()}
        print(f"{direction:10s} {w:4.1f} {n:4d} {dist:3.0f} | {r['raw']:7.1f} {r['density']:8.1f} | "
              f"{r['raw_AT_only']:6.1f} {r['density_only']:8.1f} | {te.sum()}")

# Diagnostics: how other inputs move with AT at each end.
for name, m in (("cold 2-7C", (at >= 2) & (at <= 7)), ("hot 31-36C", (at >= 31) & (at <= 36))):
    cv = np.corrcoef(at[m], v[m])[0, 1]
    slope_v = np.polyfit(at[m], v[m], 1)[0]
    print(f"{name}: rows={m.sum()} corr(AT,V)={cv:.2f} dV/dAT={slope_v:.2f} "
          f"PE sd={pe[m].std():.1f} V range={v[m].min():.1f}-{v[m].max():.1f}")
full = np.polyfit(at, v, 1)[0]
print(f"whole data dV/dAT={full:.2f}")
