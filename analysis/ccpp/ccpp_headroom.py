"""CCPP extrapolation headroom check using train targets and test inputs only.

Split follows llm_prior E16-B: train AT <= q70, test AT > q90 (quantiles of AT on Sheet1).
Test PE is never read.
"""
import sys

import numpy as np
import openpyxl

path = sys.argv[1]
book = openpyxl.load_workbook(path, read_only=True, data_only=True)
sheet = book["Sheet1"]
rows = list(sheet.iter_rows(values_only=True))
header = [str(h).strip() for h in rows[0]]
data = np.array([[float(v) for v in r[:5]] for r in rows[1:] if r[0] is not None])
col = {name: i for i, name in enumerate(header[:5])}
at, v, ap, rh = (data[:, col[k]] for k in ("AT", "V", "AP", "RH"))
q70, q90 = np.quantile(at, [0.70, 0.90])
train = at <= q70
test = at > q90
pe_train = data[train, col["PE"]]  # test PE intentionally unused

def design(kind, mask):
    t_k = at[mask] + 273.15
    if kind == "linear_AT":
        cols = [at[mask], v[mask], ap[mask], rh[mask]]
    elif kind == "inverse_T":
        cols = [1.0 / t_k, v[mask], ap[mask], rh[mask]]
    elif kind == "density_AP_over_T":
        cols = [ap[mask] / t_k, v[mask], rh[mask]]
    elif kind == "quadratic_AT":
        cols = [at[mask], at[mask] ** 2, v[mask], ap[mask], rh[mask]]
    else:
        raise ValueError(kind)
    return np.column_stack([np.ones(mask.sum())] + cols)

print(f"rows={len(data)} train={train.sum()} test={test.sum()} "
      f"train AT max={at[train].max():.2f} test AT min={at[test].min():.2f} max={at[test].max():.2f}")
preds, resid_sd = {}, {}
for kind in ("linear_AT", "inverse_T", "density_AP_over_T", "quadratic_AT"):
    beta, *_ = np.linalg.lstsq(design(kind, train), pe_train, rcond=None)
    resid = pe_train - design(kind, train) @ beta
    resid_sd[kind] = resid.std(ddof=design(kind, train).shape[1])
    preds[kind] = design(kind, test) @ beta
    print(f"{kind:20s} train residual SD={resid_sd[kind]:.3f} MW")

base = preds["linear_AT"]
pe_range = pe_train.max() - pe_train.min()
print(f"\ntrain PE range={pe_range:.1f} MW")
for kind in ("inverse_T", "density_AP_over_T", "quadratic_AT"):
    d = preds[kind] - base
    print(f"{kind:20s} vs linear at test X: RMS diff={np.sqrt(np.mean(d**2)):.3f} MW, "
          f"max |diff|={np.abs(d).max():.3f} MW, mean diff={d.mean():+.3f} MW, "
          f"RMS diff / train residual SD={np.sqrt(np.mean(d**2))/resid_sd['linear_AT']:.2f}")

# Curvature evidence inside the training range: AT^2 coefficient and its t-statistic.
X = design("quadratic_AT", train)
beta, *_ = np.linalg.lstsq(X, pe_train, rcond=None)
resid = pe_train - X @ beta
sigma2 = resid @ resid / (len(pe_train) - X.shape[1])
cov = sigma2 * np.linalg.inv(X.T @ X)
print(f"\nquadratic AT^2 coef={beta[2]:.5f} MW/C^2, t={beta[2]/np.sqrt(cov[2,2]):.1f}")
