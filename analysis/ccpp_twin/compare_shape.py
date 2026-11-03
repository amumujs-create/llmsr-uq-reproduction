"""Compare the twin's output-vs-AT shape with real CCPP on the real inputs.

Prints binned mean PE and local slopes for the real data and for each module
combination, plus a JSON blob for plotting. Real PE is read only here, for the
shape comparison; it is not used to set any simulator parameter.
"""
import json
import sys
from pathlib import Path

import numpy as np
import openpyxl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ccpp_twin import TwinConfig, load_ccpp_inputs, simulate, with_modules

path = sys.argv[1]
x = load_ccpp_inputs(path)
rows = list(openpyxl.load_workbook(path, read_only=True, data_only=True)["Sheet1"].iter_rows(values_only=True))
header = [str(h).strip() for h in rows[0]]
pe_real = np.array([float(r[header.index("PE")]) for r in rows[1:] if r[0] is not None])
at = x["AT"]

configs = {
    "base": TwinConfig(),
    "cold_cap": with_modules(cold_cap=True),
    "evap": with_modules(evap_cooling=True),
    "both": with_modules(cold_cap=True, evap_cooling=True),
}
curves = {"real": pe_real}
for name, cfg in configs.items():
    curves[name] = simulate(x["AT"], x["AP"], x["RH"], x["V"], cfg)["pe_clean"]

edges = np.arange(2.0, 37.0, 1.0)
centers, binned = [], {k: [] for k in curves}
for lo in edges:
    m = (at >= lo) & (at < lo + 1)
    if m.sum() < 30:
        continue
    centers.append(lo + 0.5)
    for k, y in curves.items():
        binned[k].append(round(float(y[m].mean()), 1))

def local_slopes(y):
    out = []
    for lo in np.arange(4.0, 34.0, 2.0):
        m = (at >= lo) & (at < lo + 3)
        out.append(round(float(np.polyfit(at[m], y[m], 1)[0]), 2))
    return out

slope_x = [round(lo + 1.5, 1) for lo in np.arange(4.0, 34.0, 2.0)]
slopes = {k: local_slopes(y) for k, y in curves.items()}
print("local slope dPE/dAT by 3C window center:", slope_x)
for k in curves:
    rmse = np.sqrt(np.mean((curves[k] - pe_real) ** 2))
    print(f"{k:8s} slopes={slopes[k]}  mean PE={curves[k].mean():.1f}  RMSE vs real={rmse:.2f}")
json.dump({"centers": centers, "binned": binned, "slope_x": slope_x, "slopes": slopes},
          open(Path(__file__).resolve().parent / "shape_comparison.json", "w"))
