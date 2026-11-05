"""Figures for the reading guide (docs/research/how-to-read-hint-assembly.md).

Usage: python guide_figures.py <Folds5x2_pp.xlsx>
Writes docs/research/figures/guide-{1..4}-*.png. Re-run after new server results to refresh figures 3 and 4.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "ccpp_twin"))
import assemble as A  # noqa: E402
from ccpp_twin import load_ccpp_inputs, simulate  # noqa: E402
from grid import build_worlds  # noqa: E402

OUT = HERE.parent.parent / "docs" / "research" / "figures"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, GRID, BAND = "#0b0b0b", "#52514e", "#e4e3df", "#f0efec"
COLD_TRUTH = {"cap_mild": 6.0, "cap_strong": 9.0, "evap+cap_mild": 6.0}

plt.rcParams.update({
    "font.family": ["AppleGothic", "Noto Sans CJK KR", "NanumGothic", "DejaVu Sans"],
    "axes.unicode_minus": False, "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "figure.facecolor": "white",
    "axes.facecolor": "white", "savefig.dpi": 160, "savefig.bbox": "tight", "lines.linewidth": 2,
})


def fig_worlds(x):
    """1. What the five twin worlds look like: binned mean clean output vs ambient temperature, one panel each."""
    worlds, _ = build_worlds(x)
    edges = np.arange(2, 37.5, 1.0)
    mid = (edges[:-1] + edges[1:]) / 2
    idx = np.digitize(x["AT"], edges) - 1
    curves = {}
    for w in worlds:
        if w["vmode"] == "realV" and w["module"] not in curves:
            pe = simulate(x["AT"], x["AP"], x["RH"], x["V"], w["cfg"])["pe_clean"]
            curves[w["module"]] = np.array([pe[idx == i].mean() if (idx == i).sum() > 20 else np.nan
                                            for i in range(len(mid))])
    marks = {"base": [], "evap": [24], "cap_mild": [6], "cap_strong": [9], "evap+cap_mild": [6, 24]}
    lo, hi = np.quantile(x["AT"], [0.005, 0.995])
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.6), sharey=True)
    for ax, (mod, ts) in zip(axes, marks.items()):
        ax.axvspan(hi - 10, hi, color=BAND, zorder=0)
        if mod != "base":
            ax.plot(mid, curves["base"], color=INK2, lw=1.2, ls=(0, (2, 2)), label="base(경계 없음)")
        ax.plot(mid, curves[mod], color=BLUE, label="이 세계")
        for t in ts:
            ax.axvline(t, color=ORANGE, lw=1.2)
            ax.text(t + 0.6, 0.06, f"{t}°C", transform=ax.get_xaxis_transform(), color=INK, fontsize=9)
        ax.set_title(mod, loc="left", color=INK)
        ax.set_xlabel("외기 온도 AT (°C)")
    axes[0].set_ylabel("순출력, 잡음 없음 (MW)")
    axes[0].text(hi - 5, 0.97, "학습 창 예", transform=axes[0].get_xaxis_transform(), ha="center", va="top",
                 color=INK2, fontsize=8.5)
    h, lab = axes[1].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, ncol=2, loc="upper right", bbox_to_anchor=(0.99, 1.02))
    fig.suptitle("그림 1. 쌍둥이 세계 다섯 개: 주황 선(경계)에서 기울기가 꺾인다. 회색 띠는 학습 창 예(hot→cold, 폭 10°C)",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "guide-1-worlds.png")
    plt.close(fig)


def fig_formula():
    """2. How we turn S2 sub-quantities into a cold boundary, and why a wrong sensitivity breaks it."""
    t = np.linspace(-15, 37, 200)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.axvspan(2, 37, color=BAND, zorder=0)
    ax.text(19.5, 0.97, "자료의 기온 범위 2–37°C", transform=ax.get_xaxis_transform(), ha="center", va="top",
            color=INK2, fontsize=8.5)
    ax.axhline(364, color=INK, lw=1.2)
    ax.text(36.5, 365.5, "발전기 한계 364 MW", ha="right", color=INK, fontsize=9)
    for sens, col, who in ((-2.2, BLUE, "감도 −2.2 MW/°C (실제에 가까움)"), (-0.8, ORANGE, "감도 −0.8 MW/°C (gemma 답)")):
        ax.plot(t, 343 + sens * (t - 15), color=col, label=who)
        at = 15 - (364 - 343) / abs(sens)
        ax.plot([at], [364], "o", ms=8, color=col, mec="white", mew=2, zorder=5)
        ax.annotate(f"경계 {at:.1f}°C", (at, 364), xytext=(at + 1, 376), color=INK, fontsize=9,
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.plot([15], [343], "o", ms=8, color=INK, mec="white", mew=2, zorder=5)
    ax.annotate("15°C에서 343 MW\n(L2 프롬프트가 주는 값)", (15, 343), xytext=(18, 352), fontsize=9, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.set_xlim(-15, 37)
    ax.set_ylim(300, 385)
    ax.set_xlabel("외기 온도 (°C)")
    ax.set_ylabel("가스터빈 출력 (MW)")
    ax.set_title("그림 2. 저온 경계 = 15 − (한계 − 15°C 출력) / |감도|", loc="left", color=INK)
    ax.legend(frameon=False, loc="lower left")
    fig.savefig(OUT / "guide-2-formula.png")
    plt.close(fig)


def s2_points(root):
    """Every S2 answer in a cap world that claims the GT limit: (tag, level, sens %, boundary error)."""
    pts = []
    for path in sorted((root / "strategy").glob("*.jsonl")):
        for line in open(path):
            r = json.loads(line)
            if r.get("strategy") != "S2" or r.get("world") not in COLD_TRUTH or r.get("level") not in ("L1", "L2"):
                continue
            g = (A._json(r["response"]) or {}).get("gt_generator_limit") or {}
            if g.get("reached_in_range") is not True:
                pts.append((r["tag"], r["level"], None, None))
                continue
            p15 = A.L2_GT_15C_MW if r["level"] == "L2" else A._num(g.get("gt_mw_at_15c"))
            lim, sens = A._num(g.get("limit_mw")), A._num(g.get("gt_mw_per_degc"))
            if None in (p15, lim, sens) or sens == 0 or not p15:
                pts.append((r["tag"], r["level"], None, None))
                continue
            at = 15 - (lim - p15) / abs(sens)
            pts.append((r["tag"], r["level"], abs(sens) / p15 * 100, at - COLD_TRUTH[r["world"]]))
    return pts


def fig_gate(root):
    """3. The gate: sensitivity plausibility (x) against boundary error (y), one panel per model."""
    pts = s2_points(root)
    tags = sorted({p[0] for p in pts}, key=lambda s: (A.split_of(s, True) != "validation", s))
    n = len(tags)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.3 * rows), sharex=True, sharey=True, squeeze=False)
    for ax, tag in zip(axes.flat, tags):
        ax.axvspan(*A.SENS_GATE_PCT, color=BAND, zorder=0)
        ax.axhspan(-3, 3, color="#dcebfa", zorder=0)
        mine = [p for p in pts if p[0] == tag]
        for lvl, col, mk in (("L1", ORANGE, "o"), ("L2", BLUE, "s")):
            q = [(a, b) for t_, l_, a, b in mine if l_ == lvl and a is not None]
            if q:
                xs, ys = np.array(q).T
                ax.scatter(np.clip(xs, 0, 1.6), np.clip(ys, -30, 30), s=34, marker=mk, color=col, alpha=0.75,
                           edgecolor="white", linewidth=0.8, label=lvl, zorder=3)
        none = sum(1 for p in mine if p[2] is None)
        split = "검증" if A.split_of(tag, True) == "validation" else "개발"
        ax.set_title(f"{tag} ({split})\n상한 주장 없음 {none}/{len(mine)}", fontsize=9, loc="left", color=INK)
        ax.set_xlim(0, 1.6)
        ax.set_ylim(-31, 31)
    for ax in axes.flat[n:]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("|감도| / 15°C 출력 (%/°C)")
    for ax in axes[:, 0]:
        ax.set_ylabel("계산한 경계 − 정답 (°C)")
    h, lab = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, ncol=2, loc="upper right", bbox_to_anchor=(0.99, 1.0))
    fig.suptitle("그림 3. 관문: 회색 세로 띠(문헌 0.4–1.2 %/°C) 안의 점만 통과, 파란 가로 띠(±3°C)가 정답",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "guide-3-gate.png")
    plt.close(fig)


def fig_rmse(root):
    """4. Result per cell: RMSE of HA against the comparators, with R0 and ORACLE as references."""
    d = json.load(open(root / "assembly" / "assembly.json"))
    cells = [(k.split("|", 1)[0], k.split("|", 1)[1], v) for k, v in d["rmse"].items() if not k.startswith("-")]
    cells.sort(key=lambda c: (c[0] != "validation", c[1]))
    fig, ax = plt.subplots(figsize=(8, 0.36 * len(cells) + 1.6))
    y = np.arange(len(cells))[::-1]
    r0, oracle = 7.99, d["oracle_rmse"]
    ax.axvline(oracle, color=AQUA, lw=1.5)
    ax.text(oracle, 1.0, " ORACLE(정답 경계)", transform=ax.get_xaxis_transform(), color=INK2, fontsize=8.5,
            va="bottom")
    for yy, (_, _, v) in zip(y, cells):
        r0 = v["R0"]
        ax.plot([min(v["HA"], v["S0maj"], v["AGR"]), max(v["HA"], v["S0maj"], v["AGR"])], [yy, yy], color=GRID, lw=2,
                zorder=1)
    ax.axvline(r0, color=INK, lw=1.5)
    ax.text(r0, 1.0, " R0(데이터만)", transform=ax.get_xaxis_transform(), color=INK2, fontsize=8.5, va="bottom")
    for key, col, mk, lab in (("AGR", YELLOW, "^", "AGR 일치도 가중"), ("S0maj", ORANGE, "o", "S0maj 직접 묻기"),
                              ("HA", BLUE, "s", "HA 조립 규칙")):
        ax.scatter([c[2][key] for c in cells], y, s=60, marker=mk, color=col, edgecolor="white", linewidth=1.5,
                   label=lab, zorder=3)
    ax.set_yticks(y, [f"{c[1]}" for c in cells], fontsize=8.5)
    nval = sum(1 for c in cells if c[0] == "validation")
    ax.axhline(y[nval - 1] - 0.5, color=INK2, lw=0.8)
    ax.text(1.01, (y[0] + y[nval - 1]) / 2, "검증", transform=ax.get_yaxis_transform(), rotation=-90, va="center",
            color=INK2)
    ax.text(1.01, (y[nval] + y[-1]) / 2, "개발", transform=ax.get_yaxis_transform(), rotation=-90, va="center",
            color=INK2)
    ax.set_xlabel("다섯 세계 평균 외삽 RMSE (MW, 왼쪽일수록 좋음)")
    ax.set_title("그림 4. 칸별 결과: 파란 네모가 검은 선(R0)의 왼쪽이면 이득, 오른쪽이면 해", loc="left", color=INK, pad=22)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.07 - 1.2 / len(cells)))
    ax.grid(axis="y", visible=False)
    fig.savefig(OUT / "guide-4-rmse.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    root = HERE / "runs"
    fig_worlds(load_ccpp_inputs(a.xlsx))
    fig_formula()
    fig_gate(root)
    fig_rmse(root)
    print("wrote", *sorted(p.name for p in OUT.glob("guide-*.png")))


if __name__ == "__main__":
    main()
