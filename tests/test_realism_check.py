"""Twin realism check (analysis/ccpp_twin/realism_check.py). Synthetic inputs only; no CCPP file is read."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis" / "ccpp_twin"))
import realism_check as rc  # noqa: E402


def _inputs(n=6000, seed=0):
    r = np.random.default_rng(seed)
    return {"AT": r.uniform(2.0, 36.0, n), "V": r.uniform(30.0, 70.0, n), "AP": r.uniform(1000.0, 1030.0, n),
            "RH": r.uniform(30.0, 100.0, n)}


def _cells(x, y, side):
    df = pd.DataFrame(rc.score(x, y, "w", side, reps=3))
    return df, df.groupby(["delta", "w"])["EDGE@0.5"].mean()


def test_linear_output_is_no_kink_like_on_both_sides():
    x = _inputs()
    y = 500.0 - 2.0 * x["AT"] + np.random.default_rng(1).normal(0.0, 1.0, len(x["AT"]))
    for side in rc.SIDES:
        df, cells = _cells(x, y, side)
        assert len(df) == rc.N_CELLS * len(rc.NS) * 3
        assert rc.classify(cells) == "no-kink-like"
        assert (df["EDGE@1.0"] > df["EDGE@0.5"]).mean() > 0.9        # taking a wrong statement in full is worse


def test_output_flat_beyond_k_is_kink_like():
    x = _inputs()
    noise = np.random.default_rng(2).normal(0.0, 1.0, len(x["AT"]))
    hot = 500.0 - 2.0 * np.minimum(x["AT"], 28.0) + noise
    cold = 500.0 - 2.0 * np.maximum(x["AT"], 8.0) + noise          # 8 degC is in the cold position sweep
    for side, y, k in (("hot", hot, 28.0), ("cold", cold, 8.0)):
        df, cells = _cells(x, y, side)
        assert rc.classify(cells) == "kink-like"
        assert df[f"POS_{k:g}@1.0"].mean() < 0.5                      # the true position, in full, is near the floor
        assert df[f"POS_{k:g}@1.0"].mean() < df["EDGE@1.0"].mean()


def test_classify_thresholds():
    assert rc.classify(pd.Series([0.9, 0.9, 0.9, 0.9, 0.9, 1.2])) == "kink-like"
    assert rc.classify(pd.Series([1.2, 1.2, 1.2, 1.2, 1.2, 0.9])) == "no-kink-like"
    assert rc.classify(pd.Series([0.9, 0.9, 0.9, 1.2, 1.2, 1.2])) == "mixed"


def test_partial_slopes_recover_a_three_segment_curve():
    x = _inputs()
    at = x["AT"]
    y = (500.0 - 1.0 * np.minimum(at, 8.0) - 2.0 * (np.clip(at, 8.0, 27.0) - 8.0) - 0.5 * np.maximum(at - 27.0, 0.0)
         + 0.3 * x["V"])
    s = rc.partial_slopes(x, y)
    assert s == pytest.approx({"below": -1.0, "middle": -2.0, "above": -0.5}, abs=1e-6)


def test_too_few_test_rows_raise():
    x = _inputs(n=200)
    with pytest.raises(RuntimeError):
        rc.score(x, 500.0 - 2.0 * x["AT"], "w", "hot", reps=1)
