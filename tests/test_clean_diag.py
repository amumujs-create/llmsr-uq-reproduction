"""Clean-window diagnostic confirmation (analysis/ccpp_twin/clean_diag.py). No collected answer is read here."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis" / "ccpp_twin"))
sys.path.insert(0, str(ROOT / "analysis" / "llm_statements"))
import clean_diag as cd  # noqa: E402
import collect_cleanworld as cc  # noqa: E402

S0_YES = '{"boundaries": [{"side": "below", "at_c": 11, "confidence": 0.8}], "rationale": "x"}'
S0_NO = '{"boundaries": [], "rationale": "x"}'
S2_CAP = json.dumps({"gt_generator_limit": {"reached_in_range": True, "gt_mw_at_15c": 343, "limit_mw": 353,
                                            "gt_mw_per_degc": -2.2},
                     "inlet_anti_icing_heating": {"active": False, "below_c": None},
                     "inlet_evaporative_cooling": {"active": True, "above_c": 25},
                     "exhaust_or_steam_side_limit": {"active": False, "above_c": None}})


def rec(tag, world, strategy, response, temp=0.7, error=None):
    return {"tag": tag, "world": world, "level": "L2", "strategy": strategy, "temperature": temp, "error": error,
            "response": response}


def write(tmp_path, tag, recs):
    d = tmp_path / "cleanworld"
    d.mkdir(exist_ok=True)
    (d / f"{tag}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))


def test_tasks_cover_the_collected_plants():
    assert set(cd.PLANTS) == set(cc.CLEAN_WORLDS)
    assert len(cd.TASKS) == 12 and len(cd.KINK) == 6 and len(cd.NOKINK) == 6
    for w, (side, k, pos, k_opp) in cd.PLANTS.items():
        assert cd.TASKS[f"{w}|{side}"] == (w, side, k, "kink")
        assert cd.TASKS[f"{w}|{cd.OPP[side]}"] == (w, cd.OPP[side], k_opp, "nokink")
        assert cd.ORACLE[w] == [(cd.WORD[side], pos)]


def test_seeds_differ_between_development_and_confirmation():
    assert cd.SEEDS["dev"] != cd.SEEDS["confirm"]


def test_judges():
    assert cd.judge("S0", 0.6, 0.0) and not cd.judge("S2", 0.6, 0.0)
    assert not cd.judge("AND", 0.6, 0.5) and cd.judge("OR", 0.6, 0.5)      # a tie is not a majority
    assert cd.judge("AND", 0.51, 0.51)


def test_load_and_shares(tmp_path):
    recs = []
    for w in cd.PLANTS:
        recs += [rec("m", w, "S0", S0_YES), rec("m", w, "S0", S0_NO), rec("m", w, "S0", S0_YES),
                 rec("m", w, "S0", S0_YES, temp=0), rec("m", w, "S0", None, error="boom"),
                 rec("m", w, "S0", "not json"), rec("m", w, "S2", S2_CAP), rec("m", w, "S2", S2_CAP, temp=0)]
    write(tmp_path, "m", recs)
    s0, s2, raw, fail, lines = cd.load_clean(tmp_path, ["m"])
    assert lines["m"] == len(recs)
    # per plant: 4 usable S0 answers of which 1 unparsed, 1 usable S2; temperature-0 and error records are skipped
    assert fail["m"] == {"S0": [6, 24], "S2": [0, 6]}
    assert len(s0[("m", "cap353")]) == 3 and len(raw[("m", "cap353")]) == 1
    sh = cd.shares(s0, raw, ["m"])
    assert sh[("m", "cap353", "below")] == (pytest.approx(2 / 3), 1.0)      # generator limit acts on the cold side
    assert sh[("m", "cap353", "above")] == (0.0, 1.0)                       # evaporative cooling claimed
    frz = cd.frozen(s0, s2, raw, ["m"])
    side, at = frz[("m", "cap353")][0]
    assert side == "below" and at == pytest.approx(15 - 10 / 2.2)


def test_shares_refuse_missing_answers(tmp_path):
    write(tmp_path, "m", [rec("m", "cap353", "S0", S0_YES)])
    s0, _, raw, _, _ = cd.load_clean(tmp_path, ["m"])
    with pytest.raises(RuntimeError):
        cd.shares(s0, raw, ["m"])


def frame(values):
    """One row per task and replicate; values maps column -> (kink value, no-kink value)."""
    rows = []
    for t, (_, _, _, kind) in cd.TASKS.items():
        for _ in range(4):
            r = {"task": t, "kind": kind, "R0_abs": 5.0, "HINGE_kept": False, "bound_ok": True}
            r.update({c: (a if kind == "kink" else b) for c, (a, b) in values.items()})
            rows.append(r)
    return pd.DataFrame(rows)


def test_verdicts_on_a_constructed_table():
    tags = ["good", "bad"]
    sh = {}
    for t in tags:
        for w, (side, *_) in cd.PLANTS.items():
            sh[(t, w, cd.WORD[side])] = (1.0, 1.0)
            sh[(t, w, cd.WORD[cd.OPP[side]])] = (0.0, 1.0 if t == "bad" else 0.0)
    vals = {"ORACLE": (0.6, 1.0), "READ": (0.8, 1.0), "EDGE_h": (0.4, 6.0), "HINGE": (1.0, 1.0), "QUAD": (1.5, 6.0)}
    for t in tags:
        unsafe = 6.0 if t == "bad" else 1.0
        vals.update({f"{t}|FROZEN": (0.9, 1.0), f"{t}|S0": (0.4, 1.0), f"{t}|S2": (0.4, unsafe),
                     f"{t}|AND": (0.4, 1.0), f"{t}|OR": (0.4, unsafe), f"{t}|CONF": (0.5, 1.0)})
    v = cd.verdicts(frame(vals), tags, sh)
    assert v["C1"]["pass"] and v["C2"]["pass"] and v["C3"]["pass"]
    assert v["C4"]["cells"] == 12 and v["C5"]["cells"] == 6
    assert v["C6"]["models"] == 1                               # only "bad" is unsafe under the S2-only rule
    assert v["C7"]["safe_models"] == 2 and v["C7"]["small_gain_models"] == 2
    assert v["P1"]["models"] == 2 and v["P2"]["models"] == 2 and v["P3"]["models"] == 2
    assert v["per_model"]["bad"]["S2"]["safe"] is False and v["per_model"]["good"]["S2"]["safe"] is True


def test_hinge_fit_keeps_a_clear_hinge_and_drops_none():
    rng = np.random.default_rng(0)
    at = rng.uniform(10, 20, 400)
    X = np.column_stack([at, rng.normal(size=(400, 3))])
    y_lin = 480 - 2.0 * at + rng.normal(0, 0.5, 400)
    y_bend = 480 - 2.0 * at + 2.0 * np.maximum(0, at - 15) + rng.normal(0, 0.5, 400)
    Xte = np.column_stack([np.array([22.0, 24.0]), np.zeros((2, 3))])
    assert cd.hinge_fit(X, y_lin, Xte, "hot")[1] is False
    pred, kept = cd.hinge_fit(X, y_bend, Xte, "hot")
    assert kept is True and abs(pred[1] - pred[0]) < 1.0        # nearly flat beyond the hinge
