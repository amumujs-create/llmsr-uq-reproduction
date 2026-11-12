"""Indirect-specification exploration (analysis/llm_statements/collect_spec.py, analysis/ccpp_twin/explore/spec_check.py)."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis" / "llm_statements"))
sys.path.insert(0, str(ROOT / "analysis" / "ccpp_twin"))
sys.path.insert(0, str(ROOT / "analysis" / "ccpp_twin" / "explore"))
import collect_cleancite as ct  # noqa: E402
import collect_spec as sp  # noqa: E402
import spec_check as sc  # noqa: E402
from prompts import L0, QUESTION, check_no_leak  # noqa: E402


def test_call_count_and_unique_ids():
    t = sp.tasks("m")
    assert len(t) == sp.N_CALLS == 608
    assert len({x[0] for x in t}) == 608
    assert sum(x[3] == "T0" for x in t) == 32 and sum(x[3] == "T2" for x in t) == 288
    assert sum(x[5] == 0.0 for x in t) == 2 * 19          # one temperature-0 call per description and question


def test_plants_and_limits():
    assert set(sp.KINK) | set(sp.NO_KINK) == set(sp.PLANTS) and not set(sp.KINK) & set(sp.NO_KINK)
    assert sp.limit_mw("g415") == pytest.approx(352.75) and sp.stated_mw("g415") == 353
    assert sp.limit_mw("n450") == pytest.approx(382.5) and sp.stated_mw("n450") == 383      # half rounds up
    assert sp.limit_mw("g415") == pytest.approx(sp.limit_mw("g392p"), abs=0.1)              # same-limit pair


def test_questions_are_the_existing_ones_unchanged():
    assert sp.QUESTIONS == {"S0": QUESTION, "S2N": ct.S2N_QUESTION}
    for kind in sp.QUESTIONS:
        assert sp.messages("g415", "T2", kind)[1]["content"].endswith(sp.QUESTIONS[kind])


def test_t2_descriptions_differ_only_by_the_two_numbers():
    texts = set()
    for plant, (mva, pf) in sp.PLANTS.items():
        d = sp.describe_spec(plant, "T2")
        assert check_no_leak(d) == []
        low = d.lower()
        for word in ("limit", "reach", "hold", "cold weather", "stays below"):       # no behaviour sentence
            assert word not in low
        assert str(sp.stated_mw(plant)) + " MW" not in d                          # the limit in MW is not given
        rating = sp.RATING.format(mva=mva, pf=pf)
        assert d.endswith(rating) and f"{pf:.2f}" in rating
        texts.add(d.replace(rating, "<rating>"))
    assert len(texts) == 1


def test_t0_and_t3_descriptions():
    assert sp.describe_spec("none", "T0") == L0
    with pytest.raises(ValueError):
        sp.describe_spec("g415", "T0")
    k, n = sp.describe_spec("g415", "T3"), sp.describe_spec("n450", "T3")
    assert "holds it at that limit" in k and k.endswith("limited to 353 MW.")
    assert "stays below the rated limit" in n and n.endswith("limited to 383 MW.")
    assert "MVA" not in k and "power factor" not in k
    for plant in sp.PLANTS:
        assert check_no_leak(sp.describe_spec(plant, "T3")) == []


def test_check_counts(tmp_path, capsys):
    recs = [{"strategy": "S0", "error": None, "response": json.dumps({"boundaries": [], "rationale": "r"})},
            {"strategy": "S0", "error": None, "response": "garbage"},
            {"strategy": "S2N", "error": None, "response": json.dumps({"gt_generator_limit": {}})},
            {"strategy": "S2N", "error": "Timeout", "response": None}]
    p = tmp_path / "x.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    assert sp.check(str(p), smoke=False) == 0
    assert "4 calls, 1 errors, S0 unparsed 1/2, S2N unparsed 0/1" in capsys.readouterr().out
    assert sp.check(str(p), smoke=True) == 1


# --- analysis side ---
DESIGN_KINK = {"g415": 10.5, "g407": 13.6, "g392p": 10.5, "g400p": 7.3, "g400": 16.3, "g425": 6.8}


def test_kink_positions_match_the_design_table():
    for plant in sp.PLANTS:
        k = sc.kink_position(sp.limit_mw(plant))
        if plant in sp.NO_KINK:
            assert k is None
        else:
            assert k == pytest.approx(DESIGN_KINK[plant], abs=0.06)


def test_twin_config_puts_the_kink_where_kink_position_says():
    import numpy as np
    import ccpp_twin as tw
    k = sc.kink_position(sp.limit_mw("g415"))
    at = np.array([k - 4.0, k - 2.0, k + 2.0, k + 4.0])
    args = (at, np.full(4, sc.MEDIAN_AP), np.full(4, sc.MEDIAN_RH), np.full(4, sc.MEDIAN_V))
    gt = tw.simulate(*args, sc.twin_config("g415"))["p_gt"]
    assert abs(gt[0] - gt[1]) < 0.1 and gt[0] == pytest.approx(352.75, abs=0.1)      # held at the limit when colder
    assert gt[2] - gt[3] > 3.0                                                         # falls with temperature when warmer


def _s2n(reached, limit=None, slope=-2.2, hot=False, ice=False):
    return json.dumps({"gt_generator_limit": {"reached_in_range": reached, "gt_mw_at_15c": 343, "limit_mw": limit,
                                              "gt_mw_per_degc": slope},
                       "inlet_anti_icing_heating": {"active": ice, "below_c": 5},
                       "inlet_evaporative_cooling": {"active": "not stated", "above_c": None},
                       "exhaust_or_steam_side_limit": {"active": hot, "above_c": 30}})


def test_read_s2n():
    a = sc.read_s2n(_s2n(True, 352.75))
    assert a["cold"] and a["gen"] and not a["hot"] and a["not_stated"] is False
    assert a["position"] == pytest.approx(15 - 9.75 / 2.2) and a["limit"] == 352.75
    b = sc.read_s2n(_s2n("not stated", 353.0, hot=True))
    assert not b["cold"] and not b["gen"] and b["hot"] and b["not_stated"] is True and b["position"] is None
    c = sc.read_s2n(_s2n(False, None, ice=True))
    assert c["cold"] and not c["gen"] and c["limit"] is None       # anti-icing is a cold-side claim, not a generator one
    assert sc.read_s2n("garbage") is None


def test_read_s0():
    text = json.dumps({"boundaries": [{"side": "below", "at_c": 6, "confidence": 1},
                                      {"side": "below", "at_c": 10, "confidence": 1},
                                      {"side": "above", "at_c": 30, "confidence": 1}], "rationale": "r"})
    a = sc.read_s0(text)
    assert a["cold"] and a["hot"] and a["position"] == 10.0
    assert sc.read_s0(json.dumps({"boundaries": [], "rationale": "r"}))["cold"] is False
    assert sc.read_s0("garbage") is None


def _records(answer, tag="m", n=5):
    """Synthetic T2 / S2N answers: answer(plant) -> response text."""
    return [{"tag": tag, "level": "T2", "strategy": "S2N", "world": p, "temperature": 0.7, "error": None,
             "response": answer(p)} for p in sp.PLANTS for _ in range(n)]


def _calc(plant):                       # multiplies, compares with the range, answers accordingly
    lim = sp.limit_mw(plant)
    return _s2n(plant in sp.KINK, lim)


def _word(plant):                       # "a generator is mentioned, so its limit is reached"; same number every time
    return _s2n(True, 353.0)


def _silent(plant):
    return _s2n("not stated", None)


def test_verdict_labels():
    def label(*answers):
        recs = [r for i, a in enumerate(answers) for r in _records(a, tag=f"m{i}")]
        df, unparsed, _ = sc.rows(recs)
        assert not unparsed
        return sc.verdict(df)
    lab, mt, pooled = label(_calc, _calc, _calc)
    assert lab == "PASS" and mt["passes"].all()
    assert pooled["kink_claim"] == 1.0 and pooled["nokink_claim"] == 0.0 and pooled["pos_within"] == 1.0
    assert label(_word, _word, _word)[0] == "FOLD: claims without calculation"
    assert label(_silent, _silent, _silent)[0] == "FOLD: no claim"
    assert label(_calc, _calc, _silent, _silent, _silent)[0] == "FOLD: no claim"      # pooled kink claim 0.4 < 0.5
    lab, mt, _ = label(_calc, _calc, _word, _word, _word)
    assert int(mt["passes"].sum()) == 2 and lab != "PASS"                          # two passing models are not enough


def test_in_between_needs_neither_fold_condition():
    recs = _records(_calc, "m0") + _records(_calc, "m1") + _records(_calc, "m2") + _records(_silent, "m3")
    df, _, _ = sc.rows(recs)
    lab, mt, pooled = sc.verdict(df)
    assert pooled["kink_claim"] == pytest.approx(0.75) and lab == "IN BETWEEN" and int(mt["passes"].sum()) == 3


def test_limit_and_pairs():
    df, _, truth = sc.rows(_records(_calc) + _records(_word, tag="w"))
    ok = df.groupby("tag")["limit_ok"].apply(lambda s: s.astype(float).mean())
    assert ok["m"] == 1.0 and ok["w"] == pytest.approx(2 / 9)          # 353 is right only for g415 and g392p
    pt = sc.pair_table(df, truth)
    assert pt.loc["m", ["move_ok", "vanish_ok", "same_ok"]].all()
    assert pt.loc["m", "move_c"] == pytest.approx(20 / 2.2)
    assert not pt.loc["w", "move_ok"] and not pt.loc["w", "vanish_ok"] and pt.loc["w", "same_ok"]


def test_missing_limit_counts_as_not_computed_and_temperature_zero_is_skipped(tmp_path):
    df, _, _ = sc.rows(_records(lambda p: _s2n(True, None), n=1))
    assert not df["limit_ok"].any() and df["position"].isna().all()
    (tmp_path / "spec").mkdir()
    recs = _records(_calc, n=1)
    recs[0]["temperature"] = 0
    recs[1]["error"] = "Timeout"
    (tmp_path / "spec" / "m.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    assert len(sc.load(str(tmp_path), ["m"])) == len(recs) - 2
    with pytest.raises(FileNotFoundError):
        sc.load(str(tmp_path), ["absent"])


def test_main_runs_on_synthetic_answers(tmp_path, monkeypatch, capsys):
    runs = tmp_path / "runs"
    (runs / "spec").mkdir(parents=True)
    models = tmp_path / "models.tsv"
    models.write_text("m\tx\t-\t-\n")
    s0 = json.dumps({"boundaries": [{"side": "below", "at_c": 10, "confidence": 1}], "rationale": "r"})
    recs = []
    for tier in ("T2", "T3"):
        for r in _records(_calc, n=2):
            recs.append({**r, "level": tier})
            recs.append({**r, "level": tier, "strategy": "S0", "response": s0})
    for kind, resp in (("S0", s0), ("S2N", _s2n(False, None))):
        recs.append({"tag": "m", "level": "T0", "strategy": kind, "world": "none", "temperature": 0.7, "error": None,
                     "response": resp})
    (runs / "spec" / "m.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    monkeypatch.setattr(sys, "argv", ["spec_check.py", "--runs", str(runs), "--models", str(models),
                                      "--out", str(tmp_path / "out")])
    sc.main()
    out = capsys.readouterr().out
    assert "§8 reading (tier T2, question S2N): IN BETWEEN" in out          # one model cannot reach three passes
    assert (tmp_path / "out" / "summary.txt").exists() and (tmp_path / "out" / "answers.csv.gz").exists()
