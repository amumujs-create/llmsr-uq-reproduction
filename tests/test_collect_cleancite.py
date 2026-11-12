"""Evidence-citation exploration stage (analysis/llm_statements/collect_cleancite.py)."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis" / "llm_statements"))
import collect_cleancite as ct  # noqa: E402
import collect_cleanworld as cc  # noqa: E402
from elicit import S2_QUESTION  # noqa: E402
from parse_score import parse  # noqa: E402
from prompts import QUESTION, check_no_leak, describe  # noqa: E402


def test_call_count_and_unique_ids():
    t = ct.tasks("m")
    assert len(t) == ct.N_CALLS == 301
    assert len({x[0] for x in t}) == 301
    assert sum(x[1] == "S2N" for x in t) == 77
    assert sum(x[4] == 0.0 for x in t) == 21          # one temperature-0 call per world and question


def test_descriptions_are_the_collected_ones():
    assert ct.describe_cite("base") == describe("base", "L2")
    for world in cc.CLEAN_WORLDS:
        assert ct.describe_cite(world) == cc.describe_clean(world)
    for world in ct.WORLDS:
        assert check_no_leak(ct.describe_cite(world)) == []


def _strip(q):
    return q.replace(ct.EVIDENCE_FIELD, "").replace(ct.EVIDENCE_NOTE, "")


def test_questions_differ_from_the_originals_only_by_the_evidence_text():
    assert _strip(ct.S0C_QUESTION) == QUESTION
    assert _strip(ct.S2C_QUESTION) == S2_QUESTION
    assert ct.S0C_QUESTION.count('"evidence": "<sentence>" or null') == 1
    assert ct.S2C_QUESTION.count('"evidence": "<sentence>" or null') == 4
    assert check_no_leak(ct.EVIDENCE_NOTE + ct.EVIDENCE_FIELD) == []
    for world in ct.WORLDS:
        assert ct.messages(world, "S0C")[1]["content"].endswith(ct.S0C_QUESTION)
        assert ct.messages(world, "S2C")[1]["content"].endswith(ct.S2C_QUESTION)
        assert ct.messages(world, "S2N")[1]["content"].endswith(ct.S2N_QUESTION)


def test_not_stated_question_differs_only_by_the_added_option():
    q = ct.S2N_QUESTION
    assert q.count(ct.NOT_STATED) == 4 and "evidence" not in q
    assert q.replace(ct.NOT_STATED_NOTE, "").replace(ct.NOT_STATED, "true or false") == S2_QUESTION
    assert check_no_leak(ct.NOT_STATED_NOTE) == []


def test_not_stated_answer_is_not_a_claim():
    obj = {"gt_generator_limit": {"reached_in_range": True},
           "inlet_anti_icing_heating": {"active": "not stated", "below_c": 5},
           "exhaust_or_steam_side_limit": {"active": "true", "above_c": 30}}
    ev = ct.s2c_evidence(obj)
    assert ev["gt_generator_limit"][0] is True
    assert ev["inlet_anti_icing_heating"][0] is False
    assert ev["exhaust_or_steam_side_limit"][0] is False      # only a JSON true counts, as in the scored rule


def test_insert_refuses_ambiguous_anchor():
    with pytest.raises(ValueError):
        ct._insert("a a", "a", "b")


def test_quote_ok():
    d = ct.describe_cite("cap353")
    assert ct.quote_ok("The gas turbine generator is limited to 353 MW.", d)
    assert ct.quote_ok('  "the gas turbine   generator is limited to 353 MW"  ', d)      # case, spaces, quotes
    assert ct.quote_ok("the control system holds it at that limit", d)                   # part of a sentence
    assert not ct.quote_ok("The gas turbine generator is limited to 357 MW.", d)         # wrong number
    assert not ct.quote_ok("The plant has an inlet evaporative cooler.", d)              # not in the description
    assert not ct.quote_ok("353 MW", d)                                                  # too short
    assert not ct.quote_ok(None, d)
    assert not ct.quote_ok(True, d)
    assert not ct.quote_ok("", d)


def test_s0c_evidence_keeps_every_boundary_and_parse_still_works():
    text = json.dumps({"boundaries": [
        {"side": "below", "at_c": 9, "confidence": 0.8, "evidence": "The gas turbine generator is limited to 353 MW."},
        {"side": "above", "at_c": 30, "confidence": 0.3, "evidence": None}], "rationale": "x"})
    assert parse(text)["ok"]
    assert ct.s0c_evidence(text) == [("below", 9.0, "The gas turbine generator is limited to 353 MW."),
                                     ("above", 30.0, None)]
    assert ct.s0c_evidence("not json") is None
    assert ct.s0c_evidence(json.dumps({"boundaries": [], "rationale": "none"})) == []


def test_s2c_evidence():
    obj = {"gt_generator_limit": {"reached_in_range": True, "evidence": "q"},
           "inlet_anti_icing_heating": {"active": True, "below_c": 5, "evidence": None},
           "inlet_evaporative_cooling": {"active": False, "evidence": "The plant has no inlet air conditioning."}}
    ev = ct.s2c_evidence(obj)
    assert ev["gt_generator_limit"] == (True, "q")
    assert ev["inlet_anti_icing_heating"] == (True, None)
    assert ev["inlet_evaporative_cooling"][0] is False
    assert ev["exhaust_or_steam_side_limit"] == (False, None)


def test_check_counts(tmp_path, capsys):
    good0 = json.dumps({"boundaries": [], "rationale": "r"})
    good2 = json.dumps({"gt_generator_limit": {"reached_in_range": False, "evidence": None}})
    recs = [{"strategy": "S0C", "error": None, "response": good0},
            {"strategy": "S0C", "error": None, "response": "garbage"},
            {"strategy": "S2C", "error": None, "response": good2},
            {"strategy": "S2N", "error": None, "response": good2},
            {"strategy": "S2C", "error": "Timeout", "response": None}]
    p = tmp_path / "x.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    assert ct.check(str(p), smoke=False) == 0
    out = capsys.readouterr().out
    assert "5 calls, 1 errors, S0C unparsed 1/2, S2C unparsed 0/1" in out and "S2N unparsed 0/1" in out
    assert ct.check(str(p), smoke=True) == 1          # an error makes the smoke test fail


# --- analysis side (analysis/ccpp_twin/explore/cite_check.py) ---
sys.path.insert(0, str(ROOT / "analysis" / "ccpp_twin"))
sys.path.insert(0, str(ROOT / "analysis" / "ccpp_twin" / "explore"))
import cite_check as ck  # noqa: E402


def _rec(strategy, world, obj, tag="m", temp=0.7):
    return {"tag": tag, "strategy": strategy, "world": world, "temperature": temp, "error": None,
            "response": json.dumps(obj)}


def test_cite_check_sides_and_quote_filter():
    d = ct.describe_cite("cap353")
    good = "The gas turbine generator is limited to 353 MW."
    s0 = json.dumps({"boundaries": [{"side": "below", "at_c": 9, "confidence": 1, "evidence": good},
                                    {"side": "above", "at_c": 30, "confidence": 1, "evidence": "made up sentence here"}],
                     "rationale": "r"})
    assert ck.sides_s0(s0) == {"below", "above"}
    assert ck.sides_s0(s0, d) == {"below"}
    assert ck.sides_s0("garbage") is None
    s2 = json.dumps({"gt_generator_limit": {"reached_in_range": True, "evidence": good},
                     "exhaust_or_steam_side_limit": {"active": True, "above_c": 30, "evidence": None}})
    assert ck.items_s2(s2) == {"gt_generator_limit", "exhaust_or_steam_side_limit"}
    assert ck.items_s2(s2, d) == {"gt_generator_limit"}
    assert ck.items_s2("garbage") is None


def test_cite_check_rows_truth_and_base():
    assert ck.true_side("cap353") == "below" and ck.true_side("evap22") == "above" and ck.true_side("base") is None
    good = "The gas turbine generator is limited to 353 MW."
    recs = [_rec("S2C", "cap353", {"gt_generator_limit": {"reached_in_range": True, "evidence": good},
                                   "inlet_evaporative_cooling": {"active": True, "above_c": 25, "evidence": None}}),
            _rec("S0C", "base", {"boundaries": [{"side": "above", "at_c": 30, "confidence": 1, "evidence": None}],
                                 "rationale": "r"}),
            _rec("S0C", "base", "x")]
    recs[2]["response"] = "garbage"
    after, unparsed, _ = ck.rows(recs, "after")
    got = {(r["world"], r["side"]): (r["truth"], r["claimed"]) for r in after}
    assert got == {("cap353", "below"): ("true", True), ("cap353", "above"): ("false", True),
                   ("base", "below"): ("false", False), ("base", "above"): ("false", True)}
    assert unparsed == {("m", "S0"): 1}
    checked, _, quoted = ck.rows(recs, "checked")
    got = {(r["world"], r["side"]): r["claimed"] for r in checked}
    assert got == {("cap353", "below"): True, ("cap353", "above"): False, ("base", "below"): False,
                   ("base", "above"): False}
    assert quoted == {"gt_generator_limit": 1}


def test_cite_check_load_requires_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        ck.load(str(tmp_path), "cleancite", ["m"])
