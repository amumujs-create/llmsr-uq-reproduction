"""Tool-hose LLM collection: prompts, collector, scoring (docs/research/robot-twin-hose-llm-preregistration.md)."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis" / "llm_statements"))
sys.path.insert(0, str(ROOT / "analysis" / "robot_twin"))
sys.path.insert(0, str(ROOT / "tests"))
import collect_hose as ch  # noqa: E402
import prompts_hose as ph  # noqa: E402
import score_hose as sh  # noqa: E402


def test_descriptions_do_not_state_the_rule():
    ph.check_no_leak()
    assert "54.5 degrees" in ph.describe("neutral", "L2")       # not the textbook 54.7 (decision 2026-11-12)
    assert all("54.7" not in ph.describe(w, lv) for w in ph.WORLDS for lv in ph.LEVELS)


def test_l0_l1_do_not_depend_on_world_and_l2_adds_only_the_angle():
    for level in ("L0", "L1"):
        assert len({ph.describe(w, level) for w in ph.WORLDS}) == 1
    base = ph.describe("shrink4", "L1")
    for w, (angle, _) in ph.WORLDS.items():
        assert ph.describe(w, "L2") == f"{base} The steel wire braid is laid at {angle:g} degrees to the hose axis."


def test_call_count_and_unique_ids():
    t = ch.tasks("m")
    assert len(t) == ch.N_CALLS == 126
    assert len({x[0] for x in t}) == 126
    assert sum(x[4] == 0.0 for x in t) == 6


def test_parse():
    ok = sh.parse('```json\n{"length_change_pct": -3, "direction": "shorter", "j6_range": "narrower", "confidence": 0.7}\n```')
    assert ok["ok"] and ok["pct"] == -3 and not ok["dir_mismatch"] and not ok["range_mismatch"]
    bad = sh.parse('{"length_change_pct": -3, "direction": "longer", "j6_range": "narrower"}')
    assert bad["ok"] and bad["dir_mismatch"]
    rng = sh.parse('{"length_change_pct": 2, "direction": "longer", "j6_range": "narrower"}')
    assert rng["range_mismatch"]
    assert not sh.parse('{"direction": "shorter"}')["ok"]
    assert not sh.parse('{"length_change_pct": "about two"}')["ok"]
    assert not sh.parse("no json here")["ok"]
    assert sh.parse('<think>{"length_change_pct": 9}</think>{"length_change_pct": -1}')["pct"] == -1


def test_structure_known_answers():
    assert sh.structure(-3.0, "shrink4") == {"correct": True, "strict_wrong": False, "mag_err": 1.0}
    assert sh.structure(-3.0, "grow2")["strict_wrong"]
    assert sh.structure(0.3, "neutral")["correct"]
    n = sh.structure(1.0, "neutral")
    assert not n["correct"] and not n["strict_wrong"]
    assert sh.structure(-3.0, "neutral")["strict_wrong"]


def test_eps_clip_round_and_statement_sets():
    assert sh.to_eps(-12.0) == -0.08 and sh.to_eps(9.0) == 0.04
    assert sh.to_eps(-2.1) == -0.02 and sh.to_eps(-2.2) == -0.0225     # nearest 0.25 percentage point
    recs = [{"tag": "m", "level": "L0", "world": "any", "temperature": 0.7, "sample": 0, "error": None,
             "parsed": sh.parse('{"length_change_pct": -2}')},
            {"tag": "m", "level": "L2", "world": "grow2", "temperature": 0.0, "sample": 20, "error": None,
             "parsed": sh.parse('{"length_change_pct": 1.9}')}]
    sets, clipped = sh.statement_sets(recs)
    assert all(sets[w]["m|L0|s0"] == -0.02 for w in ph.WORLDS)          # world-independent answer in every world
    assert sets["grow2"]["m|L2|t0"] == 0.02 and "m|L2|t0" not in sets["shrink4"]
    assert all(sets[w]["ORACLE|-|ctl"] == ph.WORLDS[w][1] for w in ph.WORLDS)
    assert clipped == [0, 2]


def test_mock_collection_end_to_end():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "mock.jsonl"
        here = ROOT / "analysis" / "llm_statements"
        subprocess.run([sys.executable, str(here / "collect_hose.py"), "--backend", "mock", "--tag", "mk",
                        "--out", str(out)], check=True, capture_output=True)
        assert sum(1 for _ in out.open()) == 126
        r = subprocess.run([sys.executable, str(here / "collect_hose.py"), "--check", str(out), "--smoke"],
                           capture_output=True, text=True)
        assert r.returncode == 0 and "unparsed 0/126" in r.stdout
        subprocess.run([sys.executable, str(here / "collect_hose.py"), "--backend", "mock", "--tag", "mk",
                        "--out", str(out)], check=True, capture_output=True)      # re-run skips finished ids
        assert sum(1 for _ in out.open()) == 126
        recs = sh.load([out])
        l2 = [sh.structure(r["parsed"]["pct"], r["world"])["correct"] for r in recs if r["level"] == "L2"]
        assert np.mean(l2) > 0.8                                                   # mock follows the braid rule


def test_external_statements_reach_the_gate_rules():
    """Known answer: exact truth, no noise -> an oracle statement under R2 is exact, R0 equals k0."""
    import hose_gate_mismatch as g
    from h2017_dynamics import load_urdf
    from test_robot_twin import _write_urdf
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "synthetic.urdf"
        _write_urdf(path)
        chain = load_urdf(path)
    sweeps = SimpleNamespace(files=[f"j5_60|1.2|{e}" for e in (-0.04, -0.02, 0.0, 0.02)])
    rows = g.run(chain, sweeps, 1, np.random.default_rng(0), 1e-6, "exact",
                 statements_fn=lambda e: {"oracle": e, "zero": 0.0})
    r2 = [err for name, rule, err, _ in rows if name == "oracle" and rule == "R2"]
    assert len(r2) == 4
    np.testing.assert_allclose(r2, 0.0, atol=1e-9)
    zero_r2 = [err for name, rule, err, _ in rows if name == "zero" and rule == "R2"]
    r0 = [err for name, rule, err, _ in rows if name == "zero" and rule == "R0"]
    np.testing.assert_allclose(zero_r2, r0)
