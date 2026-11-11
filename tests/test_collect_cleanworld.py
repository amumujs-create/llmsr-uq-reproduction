"""Clean-world collection stage (analysis/llm_statements/collect_cleanworld.py)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis" / "llm_statements"))
sys.path.insert(0, str(ROOT / "analysis" / "ccpp_twin"))
import clean_worlds as cw  # noqa: E402
import collect_cleanworld as cc  # noqa: E402
import collect_newworld as cn  # noqa: E402
from ccpp_twin import simulate  # noqa: E402
from elicit import S2_QUESTION  # noqa: E402
from prompts import QUESTION, check_no_leak, describe  # noqa: E402

NUMBER = {"cap353": "353 MW", "cap350": "350 MW", "cap346": "346 MW", "evap22": "22 degC", "evap26": "26 degC",
          "chill24": "24 degC"}
TWIN_OF = {"cap353": ("cap_strong", "357", "353"), "cap350": ("cap_strong", "357", "350"),
           "cap346": ("cap_strong", "357", "346"), "evap22": ("evap", "about 24", "about 22"),
           "evap26": ("evap", "about 24", "about 26")}


def test_call_count_and_unique_ids():
    t = cc.tasks("m")
    assert len(t) == cc.N_CALLS == 192
    assert len({x[0] for x in t}) == 192
    assert sum(x[4] == 0.0 for x in t) == 12          # one temperature-0 call per world and question


def test_descriptions_differ_from_collected_ones_by_one_number():
    for world, (old_world, a, b) in TWIN_OF.items():
        old = describe(old_world, "L2")
        assert old.count(a) == 1
        assert cc.describe_clean(world) == old.replace(a, b)
    old = cn.describe_new("chill", "L2")
    assert old.count("about 20") == 1
    assert cc.describe_clean("chill24") == old.replace("about 20", "about 24")


def test_descriptions_state_the_number_and_no_banned_word():
    for world, num in NUMBER.items():
        d = cc.describe_clean(world)
        assert num in d
        assert check_no_leak(d) == []


def test_questions_are_unchanged():
    for world in cc.CLEAN_WORLDS:
        assert cc.messages(world, "S0")[1]["content"].endswith(QUESTION)
        assert cc.messages(world, "S2")[1]["content"].endswith(S2_QUESTION)


def test_swap_refuses_ambiguous_sentence():
    with pytest.raises(ValueError):
        cc._swap("24 and 24", "24", "22")


def _rec(kind, response, error=None):
    return {"strategy": kind, "response": response, "error": error}


def test_check_counts_and_smoke(tmp_path, capsys):
    good_s0 = json.dumps({"boundaries": [{"side": "above", "at_c": 22, "confidence": 0.7}], "rationale": "x"})
    good_s2 = json.dumps({"gt_generator_limit": {"reached_in_range": False}})
    p = tmp_path / "a.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in (_rec("S0", good_s0), _rec("S0", "nonsense"), _rec("S2", good_s2))))
    assert cc.check(str(p), smoke=True) == 0
    assert "S0 unparsed 1/2" in capsys.readouterr().out
    q = tmp_path / "b.jsonl"
    q.write_text(json.dumps(_rec("S0", None, "ConnectionError: x")))
    assert cc.check(str(q), smoke=True) == 1
    assert cc.check(str(q), smoke=False) == 0


def test_world_registry_covers_every_collected_world():
    cfg = cw.configs(357.3)
    assert set(cc.CLEAN_WORLDS) <= set(cfg)
    assert set(cw.ANSWERS) == set(cfg)
    assert set(cw.SIDE) == set(cfg) - {"base"}
    for world in cc.CLEAN_WORLDS:
        assert cw.ANSWERS[world] == ("cleanworld", world, ("L2",))


def test_twin_matches_the_stated_numbers():
    """Sanity check with a known answer: far inside the regime the twin behaves as the description says."""
    cfg = cw.configs(357.3)
    ap, rh, v = np.array([1013.0]), np.array([60.0]), np.array([45.0])
    for world, mw in (("cap353", 353.0), ("cap350", 350.0), ("cap346", 346.0)):
        cold = simulate(np.array([3.0]), ap, rh, v, cfg[world])["p_gt"][0]
        assert cold == pytest.approx(mw, abs=0.6)                       # held at the stated limit
        warm = simulate(np.array([25.0]), ap, rh, v, cfg[world])["pe_clean"][0]
        assert warm == pytest.approx(simulate(np.array([25.0]), ap, rh, v, cfg["base"])["pe_clean"][0], abs=0.01)
    base = cfg["base"]
    for world, on in (("evap22", 22.0), ("evap26", 26.0), ("chill24", 24.0)):
        below = np.array([on - 3.0])
        above = np.array([on + 3.0])
        d_below = simulate(below, ap, rh, v, cfg[world])["pe_clean"][0] - simulate(below, ap, rh, v, base)["pe_clean"][0]
        d_above = simulate(above, ap, rh, v, cfg[world])["pe_clean"][0] - simulate(above, ap, rh, v, base)["pe_clean"][0]
        assert abs(d_below) < 0.1                                        # off below the stated temperature
        assert d_above > 3.0                                             # clearly on above it
