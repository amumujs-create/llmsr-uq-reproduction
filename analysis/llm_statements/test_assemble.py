"""Unit tests for the HA assembly rule (docs/research/ccpp-hint-assembly.md §3). Run: python -m pytest test_assemble.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from assemble import assemble, assemble_v1, s2_parts, split_of  # noqa: E402


def s2(sens=-2.2, lim=364, p15=343, ice=False, evap=None):
    o = {"gt_generator_limit": {"reached_in_range": True, "gt_mw_at_15c": p15, "limit_mw": lim, "gt_mw_per_degc": sens}}
    o["inlet_anti_icing_heating"] = {"active": ice, "below_c": 7 if ice else None}
    o["inlet_evaporative_cooling"] = {"active": evap is not None, "above_c": evap}
    return o


def test_cap_part_and_gate():
    p = s2_parts(s2(), "L2")
    at, ok = p["gt_limit"]
    assert abs(at - (15 - 21 / 2.2)) < 1e-9 and ok          # 0.64 %/degC passes
    _, ok_low = s2_parts(s2(sens=-0.6), "L2").get("gt_limit", (None, None))
    assert ok_low is None or ok_low is False                 # 0.17 %/degC fails (or boundary out of range)


def test_l2_uses_prompt_output():
    at, _ = s2_parts(s2(p15=300), "L2")["gt_limit"]
    assert abs(at - (15 - 21 / 2.2)) < 1e-9


def test_gate_removes_cap():
    bad = [s2_parts(s2(sens=-1.0, lim=350), "L2") for _ in range(3)]  # 0.29 %/degC
    assert assemble([], bad) == []
    assert assemble([], bad, use_gate=False)[0][0] == "below"


def test_per_mechanism_majority():
    parts = [s2_parts(s2(ice=(i == 0)), "L2") for i in range(3)]  # anti-icing claimed by 1 of 3
    st = assemble([], parts)
    assert len(st) == 1 and st[0][0] == "below" and abs(st[0][1] - 5.4545) < 1e-3


def test_unchecked_part_needs_s0_corroboration():
    parts = [s2_parts(s2(evap=24), "L2") for _ in range(3)]
    assert dict(assemble([], parts)).get("above") is None
    assert dict(assemble([[("above", 25.0)]] * 3, parts))["above"] == 24.0
    assert dict(assemble([[("above", 30.0)]] * 3, parts)).get("above") is None


def test_s2_no_boundary_is_not_filled_by_s0():
    parts = [s2_parts(s2(), "L2") for _ in range(2)]
    s0 = [[("below", 12.0), ("above", 30.0)]] * 3
    assert dict(assemble(s0, parts)) == {"below": 15 - 21 / 2.2}
    assert dict(assemble_v1(s0, parts))["above"] == 30.0   # the v1 behaviour that the amendment removed


def test_no_s2_falls_back_to_s0():
    assert assemble([[("below", 6.0)]] * 3, []) == [("below", 6.0)]


def test_split():
    assert split_of("gemma3-27b", False) == "validation"
    assert split_of("gpt-oss-20b", True) == "validation"
    assert split_of("gpt-oss-20b", False) == "dev"
    assert split_of("gpt-oss-120b", True) == "dev"
