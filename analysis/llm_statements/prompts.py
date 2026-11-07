"""Prompt templates for regime-statement collection (docs/research/llm-statement-collection.md).

Descriptions name equipment and operating facts only; they never name the regime shape.
"""
from __future__ import annotations

BANNED = ("flat", "plateau", "saturat", "regime", "level off", "levels off", "breakpoint", "piecewise")

WORLDS = ("base", "cap_mild", "cap_strong", "evap", "evap+cap_mild")
ICE_WORLDS = ("ice", "ice+cap_mild")  # anti-icing worlds, docs/research/ccpp-anti-icing-world.md
SETTINGS = ("cold->hot", "hot->cold", "mid->hot", "mid->cold")

SYSTEM = (
    "You are an experienced power-plant performance engineer. Answer with a single JSON object and nothing else."
)

L0 = (
    "A combined cycle power plant has one gas turbine, a heat recovery steam generator and a steam turbine. "
    "It runs at full load. Hourly records give net electrical output (MW) together with ambient temperature "
    "(2 to 37 degC), ambient pressure, relative humidity and condenser vacuum."
)

L1 = {
    "base": "The plant has no inlet air conditioning, and over this ambient range the gas turbine stays below "
            "the rated limit of its generator.",
    "cap": "The plant has no inlet air conditioning. In cold weather the gas turbine reaches the rated limit of "
           "its generator, and the control system holds it at that limit.",
    "evap": "The plant has an inlet evaporative cooler in front of the compressor, which operators run in warm "
            "weather. Over this ambient range the gas turbine stays below the rated limit of its generator.",
    "evap+cap": "The plant has an inlet evaporative cooler in front of the compressor, which operators run in warm "
                "weather. In cold weather the gas turbine reaches the rated limit of its generator, and the control "
                "system holds it at that limit.",
}

L2_GT = "At 15 degC and 1013 mbar the gas turbine alone produces about 343 MW."
L2_CAP = {"cap_mild": "The gas turbine generator is limited to 364 MW.",
          "cap_strong": "The gas turbine generator is limited to 357 MW."}
L2_EVAP = "Operators switch the evaporative cooler on when ambient temperature exceeds about 24 degC."

REAL = (
    "These are real records from an operating plant collected over six years while it ran at full load."
)

QUESTION = (
    "Question: over the 2 to 37 degC ambient range, is there a temperature beyond which the net output stops "
    "responding much to ambient temperature (changes far less per degC than elsewhere)? There may be none, one "
    "on the cold side, one on the hot side, or both.\n"
    "Reply with JSON only, in this form:\n"
    '{"boundaries": [{"side": "below" or "above", "at_c": <degC>, "confidence": <0 to 1>}], '
    '"rationale": "<one or two sentences>"}\n'
    '"side": "below" means the weak response holds below at_c; "above" means above at_c. '
    "Use an empty list if you expect no such temperature."
)


# Anti-icing wording (ccpp-anti-icing-world.md §6). Only used for the ice worlds and for the re-ask of the five
# original worlds (ice_note=True); the default prompts above are unchanged.
NO_ICE = "The plant has no inlet anti-icing system."
L1_ICE = "On cold, humid days an inlet anti-icing system heats the intake air with compressor bleed."
L2_ICE = "Anti-icing keeps the compressor inlet at about 7 degC when relative humidity is above about 70%."
# In the ice worlds "no inlet air conditioning" would contradict the heating sentence; say "no inlet cooling".
_ICE_BASE = {"ice": "base", "ice+cap_mild": "cap"}

QUESTION_MECH = (
    "Question: over the 2 to 37 degC ambient range, is there a temperature beyond which the net output responds to "
    "ambient temperature differently from elsewhere (much more weakly, or in the opposite direction)? There may be "
    "none, one or more on the cold side, one on the hot side, or several.\n"
    "Reply with JSON only, in this form:\n"
    '{"boundaries": [{"side": "below" or "above", "at_c": <degC>, "rh_above": <percent, or null>, '
    '"mechanism": "power_cap" or "anti_icing" or "evap_cooling" or "other", '
    '"output_trend": "rises" or "falls" or "about the same", "confidence": <0 to 1>}], '
    '"rationale": "<one or two sentences>"}\n'
    '"side": "below" means the different response holds below at_c; "above" means above at_c. '
    '"rh_above" is the relative humidity above which it applies, or null if it applies at any humidity. '
    '"mechanism" names the cause: power_cap is a generator or power limit, anti_icing is inlet anti-icing heating, '
    "evap_cooling is inlet evaporative cooling. "
    '"output_trend" says how net output changes as ambient temperature moves further beyond at_c. '
    "Use an empty list if you expect no such temperature."
)


# Shape stage (docs/research/ccpp-shape-elicitation.md §3): same fields as QUESTION_MECH, but asks how the response
# beyond at_c compares with elsewhere, without suggesting "more weakly" or "opposite" in the question itself.
SHAPE_WORLDS = ("base", "cap_mild", "cap_strong", "evap", "evap+cap_mild", "ice")
RESPONSES = ("weaker", "near_zero", "opposite", "same_or_stronger")
QUESTION_SHAPE = (
    "Question: over the 2 to 37 degC ambient range, is there a temperature beyond which the net output responds to "
    "ambient temperature differently from elsewhere? There may be none, one or more on the cold side, one on the hot "
    "side, or several.\n"
    "Reply with JSON only, in this form:\n"
    '{"boundaries": [{"side": "below" or "above", "at_c": <degC>, "rh_above": <percent, or null>, '
    '"mechanism": "power_cap" or "anti_icing" or "evap_cooling" or "other", '
    '"response_beyond": "weaker" or "near_zero" or "opposite" or "same_or_stronger", "confidence": <0 to 1>}], '
    '"rationale": "<one or two sentences>"}\n'
    '"side": "below" means the different response holds below at_c; "above" means above at_c. '
    '"rh_above" is the relative humidity above which it applies, or null if it applies at any humidity. '
    '"mechanism" names the cause: power_cap is a generator or power limit, anti_icing is inlet anti-icing heating, '
    "evap_cooling is inlet evaporative cooling. "
    '"response_beyond" compares how net output responds to ambient temperature beyond at_c with how it responds '
    "elsewhere in the range: weaker = same direction but a clearly smaller change per degC; near_zero = output "
    "barely changes with ambient temperature; opposite = output changes in the opposite direction; "
    "same_or_stronger = same direction and at least as large a change per degC. "
    "Use an empty list if you expect no such temperature."
)

def _l1_key(world: str) -> str:
    return {"base": "base", "cap_mild": "cap", "cap_strong": "cap", "evap": "evap", "evap+cap_mild": "evap+cap",
            **_ICE_BASE}[world]


def describe(world: str, level: str, ice_note: bool = False) -> str:
    """Plant description. ice_note adds the anti-icing sentence to the original worlds (always on for ice worlds)."""
    if level == "L0":
        return L0
    ice = world in ICE_WORLDS
    l1 = L1[_l1_key(world)]
    if ice:
        l1 = l1.replace("no inlet air conditioning", "no inlet cooling")
    parts = [L0, l1]
    if ice:
        parts.append(L1_ICE)
    elif ice_note:
        parts.append(NO_ICE)
    if level == "L2":
        parts.append(L2_GT)
        if world in L2_CAP:
            parts.append(L2_CAP[world])
        if world in ("evap+cap_mild", "ice+cap_mild"):
            parts.append(L2_CAP["cap_mild"])
        if "evap" in world:
            parts.append(L2_EVAP)
        if ice:
            parts.append(L2_ICE)
    return " ".join(parts)


def window_summary(summary: dict) -> str:
    return (f"Training data available: {summary['n']} hourly records with ambient temperature between "
            f"{summary['at_lo']:.1f} and {summary['at_hi']:.1f} degC. Within these records a linear fit gives "
            f"d(output)/d(temperature) = {summary['slope']:.2f} MW/degC (standard error {summary['se']:.2f}). "
            "Predictions are needed outside this temperature range.")


def build_prompt(world: str, level: str, summary: dict | None = None, real: bool = False,
                 ice_note: bool = False) -> list[dict]:
    """Ice worlds get the mechanism question; the original worlds keep QUESTION even with ice_note (§6, §13)."""
    text = describe(world, level, ice_note)
    if real:
        text = f"{text} {REAL}"
    if summary is not None:
        text = f"{text}\n{window_summary(summary)}"
    q = QUESTION_MECH if world in ICE_WORLDS else QUESTION
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"{text}\n\n{q}"}]


def check_no_leak(text: str) -> list[str]:
    low = text.lower()
    return [w for w in BANNED if w in low]


def build_shape_prompt(world: str, level: str) -> list[dict]:
    """Shape stage (ccpp-shape-elicitation.md §3): original worlds carry the 'no anti-icing' sentence."""
    text = describe(world, level, ice_note=world not in ICE_WORLDS)
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"{text}\n\n{QUESTION_SHAPE}"}]
