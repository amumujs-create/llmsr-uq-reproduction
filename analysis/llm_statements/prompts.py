"""Prompt templates for regime-statement collection (docs/research/llm-statement-collection.md).

Descriptions name equipment and operating facts only; they never name the regime shape.
"""
from __future__ import annotations

BANNED = ("flat", "plateau", "saturat", "regime", "level off", "levels off", "breakpoint", "piecewise")

WORLDS = ("base", "cap_mild", "cap_strong", "evap", "evap+cap_mild")
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


def _l1_key(world: str) -> str:
    return {"base": "base", "cap_mild": "cap", "cap_strong": "cap", "evap": "evap", "evap+cap_mild": "evap+cap"}[world]


def describe(world: str, level: str) -> str:
    if level == "L0":
        return L0
    parts = [L0, L1[_l1_key(world)]]
    if level == "L2":
        parts.append(L2_GT)
        if world in L2_CAP:
            parts.append(L2_CAP[world])
        if world == "evap+cap_mild":
            parts.append(L2_CAP["cap_mild"])
        if "evap" in world:
            parts.append(L2_EVAP)
    return " ".join(parts)


def window_summary(summary: dict) -> str:
    return (f"Training data available: {summary['n']} hourly records with ambient temperature between "
            f"{summary['at_lo']:.1f} and {summary['at_hi']:.1f} degC. Within these records a linear fit gives "
            f"d(output)/d(temperature) = {summary['slope']:.2f} MW/degC (standard error {summary['se']:.2f}). "
            "Predictions are needed outside this temperature range.")


def build_prompt(world: str, level: str, summary: dict | None = None, real: bool = False) -> list[dict]:
    text = describe(world, level)
    if real:
        text = f"{text} {REAL}"
    if summary is not None:
        text = f"{text}\n{window_summary(summary)}"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"{text}\n\n{QUESTION}"}]


def check_no_leak(text: str) -> list[str]:
    low = text.lower()
    return [w for w in BANNED if w in low]
