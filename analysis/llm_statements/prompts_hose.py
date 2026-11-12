"""Prompts for the tool-hose statement collection (docs/research/robot-twin-hose-llm-preregistration.md).

The LLM is asked how the hose length changes at working pressure (sign and size). It is not asked
for J6 angles: the twin turns the stated length change into a J6 limit with the known geometry.
Truth direction: a wire braid laid above the neutral angle (~54.7 deg) lengthens under pressure,
below it shortens (Dyrr et al., MATEC 2022). L2 gives the braid angle only, never the rule.
Magnitudes per world are twin assumptions.
"""
from __future__ import annotations

# world -> (braid angle in the L2 description, true length change at working pressure).
# The neutral world uses 54.5 deg, not 54.7, so the description does not print the textbook number
# (researcher decision 2026-11-12); 0.2 deg from neutral counts as 'about unchanged' (|eps| <= 0.5 %).
WORLDS = {"shrink4": (51.0, -0.04), "shrink2": (53.0, -0.02), "neutral": (54.5, 0.0), "grow2": (57.0, 0.02)}
LEVELS = ("L0", "L1", "L2")
NEUTRAL_TOL_PCT = 0.5       # |stated %| <= this counts as 'unchanged' in the neutral world
NEUTRAL_WRONG_PCT = 2.0     # |stated %| > this is a strict error in the neutral world

# Words that would state the rule or the answer (checked on descriptions, not on the question).
BANNED = ("neutral angle", "magic angle", "elongat", "contract", "shorten", "lengthen", "54.7",
          "zero length change", "zero-length")

SYSTEM = ("You are an experienced robotics and fluid-power engineer. "
          "Answer with a single JSON object and nothing else.")

L0 = ("A six-axis industrial collaborative robot arm (payload about 20 kg) carries a hydraulic tool on its wrist. "
      "A hydraulic supply hose runs from a clamp on the forearm to a port on the tool, so rotating the last wrist "
      "joint (J6) winds the hose around the wrist and pulls it tighter. During work the hose is pressurized.")

L1 = ("The hose is an SAE 100R2AT-type hose with two high-tensile steel wire braids and a 1/4 inch inner diameter. "
      "Its installed length between the clamp and the tool port is 1.05 to 1.35 times the straight distance "
      "between them, and during work it carries its full rated working pressure.")

L2 = "The steel wire braid is laid at {angle:g} degrees to the hose axis."

QUESTION = (
    "Question: estimate how the length of this hose changes when it is pressurized to its working pressure, "
    "compared with the unpressurized hose. Then say whether the range through which the tool can be rotated "
    "about J6 before the hose is pulled tight becomes narrower, wider, or stays the same.\n"
    "Reply with JSON only, in this form:\n"
    '{"length_change_pct": <percent, negative = shorter>, "direction": "shorter" or "longer" or "unchanged", '
    '"j6_range": "narrower" or "wider" or "unchanged", "confidence": <0 to 1>, '
    '"rationale": "<one or two sentences>"}'
)


def describe(world: str, level: str) -> str:
    if level not in LEVELS:
        raise ValueError(f'unknown level {level}')
    parts = [L0]
    if level in ("L1", "L2"):
        parts.append(L1)
    if level == "L2":
        parts.append(L2.format(angle=WORLDS[world][0]))
    return " ".join(parts)


def messages(world: str, level: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"{describe(world, level)}\n\n{QUESTION}"}]


def jobs() -> list[tuple[str, str]]:
    """(world, level): L0 and L1 do not depend on the world and are asked once ('any'); L2 once per world."""
    return [("any", "L0"), ("any", "L1")] + [(w, "L2") for w in WORLDS]


def check_no_leak() -> None:
    for world in WORLDS:
        for level in LEVELS:
            text = describe(world, level).lower()
            for b in BANNED:
                if b in text:
                    raise AssertionError(f'banned phrase {b!r} in {world}/{level}')
