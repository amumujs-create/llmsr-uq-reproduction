"""Twin configurations for the clean-window task set (docs/research/ccpp-twin-clean-collection.md).

DRAFT until the experiment is registered: these are the candidates found by explore/clean_tasks.py. A world is
"sharp" when its regime switches on quickly (cap_softness_mw 0.5 instead of 2.0, evap_ramp_c 0.3 instead of 3.0),
so that a training window short of the kink carries no trace of the regime.

ANSWERS says where the LLM answers for each world come from: (stage, world name in that stage, levels).
"""
from ccpp_twin import TwinConfig, with_modules

SHARP_CAP = dict(cold_cap=True, cap_softness_mw=0.5)
SHARP_EVAP = dict(evap_cooling=True, evap_ramp_c=0.3)


def configs(cap_strong_mw: float) -> dict[str, TwinConfig]:
    """cap_strong_mw is the registered cap_strong limit (grid.build_worlds; about 357.3 MW, described as 357 MW)."""
    return {
        "base": with_modules(),
        "cap357": with_modules(**SHARP_CAP, gt_cap_mw=cap_strong_mw),
        "cap353": with_modules(**SHARP_CAP, gt_cap_mw=353.0),
        "cap350": with_modules(**SHARP_CAP, gt_cap_mw=350.0),
        "cap346": with_modules(**SHARP_CAP, gt_cap_mw=346.0),
        "evap22": with_modules(**SHARP_EVAP, evap_on_c=22.0),
        "evap24": with_modules(**SHARP_EVAP, evap_on_c=24.0),
        "evap26": with_modules(**SHARP_EVAP, evap_on_c=26.0),
        "chill20": with_modules(inlet_chill=True, chill_set_c=20.0),
        "chill24": with_modules(inlet_chill=True, chill_set_c=24.0),
    }


SIDE = {"cap357": "cold", "cap353": "cold", "cap350": "cold", "cap346": "cold", "evap22": "hot", "evap24": "hot",
        "evap26": "hot", "chill20": "hot", "chill24": "hot"}

# world -> (stage directory under runs/, world name used there, levels available)
ANSWERS = {
    "base": ("main+strategy", "base", ("L1", "L2")),
    "cap357": ("main+strategy", "cap_strong", ("L1", "L2")),      # same description text
    "evap24": ("main+strategy", "evap", ("L1", "L2")),            # same description text
    "chill20": ("newworld", "chill", ("L1", "L2")),
    "cap353": ("cleanworld", "cap353", ("L2",)),
    "cap350": ("cleanworld", "cap350", ("L2",)),
    "cap346": ("cleanworld", "cap346", ("L2",)),
    "evap22": ("cleanworld", "evap22", ("L2",)),
    "evap26": ("cleanworld", "evap26", ("L2",)),
    "chill24": ("cleanworld", "chill24", ("L2",)),
}
