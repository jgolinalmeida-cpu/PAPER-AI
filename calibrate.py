"""Module E: Acemoglu (2024) back-of-envelope, extended with shadow AI.

Acemoglu's equation (14):  TFP gain = (GDP share of tasks impacted) x (average cost saving).
The extension splits impacted tasks into officially adopted and shadow tasks.
Shadow tasks get a pass-through below one: the worker saves time, but only part
of the saving shows up in firm output or cost (NANDA: individual productivity, not P&L).
The pass-through is an assumption, not an estimate. Replace it with your
Module C estimate once you have real data.
"""
from __future__ import annotations
import pandas as pd
from . import config as C


def tfp_gain(exposed_share, adopt_share, labor_cost_saving, labor_share,
             shadow_extra_share=0.0, shadow_pass_through=0.0) -> dict:
    saving = labor_cost_saving * labor_share
    official = exposed_share * adopt_share * saving
    shadow = exposed_share * shadow_extra_share * saving * shadow_pass_through
    return dict(official_pct=100 * official, shadow_pct=100 * shadow, total_pct=100 * (official + shadow))


def scenarios(shadow_extra_share=0.30, pass_throughs=(0.0, 0.25, 0.5, 1.0)) -> pd.DataFrame:
    """Baseline reproduces Acemoglu's 0.66%. Other rows add shadow use on a further
    `shadow_extra_share` of exposed tasks with different pass-through rates."""
    rows = [dict(scenario="Acemoglu baseline (official adoption only)", **tfp_gain(**C.ACEMOGLU))]
    for p in pass_throughs[1:]:
        rows.append(dict(scenario=f"+ shadow use, pass-through {int(p * 100)}%",
                         **tfp_gain(**C.ACEMOGLU, shadow_extra_share=shadow_extra_share, shadow_pass_through=p)))
    return pd.DataFrame(rows)
