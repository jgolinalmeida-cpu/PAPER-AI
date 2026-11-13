"""Shadow AI measurement.

NANDA (2025) reports that about 40% of companies bought an official LLM
subscription while workers at over 90% of companies used personal AI tools.
We turn that idea into a variable:

    shadow_ai = worker_ai - official_ai      (floored at zero)

worker_ai   = share of workers who say they use AI at work (worker surveys).
official_ai = share of workers covered by an employer-approved tool (business surveys).
The gap is a LOWER BOUND on unapproved use, because some covered workers also
use personal tools.
"""
from __future__ import annotations
import pandas as pd


def add_shadow(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    df["shadow_ai"] = (df["worker_ai"] - df["official_ai"]).clip(lower=0)
    df["shadow_share"] = (df["shadow_ai"] / df["worker_ai"]).where(df["worker_ai"] > 0.01)
    # pre-determined shadow intensity: unit mean over the sample (used in interactions)
    df["shadow_base"] = df.groupby("unit")["shadow_ai"].transform("mean")
    return df


def shadow_summary(panel: pd.DataFrame, by: str = "sector") -> pd.DataFrame:
    """Latest-period official use, worker use and shadow gap by group."""
    last = panel[panel["date"] == panel["date"].max()]
    out = last.groupby(by)[["official_ai", "worker_ai", "shadow_ai"]].mean()
    return out.sort_values("shadow_ai", ascending=False)


def shadow_over_time(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.groupby("date")[["official_ai", "worker_ai", "shadow_ai"]].mean()
