"""Module A: the aggregate AI shock.

Step 1 builds the shock: the part of the AI basket return that the market
return, its own lags and lagged macro conditions cannot explain.
Step 2 traces the response of macro outcomes with local projections
(Jorda 2005), using puremacro.lp.lp_hac. A Cholesky VAR is a cross-check.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from puremacro.lp import lp_hac
from puremacro.var.identify import cholesky_svar
from . import config as C


def build_ai_shock(macro: pd.DataFrame, controls=(), n_lags: int = 2) -> pd.Series:
    """Residual of ai_ret on mkt_ret, lags of both and lagged controls. Unit variance."""
    X = {"const": 1.0, "mkt_ret": macro["mkt_ret"]}
    for l in range(1, n_lags + 1):
        X[f"ai_l{l}"] = macro["ai_ret"].shift(l)
        X[f"mkt_l{l}"] = macro["mkt_ret"].shift(l)
        for c in controls:
            X[f"{c}_l{l}"] = macro[c].diff().shift(l)
    X = pd.DataFrame(X, index=macro.index)
    ok = X.notna().all(axis=1) & macro["ai_ret"].notna()
    b, *_ = np.linalg.lstsq(X[ok].values, macro.loc[ok, "ai_ret"].values, rcond=None)
    e = macro.loc[ok, "ai_ret"] - X[ok].values @ b
    return (e / e.std()).rename("ai_shock").reindex(macro.index)


def macro_irfs(macro: pd.DataFrame, outcomes, shock: str = "ai_shock",
               horizons=C.MACRO_HORIZONS, n_lags=C.N_LAGS, ci=C.CI) -> pd.DataFrame:
    """Local projection of each outcome on the AI shock. Returns a long table."""
    out = []
    for y in outcomes:
        d = macro[[y, shock]].dropna()
        r = pd.DataFrame(lp_hac(d, y=y, x=shock, horizons=horizons, n_lags=n_lags, alpha=1 - ci))
        r["outcome"] = y
        out.append(r)
    return pd.concat(out, ignore_index=True)


def bust_scenario(irfs: pd.DataFrame, macro: pd.DataFrame, size: float = C.BUST_SIZE) -> pd.DataFrame:
    """Scale the IRFs to a fall of `size` (log points) in the AI basket relative to the market.

    This is a linear extrapolation. It assumes a bust works like a large
    negative version of the shocks seen so far.
    """
    X = np.column_stack([np.ones(len(macro)), macro["mkt_ret"].fillna(0)])
    ok = macro["ai_ret"].notna().values
    b, *_ = np.linalg.lstsq(X[ok], macro["ai_ret"].values[ok], rcond=None)
    sd = float(np.std(macro["ai_ret"].values[ok] - X[ok] @ b))
    k = size / sd                                  # size of the bust in shock standard deviations
    s = irfs.copy()
    for c in ["beta", "lo", "hi"]:
        s[c] = irfs[c] * k
    s[["lo", "hi"]] = np.sort(s[["lo", "hi"]].values, axis=1)
    s.attrs["n_sd"] = k
    return s


def var_crosscheck(macro: pd.DataFrame, outcome: str, shock: str = "ai_shock", p: int = 2, horizon: int = 12):
    """Cholesky VAR with the shock ordered first. Returns the outcome's cumulated IRF."""
    d = macro[[shock, outcome]].dropna()
    d = d.assign(**{outcome: d[outcome].diff()}).dropna()
    r = cholesky_svar(d.values, p=p, horizon=horizon, n_boot=300, ci=C.CI, seed=0)
    # point estimate only: cumulating pointwise bands would not give a valid band
    return pd.DataFrame({"h": range(horizon + 1), "beta": np.cumsum(r.irf_point[:, 1, 0])})
