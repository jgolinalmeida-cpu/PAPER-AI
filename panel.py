"""Modules B and C: who gains from AI, and what shadow AI does.

B1  heterogeneous_irfs : panel local projections. The common AI shock is
    interacted with each unit's AI exposure and its shadow intensity. Unit and
    time fixed effects absorb everything common, so the estimate is the
    DIFFERENCE in response between more and less exposed units.
B2  adoption_effects   : staggered difference-in-differences (Callaway and
    Sant'Anna 2021) for the profit effect of integrating AI, and separately
    for pilots that never integrate. LP-DiD (Dube et al. 2023) as a cross-check.
C   shadow_regression  : two-way fixed effects regression of an outcome on
    official use, shadow use and their interaction, clustered by unit.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from puremacro.lp import panel_lp, lp_did
from puremacro.did import callaway_santanna
from . import config as C


# ----------------------------------------------------------------------- B1
def heterogeneous_irfs(panel: pd.DataFrame, y: str = "profit_margin", shock: str = "ai_shock",
                       by=("exposure", "shadow_base"), horizons=C.PANEL_HORIZONS,
                       n_lags=C.N_LAGS, ci=C.CI, cov_type="driscoll-kraay",
                       controls=("official_ai",)) -> pd.DataFrame:
    """Standard errors are Driscoll-Kraay by default. The shock is common to all
    units, so errors are correlated across units and clustering by unit alone
    makes the bands too narrow."""
    df = panel.copy()
    for v in by:
        df[f"x_{v}"] = df[shock] * (df[v] - df[v].mean())
    out = []
    for v in by:
        others = [f"x_{o}" for o in by if o != v] + [c for c in controls if c in df]
        r = pd.DataFrame(panel_lp(df, y=y, x=f"x_{v}", horizons=horizons, n_lags=n_lags,
                                  controls=others, alpha=1 - ci, unit_col="unit", time_col="date",
                                  cov_type=cov_type))
        r["interaction"] = v
        out.append(r)
    return pd.concat(out, ignore_index=True)


def two_step_irfs(panel: pd.DataFrame, y: str = "profit_margin", shock: str = "ai_shock",
                  by=("exposure", "shadow_base"), controls=("official_ai",), horizons=C.PANEL_HORIZONS,
                  n_lags=C.N_LAGS, ci=C.CI) -> pd.DataFrame:
    """Preferred estimator for the heterogeneous response. Inference runs on the time dimension.

    Step 1 (each quarter t, horizon h): cross-section regression of
            y[i,t+h] - y[i,t-1] on the unit characteristics in `by` (plus controls).
            The slopes say how much more exposed units gained in that quarter.
    Step 2: time-series regression of those slopes on the AI shock and its lags,
            Newey-West errors, Student-t critical values.
    The shock is common to all units, so the real sample size is the number of
    quarters. This estimator makes that explicit and avoids bands that are too narrow.
    """
    from scipy import stats
    df = panel.sort_values(["unit", "date"]).copy()
    by, controls = list(by), [c for c in controls if c in df]
    dates = np.array(sorted(df["date"].unique()))
    a = df.groupby("date")[shock].first().reindex(dates).values
    ylag = df.groupby("unit")[y].shift(1)
    rows = []
    for h in horizons:
        df["_dy"] = df.groupby("unit")[y].shift(-h) - ylag
        G = np.full((len(dates), len(by)), np.nan)
        for j, d in enumerate(dates):
            c = df.loc[df["date"] == d, ["_dy"] + by + controls].dropna()
            if len(c) < 5 * (len(by) + len(controls) + 1):
                continue
            keep = [k for k in controls if c[k].std() > 1e-8]
            X = np.column_stack([np.ones(len(c)), c[by + keep].values])
            G[j] = np.linalg.lstsq(X, c["_dy"].values, rcond=None)[0][1:1 + len(by)]
        L = np.column_stack([np.ones(len(a))] + [np.roll(a, l) for l in range(0, n_lags + 1)])
        L[:n_lags] = np.nan
        for k, v in enumerate(by):
            ok = ~np.isnan(G[:, k]) & ~np.isnan(L).any(axis=1)
            X, g = L[ok], G[ok, k]
            n, p = X.shape
            XtXi = np.linalg.inv(X.T @ X)
            b = XtXi @ X.T @ g
            u = X * (g - X @ b)[:, None]
            S = u.T @ u
            for l in range(1, h + 1):                      # Newey-West, bandwidth h
                w = 1 - l / (h + 1)
                Gm = u[l:].T @ u[:-l]
                S += w * (Gm + Gm.T)
            se = np.sqrt(np.diag(XtXi @ S @ XtXi)[1] * n / (n - p))
            crit = stats.t.ppf(0.5 + ci / 2, n - p)
            rows.append(dict(h=h, beta=b[1], se=se, t=b[1] / se, lo=b[1] - crit * se, hi=b[1] + crit * se,
                             interaction=v, n_periods=int(n)))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------- B2
def _did_frame(panel, treat_col, drop_col):
    """Treated = units with a date in treat_col. Controls = never treated in either way."""
    df = panel[panel[drop_col].isna()].copy() if drop_col in panel else panel.copy()
    periods = {d: i for i, d in enumerate(sorted(df["date"].unique()))}
    df["t"] = df["date"].map(periods)
    df["g"] = df[treat_col].map(periods)          # NaN for never treated
    return df


def adoption_effects(panel: pd.DataFrame, y: str = "profit_margin", n_boot: int = 200, ci=C.CI) -> dict:
    """Event-study ATT of (i) integrated AI and (ii) pilot-only adoption on y."""
    res = {}
    for label, tcol, dcol in [("integrated", "adopt_date", "pilot_date"), ("pilot_only", "pilot_date", "adopt_date")]:
        df = _did_frame(panel, tcol, dcol)
        if df["g"].notna().sum() == 0:
            continue
        r = callaway_santanna(df, unit="unit", time="t", outcome=y, treat_time="g",
                              control="never_treated", n_boot=n_boot, alpha=1 - ci, seed=0)
        es = r.att_event_study.copy()
        es["group"] = label
        post = es[es["event_time"] >= 0]
        res[label] = dict(event_study=es, overall=r.att_overall, se=r.att_overall_se,
                          lo=r.att_overall_lo, hi=r.att_overall_hi,
                          n_treated=int(df.loc[df["g"].notna(), "unit"].nunique()),
                          n_control=int(df.loc[df["g"].isna(), "unit"].nunique()),
                          pretrend_max_abs=float(es.loc[es["event_time"] < -1, "att"].abs().max()),
                          att_last=float(post["att"].iloc[-1]) if len(post) else np.nan)
    return res


def lpdid_crosscheck(panel: pd.DataFrame, y: str = "profit_margin", horizons=range(0, 7)) -> pd.DataFrame:
    df = _did_frame(panel, "adopt_date", "pilot_date")
    df["D"] = (df["t"] >= df["g"]).astype(int)
    r = lp_did(df, y=y, treat="D", unit="unit", time="t", horizons=horizons, pre_window=4)
    return pd.DataFrame(r.estimates)


# ------------------------------------------------------------------------ C
def twfe(panel: pd.DataFrame, y: str, x: list[str], unit="unit", time="date",
         time_varying_loading: str | None = None) -> pd.DataFrame:
    """Two-way fixed effects OLS with standard errors clustered by unit.

    `time_varying_loading="exposure"` adds exposure x period effects. They absorb
    any common shock that hits exposed units harder, which would otherwise be
    picked up by regressors that trend in exposed units (such as shadow use).
    """
    d = panel[[unit, time, y] + x + ([time_varying_loading] if time_varying_loading else [])].dropna().copy()
    Z = d[[y] + x].astype(float)
    nuis = []
    if time_varying_loading:
        dums = pd.get_dummies(d[time], drop_first=True, dtype=float)
        dums = dums.mul(d[time_varying_loading].values, axis=0)
        dums.columns = nuis = [f"_tv{i}" for i in range(dums.shape[1])]
        Z = pd.concat([Z, dums], axis=1)
    for _ in range(50):                               # alternating projections (handles unbalanced panels)
        Z = Z - Z.groupby(d[unit]).transform("mean")
        Z = Z - Z.groupby(d[time]).transform("mean")
    if nuis:                                          # partial out the exposure x period effects
        N = Z[nuis].values
        Z[[y] + x] = Z[[y] + x].values - N @ np.linalg.lstsq(N, Z[[y] + x].values, rcond=None)[0]
    Y, X = Z[y].values, Z[x].values
    XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ X.T @ Y
    e = Y - X @ b
    codes = pd.factorize(d[unit])[0]
    G = codes.max() + 1
    S = np.zeros((G, X.shape[1]))
    np.add.at(S, codes, X * e[:, None])
    n, k = X.shape
    adj = G / (G - 1) * (n - 1) / (n - k - G - d[time].nunique() - len(nuis))
    V = XtXi @ (S.T @ S) @ XtXi * adj
    se = np.sqrt(np.diag(V))
    out = pd.DataFrame({"coef": b, "se": se, "t": b / se}, index=x)
    out["lo"], out["hi"] = out["coef"] - 1.645 * out["se"], out["coef"] + 1.645 * out["se"]
    out.attrs.update(n_obs=n, n_units=G, outcome=y)
    return out


def shadow_regression(panel: pd.DataFrame, outcomes=("log_productivity", "profit_margin")) -> pd.DataFrame:
    df = panel.copy()
    df["official_x_shadow"] = df["official_ai"] * df["shadow_ai"]
    out = []
    for y in outcomes:
        r = twfe(df, y, ["official_ai", "shadow_ai", "official_x_shadow"],
                 time_varying_loading="exposure" if "exposure" in df else None)
        r.insert(0, "outcome", y)
        out.append(r.reset_index(names="regressor"))
    return pd.concat(out, ignore_index=True)
