"""Module D: is there a bubble, and what did the NVIDIA case do?

D1  psy_test    : Phillips, Shi and Yu (2015) GSADF test for explosive behavior
                  with date-stamping (BSADF sequence against simulated critical values).
D2  event_study : market-model abnormal returns around the event dates
                  (MacKinlay 1997) for NVIDIA, the AI basket and a non-AI basket.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import config as C


def _bsadf(y: np.ndarray, w0: int) -> np.ndarray:
    """Backward sup ADF sequence (no lag augmentation), fully vectorized.

    For each end point r2 it takes the largest Dickey-Fuller t statistic over
    all start points r1 with a window of at least w0 observations.
    """
    x, d = y[:-1], np.diff(y)
    c = lambda v: np.concatenate([[0.0], np.cumsum(v)])
    Sx, Sxx, Sd, Sdd, Sxd = c(x), c(x * x), c(d), c(d * d), c(x * d)
    m = len(d)
    out = np.full(len(y), np.nan)
    for e in range(w0, m + 1):                      # window uses d[s:e]
        s = np.arange(0, e - w0 + 1)
        n = (e - s).astype(float)
        sx, sxx, sd, sdd, sxd = Sx[e] - Sx[s], Sxx[e] - Sxx[s], Sd[e] - Sd[s], Sdd[e] - Sdd[s], Sxd[e] - Sxd[s]
        den = n * sxx - sx ** 2
        beta = (n * sxd - sx * sd) / den
        alpha = (sd - beta * sx) / n
        ssr = np.maximum(sdd - alpha * sd - beta * sxd, 1e-300)
        t = beta / np.sqrt(ssr / (n - 2) * n / den)
        out[e] = np.nanmax(t)
    return out


def psy_test(series: pd.Series, n_sim: int = 500, level: float = 0.95, seed: int = 0) -> dict:
    """GSADF test and bubble dates for a (log) price or price-to-fundamental series."""
    s = series.dropna()
    y, T = s.values.astype(float), len(s)
    w0 = int(np.floor(T * (0.01 + 1.8 / np.sqrt(T))))          # PSY minimum window rule
    stat = _bsadf(y, w0)
    rng = np.random.default_rng(seed)
    sims = np.array([_bsadf(np.cumsum(rng.normal(1.0 / T, 1.0, T)), w0) for _ in range(n_sim)])
    cv_seq = np.nanquantile(sims, level, axis=0)               # critical value for each date
    gs_null = np.nanmax(sims, axis=1)
    gsadf = float(np.nanmax(stat))
    seq = pd.DataFrame({"bsadf": stat, "cv": cv_seq}, index=s.index)
    seq["explosive"] = seq["bsadf"] > seq["cv"]
    # date-stamping: keep episodes that last at least log(T) periods
    min_len = int(np.ceil(np.log(T)))
    run = (seq["explosive"] != seq["explosive"].shift()).cumsum()
    episodes = []
    for _, g in seq[seq["explosive"]].groupby(run):
        if len(g) >= min_len:
            episodes.append((g.index[0], g.index[-1], len(g)))
    return dict(gsadf=gsadf, cv=float(np.quantile(gs_null, level)),
                p_value=float((gs_null >= gsadf).mean()), sequence=seq,
                episodes=pd.DataFrame(episodes, columns=["start", "end", "periods"]),
                ongoing=bool(seq["explosive"].iloc[-1]), min_window=w0)


def event_study(daily: pd.DataFrame, events=None, assets=("NVDA", "AI", "NONAI"),
                window=C.EVENT_WINDOW, est=C.ESTIMATION_WINDOW) -> pd.DataFrame:
    """Cumulative abnormal return (CAR) per asset and event, with a t statistic."""
    events = dict(events or C.EVENTS)
    rows = []
    if len(events) > 1:                              # whole episode: first event day to the last price date
        first = min(events)
        span = int((daily.index >= pd.Timestamp(first)).sum()) - 1
        todo = list(events.items()) + [(first, ("Whole episode", (0, span)))]
    else:
        todo = list(events.items())
    for date, label in todo:
        if isinstance(label, tuple):
            label, window = label
        d = pd.Timestamp(date)
        if d not in daily.index:
            pos = daily.index.searchsorted(d)
            if pos >= len(daily.index):
                continue                             # event after the last price date
            d = daily.index[pos]
        i = daily.index.get_loc(d)
        a, b = i + est[0], i + est[1]
        lo, hi = i + window[0], min(i + window[1], len(daily) - 1)
        if a < 0:
            continue
        for asset in assets:
            X = np.column_stack([np.ones(b - a), daily["MKT"].values[a:b]])
            yv = daily[asset].values[a:b]
            coef, *_ = np.linalg.lstsq(X, yv, rcond=None)
            sig = np.std(yv - X @ coef, ddof=2)
            ar = daily[asset].values[lo:hi + 1] - coef[0] - coef[1] * daily["MKT"].values[lo:hi + 1]
            car = ar.sum()
            rows.append(dict(event="episode" if label == "Whole episode" else str(d.date()), label=label, asset=asset, days=len(ar),
                             car=car, t=car / (sig * np.sqrt(len(ar))), beta=coef[1]))
    return pd.DataFrame(rows)
