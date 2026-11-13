"""Data layer.

Three tidy tables feed every model:

macro   : monthly DataFrame, DatetimeIndex (month end). Columns: ai_ret, mkt_ret,
          log_ai_price, plus macro outcomes (log_ip, log_emp, ...).
daily   : daily DataFrame of log returns. Columns: MKT, NVDA, AI, NONAI.
panel   : long DataFrame, one row per (unit, date). Units are firms or sectors.
          Required columns:
            unit, date (quarter end), sector
            official_ai  share of workers covered by an employer-approved AI tool (0-1)
            worker_ai    share of workers who use AI at work, approved or not (0-1)
            exposure     time-invariant AI exposure of the unit (0-1)
            adopt_date   first quarter with an AI system integrated in workflows (NaT if never)
            pilot_date   first quarter with an official pilot only (NaT if never)
            profit_margin (percent), log_productivity

`load_real_*` functions download public data. `make_demo_data` simulates all
three tables with KNOWN true effects so you can test that the pipeline works.
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from . import config as C

# --------------------------------------------------------------------- real
def load_fred(series: dict, freq: str = "M") -> pd.DataFrame:
    """Download FRED series as CSV (no API key needed)."""
    out = {}
    for name, (sid, how) in series.items():
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
        try:
            s = pd.read_csv(url, index_col=0, parse_dates=True).iloc[:, 0]
            s = pd.to_numeric(s, errors="coerce").dropna()
            s = s.resample("ME" if freq == "M" else "QE").mean()
            out[name] = np.log(s) if how == "log" else s
        except Exception as e:  # keep going if one series fails
            warnings.warn(f"FRED series {sid} skipped: {e}")
    return pd.DataFrame(out)


def load_prices(tickers, start=C.PRICE_START) -> pd.DataFrame:
    """Daily adjusted close prices from Yahoo Finance (pip install yfinance)."""
    import yfinance as yf
    px = yf.download(list(tickers), start=start, auto_adjust=True, progress=False)["Close"]
    return px.dropna(how="all")


def load_real_market():
    """Return (daily returns table, monthly market table) from Yahoo Finance."""
    tick = sorted(set([C.MARKET, C.NVDA] + C.AI_BASKET + C.NON_AI_BASKET))
    r = np.log(load_prices(tick)).diff()
    daily = pd.DataFrame({
        "MKT": r[C.MARKET], "NVDA": r[C.NVDA],
        "AI": r[C.AI_BASKET].mean(axis=1), "NONAI": r[C.NON_AI_BASKET].mean(axis=1),
    }).dropna()
    return daily, market_monthly(daily)


def market_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    m = daily.resample("ME").sum()
    return pd.DataFrame({"ai_ret": m["AI"], "mkt_ret": m["MKT"],
                         "log_ai_price": m["AI"].cumsum(), "log_mkt_price": m["MKT"].cumsum()})


def load_real_macro() -> tuple[pd.DataFrame, pd.DataFrame]:
    daily, mk = load_real_market()
    macro = mk.join(load_fred(C.FRED_MONTHLY, "M"), how="left")
    return daily, macro


def load_panel_csv(path: str) -> pd.DataFrame:
    """Load your own firm or sector panel. See data/panel_template.csv."""
    df = pd.read_csv(path, parse_dates=["date", "adopt_date", "pilot_date"])
    need = {"unit", "date", "official_ai", "worker_ai", "exposure", "profit_margin"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"panel is missing columns: {sorted(miss)}")
    return df.sort_values(["unit", "date"]).reset_index(drop=True)


# --------------------------------------------------------------------- demo
# True parameters of the simulated world. The test script checks that each
# estimator recovers them.
TRUTH = dict(
    macro_irf_peak=0.40,        # log_invest response (x100) to a 1 sd AI shock, long run
    panel_exposure_beta0=0.60,  # margin response at h=0 per unit of exposure
    panel_shadow_beta0=0.00,    # shadow use does not change the margin response
    panel_decay=0.70,
    att_integrated=2.0,         # long-run margin gain (pp) from integrated AI
    att_pilot=0.0,              # pilots that never integrate: no gain
    prod_official=0.10,         # log productivity per unit of official use
    prod_shadow=0.06,           # log productivity per unit of shadow use
    margin_shadow=0.0,          # shadow use raises output per worker, not margins
    bubble_start="2023-01", bubble_end="2026-10",
    event_car_nvda=-0.12,
)

SECTORS = {  # sector: (AI exposure, shadow intensity). Ordering follows the NANDA disruption index.
    "Technology": (0.75, 0.55), "Media & Telecom": (0.60, 0.50),
    "Professional Services": (0.55, 0.60), "Financial Services": (0.45, 0.40),
    "Healthcare & Pharma": (0.30, 0.35), "Consumer & Retail": (0.25, 0.30),
    "Advanced Industries": (0.20, 0.20), "Energy & Materials": (0.10, 0.10),
}


def make_demo_data(seed: int = 7, n_firms: int = 320):
    """Simulate daily, macro and panel tables with the effects in TRUTH."""
    rng = np.random.default_rng(seed)
    T = TRUTH

    # ---- daily market: AI basket has a mildly explosive phase, then event drops
    days = pd.bdate_range("2015-01-02", "2026-11-13")
    n = len(days)
    mkt = rng.normal(0.0003, 0.009, n)
    ai_spec = rng.normal(0, 0.008, n)                 # AI-specific news
    bub = (days >= T["bubble_start"]) & (days <= "2026-10-31")
    logp_ai = np.zeros(n)
    dev = 0.0                                          # deviation from fundamentals
    for t in range(1, n):
        if bub[t]:                                     # mildly explosive, as in Phillips-Shi-Yu
            dev = 1.0012 * dev + 0.0006 + 0.6 * ai_spec[t]
        else:
            dev = dev + 0.6 * ai_spec[t]
        logp_ai[t] = dev
    ai_ret = np.diff(logp_ai, prepend=0) + 1.1 * mkt
    nvda = 1.3 * ai_ret + rng.normal(0, 0.006, n)
    nonai = 0.8 * mkt + rng.normal(0, 0.005, n)
    for d, (jn, ja, jo) in {"2026-11-09": (-0.045, -0.02, 0.0), "2026-11-10": (-0.02, -0.01, 0.0),
                            "2026-11-12": (-0.055, -0.025, -0.003)}.items():
        i = days.get_loc(pd.Timestamp(d))
        nvda[i] += jn; ai_ret[i] += ja; nonai[i] += jo
    daily = pd.DataFrame({"MKT": mkt, "NVDA": nvda, "AI": ai_ret, "NONAI": nonai}, index=days)
    macro = market_monthly(daily)

    # ---- macro outcomes respond to the AI-specific monthly shock
    m_spec = pd.Series(ai_spec, index=days).resample("ME").sum()
    s = ((m_spec - m_spec.mean()) / m_spec.std()).values
    nm = len(s)
    phi = T["macro_irf_peak"] * 0.25 * 0.75 ** np.arange(24)   # growth response, sums to the peak
    for name, scale in {"log_invest": 1.0, "log_ip": 0.5, "log_emp": 0.2}.items():
        g = np.convolve(s, phi)[:nm] * scale / 100 + rng.normal(0.002, 0.0015 * scale + 0.0005, nm)
        macro[name] = np.cumsum(g)
    macro["credit_spread"] = 2.0 - 0.10 * np.convolve(s, 0.8 ** np.arange(12))[:nm] + rng.normal(0, 0.15, nm)
    macro["true_shock"] = s

    # ---- panel: firms in 8 sectors, quarterly 2019Q1-2026Q3
    q = pd.date_range("2019-03-31", "2026-09-30", freq="QE")
    nq = len(q)
    qs = m_spec.resample("QE").sum().reindex(q)
    a = ((qs - qs.mean()) / qs.std()).values           # common quarterly AI shock
    post = np.maximum(np.arange(nq) - q.get_loc(pd.Timestamp("2022-12-31")), 0)  # quarters since ChatGPT
    rows = []
    names = list(SECTORS)
    for i in range(n_firms):
        sec = names[i % len(names)]
        ex0, sh0 = SECTORS[sec]
        expo = float(np.clip(ex0 + rng.normal(0, 0.10), 0.02, 0.98))
        u = rng.random()
        # NANDA funnel: few integrate, many pilot, most have no official system
        p_int = 0.04 + 0.14 * ex0
        kind = "integrated" if u < p_int else ("pilot" if u < p_int + 0.35 else "none")
        start = int(rng.integers(q.get_loc(pd.Timestamp("2023-06-30")), nq - 3)) if kind != "none" else None
        D = np.zeros(nq)
        if start is not None:
            D[start:] = 1
        ramp = np.minimum(np.maximum(np.arange(nq) - (start or 0) + 1, 0) / 4, 1) * D
        official = np.clip(0.15 * D + 0.55 * ramp * (kind == "integrated") + rng.normal(0, 0.02, nq), 0, 1)
        shadow_true = np.clip(sh0 * (1 - np.exp(-post / 5)) * rng.uniform(0.6, 1.3) + rng.normal(0, 0.02, nq), 0, 1) * (post > 0)
        worker = np.clip(official + shadow_true, 0, 1)
        fe_m, fe_y = rng.normal(10, 3), rng.normal(0, 0.3)
        resp = np.convolve(a, T["panel_exposure_beta0"] * T["panel_decay"] ** np.arange(12))[:nq] * expo
        common = 0.3 * np.convolve(a, 0.7 ** np.arange(12))[:nq]
        margin = (fe_m + common + resp + T["att_integrated"] * ramp * (kind == "integrated")
                  + T["margin_shadow"] * shadow_true + rng.normal(0, 0.6, nq))
        prod = (fe_y + 0.004 * np.arange(nq) + T["prod_official"] * official
                + T["prod_shadow"] * shadow_true + rng.normal(0, 0.02, nq))
        for t in range(nq):
            rows.append((f"F{i:03d}", q[t], sec, expo, official[t], worker[t], kind,
                         q[start] if kind == "integrated" else pd.NaT,
                         q[start] if kind == "pilot" else pd.NaT, margin[t], prod[t], a[t]))
    panel = pd.DataFrame(rows, columns=["unit", "date", "sector", "exposure", "official_ai", "worker_ai",
                                        "kind", "adopt_date", "pilot_date", "profit_margin",
                                        "log_productivity", "ai_shock"])
    return daily, macro, panel
