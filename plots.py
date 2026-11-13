"""Figures. One axis per chart, thin marks, direct titles, light grid."""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, ORANGE, AQUA, INK, MUTED, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c9c8c2",
                     "axes.grid": True, "grid.color": "#e6e5df", "grid.linewidth": 0.6,
                     "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "axes.labelsize": 9, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "font.family": "DejaVu Sans",
                     "legend.frameon": False, "legend.fontsize": 8.5, "figure.dpi": 150})


def _band(ax, r, color, label=None, scale=1.0):
    ax.fill_between(r["h"], r["lo"] * scale, r["hi"] * scale, color=color, alpha=0.15, linewidth=0)
    ax.plot(r["h"], r["beta"] * scale, color=color, linewidth=2, label=label)
    ax.axhline(0, color=MUTED, linewidth=0.8)


def _save(fig, path, note=None):
    if note:
        fig.text(0.01, 0.005, note, fontsize=7.5, color=MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03 if note else 0, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def macro_irf(irfs, path, labels=None, note=None, xlabel="Months after the shock", title_prefix="", scale=100.0):
    outs = list(irfs["outcome"].unique())
    fig, axes = plt.subplots(1, len(outs), figsize=(3.5 * len(outs), 3.1), squeeze=False)
    for ax, o in zip(axes[0], outs):
        r = irfs[irfs["outcome"] == o]
        sc = scale if o.startswith("log_") else 1.0
        _band(ax, r, BLUE, scale=sc)
        ax.set_title(title_prefix + (labels or {}).get(o, o))
        ax.set_xlabel(xlabel)
        ax.set_ylabel("percent" if o.startswith("log_") else "level")
    _save(fig, path, note)


def hetero_irf(het, path, note=None):
    names = {"exposure": "Extra response of AI-exposed units", "shadow_base": "Extra response of high shadow-use units"}
    keys = list(het["interaction"].unique())
    fig, axes = plt.subplots(1, len(keys), figsize=(4.4 * len(keys), 3.2), squeeze=False, sharey=True)
    for ax, k, col in zip(axes[0], keys, [BLUE, ORANGE]):
        _band(ax, het[het["interaction"] == k], col)
        ax.set_title(names.get(k, k))
        ax.set_xlabel("Quarters after the AI shock")
    axes[0][0].set_ylabel("profit margin, percentage points")
    _save(fig, path, note)


def adoption(res, path, note=None):
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for (k, lab), col in zip({"integrated": "Integrated AI", "pilot_only": "Pilot only"}.items(), [BLUE, ORANGE]):
        if k not in res:
            continue
        es = res[k]["event_study"]
        es = es[(es["event_time"] >= -6) & (es["event_time"] <= 8)]
        ax.fill_between(es["event_time"], es["lo"], es["hi"], color=col, alpha=0.15, linewidth=0)
        ax.plot(es["event_time"], es["att"], color=col, linewidth=2, marker="o", markersize=3.5, label=lab)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.axvline(-0.5, color=MUTED, linewidth=0.8, linestyle=":")
    ax.set_title("Profit margin before and after AI adoption")
    ax.set_xlabel("Quarters relative to adoption")
    ax.set_ylabel("percentage points vs. never adopters")
    ax.legend(loc="upper left")
    _save(fig, path, note)


def shadow_gap(over_time, by_sector, path, note=None):
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.5), gridspec_kw={"width_ratios": [1.1, 1]})
    a.plot(over_time.index, 100 * over_time["worker_ai"], color=BLUE, linewidth=2, label="Workers using AI at work")
    a.plot(over_time.index, 100 * over_time["official_ai"], color=ORANGE, linewidth=2, label="Covered by an approved tool")
    a.fill_between(over_time.index, 100 * over_time["official_ai"], 100 * over_time["worker_ai"], color=BLUE, alpha=0.10, linewidth=0)
    a.set_title("The shadow gap over time")
    a.set_ylabel("percent of workers")
    a.legend(loc="upper left")
    s = by_sector.sort_values("shadow_ai")
    b.barh(s.index, 100 * s["shadow_ai"], color=BLUE, height=0.6)
    b.set_title("Shadow AI use by sector, latest quarter")
    b.set_xlabel("percentage points (worker use minus approved use)")
    b.grid(axis="y", visible=False)
    _save(fig, path, note)


def bubble(res, price, path, events=None, note=None):
    seq = res["sequence"]
    fig, (a, b) = plt.subplots(2, 1, figsize=(7.2, 5.2), sharex=True, gridspec_kw={"height_ratios": [1, 1]})
    a.plot(price.index, price.values, color=INK, linewidth=1.6)
    a.set_title("AI basket relative to the market, log price")
    b.plot(seq.index, seq["bsadf"], color=BLUE, linewidth=2, label="Explosiveness statistic (BSADF)")
    b.plot(seq.index, seq["cv"], color=ORANGE, linewidth=1.5, linestyle="--", label="95% critical value")
    for ax in (a, b):
        for _, e in res["episodes"].iterrows():
            ax.axvspan(e["start"], e["end"], color=BLUE, alpha=0.10, linewidth=0)
        for d in (events or {}):
            ax.axvline(pd.Timestamp(d), color=MUTED, linewidth=0.8, linestyle=":")
    b.set_title("Bubble test: shaded periods are flagged as explosive")
    b.legend(loc="upper left")
    _save(fig, path, note)


def event_cars(ev, path, note=None):
    names = {"NVDA": "NVIDIA", "AI": "AI basket", "NONAI": "Non-AI basket"}
    events = list(ev["event"].unique())
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    w = 0.25
    for j, (asset, col) in enumerate(zip(["NVDA", "AI", "NONAI"], [BLUE, ORANGE, AQUA])):
        r = ev[ev["asset"] == asset].set_index("event").reindex(events)
        bars = ax.bar(np.arange(len(events)) + (j - 1) * (w + 0.02), 100 * r["car"], width=w, color=col, label=names[asset])
        for rect, v in zip(bars, 100 * r["car"]):
            ax.annotate(f"{v:.1f}", (rect.get_x() + rect.get_width() / 2, v), ha="center",
                        va="top" if v < 0 else "bottom", fontsize=8, color=INK,
                        xytext=(0, -3 if v < 0 else 3), textcoords="offset points")
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xticks(range(len(events)), events)
    ax.set_ylabel("cumulative abnormal return, percent")
    ax.set_title("Abnormal returns around the NVIDIA events")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower left", ncol=3)
    ax.margins(y=0.18)
    _save(fig, path, note)


def calibration(sc, path, note=None):
    fig, ax = plt.subplots(figsize=(7, 3))
    y = np.arange(len(sc))[::-1]
    ax.barh(y, sc["official_pct"], color=BLUE, height=0.55, label="Official adoption")
    ax.barh(y, sc["shadow_pct"], left=sc["official_pct"] + 0.004, color=ORANGE, height=0.55, label="Shadow use")
    for yi, v in zip(y, sc["total_pct"]):
        ax.text(v + 0.02, yi, f"{v:.2f}%", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(y, sc["scenario"])
    ax.set_xlabel("TFP gain over 10 years, percent")
    ax.set_title("Acemoglu's estimate with a shadow AI margin")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.12), ncol=2)
    ax.margins(x=0.15)
    _save(fig, path, note)
