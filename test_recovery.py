"""Check that every estimator recovers the true effects built into the demo data.

    python test_recovery.py

Run this after any change to the code. If a check fails, do not trust real-data results.
"""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from ai_shock import data, shadow, shock, panel as pn, bubble, calibrate
T = data.TRUTH
daily, macro, panel = data.make_demo_data()
panel = shadow.add_shadow(panel)
checks = []
def check(name, est, truth, tol):
    ok = abs(est - truth) <= tol
    checks.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name:<58} estimate {est:8.3f}   truth {truth:8.3f}   tolerance {tol}")

macro["ai_shock"] = shock.build_ai_shock(macro)
check("A  shock recovered (correlation with true shock)", macro[["ai_shock", "true_shock"]].corr().iloc[0, 1], 1.0, 0.2)
irf = shock.macro_irfs(macro, ["log_invest"])
check("A  investment response at 12 months, percent", 100 * irf.loc[irf.h == 12, "beta"].item(), T["macro_irf_peak"] * (1 - 0.75 ** 13), 0.15)
het = pn.two_step_irfs(panel, horizons=range(0, 2))
e0 = het[(het.h == 0) & (het.interaction == "exposure")].iloc[0]
s0 = het[(het.h == 0) & (het.interaction == "shadow_base")].iloc[0]
check("B1 extra margin response of exposed units, impact", e0.beta, T["panel_exposure_beta0"], 3 * e0.se)
check("B1 extra margin response of shadow-heavy units, impact", s0.beta, T["panel_shadow_beta0"], 3 * s0.se)
ad = pn.adoption_effects(panel, n_boot=100)
es = ad["integrated"]["event_study"]
check("B2 margin gain from integrated AI after 4+ quarters, pp", es.loc[es.event_time.between(4, 6), "att"].mean(), T["att_integrated"], 0.5)
check("B2 margin gain from pilot only, pp", ad["pilot_only"]["overall"], T["att_pilot"], 0.3)
pre = es.loc[es.event_time.between(-5, -2), "att"].mean()
check("B2 pre-trend before integration (should be zero), pp", pre, 0.0, 0.3)
reg = pn.shadow_regression(panel).set_index(["outcome", "regressor"])
check("C  productivity effect of official use", reg.loc[("log_productivity", "official_ai"), "coef"], T["prod_official"], 0.03)
check("C  productivity effect of shadow use", reg.loc[("log_productivity", "shadow_ai"), "coef"], T["prod_shadow"], 0.02)
r = reg.loc[("profit_margin", "shadow_ai")]
check("C  margin effect of shadow use (should be zero)", r.coef, T["margin_shadow"], 3 * r.se)
psy = bubble.psy_test(macro["log_ai_price"] - macro["log_mkt_price"], n_sim=300)
check("D1 bubble detected (1 = GSADF above 95% critical value)", float(psy["gsadf"] > psy["cv"]), 1.0, 0.0)
ep = psy["episodes"]
inside = len(ep) > 0 and ep.start.min() >= pd.Timestamp(T["bubble_start"]) and ep.start.min() <= pd.Timestamp("2025-06-30")
check("D1 first flagged date falls inside the true bubble (1 = yes)", float(inside), 1.0, 0.0)
rw = pd.Series(np.cumsum(np.random.default_rng(1).normal(0, 1, 140)))
check("D1 no bubble flagged in a pure random walk (1 = correct)", float(bubble.psy_test(rw, n_sim=300)["p_value"] > 0.05), 1.0, 0.0)
ev = bubble.event_study(daily).set_index(["event", "asset"])
check("D2 NVIDIA abnormal return over the whole episode", ev.loc[("episode", "NVDA"), "car"], T["event_car_nvda"], 0.04)
check("D2 non-AI basket abnormal return (should be near zero)", ev.loc[("episode", "NONAI"), "car"], -0.003, 0.03)
check("E  Acemoglu baseline TFP gain, percent", calibrate.scenarios().total_pct.iloc[0], 0.66, 0.005)
print(f"\n{sum(checks)} of {len(checks)} checks passed")
raise SystemExit(0 if all(checks) else 1)
