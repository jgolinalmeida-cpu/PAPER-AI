"""Run the whole analysis.

    python run_all.py --demo                      # simulated data with known true effects
    python run_all.py --real                      # real market + FRED data (needs internet)
    python run_all.py --real --panel data/my_panel.csv   # also run the firm/sector modules

Tables go to output/tables, figures to output/figures.
"""
import argparse, os, json
import numpy as np
import pandas as pd
from ai_shock import config as C, data, shadow, shock, panel as pn, bubble, calibrate, plots

ap = argparse.ArgumentParser()
ap.add_argument("--demo", action="store_true")
ap.add_argument("--real", action="store_true")
ap.add_argument("--panel", default=None)
ap.add_argument("--out", default="output")
args = ap.parse_args()
demo = args.demo or not args.real
NOTE = "SIMULATED DEMO DATA. Not a finding about the real economy." if demo else None
fig, tab = os.path.join(args.out, "figures"), os.path.join(args.out, "tables")
os.makedirs(fig, exist_ok=True); os.makedirs(tab, exist_ok=True)
summary = {"mode": "demo" if demo else "real"}

if demo:
    daily, macro, panel = data.make_demo_data()
else:
    daily, macro = data.load_real_macro()
    panel = data.load_panel_csv(args.panel) if args.panel else None

# ---- A. aggregate AI shock --------------------------------------------------
macro["ai_shock"] = shock.build_ai_shock(macro)
outcomes = [c for c in ["log_invest", "log_ip", "log_emp", "credit_spread"] if c in macro]
labels = {"log_invest": "Investment", "log_ip": "Industrial production", "log_emp": "Employment",
          "credit_spread": "Credit spread"}
irfs = shock.macro_irfs(macro, outcomes)
irfs.to_csv(f"{tab}/A_macro_irfs.csv", index=False)
plots.macro_irf(irfs, f"{fig}/A_macro_irfs.png", labels, NOTE, title_prefix="")
bust = shock.bust_scenario(irfs, macro)
bust.to_csv(f"{tab}/A_bust_scenario.csv", index=False)
plots.macro_irf(bust, f"{fig}/A_bust_scenario.png", labels,
                (NOTE + " " if NOTE else "") + f"Scenario: AI basket falls {abs(C.BUST_SIZE):.0%} relative to the market. Linear extrapolation.")
summary["bust_size_in_sd"] = round(float(bust.attrs["n_sd"]), 2)
if "true_shock" in macro:
    summary["shock_recovery_corr"] = round(float(macro[["ai_shock", "true_shock"]].corr().iloc[0, 1]), 3)

# ---- D. bubble test and event study ----------------------------------------
series = macro["log_ai_price"] - macro["log_mkt_price"]      # AI basket relative to the market
psy = bubble.psy_test(series)
psy["sequence"].to_csv(f"{tab}/D_bsadf_sequence.csv")
psy["episodes"].to_csv(f"{tab}/D_bubble_episodes.csv", index=False)
plots.bubble(psy, series, f"{fig}/D_bubble_test.png", C.EVENTS, NOTE)
ev = bubble.event_study(daily)
ev.to_csv(f"{tab}/D_event_study.csv", index=False)
if len(ev):
    plots.event_cars(ev, f"{fig}/D_event_study.png", NOTE)
summary["gsadf"] = dict(stat=round(psy["gsadf"], 2), cv95=round(psy["cv"], 2), p=psy["p_value"],
                        episodes=psy["episodes"].astype(str).values.tolist(), still_explosive=psy["ongoing"])

# ---- E. Acemoglu calibration with shadow margin ----------------------------
sc = calibrate.scenarios()
sc.to_csv(f"{tab}/E_tfp_scenarios.csv", index=False)
plots.calibration(sc, f"{fig}/E_tfp_scenarios.png",
                  "Baseline from Acemoglu (2024). The shadow rows are assumptions to replace with your own estimates.")

# ---- B, C. panel modules -----------------------------------------------------
if panel is not None:
    panel = shadow.add_shadow(panel)
    if "ai_shock" not in panel:                               # quarterly sum of the monthly shock
        q = macro["ai_shock"].resample("QE").sum()
        panel["ai_shock"] = panel["date"].map(q / q.std())
    by_sec, over_t = shadow.shadow_summary(panel), shadow.shadow_over_time(panel)
    by_sec.to_csv(f"{tab}/C_shadow_by_sector.csv")
    plots.shadow_gap(over_t, by_sec, f"{fig}/C_shadow_gap.png", NOTE)
    reg = pn.shadow_regression(panel)
    reg.to_csv(f"{tab}/C_shadow_regression.csv", index=False)
    het = pn.two_step_irfs(panel)                              # preferred: inference on the time dimension
    het.to_csv(f"{tab}/B_heterogeneous_irfs.csv", index=False)
    pn.heterogeneous_irfs(panel).to_csv(f"{tab}/B_heterogeneous_irfs_panel_lp_crosscheck.csv", index=False)
    pn.lpdid_crosscheck(panel).to_csv(f"{tab}/B_adoption_lpdid_crosscheck.csv", index=False)
    shock.var_crosscheck(macro, outcomes[0]).to_csv(f"{tab}/A_var_crosscheck.csv", index=False)
    plots.hetero_irf(het, f"{fig}/B_heterogeneous_irfs.png", NOTE)
    ad = pn.adoption_effects(panel)
    pd.concat([v["event_study"] for v in ad.values()]).to_csv(f"{tab}/B_adoption_event_study.csv", index=False)
    plots.adoption(ad, f"{fig}/B_adoption_effects.png", NOTE)
    summary["adoption"] = {k: {m: (round(float(v[m]), 3) if isinstance(v[m], (float, np.floating)) else v[m])
                               for m in ["overall", "se", "att_last", "pretrend_max_abs", "n_treated", "n_control"]}
                           for k, v in ad.items()}
    summary["shadow_regression"] = reg.round(4).to_dict("records")
    summary["hetero_h0"] = het[het["h"] == 0].round(3).to_dict("records")

json.dump(summary, open(f"{args.out}/summary.json", "w"), indent=2, default=str)
print(json.dumps(summary, indent=2, default=str))
