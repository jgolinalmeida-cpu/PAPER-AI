# AI Shock Lab

A small econometric system to study three questions:

1. How much AI use is official and how much is shadow use, and what each does.
2. Whether firms that use AI earn more, and which kind of use pays.
3. Whether AI stocks behave like a bubble, and what the NVIDIA CEO case did to them.

It is built on [puremacro](https://jalonso1979.github.io/puremacro/docs/quickstart/). `METHODS.md` explains the models, the data you need and every source.

## Run it

```
pip install -r requirements.txt
python test_recovery.py        # 16 checks on simulated data with known true effects
python run_all.py --demo       # full run on simulated data
python run_all.py --real       # real stock prices (Yahoo Finance) and macro data (FRED)
python run_all.py --real --panel data/my_panel.csv   # adds the firm or sector modules
```

In Google Colab, upload the folder, then run `!pip install -r requirements.txt` and `!python run_all.py --demo` in a cell.

Results go to `output/tables` (CSV) and `output/figures` (PNG). `output/summary.json` holds the headline numbers.

## What is tested and what is not

The demo data is simulated. Its figures say nothing about the real economy. Its only job is to prove that each estimator finds an effect that was put there on purpose.

The real-data loaders (`--real`) were written but could not be run in the environment where this was built, because it had no access to Yahoo Finance or FRED. Expect to fix small things on the first run, such as a FRED series ID.

## Files

| File | What it does |
|---|---|
| `ai_shock/config.py` | Tickers, FRED series, event dates, Acemoglu parameters. Edit this first. |
| `ai_shock/data.py` | Real-data loaders and the demo simulator. |
| `ai_shock/shadow.py` | Builds the shadow AI variable. |
| `ai_shock/shock.py` | Module A. The AI shock, macro responses, bust scenario. |
| `ai_shock/panel.py` | Modules B and C. Who gains, adoption effects, shadow regressions. |
| `ai_shock/bubble.py` | Module D. Bubble test and event study. |
| `ai_shock/calibrate.py` | Module E. Acemoglu's formula with a shadow AI margin. |
| `data/panel_template.csv` | Column layout for your own panel. |
