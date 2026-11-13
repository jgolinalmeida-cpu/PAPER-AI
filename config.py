"""All settings in one place. Edit this file, not the model code."""

# ---- Market data (Yahoo Finance tickers) ---------------------------------
MARKET = "SPY"                      # broad market benchmark
NVDA = "NVDA"
AI_BASKET = ["NVDA", "MSFT", "GOOGL", "AMZN", "META", "AVGO", "AMD", "ORCL", "TSM"]
NON_AI_BASKET = ["XLP", "XLU", "XLE", "XLV", "XLI"]   # sector ETFs with low AI exposure
PRICE_START = "2015-01-01"

# ---- Macro data (FRED series IDs). Check each ID on fred.stlouisfed.org ---
# A series that fails to download is skipped with a warning.
FRED_MONTHLY = {
    "log_ip": ("INDPRO", "log"),          # industrial production
    "log_emp": ("PAYEMS", "log"),         # nonfarm payrolls
    "credit_spread": ("BAA10Y", "level"), # Baa minus 10y Treasury
    "vix": ("VIXCLS", "level"),
    "sentiment": ("UMCSENT", "level"),
}
FRED_QUARTERLY = {
    "log_gdp": ("GDPC1", "log"),
    "log_invest": ("PNFI", "log"),            # private nonresidential fixed investment
    "log_it_invest": ("A679RC1Q027SBEA", "log"),  # information processing equipment and software
    "log_profits": ("CP", "log"),             # corporate profits after tax
}

# ---- NVIDIA CEO case: event dates (first trading day the news could move prices)
# Sources are listed in METHODS.md. Add or correct dates as the case develops.
EVENTS = {
    "2026-11-09": "Board opens investigation; CEO admits relationship (news on Sunday Nov 8)",
    "2026-11-12": "CEO leaves; CFO named interim CEO",
}
EVENT_WINDOW = (0, 1)         # trading days around each event (kept short so windows do not overlap)
ESTIMATION_WINDOW = (-250, -20)

# ---- Acemoglu (2024) calibration, Section 4 of the paper -------------------
ACEMOGLU = dict(
    exposed_share=0.199,      # GDP share of tasks exposed to AI (Eloundou et al.)
    adopt_share=0.23,         # share of exposed tasks profitably automated in 10 years (Svanberg et al.)
    labor_cost_saving=0.27,   # average labor cost saving (Noy-Zhang; Brynjolfsson et al.)
    labor_share=0.535,        # exposure-adjusted labor share
)

# ---- Local projection settings ---------------------------------------------
MACRO_HORIZONS = range(0, 13)   # months
PANEL_HORIZONS = range(0, 9)    # quarters
N_LAGS = 2
CI = 0.90
BUST_SIZE = -0.30               # scenario: AI basket falls 30% relative to the market
