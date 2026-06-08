# Credit Regime Detector

A quantitative framework for detecting macroeconomic regimes in European credit markets using a Gaussian Mixture Model (GMM) and lagged macro signals.

## What this project does

European credit markets cycle through distinct regimes - periods of compressed spreads and calm volatility, stress periods of rapid widening, and transitional states in between. This project builds a systematic model that identifies which regime the market is in on any given day, and tests whether macro variables observed in the preceding week can predict regime transitions before they fully reprice.

## Structure

| Script | Description |
|---|---|
| step1_collect_data.py | Pulls EUR HY/IG spreads, VIX, yields, EUR/USD from FRED |
| step2_stationarity.py | ADF stationarity tests, differencing, feature engineering |
| step3_hmm_model.py | Fits 3-state GMM, labels regimes, produces regime chart |
| step4_macro_linkage.py | Logistic regression of lagged macro variables on regime labels |

## Data sources

- ICE BofA EUR HY and IG OAS via FRED
- CBOE VIX via FRED
- US 10y Treasury yield via FRED
- EUR/USD via FRED

## Setup

```bash
git clone https://github.com/0aarontahiri/credit-regime-detector.git
cd credit-regime-detector
pip install -r requirements.txt
python step1_collect_data.py
```

## Research note

See `credit_regime_research_note.pdf` for the full write-up of methodology, results, and implications.

## Author

Aaron Tahiri - BSc Mathematics, Queen Mary University of London, 2026