import pandas as pd
import requests
from io import StringIO
import os

OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

START = "2019-01-01"
END   = "2024-12-31"


def fetch_fred(series_id, name, start=START, end=END):
    # Download a single series from FRED as CSV, no API key needed
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    print(f"  Fetching {name}...", end=" ")
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        df = pd.read_csv(
            StringIO(r.text),
            parse_dates=["observation_date"],
            index_col="observation_date",
            na_values="."
        )
        df.columns = [name]
        df.index.name = "date"
        df = df.loc[start:end]
        df[name] = pd.to_numeric(df[name], errors="coerce")
        df = df.resample("B").last().ffill()
        print(f"OK - {len(df)} rows")
        return df[name]
    except Exception as e:
        print(f"FAILED: {e}")
        return pd.Series(dtype=float, name=name)


# FRED series to pull
# EUR_HY_OAS  : euro high yield spread, proxy for iTraxx Crossover
# EUR_IG_OAS  : euro investment grade spread, proxy for iTraxx Main
# VIX         : volatility, fear gauge
# DE10Y       : german 10y yield, european risk-free rate
# EURUSD      : fx, broad risk sentiment
# PMI_MFG     : manufacturing PMI, real economy leading indicator
SERIES = {
    # Euro high yield OAS - correct FRED ID
    "EUR_HY_OAS": ("BAMLHE00EHYIOAS", "EUR HY OAS"),
    # US investment grade OAS - best available IG proxy on FRED
    "EUR_IG_OAS": ("BAMLC0A0CM",      "EUR IG OAS"),
    "VIX":        ("VIXCLS",          "VIX"),
    "DE10Y": ("DGS10", "US 10y Yield"),
     "EURUSD":     ("DEXUSEU",         "EUR/USD"),
    
}

print(f"\nFetching data from FRED ({START} to {END})\n")

series_dict = {}
for key, (fred_id, label) in SERIES.items():
    s = fetch_fred(fred_id, label)
    if not s.empty:
        series_dict[key] = s

# Try VSTOXX from Yahoo, fall back to VIX if it fails
# VSTOXX not available via free APIs, VIX used as volatility proxy


# Combine all series into one dataframe
df = pd.DataFrame(series_dict)
df = df.dropna(how="all")
df = df.ffill().bfill()

print(f"\nShape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Range: {df.index[0].date()} to {df.index[-1].date()}")
print(f"\nMissing values:\n{df.isna().sum().to_string()}")
print(f"\nLast 5 rows:\n{df.tail().to_string()}")

output_path = os.path.join(OUTPUT_DIR, "raw_data.csv")
df.to_csv(output_path)
print(f"\nSaved to {output_path}")

# Plot a 3-panel overview chart and save it
print("\nGenerating chart...")
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Raw Data Overview", fontsize=13, y=0.98)

    ax1 = axes[0]
    if "EUR_HY_OAS" in df.columns:
        ax1.plot(df.index, df["EUR_HY_OAS"], color="#D85A30", linewidth=1, label="EUR HY OAS")
    if "EUR_IG_OAS" in df.columns:
        ax1b = ax1.twinx()
        ax1b.plot(df.index, df["EUR_IG_OAS"], color="#185FA5", linewidth=1, label="EUR IG OAS", alpha=0.8)
        ax1b.set_ylabel("IG OAS (bps)", fontsize=9, color="#185FA5")
    ax1.set_ylabel("HY OAS (bps)", fontsize=9, color="#D85A30")
    ax1.set_title("Credit Spreads", fontsize=10, loc="left")
    ax1.legend(loc="upper left", fontsize=8)

    ax2 = axes[1]
    vol_col = "VSTOXX" if "VSTOXX" in df.columns else "VIX"
    if vol_col in df.columns:
        ax2.fill_between(df.index, df[vol_col], alpha=0.3, color="#534AB7")
        ax2.plot(df.index, df[vol_col], color="#534AB7", linewidth=0.8)
    ax2.set_ylabel(vol_col, fontsize=9)
    ax2.set_title("Volatility", fontsize=10, loc="left")
    ax2.axvspan(pd.Timestamp("2020-02-20"), pd.Timestamp("2020-05-01"), alpha=0.12, color="red", label="COVID")
    ax2.axvspan(pd.Timestamp("2022-02-01"), pd.Timestamp("2022-12-01"), alpha=0.12, color="orange", label="Rate shock")
    ax2.legend(loc="upper right", fontsize=8)

    ax3 = axes[2]
    if "DE10Y" in df.columns:
        ax3.plot(df.index, df["DE10Y"], color="#1D9E75", linewidth=1, label="German 10y")
        ax3.axhline(0, color="#888780", linewidth=0.5, linestyle="--")
    if "EURUSD" in df.columns:
        ax3b = ax3.twinx()
        ax3b.plot(df.index, df["EURUSD"], color="#BA7517", linewidth=1, label="EUR/USD", alpha=0.7)
        ax3b.set_ylabel("EUR/USD", fontsize=9, color="#BA7517")
    ax3.set_ylabel("Yield (%)", fontsize=9, color="#1D9E75")
    ax3.set_title("Rates and FX", fontsize=10, loc="left")
    ax3.legend(loc="upper left", fontsize=8)

    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax3.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=0, fontsize=9)

    plt.tight_layout()
    chart_path = os.path.join(OUTPUT_DIR, "raw_data_overview.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Chart saved to {chart_path}")

except Exception as e:
    print(f"Chart failed: {e}")

print("\nStep 1 done. Run step2_define_regimes.py next.")