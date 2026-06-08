import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

DATA_PATH  = "data/raw_data.csv"
OUTPUT_DIR = "data"
CHART_DIR  = "outputs"
os.makedirs(CHART_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH, index_col="date", parse_dates=True)
print(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns\n")


def adf_test(series, name):
    # ADF test checks if a series is stationary.
    # Null hypothesis: series is non-stationary (has a trend).
    # If p-value < 0.05 we reject that and call it stationary.
    result     = adfuller(series.dropna(), autolag="AIC")
    p_value    = result[1]
    stationary = p_value < 0.05
    status     = "STATIONARY" if stationary else "NON-STATIONARY"
    print(f"  {name:<20} p={p_value:.4f}  {status}")
    return stationary


print("=== ADF Stationarity Tests (levels) ===")
stationarity = {}
for col in df.columns:
    stationarity[col] = adf_test(df[col], col)


print("\n=== First-differencing non-stationary series ===")
df_diff = pd.DataFrame(index=df.index)

for col in df.columns:
    if stationarity[col]:
        df_diff[col] = df[col]
        print(f"  {col:<20} kept as-is")
    else:
        # Differencing removes the trend by computing day-to-day changes
        df_diff[f"{col}_diff"] = df[col].diff()
        print(f"  {col:<20} differenced -> {col}_diff")

df_diff = df_diff.dropna()
print(f"\nShape after differencing: {df_diff.shape}")


print("\n=== ADF Tests (after differencing) ===")
for col in df_diff.columns:
    adf_test(df_diff[col], col)


# We give the HMM two features to learn from.
# Spread change captures how much credit stress moved each day.
# Log VIX captures market fear independently of the credit signal.
# Two clean features is enough, more would add noise not signal.

print("\n=== Engineering HMM features ===")

features = pd.DataFrame(index=df_diff.index)

# Use the raw spread level rather than the daily change.
# The daily change series is flat pre-2022 due to data frequency issues.
# The level varies meaningfully across the full sample period.
features["spread_level"] = df["EUR_HY_OAS"].reindex(df_diff.index)

vix_col = "VIX_diff" if "VIX_diff" in df_diff.columns else "VIX"
features["vix"] = np.log(df[vix_col].reindex(features.index))

features = features.dropna()

print(f"  Features: {list(features.columns)}")
print(f"  Shape: {features.shape}")
print(f"\nFeature summary statistics:")
print(features.describe().round(4).to_string())


# Normalise so the HMM treats both features equally regardless of scale
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X      = scaler.fit_transform(features)
X_df   = pd.DataFrame(X, index=features.index, columns=features.columns)

print(f"\nFeatures normalised. Mean: {X_df.mean().round(4).to_string()}")
print(f"Std: {X_df.std().round(4).to_string()}")


features.to_csv(f"{OUTPUT_DIR}/features_raw.csv")
X_df.to_csv(f"{OUTPUT_DIR}/features_scaled.csv")
print(f"\nSaved features to {OUTPUT_DIR}/")


print("\nGenerating feature chart...")

fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
fig.suptitle("HMM Input Features", fontsize=13)

axes[0].plot(features.index, features["spread_level"],
             color="#D85A30", linewidth=0.8, alpha=0.9)
axes[0].axhline(0, color="#888780", linewidth=0.5, linestyle="--")
axes[0].set_ylabel("Spread level (bps)", fontsize=9)
axes[0].set_title("EUR HY OAS level", fontsize=10, loc="left")
axes[0].axvspan(pd.Timestamp("2020-02-20"), pd.Timestamp("2020-05-01"),
                alpha=0.15, color="red", label="COVID")
axes[0].axvspan(pd.Timestamp("2022-02-01"), pd.Timestamp("2022-12-01"),
                alpha=0.15, color="orange", label="Rate shock")
axes[0].legend(fontsize=8)

axes[1].plot(features.index, features["vix"],
             color="#534AB7", linewidth=0.8, alpha=0.9)
axes[1].set_ylabel("log(VIX)", fontsize=9)
axes[1].set_title("VIX (log scale)", fontsize=10, loc="left")

plt.tight_layout()
chart_path = f"{CHART_DIR}/features_overview.png"
plt.savefig(chart_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Chart saved to {chart_path}")

print("\nStep 2 done. Run step3_hmm_model.py next.")