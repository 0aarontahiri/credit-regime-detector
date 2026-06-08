import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

DATA_PATH   = "data/raw_data.csv"
REGIME_PATH = "data/regime_labels.csv"
CHART_DIR   = "outputs"
os.makedirs(CHART_DIR, exist_ok=True)

df     = pd.read_csv(DATA_PATH, index_col="date", parse_dates=True)
regime = pd.read_csv(REGIME_PATH, index_col="date", parse_dates=True)
regime.columns = ["regime"]

# Align both dataframes to the same dates
data = df.join(regime, how="inner").dropna()
print(f"Loaded {len(data)} observations\n")


# Build macro features with lags.
# The core question: do macro variables today predict the regime in N days?
# We test lags of 5, 10, and 20 trading days (1 week, 2 weeks, 1 month).
# If a lagged variable is predictive it means it leads regime transitions
# which is what a portfolio manager actually needs.

LAGS = [5, 10, 20]

feature_df = pd.DataFrame(index=data.index)

for lag in LAGS:
    feature_df[f"vix_lag{lag}"]         = data["VIX"].shift(lag)
    feature_df[f"hy_spread_lag{lag}"]   = data["EUR_HY_OAS"].shift(lag)
    feature_df[f"ig_spread_lag{lag}"]   = data["EUR_IG_OAS"].shift(lag)
    feature_df[f"yield_lag{lag}"]       = data["DE10Y"].shift(lag)
    feature_df[f"eurusd_lag{lag}"]      = data["EURUSD"].shift(lag)

feature_df = feature_df.dropna()
y_raw      = data["regime"].reindex(feature_df.index)

print(f"Feature matrix shape: {feature_df.shape}")
print(f"Regime distribution:\n{y_raw.value_counts().to_string()}\n")


# Encode regime labels as numbers for the classifier
regime_map     = {"risk-on": 0, "transition": 1, "stress": 2}
regime_map_inv = {v: k for k, v in regime_map.items()}
y              = y_raw.map(regime_map)


# Normalise features
scaler = StandardScaler()
X      = scaler.fit_transform(feature_df)


# Fit a logistic regression to predict regime from lagged macro variables.
# We use a simple model deliberately - interpretability matters more than
# accuracy here. We want to know which variables matter, not just predict well.
print("=== Logistic Regression: lagged macro -> regime ===\n")
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X, y)

y_pred = clf.predict(X)
print(classification_report(
    y, y_pred,
    target_names=["risk-on", "transition", "stress"]
))


# Extract feature importance from the coefficients.
# Each regime has its own set of coefficients - one per feature.
# A large positive coefficient means that variable strongly predicts that regime.
print("=== Most predictive lagged variables per regime ===\n")
coef_df = pd.DataFrame(
    clf.coef_,
    columns=feature_df.columns,
    index=["risk-on", "transition", "stress"]
)

for regime_name in ["risk-on", "transition", "stress"]:
    top = coef_df.loc[regime_name].abs().sort_values(ascending=False).head(3)
    print(f"{regime_name}:")
    for feat, val in top.items():
        direction = "+" if coef_df.loc[regime_name, feat] > 0 else "-"
        print(f"  {direction} {feat:<25} coefficient: {val:.3f}")
    print()


# Plot: which lag length is most predictive overall?
# We refit the model using only features from each lag and compare accuracy.
print("=== Accuracy by lag length ===\n")
from sklearn.metrics import accuracy_score

for lag in LAGS:
    lag_cols = [c for c in feature_df.columns if f"lag{lag}" in c]
    X_lag    = scaler.fit_transform(feature_df[lag_cols])
    clf_lag  = LogisticRegression(max_iter=1000, random_state=42)
    clf_lag.fit(X_lag, y)
    acc = accuracy_score(y, clf_lag.predict(X_lag))
    print(f"  Lag {lag:>2} days: accuracy = {acc:.3f}")


# Plot the regime probabilities over time.
# Rather than just the hard label, this shows how confident the model
# is in each regime on each day - more useful for a portfolio manager.
print("\nGenerating probability chart...")

probs   = clf.predict_proba(X)
prob_df = pd.DataFrame(probs, index=feature_df.index,
                       columns=["p_risk_on", "p_transition", "p_stress"])

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
fig.suptitle("Regime Probabilities from Lagged Macro Variables", fontsize=13)

ax1 = axes[0]
ax1.stackplot(
    prob_df.index,
    prob_df["p_risk_on"],
    prob_df["p_transition"],
    prob_df["p_stress"],
    labels=["risk-on", "transition", "stress"],
    colors=["#2ECC71", "#F39C12", "#E74C3C"],
    alpha=0.8
)
ax1.set_ylabel("Probability", fontsize=9)
ax1.set_title("Stacked regime probabilities", fontsize=10, loc="left")
ax1.legend(loc="upper right", fontsize=8)

ax2 = axes[1]
ax2.plot(data.index, data["EUR_HY_OAS"], color="black", linewidth=0.8)
ax2.set_ylabel("EUR HY OAS (bps)", fontsize=9)
ax2.set_title("Credit spread for reference", fontsize=10, loc="left")

plt.tight_layout()
chart_path = f"{CHART_DIR}/regime_probabilities.png"
plt.savefig(chart_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Chart saved to {chart_path}")

print("\nStep 4 done. Project complete.")