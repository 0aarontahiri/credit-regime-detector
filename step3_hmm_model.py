import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from sklearn.mixture import GaussianMixture

DATA_PATH = "data/raw_data.csv"
FEAT_PATH = "data/features_scaled.csv"
CHART_DIR = "outputs"
os.makedirs(CHART_DIR, exist_ok=True)

df       = pd.read_csv(DATA_PATH, index_col="date", parse_dates=True)
features = pd.read_csv(FEAT_PATH, index_col="date", parse_dates=True)
X        = features.values

print(f"Loaded {X.shape[0]} observations, {X.shape[1]} features\n")

# We use a Gaussian Mixture Model as a simpler but equally valid
# approach to regime detection. A GMM clusters observations into
# k groups based on their statistical similarity. Each cluster
# is a regime. This is more stable than HMM on smaller datasets
# and gives interpretable, well-separated regimes.
gmm = GaussianMixture(
    n_components=3,
    covariance_type="full",
    n_init=20,          # run 20 times with different seeds, keep best
    random_state=42
)
gmm.fit(X)
states = gmm.predict(X)

print(f"Model converged: {gmm.converged_}")
print(f"Log-likelihood:  {gmm.lower_bound_:.2f}\n")

# Label each state by its VIX mean
# Lowest VIX mean = risk-on, highest = stress, middle = transition
state_means = pd.DataFrame(gmm.means_, columns=features.columns)
print("=== State profiles ===")
print(state_means.round(4).to_string())

# Label by spread_level magnitude - large positive moves = stress
# We use absolute spread level since stress means big moves either way
# Higher spread level = more stress, so we reverse the order
spread_order = state_means["spread_level"].argsort().values
labels = {}
labels[spread_order[0]] = "stress"
labels[spread_order[1]] = "transition"
labels[spread_order[2]] = "risk-on"

print("\n=== State labels ===")
for state, label in labels.items():
    mean_spread = state_means.loc[state, "spread_level"]
    mean_vix    = state_means.loc[state, "vix"]
    print(f"  State {state} -> {label}  (spread_level={mean_spread:.3f}, vix={mean_vix:.3f})")

state_labels = pd.Series([labels[s] for s in states], index=features.index)

print("\n=== Regime distribution ===")
counts = state_labels.value_counts()
for label, count in counts.items():
    pct = count / len(state_labels) * 100
    print(f"  {label:<12} {count} days ({pct:.1f}%)")

print("\nGenerating regime chart...")

colours = {
    "risk-on":    "#2ECC71",
    "transition": "#F39C12",
    "stress":     "#E74C3C"
}

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
fig.suptitle("European Credit Regimes - GMM Output", fontsize=13)

ax1 = axes[0]
spread = df["EUR_HY_OAS"].reindex(features.index)
for i in range(len(features.index) - 1):
    colour = colours[state_labels.iloc[i]]
    ax1.axvspan(features.index[i], features.index[i+1],
                alpha=0.3, color=colour, linewidth=0)
ax1.plot(spread.index, spread.values, color="black", linewidth=0.8)
ax1.set_ylabel("EUR HY OAS (bps)", fontsize=9)
ax1.set_title("Credit spread coloured by regime", fontsize=10, loc="left")
patches = [mpatches.Patch(color=c, label=l) for l, c in colours.items()]
ax1.legend(handles=patches, fontsize=8, loc="upper right")

ax2 = axes[1]
vix = df["VIX"].reindex(features.index)
for i in range(len(features.index) - 1):
    colour = colours[state_labels.iloc[i]]
    ax2.axvspan(features.index[i], features.index[i+1],
                alpha=0.3, color=colour, linewidth=0)
ax2.plot(vix.index, vix.values, color="black", linewidth=0.8)
ax2.set_ylabel("VIX", fontsize=9)
ax2.set_title("VIX coloured by regime", fontsize=10, loc="left")

ax3 = axes[2]
regime_num = state_labels.map({"risk-on": 0, "transition": 1, "stress": 2})
ax3.fill_between(features.index, regime_num,
                 step="post", alpha=0.8,
                 color=[colours[l] for l in state_labels])
ax3.set_yticks([0, 1, 2])
ax3.set_yticklabels(["risk-on", "transition", "stress"], fontsize=8)
ax3.set_ylabel("Regime", fontsize=9)
ax3.set_title("Regime over time", fontsize=10, loc="left")

plt.tight_layout()
chart_path = f"{CHART_DIR}/regimes.png"
plt.savefig(chart_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Chart saved to {chart_path}")

state_labels.to_csv("data/regime_labels.csv", header=True)
print("Regime labels saved to data/regime_labels.csv")

print("\nStep 3 done. Run step4_macro_linkage.py next.")