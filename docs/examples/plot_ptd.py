"""
PTDMeanEstimator
=========================================

PTD combines a small set of expensive true evaluation labels with a large pool
of cheap proxy evaluation labels to produce a statistically valid, bias-corrected
quality metric.
"""

##############################################################################
# Create a simulated dataset with 2200 conversations, all of which are labeled
# by an LLM-judge and a subset of 200 samples are human-labeled. The
# ground-truth hallucination rate is 10%.

import matplotlib.pyplot as plt

from glide.core.simulated_datasets import generate_binary_dataset
from glide.estimators import PTDMeanEstimator

C_GLIDE = "#27AE60"
C_TRUTH = "#2C3E50"

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "#FAFAFA",
        "axes.grid": True,
        "grid.color": "#E5E5E5",
        "grid.linewidth": 0.8,
        "axes.titlesize": 14,
        "axes.titlepad": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    }
)

y_true, y_proxy = generate_binary_dataset(
    n=200,  # human labeled samples
    N=2000,  # samples labeled only by the LLM-judge
    true_mean=0.10,
    proxy_mean=0.05,
    correlation=0.65,
    random_seed=42,
)

##############################################################################
# Compute the PTD estimate, which corrects for proxy bias by combining both
# label sources.

TRUE_RATE = 0.10

ptd_result = PTDMeanEstimator().estimate(y_true, y_proxy, metric_name="Hallucination Rate")

##############################################################################
# Plot the PTD estimate with its confidence interval.

label = f"PTD (GLIDE)\n({ptd_result.n_true}  +  {ptd_result.n_proxy})"
mean = ptd_result.mean
lo = ptd_result.confidence_interval.lower_bound
hi = ptd_result.confidence_interval.upper_bound

fig, ax = plt.subplots(figsize=(9, 3))

ax.plot([lo, hi], [0, 0], color=C_GLIDE, linewidth=4, solid_capstyle="round", zorder=3)
for xc in [lo, hi]:
    ax.plot([xc, xc], [-0.2, 0.2], color=C_GLIDE, linewidth=2.5, zorder=3)
ax.scatter(mean, 0, s=200, color=C_GLIDE, zorder=5, edgecolors="white", linewidths=2.5)
ax.text(mean, 0.34, f"{mean:.1%}", ha="center", va="bottom", fontsize=12, color=C_GLIDE, fontweight="bold")
ax.text(mean, -0.34, f"[{lo:.1%},  {hi:.1%}]", ha="center", va="top", fontsize=11, color="#888888")

ax.axvline(TRUE_RATE, color=C_TRUTH, linestyle="--", linewidth=2.5, zorder=4)
ax.text(TRUE_RATE + 0.004, 0.52, "True rate  10%", color=C_TRUTH, fontsize=10.5, fontweight="bold")
ax.set_yticks([0])
ax.set_yticklabels([label], fontsize=11)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_xlabel("Estimated Hallucination Rate", fontsize=12)
ax.set_title("PTD Delivers an Unbiased, Precise Estimate", fontsize=14, fontweight="bold")
ax.set_xlim(-0.01, 0.26)
ax.set_ylim(-0.8, 0.9)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.tick_params(left=False)
plt.tight_layout()
plt.show()
