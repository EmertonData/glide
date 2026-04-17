"""
Reliable Metric Estimation with PTD
=========================================

PTD combines a small set of expensive true evaluation labels with a large pool
of cheap proxy evaluation labels to produce a statistically valid, bias-corrected
quality metric.
"""

##############################################################################
# Create a simulated dataset with 200 human-labeled and 2000 LLM-judge-labeled
# conversations, with a known ground-truth hallucination rate of 10%.

import matplotlib.pyplot as plt
import numpy as np

from glide.core.simulated_datasets import generate_binary_dataset
from glide.estimators import ClassicalMeanEstimator, PTDMeanEstimator

C_JUDGE = "#E74C3C"
C_HUMAN = "#2E86AB"
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
    n=200,
    N=2000,
    true_mean=0.10,
    proxy_mean=0.05,
    correlation=0.65,
    random_seed=42,
)

##############################################################################
# The LLM judge under-reports hallucinations on the labeled subset compared to
# human annotations, revealing a systematic bias.

labeled_mask = ~np.isnan(y_true)
p_mean = np.mean(y_proxy[labeled_mask])
t_mean = np.mean(y_true[labeled_mask])
bias = p_mean - t_mean

print(f"LLM judge mean (annotated subset): {p_mean:.1%}")
print(f"Human annotation mean:             {t_mean:.1%}")
print(f"Proxy bias:                        {bias:+.1%}")

##############################################################################
# Compute baseline estimates using a classical mean estimator on each source
# independently, then compute the PTD estimate which corrects for proxy bias.

TRUE_RATE = 0.10

judge_estimate = ClassicalMeanEstimator().estimate(y_proxy)
human_estimate = ClassicalMeanEstimator().estimate(y_true)
ptd_result = PTDMeanEstimator().estimate(y_true, y_proxy, metric_name="Hallucination Rate")

##############################################################################
# Plot the three estimates side by side.

estimates = [
    (
        f"LLM Judge\n(N={ptd_result.n_proxy}  |  raw proxy)",
        judge_estimate.mean,
        judge_estimate.confidence_interval.lower_bound,
        judge_estimate.confidence_interval.upper_bound,
        C_JUDGE,
    ),
    (
        f"Human Annotation\n(n={ptd_result.n_true}  |  small sample)",
        human_estimate.mean,
        human_estimate.confidence_interval.lower_bound,
        human_estimate.confidence_interval.upper_bound,
        C_HUMAN,
    ),
    (
        f"PTD (GLIDE)\n(n={ptd_result.n_true}  +  N={ptd_result.n_proxy})\n(full data exploited)",
        ptd_result.mean,
        ptd_result.confidence_interval.lower_bound,
        ptd_result.confidence_interval.upper_bound,
        C_GLIDE,
    ),
]

fig, ax = plt.subplots(figsize=(11, 5.5))
y_pos = [2, 1, 0]

for y, (label, mean, lo, hi, color) in zip(y_pos, estimates):
    ax.plot([lo, hi], [y, y], color=color, linewidth=4, solid_capstyle="round", zorder=3)
    for xc in [lo, hi]:
        ax.plot([xc, xc], [y - 0.2, y + 0.2], color=color, linewidth=2.5, zorder=3)
    ax.scatter(mean, y, s=200, color=color, zorder=5, edgecolors="white", linewidths=2.5)
    ax.text(mean, y + 0.34, f"{mean:.1%}", ha="center", va="bottom", fontsize=12, color=color, fontweight="bold")
    ax.text(mean, y - 0.34, f"[{lo:.1%},  {hi:.1%}]", ha="center", va="top", fontsize=11, color="#888888")

ax.axvline(TRUE_RATE, color=C_TRUTH, linestyle="--", linewidth=2.5, zorder=4)
ax.text(TRUE_RATE + 0.004, 2.72, "True rate  10%", color=C_TRUTH, fontsize=10.5, fontweight="bold")
ax.set_yticks(y_pos)
ax.set_yticklabels([e[0] for e in estimates], fontsize=11)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_xlabel("Estimated Hallucination Rate", fontsize=12)
ax.set_title("GLIDE PTD Delivers an Unbiased, Precise Estimate", fontsize=14, fontweight="bold")
ax.set_xlim(-0.01, 0.26)
ax.set_ylim(-0.8, 3.2)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.tick_params(left=False)
plt.tight_layout()
plt.show()
