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

import plotly.graph_objects as go
import plotly.io as pio

from glide.core.simulated_datasets import generate_binary_dataset
from glide.estimators import PTDMeanEstimator

pio.renderers.default = "sphinx_gallery"


C_GLIDE = "#27AE60"
C_TRUTH = "#2C3E50"

TRUE_RATE = 0.10

y_true, y_proxy = generate_binary_dataset(
    n=200,  # human labeled samples
    N=2000,  # samples labeled only by the LLM-judge
    true_mean=TRUE_RATE,
    proxy_mean=0.05,
    correlation=0.65,
    random_seed=42,
)

##############################################################################
# Compute the PTD estimate, which corrects for proxy bias by combining both
# label sources.

ptd_result = PTDMeanEstimator().estimate(y_true, y_proxy, metric_name="Hallucination Rate")

##############################################################################
# Plot the PTD estimate with its confidence interval next to the true rate.

mean = ptd_result.mean
lo = ptd_result.confidence_interval.lower_bound
hi = ptd_result.confidence_interval.upper_bound
ptd_label = "PTD (GLIDE)"

fig = go.Figure()

fig.add_trace(
    go.Bar(
        x=[ptd_label, "True Rate"],
        y=[mean, TRUE_RATE],
        marker_color=[C_GLIDE, C_TRUTH],
        width=0.4,
        error_y=dict(
            type="data",
            symmetric=False,
            array=[hi - mean, 0],
            arrayminus=[mean - lo, 0],
            color="black",
            thickness=2,
            width=8,
        ),
        text=[f"{mean:.1%}<br>[{lo:.1%}, {hi:.1%}]", ""],
        textposition="outside",
        textfont=dict(size=13, color=C_GLIDE),
        showlegend=False,
    )
)

fig.add_shape(
    type="line",
    x0=-0.5,
    x1=1.5,
    y0=TRUE_RATE,
    y1=TRUE_RATE,
    line=dict(color=C_TRUTH, width=2, dash="dash"),
)

fig.update_layout(
    title=dict(text="PTD Delivers an Unbiased, Precise Estimate", font=dict(size=14)),
    yaxis=dict(
        title="Hallucination Rate",
        tickformat=".0%",
        range=[0, max(hi, TRUE_RATE) * 1.6],
        gridcolor="#E5E5E5",
    ),
    plot_bgcolor="#FAFAFA",
    paper_bgcolor="white",
    width=600,
    height=400,
)

fig
