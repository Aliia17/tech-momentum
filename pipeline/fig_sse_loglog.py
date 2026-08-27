"""Figure — SSE vs K on log-log axes (reviewer point 3, made visible).

Pools the three bge-large SSE grids (K = 50..2000, same embedding space),
fits the log-log slope, and draws the picture that would have prevented
the 'elbow' claim: a straight line on log-log axes = scale-free power law
= no data-privileged K.

Output: results/sse_loglog.png
Run:    python pipeline/fig_sse_loglog.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

BLUE = "#2563EB"      # single series -> one hue; text stays in neutral ink
INK = "#1F2937"
MUTED = "#6B7280"
GRID = "#E5E7EB"

files = ["kmeans_sse_large.csv", "kmeans_sse_large_k250.csv",
         "kmeans_sse_large_k1000.csv"]
sse = (pd.concat([pd.read_csv(config.RESULTS / f) for f in files])
       .drop_duplicates("k").sort_values("k"))

logk, logs = np.log(sse["k"]), np.log(sse["sse"])
slope, intercept = np.polyfit(logk, logs, 1)
fit = np.exp(intercept) * sse["k"] ** slope

fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
ax.loglog(sse["k"], fit, "--", color=MUTED, lw=1.2, zorder=1,
          label=f"power-law fit, slope {slope:.3f}")
ax.loglog(sse["k"], sse["sse"], "-", color=BLUE, lw=2, zorder=2)
ax.scatter(sse["k"], sse["sse"], s=42, color=BLUE, zorder=3)

k500 = sse[sse["k"] == 500]
ax.scatter(k500["k"], k500["sse"], s=90, facecolor="white",
           edgecolor=INK, lw=1.6, zorder=4)
ax.annotate("K = 500\n(paper convention,\nnot an optimum)",
            xy=(500, float(k500["sse"].iloc[0])), xytext=(620, 34300),
            fontsize=9, color=INK,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
ax.annotate(f"straight line on log-log axes:\nSSE ≈ c·K^{slope:.2f} "
            "everywhere — no elbow at any scale",
            xy=(0.03, 0.10), xycoords="axes fraction",
            fontsize=9.5, color=MUTED)

ax.set_xlabel("K — number of clusters (log scale)", fontsize=10, color=INK)
ax.set_ylabel("K-means SSE (log scale)", fontsize=10, color=INK)
ax.set_title("Cluster-count diagnostic: SSE follows a scale-free power law",
             fontsize=11.5, color=INK, loc="left", pad=12)
ax.grid(True, which="both", color=GRID, lw=0.6, alpha=0.7)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(MUTED)
ax.tick_params(colors=MUTED, labelsize=9)
ax.legend(frameon=False, fontsize=9, loc="upper right")

out = config.RESULTS / "sse_loglog.png"
fig.tight_layout()
fig.savefig(out, facecolor="white")
print(f"saved {out}")
print(f"pooled K range: {sse['k'].min()}..{sse['k'].max()} | "
      f"log-log slope {slope:.4f} | "
      f"SSE drop 500->2000: "
      f"{1 - sse.loc[sse['k'] == 2000, 'sse'].iloc[0] / k500['sse'].iloc[0]:.1%}")
