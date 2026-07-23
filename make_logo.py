"""
Generate the ceREEBerus logo from the dancing man Reeb graph.
Run from the repo root:  python make_logo.py
Outputs: doc_source/images/logo.png  (and logo_with_text.png)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "cereeberus"))

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from cereeberus.data import ex_reebgraphs as ex_rg

# ── Build the Reeb graph ──────────────────────────────────────────────────────
R = ex_rg.dancing_man(seed=5)

# ── Shared style ──────────────────────────────────────────────────────────────
NODE_SIZE   = 450
EDGE_COLOR  = "#777777"
EDGE_WIDTH  = 3.5
CMAP        = mpl.colormaps["viridis"].resampled(16)

def _draw_graph(ax, R):
    """Draw the Reeb graph onto ax, no axes/labels."""
    pos = R.pos_f

    # Edges (straight lines only — no loops in dancing_man)
    for u, v in R.edges():
        x = [pos[u][0], pos[v][0]]
        y = [pos[u][1], pos[v][1]]
        ax.plot(x, y, color=EDGE_COLOR, linewidth=EDGE_WIDTH, zorder=0, solid_capstyle="round")

    # Nodes coloured by function value
    fx_vals = np.array([pos[v][1] for v in R.nodes()])
    norm = mpl.colors.Normalize(vmin=fx_vals.min(), vmax=fx_vals.max())
    colors = [CMAP(norm(f)) for f in fx_vals]
    xs = [pos[v][0] for v in R.nodes()]
    ys = [pos[v][1] for v in R.nodes()]
    sc = ax.scatter(xs, ys, s=NODE_SIZE, c=colors, zorder=2, edgecolors="white", linewidths=1.5)
    sc.set_clip_on(False)

    ax.margins(0.15)
    ax.set_aspect("equal")
    ax.axis("off")


# ── Version 1: graph only ─────────────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(4, 4), facecolor="white")
_draw_graph(ax1, R)
fig1.tight_layout(pad=0.3)
out1 = os.path.join(os.path.dirname(__file__), "doc_source", "images", "logo.png")
fig1.savefig(out1, dpi=200, bbox_inches="tight", pad_inches=0.05, transparent=True)
from PIL import Image
img1 = Image.open(out1)
img1 = img1.crop(img1.getbbox())
img1.save(out1)
print(f"Saved: {out1}")

# ── Version 2: graph + package name ──────────────────────────────────────────
# Stacked layout: graph on top, text centered below
fig2 = plt.figure(figsize=(4, 5), facecolor="white")

# Top 70% of figure: graph
ax2 = fig2.add_axes([0.0, 0.28, 1.0, 0.72], frameon=False)
_draw_graph(ax2, R)

# Bottom strip: text axis
# Between the EE's
# ax_txt = fig2.add_axes([0.103, 0.0, 1.0, 1], frameon=False)
ax_txt = fig2.add_axes([0.27, 0.0, 1.0, 1], frameon=False)
ax_txt.axis("off")

GREEN = CMAP(0.3)
DARK  = "#333333"

segments = [
    ("ce",   DARK),
    ("REEB", GREEN),
    ("erus", DARK),
]

FONTSIZE = 38
FONTWEIGHT = "bold"

# Render once to get a renderer for measuring text widths
renderer = fig2.canvas.get_renderer()

# First pass: measure total width of all segments so we can centre them
fig2.draw(renderer)
widths = []
for text_str, color in segments:
    t = ax_txt.text(0, 0.5, text_str, ha="left", va="center",
                    fontsize=FONTSIZE, fontweight=FONTWEIGHT, color=color,
                    transform=ax_txt.transAxes)
    fig2.draw(renderer)
    bb = t.get_window_extent(renderer=renderer)
    ax_w = ax_txt.get_window_extent(renderer=renderer).width
    widths.append(bb.width / ax_w)
    t.remove()

total_width = sum(widths)
x_cursor = 0.5 - total_width / 2  # start so text is centred

# Second pass: place for real
for (text_str, color), w in zip(segments, widths):
    t = ax_txt.text(
        x_cursor, 0.5, text_str,
        ha="left", va="center",
        fontsize=FONTSIZE, fontweight=FONTWEIGHT,
        color=color,
        transform=ax_txt.transAxes,
    )
    t.set_path_effects([
        pe.withStroke(linewidth=1.5, foreground="#E5E5E5"),
        pe.Normal(),
    ])
    x_cursor += w

out2 = os.path.join(os.path.dirname(__file__), "doc_source", "images", "logo_with_text.png")
fig2.savefig(out2, dpi=200, bbox_inches="tight", pad_inches=0.05, transparent=True)

from PIL import Image
img = Image.open(out2)
img = img.crop(img.getbbox())
img.save(out2)
print(f"Saved: {out2}")

# plt.show()
