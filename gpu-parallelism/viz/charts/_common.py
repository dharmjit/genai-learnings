"""Shared parsing + framing for the multi-GPU posts (05-07)."""
import re
import sys
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "viz"))
from theme import THEMES  # noqa: E402,F401

SRC = ROOT / "results" / "RESULTS.txt"
OUT = ROOT / "viz" / "charts"


def section(name):
    return SRC.read_text().split(f"=============== {name} ===============")[1] \
              .split("\n===============")[0]


def frame(t, w, h, top=0.755, left=0.13, right=0.955, bottom=0.145):
    fig, ax = plt.subplots(figsize=(w, h), dpi=200)
    fig.subplots_adjust(top=top, left=left, right=right, bottom=bottom)
    ax.set_facecolor(t["bg"])
    fig.patch.set_facecolor(t["bg"])
    return fig, ax


def head(fig, t, title, sub, y=0.925, ysub=0.868, x=0.026):
    fig.text(x, y, title, color=t["ink"], fontsize=13.5, fontweight="bold",
             va="bottom")
    fig.text(x, ysub, sub, color=t["soft"], fontsize=9.3, va="bottom")


def clean(ax, t, xlabel, ylabel, axis="both"):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(t["grid"])
    ax.grid(True, axis=axis, color=t["grid"], lw=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors=t["soft"], labelsize=9, length=0)
    ax.set_xlabel(xlabel, color=t["mid"], fontsize=9.5, labelpad=9)
    ax.set_ylabel(ylabel, color=t["mid"], fontsize=9.5, labelpad=9)
