#!/usr/bin/env python3
"""Charts for post 04 — "Two Cards, One Wire".

Parses the p2p_bandwidth and overlap_streams sections of results/RESULTS.txt.

    python3 viz/charts/post04_charts.py
"""
import re
import sys
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "viz"))
from theme import THEMES  # noqa: E402

SRC = ROOT / "results" / "RESULTS.txt"
OUT = ROOT / "viz" / "charts"


def section(name):
    txt = SRC.read_text()
    return txt.split(f"=============== {name} ===============")[1] \
              .split("\n===============")[0]


def parse_rows(block):
    rows = []
    for line in block.splitlines():
        m = re.match(r"(.+?)\s{2,}([\d.]+)\s+([\d.]+)\s+([\d.]+)%\s+\w+\s*$", line)
        if m:
            rows.append((m.group(1).strip(), float(m.group(3)), float(m.group(4))))
    return rows


def chart_ladder():
    """One picture of why multi-GPU is hard: the wire is the slow part."""
    rows = parse_rows(section("p2p_bandwidth"))[::-1]
    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    local = max(vals)
    for mode, t in THEMES.items():
        fig, ax = plt.subplots(figsize=(7.8, 4.6), dpi=200)
        fig.subplots_adjust(top=0.775, left=0.315, right=0.955, bottom=0.135)
        ax.set_facecolor(t["bg"])
        fig.patch.set_facecolor(t["bg"])
        fig.text(0.026, 0.925, "Everything is slower than local memory",
                 color=t["ink"], fontsize=13.5, fontweight="bold", va="bottom")
        fig.text(0.026, 0.872, "256 MiB moved each way. The gap between the "
                               "top bar and the rest is the whole problem.",
                 color=t["soft"], fontsize=9.3, va="bottom")

        colors = [t["data"] if "D2D" in n else t["hi"] for n in names]
        y = list(range(len(names)))
        ax.barh(y, vals, height=0.6, color=colors, zorder=3)
        for i, v in enumerate(vals):
            ax.text(v * 1.05, i, f"{v:,.0f}", color=t["ink"], fontsize=9,
                    va="center", fontweight="bold")
            if vals[i] != local:
                ax.text(v * 1.05, i - 0.32, f"{local/v:.0f}x slower than local",
                        color=t["soft"], fontsize=7.4, va="center")

        ax.set_xscale("log")
        ax.set_xlim(15, 3000)
        ax.set_xticks([20, 50, 100, 200, 500, 1000])
        ax.set_xticklabels(["20", "50", "100", "200", "500", "1000"])
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=9)
        ax.set_ylim(-0.6, len(names) - 0.4)
        ax.set_xlabel("GB/s  (log scale)", color=t["mid"], fontsize=9.5,
                      labelpad=9)
        ax.tick_params(colors=t["soft"], labelsize=9, length=0)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(t["grid"])
        ax.grid(True, axis="x", color=t["grid"], lw=0.8, alpha=0.7)
        ax.set_axisbelow(True)
        fig.savefig(OUT / f"04-bandwidth-ladder-{mode}.png", facecolor=t["bg"])
        plt.close(fig)


def chart_overlap():
    """Pipelining recovers most of what the serial version wastes."""
    rows = parse_rows(section("overlap_streams"))
    names = [r[0] for r in rows]
    speed = [r[2] / 100.0 for r in rows]
    for mode, t in THEMES.items():
        fig, ax = plt.subplots(figsize=(7.8, 4.0), dpi=200)
        fig.subplots_adjust(top=0.735, left=0.285, right=0.955, bottom=0.16)
        ax.set_facecolor(t["bg"])
        fig.patch.set_facecolor(t["bg"])
        fig.text(0.026, 0.915, "Copy and compute at the same time, or wait twice",
                 color=t["ink"], fontsize=13.5, fontweight="bold", va="bottom")
        fig.text(0.026, 0.855, "512 MiB through H2D → compute → D2H. Identical "
                               "work in every row.",
                 color=t["soft"], fontsize=9.3, va="bottom")

        y = list(range(len(names)))[::-1]
        colors = [t["hi"] if "2 GPUs" in n else t["data"] for n in names]
        ax.barh(y, speed, height=0.58, color=colors, zorder=3)
        ax.axvline(1.0, color=t["soft"], lw=1.4, ls=(0, (5, 4)), zorder=2)
        for i, v in zip(y, speed):
            ax.text(v + 0.03, i, f"{v:.2f}x", color=t["ink"], fontsize=9,
                    va="center", fontweight="bold")
        ax.set_yticks(y)
        ax.set_yticklabels([n[2:] if n[1] == " " else n for n in names],
                           fontsize=9)
        ax.set_xlim(0, 2.6)
        ax.set_xlabel("speedup over the serial version", color=t["mid"],
                      fontsize=9.5, labelpad=9)
        ax.tick_params(colors=t["soft"], labelsize=9, length=0)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(t["grid"])
        ax.grid(True, axis="x", color=t["grid"], lw=0.8, alpha=0.7)
        ax.set_axisbelow(True)
        fig.savefig(OUT / f"04-stream-overlap-{mode}.png", facecolor=t["bg"])
        plt.close(fig)


if __name__ == "__main__":
    chart_ladder()
    chart_overlap()
    for p in sorted(OUT.glob("04-*.png")):
        print(f"  wrote {p.relative_to(ROOT)}  {p.stat().st_size//1024} KiB")
