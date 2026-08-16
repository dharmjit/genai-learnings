#!/usr/bin/env python3
"""Charts for post 02 — "The Memory Wall".

Parses results/RESULTS.txt (the transpose section). Nothing hand-entered.

    python3 viz/charts/post02_charts.py
"""
import re
import sys
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "viz"))
from theme import THEMES, new_fig, title  # noqa: E402

SRC = ROOT / "results" / "RESULTS.txt"
OUT = ROOT / "viz" / "charts"


def parse():
    """Both transpose regimes: {label: [(kernel, GB/s, pct_of_copy), ...]}."""
    txt = SRC.read_text()
    # split on the FULL banner: splitting on the bare rule re-matches the
    # header's own trailing "===============" and yields an empty block.
    block = txt.split("=============== transpose ===============")[1]
    block = block.split("\n===============")[0]
    out, cur = {}, None
    for line in block.splitlines():
        m = re.match(r"transpose (\d+) x \d+\s+\((.*?),\s*(.*?)\)", line.strip())
        if m:
            cur = f"{m.group(1)}²  ({m.group(3).strip()})"
            out[cur] = []
            continue
        m = re.match(r"(.+?)\s{2,}([\d.]+)\s+([\d.]+)\s+([\d.]+)%\s+\w+\s*$", line)
        if m and cur:
            out[cur].append((m.group(1).strip(), float(m.group(3)),
                             float(m.group(4))))
    return out


def chart_regimes(data):
    """The post's whole argument in one figure: identical kernels, two
    working-set sizes, and an optimisation worth 1.6x in one and nothing in
    the other."""
    keys = list(data)
    # "exceeds L2 -> DRAM bound" also contains "L2", so match on DRAM and take
    # the other key. Matching both on "L2" plotted the same series twice.
    dram = [k for k in keys if "DRAM" in k][0]
    l2 = [k for k in keys if k != dram][0]

    names = [n for n, _, _ in data[dram]]
    keep = [i for i, n in enumerate(names) if "cublas" not in n][::-1]
    labels = [names[i].replace(" (speed of light, no T)", "") for i in keep]
    a = [data[dram][i][2] for i in keep]
    b = [data[l2][i][2] for i in keep]

    for mode, t in THEMES.items():
        fig, ax = plt.subplots(figsize=(7.6, 4.7), dpi=200)
        fig.subplots_adjust(top=0.755, left=0.29, right=0.955, bottom=0.135)
        ax.set_facecolor(t["bg"])
        fig.patch.set_facecolor(t["bg"])

        # figure coords so a long title is not boxed in by the axes
        fig.text(0.028, 0.925, "The same optimisation, worth 1.6x "
                               "\u2014 or nothing at all",
                 color=t["ink"], fontsize=13.5, fontweight="bold", va="bottom")
        fig.text(0.028, 0.872, "Identical kernels, identical bytes moved. "
                               "Only the working-set size changes.",
                 color=t["soft"], fontsize=9.3, va="bottom")

        fig.text(0.028, 0.795, "\u25a0 8192\u00b2 · exceeds L2",
                 color=t["data"], fontsize=8.8, fontweight="bold", va="bottom")
        fig.text(0.245, 0.795, "\u25a0 2048\u00b2 · fits in the 128 MiB L2",
                 color=t["hi"], fontsize=8.8, fontweight="bold", va="bottom")

        y = list(range(len(labels)))
        h = 0.34
        ax.barh([i + h / 2 for i in y], a, height=h, color=t["data"], zorder=3)
        ax.barh([i - h / 2 for i in y], b, height=h, color=t["hi"], zorder=3)
        ax.axvline(100, color=t["soft"], lw=1.4, ls=(0, (5, 4)), zorder=2)
        ax.text(99, -0.55, "speed of light", color=t["soft"], fontsize=8.2,
                ha="right", va="center")

        for i, (va, vb) in enumerate(zip(a, b)):
            ax.text(va + 1.5, i + h / 2, f"{va:.0f}%", color=t["mid"],
                    fontsize=8.6, va="center")
            ax.text(vb + 1.5, i - h / 2, f"{vb:.0f}%", color=t["mid"],
                    fontsize=8.6, va="center")


        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_ylim(-0.7, len(labels) - 0.25)
        ax.set_xlim(0, 114)
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.set_xlabel("percent of a plain copy (same bytes, no transpose)",
                      color=t["mid"], fontsize=9.5, labelpad=9)
        ax.tick_params(colors=t["soft"], labelsize=9, length=0)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(t["grid"])
        ax.grid(True, axis="x", color=t["grid"], lw=0.8, alpha=0.7)
        ax.set_axisbelow(True)
        fig.savefig(OUT / f"02-two-regimes-{mode}.png", facecolor=t["bg"])
        plt.close(fig)


if __name__ == "__main__":
    d = parse()
    for k, v in d.items():
        print(f"  parsed {k}: {len(v)} kernels")
    chart_regimes(d)
    for p in sorted(OUT.glob("02-*.png")):
        print(f"  wrote {p.relative_to(ROOT)}  {p.stat().st_size//1024} KiB")
