#!/usr/bin/env python3
"""Charts for post 03 — "Arithmetic Intensity: Climbing the Matmul Ladder".

Parses the matmul section of results/RESULTS.txt. Nothing hand-entered.

    python3 viz/charts/post03_charts.py
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

# Reuse factor per rung, derived from the tile shape in the source, not
# measured: with a BM x BN tile, A is re-read N/BN times and B is re-read
# M/BM times, so each loaded value feeds ~0.5/(1/BM + 1/BN) FMAs.
REUSE = {
    "1 naive (uncoalesced)":     "every FMA reloads both operands",
    "2 coalesced":               "same reuse — the lanes just cooperate",
    "3 shared tiling 32x32":     "a 32x32 tile, loaded once per block",
    "4 1D reg tiling (TM=8)":    "8 outputs per thread, kept in registers",
    "5 2D reg tiling + float4":  "an 8x8 patch per thread",
    "6 cuBLAS SGEMM (fp32 ref)": "the same ideas, tuned harder",
    "7 WMMA 16x16x16 (fp16)":    "tensor cores, naive scheduling",
    "8 cuBLAS fp16 TC (ref)":    "tensor cores, tuned harder",
}


def parse():
    txt = SRC.read_text()
    block = txt.split("=============== matmul ===============")[1]
    block = block.split("\n===============")[0]
    gflop = float(re.search(r"([\d.]+) GFLOP per call", block).group(1))
    rows = []
    for line in block.splitlines():
        m = re.match(r"([1-8] .+?)\s{2,}([\d.]+)\s+([\d.]+)\s+([\d.]+)%", line)
        if m:
            rows.append((m.group(1).strip(), float(m.group(3))))
    return rows, gflop


def chart_ladder(rows, gflop):
    """Eight rungs, three orders of magnitude, identical arithmetic."""
    names = [n for n, _ in rows][::-1]
    vals = [v for _, v in rows][::-1]
    for mode, t in THEMES.items():
        fig, ax = plt.subplots(figsize=(7.8, 5.0), dpi=200)
        fig.subplots_adjust(top=0.775, left=0.335, right=0.955, bottom=0.125)
        ax.set_facecolor(t["bg"])
        fig.patch.set_facecolor(t["bg"])

        span = max(vals) / min(vals)
        fig.text(0.026, 0.925, f"Same matmul. Same {gflop:.0f} GFLOP. "
                               f"{span:.0f}x apart.",
                 color=t["ink"], fontsize=13.5, fontweight="bold", va="bottom")
        fig.text(0.026, 0.872, "Every rung buys the same thing: more uses of "
                               "each byte before it is thrown away.",
                 color=t["soft"], fontsize=9.3, va="bottom")
        fig.text(0.026, 0.812, "■ fp32", color=t["data"], fontsize=8.8,
                 fontweight="bold", va="bottom")
        fig.text(0.115, 0.812, "■ fp16 tensor cores", color=t["hi"],
                 fontsize=8.8, fontweight="bold", va="bottom")

        colors = [t["hi"] if n.startswith(("7", "8")) else t["data"]
                  for n in names]
        y = list(range(len(names)))
        ax.barh(y, vals, height=0.62, color=colors, zorder=3)
        for i, (n, v) in enumerate(zip(names, vals)):
            ax.text(v * 1.06, i, f"{v:,.1f}", color=t["ink"], fontsize=9,
                    va="center", fontweight="bold")

        ax.set_xscale("log")
        ax.set_xlim(0.4, 900)
        ax.set_xticks([1, 10, 100])
        ax.set_xticklabels(["1", "10", "100"])
        ax.set_yticks(y)
        ax.set_yticklabels([f"{n[2:]}\n{REUSE.get(n, '')}" for n in names],
                           fontsize=9)
        for lab in ax.get_yticklabels():
            lab.set_linespacing(1.6)
        ax.set_ylim(-0.6, len(names) - 0.35)
        ax.set_xlabel("TFLOP/s  (log scale)", color=t["mid"], fontsize=9.5,
                      labelpad=9)
        ax.tick_params(colors=t["soft"], labelsize=9, length=0)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(t["grid"])
        ax.grid(True, axis="x", color=t["grid"], lw=0.8, alpha=0.7)
        ax.set_axisbelow(True)
        fig.savefig(OUT / f"03-matmul-ladder-{mode}.png", facecolor=t["bg"])
        plt.close(fig)


if __name__ == "__main__":
    r, gflop = parse()
    print(f"  parsed {len(r)} kernels, {gflop} GFLOP/call")
    chart_ladder(r, gflop)
    for p in sorted(OUT.glob("03-*.png")):
        print(f"  wrote {p.relative_to(ROOT)}  {p.stat().st_size//1024} KiB")
