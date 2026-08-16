#!/usr/bin/env python3
"""Charts for post 01 — "Your GPU Is Not a Fast CPU".

Every number is parsed out of results/warp_lab.txt. Nothing is hand-entered,
so re-running the lab and re-running this script is the only way a figure
changes. Emits light and dark variants of each chart.

    python3 viz/charts/post01_charts.py
"""
import re
import sys
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "viz"))
from theme import THEMES, new_fig, style, title, si  # noqa: E402

SRC = ROOT / "results" / "warp_lab.txt"
OUT = ROOT / "viz" / "charts"

NUM = r"[-+]?\d[\d,]*\.?\d*"


def parse():
    """Pull the three tables out of the lab's stdout."""
    text = SRC.read_text()
    sec = {}
    cur = None
    for line in text.splitlines():
        m = re.match(r"\s*([1-4])\.\s+([A-Z ]+)", line)
        if m:
            cur = int(m.group(1))
            sec[cur] = []
            continue
        if cur is None:
            continue
        nums = re.findall(NUM, line.replace("%", " ").replace("x", " "))
        if nums and re.match(r"^\s+\d", line):
            sec[cur].append([float(n.replace(",", "")) for n in nums])
    return sec


def chart_crossover(rows, name):
    threads = [r[0] for r in rows]
    gflops = [r[3] for r in rows]
    for mode, t in THEMES.items():
        fig, ax = new_fig(t)
        style(ax, t, "threads doing the work (log)", "GFLOP/s (log)")
        title(ax, t, "One thread is slow. A million of them are not.",
              "Identical total work — 200M dependent FMAs — spread across more threads")
        ax.plot(threads, gflops, color=t["data"], lw=2, marker="o", ms=7,
                mfc=t["bg"], mew=2, zorder=3, clip_on=False)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(si))
        ax.yaxis.set_major_formatter(FuncFormatter(si))

        peak = max(range(len(gflops)), key=lambda i: gflops[i])
        ax.annotate(f"{gflops[peak]:,.0f} GFLOP/s\nat {si(threads[peak])} threads",
                    (threads[peak], gflops[peak]), textcoords="offset points",
                    xytext=(-10, -36), color=t["hi"], fontsize=9.5,
                    fontweight="bold", ha="right")
        ax.annotate(f"{gflops[0]:.1f} GFLOP/s\n1 thread", (threads[0], gflops[0]),
                    textcoords="offset points", xytext=(14, 8), color=t["mid"],
                    fontsize=9.5)
        fig.savefig(OUT / f"{name}-{mode}.png", facecolor=t["bg"])
        plt.close(fig)


def chart_divergence(rows, name):
    br = [r[0] for r in rows]
    slow = [r[2] for r in rows]
    for mode, t in THEMES.items():
        fig, ax = new_fig(t)
        style(ax, t, "distinct branches taken within one warp",
              "slowdown vs. no divergence")
        title(ax, t, "Every lane did the same work. The warp paid for all of it.",
              "32 lanes execute in lockstep, so the warp walks every branch its lanes take")
        # Reference line is recessive and directly labelled, not a peer series.
        ax.plot(br, br, color=t["soft"], lw=1.5, ls=(0, (5, 4)), zorder=2)
        ax.annotate("perfect serialisation (y = x)", (br[2], br[2]),
                    textcoords="offset points", xytext=(-6, 14),
                    color=t["soft"], fontsize=8.5, ha="right")
        ax.plot(br, slow, color=t["data"], lw=2, marker="o", ms=7, mfc=t["bg"],
                mew=2, zorder=3, clip_on=False)
        ax.annotate(f"{slow[-1]:.1f}x slower", (br[-1], slow[-1]),
                    textcoords="offset points", xytext=(-10, -30),
                    color=t["hi"], fontsize=10, fontweight="bold", ha="right")
        ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
        ax.set_xticks(br); ax.set_xticklabels([f"{int(b)}" for b in br])
        ax.set_yticks([1, 2, 4, 8, 16, 32])
        ax.set_yticklabels(["1x", "2x", "4x", "8x", "16x", "32x"])
        fig.savefig(OUT / f"{name}-{mode}.png", facecolor=t["bg"])
        plt.close(fig)


def chart_latency(rows, name):
    threads = [r[0] for r in rows]
    gbs = [r[3] for r in rows]
    peak = max(gbs)
    knee = next(i for i, v in enumerate(gbs) if v >= 0.98 * peak)
    for mode, t in THEMES.items():
        fig, ax = new_fig(t)
        style(ax, t, "threads resident on the GPU (log)", "achieved read bandwidth, GB/s")
        title(ax, t, "The memory system is only fast if enough warps are asking.",
              "1 GiB streaming read — the hardware does not get faster, it gets busier")
        ax.axhline(peak, color=t["soft"], lw=1.5, ls=(0, (5, 4)), zorder=2)
        ax.annotate(f"{peak:,.0f} GB/s ceiling", (threads[0], peak),
                    textcoords="offset points", xytext=(2, 8), color=t["soft"],
                    fontsize=8.5)
        ax.plot(threads, gbs, color=t["data"], lw=2, marker="o", ms=7,
                mfc=t["bg"], mew=2, zorder=3, clip_on=False)
        ax.set_xscale("log")
        ax.set_xlim(threads[0] * 0.6, threads[-1] * 1.6)
        ax.xaxis.set_major_formatter(FuncFormatter(si))
        ax.annotate(f"saturated at {si(threads[knee])} threads\n"
                    f"— only {100*threads[knee]/288768:.0f}% of what fits",
                    (threads[knee], gbs[knee]), textcoords="offset points",
                    xytext=(-6, -46), color=t["hi"], fontsize=9.5,
                    fontweight="bold", ha="right")
        ax.annotate(f"{gbs[0]:.0f} GB/s\n{int(threads[0])} threads",
                    (threads[0], gbs[0]), textcoords="offset points",
                    xytext=(6, 14), color=t["mid"], fontsize=9.5)
        fig.savefig(OUT / f"{name}-{mode}.png", facecolor=t["bg"])
        plt.close(fig)


if __name__ == "__main__":
    s = parse()
    chart_crossover(s[2], "01-throughput-crossover")
    chart_divergence(s[3], "01-warp-divergence")
    chart_latency(s[4], "01-latency-hiding")
    print("wrote:")
    for p in sorted(OUT.glob("01-*.png")):
        print(f"  {p.relative_to(ROOT)}  {p.stat().st_size//1024} KiB")
