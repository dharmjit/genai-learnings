#!/usr/bin/env python3
"""Charts for post 05 — Data Parallelism."""
import re
from _common import THEMES, OUT, ROOT, section, frame, head, clean, plt


def parse():
    rows = []
    for line in section("dp_matmul").splitlines():
        m = re.match(r"\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+"
                     r"([\d.]+)x\s+(\d+)%", line)
        if m:
            g = [float(x) for x in m.groups()]
            rows.append(dict(batch=int(g[0]), one=g[1], two=g[2], nocomm=g[3],
                             comm=g[4], speedup=g[5], eff=g[6]))
    return rows


def chart_flat(rows):
    b = [r["batch"] for r in rows]
    for mode, t in THEMES.items():
        fig, ax = frame(t, 7.6, 4.5)
        head(fig, t, "The all-reduce does not care how big your batch is",
             "Two GPUs, one weight matrix. Compute grows; communication does not.")
        ax.plot(b, [r["nocomm"] for r in rows], color=t["data"], lw=2,
                marker="o", ms=7, mfc=t["bg"], mew=2, zorder=3, clip_on=False)
        ax.plot(b, [r["comm"] for r in rows], color=t["hi"], lw=2, marker="o",
                ms=7, mfc=t["bg"], mew=2, zorder=3, clip_on=False)
        ax.annotate("compute — scales with the batch",
                    (b[-1], rows[-1]["nocomm"]), textcoords="offset points",
                    xytext=(-8, 12), color=t["data"], fontsize=8.8,
                    fontweight="bold", ha="right")
        ax.annotate("all-reduce — flat, it moves the WEIGHTS",
                    (b[-1], rows[-1]["comm"]), textcoords="offset points",
                    xytext=(-8, -22), color=t["hi"], fontsize=8.8,
                    fontweight="bold", ha="right")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xticks(b); ax.set_xticklabels([f"{x:,}" for x in b])
        clean(ax, t, "batch size (log)", "milliseconds per step (log)")
        fig.savefig(OUT / f"05-flat-comm-{mode}.png", facecolor=t["bg"])
        plt.close(fig)


def chart_eff(rows):
    b = [r["batch"] for r in rows]
    e = [r["eff"] for r in rows]
    for mode, t in THEMES.items():
        fig, ax = frame(t, 7.6, 3.9, top=0.72)
        head(fig, t, "Scaling efficiency is a batch-size problem",
             "The same fixed comm cost, amortised over more and more work.",
             y=0.905, ysub=0.838)
        ax.axhline(100, color=t["soft"], lw=1.4, ls=(0, (5, 4)), zorder=2)
        ax.text(b[0], 103, "perfect 2-GPU scaling", color=t["soft"],
                fontsize=8.4, va="bottom")
        ax.plot(b, e, color=t["data"], lw=2, marker="o", ms=7, mfc=t["bg"],
                mew=2, zorder=3, clip_on=False)
        for x, y in zip(b, e):
            ax.annotate(f"{y:.0f}%", (x, y), textcoords="offset points",
                        xytext=(0, -20), ha="center", color=t["mid"],
                        fontsize=8.8, fontweight="bold")
        ax.set_xscale("log")
        ax.set_xticks(b); ax.set_xticklabels([f"{x:,}" for x in b])
        ax.set_ylim(0, 125)
        clean(ax, t, "batch size (log)", "efficiency vs 1 GPU")
        fig.savefig(OUT / f"05-efficiency-{mode}.png", facecolor=t["bg"])
        plt.close(fig)


if __name__ == "__main__":
    r = parse(); print(f"  parsed {len(r)} batches")
    chart_flat(r); chart_eff(r)
    for p in sorted(OUT.glob("05-*.png")):
        print(f"  wrote {p.relative_to(ROOT)}")
