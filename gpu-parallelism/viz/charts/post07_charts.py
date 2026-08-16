#!/usr/bin/env python3
"""Charts for post 07 — Pipeline parallelism and the strategy shootout."""
import re
from _common import THEMES, OUT, ROOT, section, frame, head, clean, plt


def parse_bubble():
    rows = []
    for line in section("pp_matmul").splitlines():
        m = re.match(r"\s+(\d+)\s+([\d.]+)\s+([\d.]+)x\s+(\d+)%\s+(\d+)%\s+(\d+)%",
                     line)
        if m:
            g = m.groups()
            rows.append(dict(m=int(g[0]), ms=float(g[1]), speedup=float(g[2]),
                             meas=int(g[4]), pred=int(g[5])))
    return rows


def parse_final():
    rows = []
    for line in section("mlp_2gpu").splitlines():
        m = re.match(r"(.+?)\s{2,}([\d.]+)\s+([\d.]+)x\s+(\d+)\s+(\d+)\s+\w+", line)
        if m:
            rows.append(dict(name=m.group(1).strip(), ms=float(m.group(2)),
                             speedup=float(m.group(3)), comm=int(m.group(4)),
                             mem=int(m.group(5))))
    return rows


def chart_bubble(rows):
    x = [r["m"] for r in rows]
    for mode, t in THEMES.items():
        fig, ax = frame(t, 7.6, 4.4)
        head(fig, t, "Microbatching fills the pipe — until it doesn't",
             "Theory says the bubble is 1/(m+1). Reality agrees, then stops agreeing.")
        ax.plot(x, [r["pred"] for r in rows], color=t["soft"], lw=1.6,
                ls=(0, (5, 4)), marker="s", ms=5, mfc=t["bg"], zorder=2)
        ax.plot(x, [r["meas"] for r in rows], color=t["data"], lw=2, marker="o",
                ms=7, mfc=t["bg"], mew=2, zorder=3, clip_on=False)
        ax.annotate("predicted  (P-1)/(m+P-1)", (x[-1], rows[-1]["pred"]),
                    textcoords="offset points", xytext=(-6, -20),
                    color=t["soft"], fontsize=8.6, ha="right")
        ax.annotate("measured", (x[2], rows[2]["meas"]),
                    textcoords="offset points", xytext=(10, 8),
                    color=t["data"], fontsize=8.8, fontweight="bold")
        best = min(range(len(rows)), key=lambda i: rows[i]["ms"])
        ax.annotate(f"best: {rows[best]['speedup']:.2f}x at m={rows[best]['m']}",
                    (x[best], rows[best]["meas"]), textcoords="offset points",
                    xytext=(-4, -30), color=t["hi"], fontsize=9,
                    fontweight="bold", ha="center")
        ax.annotate("microbatches too small\nto fill 188 SMs",
                    (x[-1], rows[-1]["meas"]), textcoords="offset points",
                    xytext=(-6, 16), color=t["hi"], fontsize=8.4, ha="right")
        ax.set_xscale("log", base=2)
        ax.set_xticks(x); ax.set_xticklabels([str(v) for v in x])
        ax.set_ylim(0, 62)
        clean(ax, t, "microbatches", "idle time in the pipeline (%)")
        fig.savefig(OUT / f"07-bubble-{mode}.png", facecolor=t["bg"])
        plt.close(fig)


def chart_final(rows):
    order = rows[::-1]
    names = [r["name"] for r in order]
    for mode, t in THEMES.items():
        fig, ax = frame(t, 7.8, 4.2, top=0.71, left=0.38, bottom=0.15)
        head(fig, t, "No best strategy — only a best fit",
             "One 8-layer MLP, 8192 batch. Read speed together with memory.",
             y=0.90, ysub=0.832)
        y = list(range(len(names)))
        colors = [t["hi"] if r["comm"] == 0 and r["speedup"] > 1 else t["data"]
                  for r in order]
        ax.barh(y, [r["speedup"] for r in order], height=0.55, color=colors,
                zorder=3)
        ax.axvline(2.0, color=t["soft"], lw=1.4, ls=(0, (5, 4)), zorder=2)
        ax.text(1.98, -0.66, "perfect 2-GPU scaling", color=t["soft"],
                fontsize=8.2, ha="right", va="center")
        for i, r in enumerate(order):
            ax.text(r["speedup"] + 0.04, i, f"{r['speedup']:.2f}x",
                    color=t["ink"], fontsize=9.5, va="center",
                    fontweight="bold")
        ax.set_yticks(y)
        ax.set_yticklabels(
            [f"{r['name']}\n{r['comm']} MiB wire · {r['mem']} MiB/GPU"
             for r in order], fontsize=9)
        for lab in ax.get_yticklabels():
            lab.set_linespacing(1.6)
        ax.set_xlim(0, 2.35); ax.set_ylim(-0.8, len(names) - 0.4)
        ax.spines["left"].set_visible(False)
        clean(ax, t, "speedup over one GPU", "", axis="x")
        fig.savefig(OUT / f"07-strategies-{mode}.png", facecolor=t["bg"])
        plt.close(fig)


if __name__ == "__main__":
    b, f = parse_bubble(), parse_final()
    print(f"  parsed {len(b)} microbatch rows, {len(f)} strategies")
    chart_bubble(b); chart_final(f)
    for p in sorted(OUT.glob("07-*.png")):
        print(f"  wrote {p.relative_to(ROOT)}")
