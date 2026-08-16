#!/usr/bin/env python3
"""Charts for post 06 — Tensor Parallelism."""
import re
from _common import THEMES, OUT, ROOT, section, frame, head, clean, plt


def parse():
    blk = section("tp_matmul")
    comm = dict(re.findall(r"strategy ([AB]) \(.*?\):\s*([\d.]+) MiB", blk))
    rows = []
    for line in blk.splitlines():
        m = re.match(r"(.+?)\s{2,}([\d.]+)\s+([\d.]+)\s+([\d.]+)%\s+\w+", line)
        if m:
            rows.append((m.group(1).strip(), float(m.group(2)),
                         float(m.group(4))))
    return rows, {k: float(v) for k, v in comm.items()}


def chart(rows, comm):
    order = [r for r in rows][::-1]
    names = [r[0] for r in order]
    sp = [r[2] / 100 for r in order]
    note = {"TP-B Megatron (all-reduce Y)": f"moves {comm.get('B', 0):.0f} MiB",
            "TP-A naive (all-gather Z)": f"moves {comm.get('A', 0):.0f} MiB",
            "1 GPU (no parallelism)": "moves nothing"}
    for mode, t in THEMES.items():
        fig, ax = frame(t, 7.6, 3.8, top=0.71, left=0.38, bottom=0.16)
        head(fig, t, "Same maths, same split — 4x less data on the wire",
             "Column-split W1 with row-split W2 makes the shard boundaries line up.",
             y=0.90, ysub=0.825)
        colors = [t["hi"] if "TP-B" in n else t["data"] for n in names]
        y = list(range(len(names)))
        ax.barh(y, sp, height=0.55, color=colors, zorder=3)
        ax.axvline(2.0, color=t["soft"], lw=1.4, ls=(0, (5, 4)), zorder=2)
        ax.text(1.98, -0.62, "perfect 2-GPU scaling", color=t["soft"],
                fontsize=8.2, ha="right", va="center")
        for i, (n, v) in enumerate(zip(names, sp)):
            ax.text(v + 0.04, i, f"{v:.2f}x", color=t["ink"], fontsize=9.5,
                    va="center", fontweight="bold")
        ax.set_yticks(y)
        ax.set_yticklabels([f"{n}\n{note.get(n, '')}" for n in names],
                           fontsize=9)
        for lab in ax.get_yticklabels():
            lab.set_linespacing(1.6)
        ax.set_xlim(0, 2.3); ax.set_ylim(-0.75, len(names) - 0.4)
        for sp_ in ("left",):
            ax.spines[sp_].set_visible(False)
        clean(ax, t, "speedup over one GPU", "", axis="x")
        fig.savefig(OUT / f"06-tp-strategies-{mode}.png", facecolor=t["bg"])
        plt.close(fig)


if __name__ == "__main__":
    r, c = parse(); print(f"  parsed {len(r)} strategies, comm {c}")
    chart(r, c)
    for p in sorted(OUT.glob("06-*.png")):
        print(f"  wrote {p.relative_to(ROOT)}")
