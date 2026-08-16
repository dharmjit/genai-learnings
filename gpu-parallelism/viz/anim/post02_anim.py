#!/usr/bin/env python3
"""Figures for post 02 — "The Memory Wall: Why Your Kernel Is Slow".

    memory hierarchy   the tiers, with this card's measured numbers

    python3 viz/anim/post02_anim.py
"""
import sys
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "viz"))
from theme import THEMES, bare  # noqa: E402

FPS = 12

OUT = ROOT / "viz" / "anim"


def fig_memory(mode):
    """Static, but the numbers are ours -- measured, not from a spec sheet.

    Lives in post 02, not 01: post 01 measures EXECUTION (threads, warps,
    scheduling) and never touches shared memory, registers or L2. Post 02 is
    where the 128 MiB L2 becomes the punchline.
    """
    t = THEMES[mode]
    TIERS = [
        ("registers",     "~64K per SM",  "fastest — private to one thread", 0.34),
        ("shared memory", "48 KiB/block", "software-managed, per block",     0.50),
        ("L2 cache",      "128 MiB",      "unusually large — see post 02",   0.72),
        ("device memory", "95 GiB",       "1531 GB/s measured",              1.00),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 4.0), dpi=200)
    fig.subplots_adjust(top=0.76, bottom=0.06, left=0.045, right=0.955)
    fig.patch.set_facecolor(t["bg"])
    bare(ax, t)
    ax.set_xlim(-0.02, 1.06)
    ax.set_ylim(-0.3, len(TIERS))

    ax.text(0, len(TIERS) + 0.62, "The memory hierarchy, measured on this card",
            color=t["ink"], fontsize=13.5, fontweight="bold", va="bottom")
    ax.text(0, len(TIERS) + 0.22, "Capacities span six orders of magnitude — bars are indicative, not to scale.",
            color=t["soft"], fontsize=9.3, va="bottom")

    for i, (name, cap, note, w) in enumerate(TIERS):
        y = len(TIERS) - 1 - i
        hot = (i == len(TIERS) - 1)
        ax.add_patch(FancyBboxPatch(
            (0, y + 0.18), w, 0.62,
            boxstyle="round,pad=0,rounding_size=0.012",
            facecolor=t["data"] if hot else t["faint"],
            edgecolor=t["data"] if not hot else "none",
            lw=1.4, alpha=1.0 if hot else 0.9, zorder=3))
        ax.text(0.014, y + 0.585, name, color=t["bg"] if hot else t["ink"],
                fontsize=10.5, fontweight="bold", va="center", zorder=4)
        ax.text(0.014, y + 0.33, note, color=t["bg"] if hot else t["soft"],
                fontsize=8.4, va="center", zorder=4,
                alpha=0.85 if hot else 1.0)
        ax.text(w + 0.012, y + 0.46, cap, color=t["mid"], fontsize=9.5,
                va="center", family="monospace")
    fig.savefig(OUT / f"02-memory-hierarchy-{mode}.png", facecolor=t["bg"])
    plt.close(fig)
    print(f"  viz/anim/02-memory-hierarchy-{mode}.png")




# ──────────────────────────────────────────────────── coalescing (hero)
def anim_coalescing(mode):
    """What a warp actually asks memory for.

    Memory is 32 cache lines, each 128 B (32 slots of 4 B). The same 32 lanes
    request the same 128 bytes of USEFUL data twice: once packed into one line,
    once scattered one slot per line. Solid = what the lane asked for; pale =
    dragged in anyway, because a line is the smallest unit memory serves.

    Base asset for the series -- posts 03, 05 and 06 reuse this grid.
    """
    t = THEMES[mode]
    LINES, SLOTS = 32, 32
    PAUSE = 16
    reveal = [1, 4, 8, 14, 20, 26, 32]
    script = ([(0, n) for n in reveal] + [(0, SLOTS)] * PAUSE
              + [(1, n) for n in reveal] + [(1, LINES)] * PAUSE)

    fig, ax = plt.subplots(figsize=(7.4, 5.2), dpi=170)
    fig.subplots_adjust(top=0.74, bottom=0.11, left=0.075, right=0.955)
    fig.patch.set_facecolor(t["bg"])
    bare(ax, t)
    ax.set_xlim(-0.4, SLOTS + 0.4)
    ax.set_ylim(LINES + 0.4, -1.6)          # inverted: line 0 at the top

    fig.text(0.075, 0.915, "A warp does not read variables. It reads lines.",
             color=t["ink"], fontsize=13.5, fontweight="bold", va="bottom")
    fig.text(0.075, 0.855, "128 bytes is the smallest thing memory will hand "
                           "over. The same 32 lanes, asking twice.",
             color=t["soft"], fontsize=9.3, va="bottom")
    ax.text(-0.4, -1.0, "MEMORY \u2014 32 cache lines \u00d7 128 B",
            color=t["soft"], fontsize=7.6, family="monospace", va="center")

    # a visible border is essential: the faint fill alone is ~1.05 contrast
    # against the page and the whole grid disappears.
    cells = [[ax.add_patch(Rectangle((c + 0.04, r + 0.08), 0.92, 0.84,
                                     facecolor=t["faint"], edgecolor=t["grid"],
                                     lw=0.3, zorder=2))
              for c in range(SLOTS)] for r in range(LINES)]

    caption = ax.text(-0.4, LINES + 1.6, "", color=t["ink"], fontsize=10.5,
                      va="center")
    eff = ax.text(SLOTS + 0.4, LINES + 1.6, "", color=t["hi"], fontsize=11,
                  fontweight="bold", ha="right", va="center")

    def draw(k):
        phase, n = script[k]
        for row in cells:
            for cell in row:
                cell.set_facecolor(t["faint"])
                cell.set_alpha(1.0)
        if phase == 0:
            for c in range(n):
                cells[0][c].set_facecolor(t["data"])
            caption.set_text(f"COALESCED \u2014 {n} lanes, all in line 0")
            if n == SLOTS:
                eff.set_text("1 line fetched \u00b7 100% of it used")
        else:
            for r in range(n):
                for c in range(SLOTS):
                    cells[r][c].set_facecolor(t["hi"])
                    cells[r][c].set_alpha(0.18)
                cells[r][0].set_facecolor(t["hi"])
                cells[r][0].set_alpha(1.0)
            caption.set_text(f"STRIDED \u2014 {n} lanes, {n} different lines")
            if n == LINES:
                eff.set_text("32 lines fetched \u00b7 3% of them used")
        return [caption, eff]

    anim = FuncAnimation(fig, draw, frames=len(script), interval=1000 / FPS,
                         blit=False)
    path = OUT / f"02-coalescing-{mode}.gif"
    anim.save(path, writer=PillowWriter(fps=FPS),
              savefig_kwargs={"facecolor": fig.get_facecolor()})
    plt.close(fig)
    print(f"  {path.relative_to(ROOT)}  {len(script)} frames  "
          f"{path.stat().st_size//1024} KiB")


if __name__ == "__main__":
    print("rendering post 02 figures:")
    for mode in ("light", "dark"):
        fig_memory(mode)
        anim_coalescing(mode)
