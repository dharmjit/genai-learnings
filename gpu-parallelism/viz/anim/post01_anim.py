#!/usr/bin/env python3
"""Animations for post 01 — the two mechanisms a static chart cannot show.

    lockstep   why 32 lanes that disagree cost 32 passes
    hiding     why a GPU needs thousands of threads to reach full bandwidth

The charts in viz/charts/ show the CONSEQUENCES of these two mechanisms.
These show the mechanism itself. Both read their numbers from results/, so
the animation and the chart can never drift apart.

Output: looping GIFs, light and dark, sized for inline Substack embedding.

    python3 viz/anim/post01_anim.py
"""
import re
import sys
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "viz"))
from theme import THEMES, bare  # noqa: E402

OUT = ROOT / "viz" / "anim"
FPS = 12


def measured_divergence():
    """Read the 32-branch slowdown from the lab output.

    The end card used to hardcode "30.1x". Run-to-run variance moved the real
    figure to 28.2x and the GIF silently disagreed with the charts, which DO
    parse results/. Same rule for animations now: no hand-entered numbers.
    """
    txt = (ROOT / "results" / "warp_lab.txt").read_text()
    sec = txt.split("3. WARP DIVERGENCE")[1].split("-->")[0]
    rows = re.findall(r"^\s+(\d+)\s+[\d.]+\s+([\d.]+)x", sec, re.M)
    return float(dict(rows)["32"])


def save(fig, anim, name, mode, frames):
    path = OUT / f"{name}-{mode}.gif"
    anim.save(path, writer=PillowWriter(fps=FPS), savefig_kwargs={
        "facecolor": fig.get_facecolor()})
    plt.close(fig)
    kb = path.stat().st_size // 1024
    print(f"  {path.relative_to(ROOT)}  {frames} frames  {kb} KiB")


# ─────────────────────────────────────────────────────────── 1. lockstep
def anim_lockstep(mode):
    """32 lanes, one instruction at a time. Watch the warp walk every branch."""
    t = THEMES[mode]
    LANES = 32
    # (branch_count, label) — each scenario runs branch_count passes
    SCEN = [(1, "all 32 lanes agree"),
            (2, "lanes split two ways"),
            (4, "lanes split four ways")]
    DIVERGE = measured_divergence()
    HOLD = 7          # frames per pass
    GAP = 4           # frames between scenarios

    # Precompute the frame script: (divisor, pass_index, cost_so_far, note)
    script = []
    for d, note in SCEN:
        for p in range(d):
            for _ in range(HOLD):
                script.append((d, p, p + 1, note))
        for _ in range(GAP):
            script.append((d, d - 1, d, note))
    for _ in range(FPS * 2):   # end card
        script.append((32, -1, 32, "all 32 lanes disagree"))

    fig, ax = plt.subplots(figsize=(7.4, 3.7), dpi=170)
    fig.subplots_adjust(top=0.70, bottom=0.06, left=0.045, right=0.955)
    fig.patch.set_facecolor(t["bg"])
    bare(ax, t)
    ax.set_xlim(0, LANES)
    ax.set_ylim(-2.6, 1.5)

    # Figure coords, not data coords: text placed above ylim gets clipped by
    # the canvas edge no matter how much axes margin you leave.
    fig.text(0.045, 0.905, "One warp = 32 lanes, one instruction at a time",
             color=t["ink"], fontsize=13.5, fontweight="bold", va="bottom")
    fig.text(0.045, 0.815, "Lanes that disagree cannot run at the same time. "
                           "The warp walks every branch its lanes take.",
             color=t["soft"], fontsize=9.3, va="bottom")

    cells = []
    for i in range(LANES):
        r = FancyBboxPatch((i + 0.11, 0), 0.78, 0.9,
                           boxstyle="round,pad=0,rounding_size=0.07",
                           lw=1.2, zorder=3)
        ax.add_patch(r)
        cells.append(r)

    ax.text(0, 1.16, "LANE 0 → 31", color=t["soft"], fontsize=7.2,
            va="bottom", family="monospace")

    bar_bg = Rectangle((0, -1.5), LANES, 0.42, facecolor=t["faint"],
                       edgecolor="none", zorder=2)
    ax.add_patch(bar_bg)
    bar = Rectangle((0, -1.5), 0, 0.42, facecolor=t["hi"], edgecolor="none",
                    zorder=3)
    ax.add_patch(bar)
    ax.text(0, -1.02, "TIME SPENT BY THE WARP", color=t["soft"], fontsize=7.2,
            va="bottom", family="monospace")

    status = ax.text(0, -2.35, "", color=t["ink"], fontsize=10.5, va="bottom")
    cost = ax.text(LANES, -2.35, "", color=t["hi"], fontsize=11.5,
                   fontweight="bold", va="bottom", ha="right")

    def draw(k):
        d, p, spent, note = script[k]
        end = (p == -1)
        for i, r in enumerate(cells):
            # On the end card light exactly one lane: 32 passes,
            # one lane doing useful work in each.
            active = (i == 0) if end else (i % d == p)
            r.set_facecolor(t["data"] if active else t["bg"])
            r.set_edgecolor(t["data"] if active else t["grid"])
            r.set_alpha(1.0 if active else 0.55)
        bar.set_width(LANES * spent / 32.0)
        if end:
            status.set_text("32 branches → 32 sequential passes")
            cost.set_text(f"measured: {DIVERGE:.1f}x slower")
        else:
            live = LANES // d
            status.set_text(f"{note} — pass {p+1} of {d},  "
                            f"{live}/32 lanes doing work")
            cost.set_text(f"{spent}x")
        return cells + [bar, status, cost]

    anim = FuncAnimation(fig, draw, frames=len(script), interval=1000 / FPS,
                         blit=False)
    save(fig, anim, "01-warp-lockstep", mode, len(script))


# ──────────────────────────────────────────────────────── 2. latency hiding
def anim_hiding(mode):
    """The same slow memory, twice. Once with one warp, once with eight."""
    t = THEMES[mode]
    WARPS = 8
    ISSUE = 2        # cycles a warp issues before it needs memory
    STALL = 14       # cycles waiting on DRAM
    PERIOD = ISSUE * WARPS   # 16 — with 8 warps the SM never idles
    TOTAL = 64       # cycles shown

    def busy(nwarps, cycle):
        """Which warp, if any, is issuing on this cycle? Round-robin."""
        slot = (cycle % PERIOD) // ISSUE
        return slot if slot < nwarps else None

    frames = TOTAL + FPS * 2
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 5.0), dpi=170,
                             gridspec_kw={"height_ratios": [1, 2.2]})
    fig.subplots_adjust(top=0.78, bottom=0.10, left=0.13, right=0.97, hspace=0.55)
    fig.patch.set_facecolor(t["bg"])

    fig.text(0.045, 0.945, "A stalled warp is not a stalled GPU",
             color=t["ink"], fontsize=13.5, fontweight="bold", va="bottom")
    fig.text(0.045, 0.885, "Each warp issues for 2 cycles, then waits 14 on memory. "
                           "Eight warps exactly cover each other's gaps.",
             color=t["soft"], fontsize=9.3, va="bottom")

    panels = []
    for ax, n, lab in ((axes[0], 1, "1 warp resident"),
                       (axes[1], WARPS, f"{WARPS} warps resident")):
        bare(ax, t)
        ax.set_xlim(0, TOTAL)
        ax.set_ylim(-1.35, n)
        ax.text(0, n + 0.28, lab, color=t["ink"], fontsize=10.5,
                fontweight="bold", va="bottom")
        for w in range(n):
            ax.add_patch(Rectangle((0, w + 0.12), TOTAL, 0.72,
                                   facecolor=t["faint"], edgecolor="none"))
            ax.text(-0.8, w + 0.48, f"w{w}", color=t["soft"], fontsize=7.5,
                    ha="right", va="center", family="monospace")
        ax.add_patch(Rectangle((0, -1.15), TOTAL, 0.5, facecolor=t["faint"],
                               edgecolor="none"))
        ax.text(-0.8, -0.9, "SM", color=t["soft"], fontsize=7.5, ha="right",
                va="center", family="monospace")
        util = ax.text(TOTAL, -1.95, "", color=t["hi"], fontsize=11,
                       fontweight="bold", ha="right", va="bottom")
        panels.append((ax, n, util, []))

    def draw(k):
        cyc = min(k, TOTAL)
        arts = []
        for ax, n, util, drawn in panels:
            while drawn and drawn[-1][0] >= cyc:
                _, a, b = drawn.pop()
                a.remove()
                if b is not None:
                    b.remove()
            for c in range(len(drawn), cyc):
                w = busy(n, c)
                if w is None:
                    drawn.append((c, ax.add_patch(Rectangle(
                        (c, -1.15), 1, 0.5, facecolor=t["bg"], alpha=0.0,
                        edgecolor="none")), None))
                    continue
                lane = ax.add_patch(Rectangle(
                    (c, w + 0.12), 1, 0.72, facecolor=t["data"],
                    edgecolor="none", zorder=3))
                sm = ax.add_patch(Rectangle(
                    (c, -1.15), 1, 0.5, facecolor=t["data"], edgecolor="none",
                    zorder=3))
                drawn.append((c, lane, sm))
            live = sum(1 for c in range(cyc) if busy(n, c) is not None)
            pct = 100.0 * live / cyc if cyc else 0
            util.set_text(f"SM busy {pct:.0f}%")
            arts.append(util)
        return arts

    anim = FuncAnimation(fig, draw, frames=frames, interval=1000 / FPS,
                         blit=False)
    save(fig, anim, "01-latency-hiding", mode, frames)


# ─────────────────────────────────────────────── 4. execution hierarchy
def fig_execution(mode):
    """Defines thread / lane / warp / block / SM in one glance.

    Reused by every post in the series -- the vocabulary is introduced once
    here and never re-explained. Counts are this card's, from device_probe.
    Fixed internal y positions per card, so a longer label in one card cannot
    shove text into its neighbour.
    """
    t = THEMES[mode]
    fig, ax = plt.subplots(figsize=(7.4, 3.4), dpi=200)
    fig.subplots_adjust(top=0.80, bottom=0.03, left=0.03, right=0.97)
    fig.patch.set_facecolor(t["bg"])
    bare(ax, t)
    ax.set_xlim(0, 100)
    ax.set_ylim(-2.5, 32)

    fig.text(0.03, 0.905, "The four words you need", color=t["ink"],
             fontsize=13.5, fontweight="bold", va="bottom")
    fig.text(0.03, 0.835, "Each one contains the last. Counts are this card's, "
                          "from ./bin/device_probe.",
             color=t["soft"], fontsize=9.3, va="bottom")

    Y0, H = 2.0, 27.0          # card box
    Y_NAME, Y_COUNT, Y_NOTE = 26.0, 22.4, 16.2
    W = 21

    COLS = [
        (2,  "THREAD", "1",             "one instance of\nyour kernel code"),
        (26, "WARP",   "32 threads",    "the real unit of\nexecution — they\nmove together"),
        (50, "BLOCK",  "256 threads",   "8 warps sharing\n48 KiB of scratchpad;\nruns on one SM"),
        (74, "SM",     "1536 threads",  "188 of them —\n288,768 threads\nresident at once"),
    ]

    for x, name, count, note in COLS:
        ax.add_patch(FancyBboxPatch(
            (x, Y0), W, H, boxstyle="round,pad=0,rounding_size=0.6",
            facecolor=t["faint"], edgecolor=t["grid"], lw=1.2, zorder=2))
        ax.text(x + 1.5, Y_NAME, name, color=t["hi"], fontsize=8.6,
                fontweight="bold", family="monospace", va="center")
        ax.text(x + 1.5, Y_COUNT, count, color=t["ink"], fontsize=10.2,
                fontweight="bold", va="center")
        ax.text(x + 1.5, Y_NOTE, note, color=t["soft"], fontsize=8.0,
                va="center", linespacing=1.55)

    def sq(x, y, w, h, alpha=1.0):
        ax.add_patch(Rectangle((x, y), w, h, zorder=4, facecolor=t["data"],
                               edgecolor="none", alpha=alpha))

    sq(3.5, 5.6, 1.6, 1.6)                                  # THREAD: one
    for i2 in range(32):                                    # WARP: 2 x 16
        sq(27.5 + (i2 % 16) * 1.15, 5.0 + (i2 // 16) * 1.9, 0.95, 1.5)
    for w in range(8):                                      # BLOCK: 8 warps
        sq(51.5, 4.6 + w * 0.9, 18.0, 0.62, 1.0 if w == 0 else 0.38)
    for b in range(6):                                      # SM: resident blocks
        sq(75.5, 4.6 + b * 1.2, 18.0, 0.85, 0.38)

    for x in (24.4, 48.4, 72.4):
        ax.annotate("", xy=(x + 1.3, 15), xytext=(x - 1.3, 15),
                    arrowprops=dict(arrowstyle="-|>", color=t["soft"], lw=1.3))

    ax.text(2, -1.6, "A lane is one thread's slot inside its warp — lane 0 through lane 31. "
                     "Divergence is lanes inside ONE warp disagreeing.",
            color=t["mid"], fontsize=8.4, va="center")
    fig.savefig(OUT / f"01-execution-hierarchy-{mode}.png", facecolor=t["bg"])
    plt.close(fig)
    print(f"  viz/anim/01-execution-hierarchy-{mode}.png")


if __name__ == "__main__":
    print("rendering post 01 animations (this takes a minute):")
    for mode in ("light", "dark"):
        anim_lockstep(mode)
        anim_hiding(mode)
        fig_execution(mode)
