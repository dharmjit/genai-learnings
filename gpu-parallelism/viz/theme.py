"""Shared visual kit for every figure in the series.

Defined once so seven posts read as one authored piece of work. Charts and
animations import from here; nothing downstream hardcodes a colour.

The two-hue categorical palette was checked with the dataviz palette validator
(lightness band, chroma floor, CVD separation, normal-vision floor, contrast)
and passes all checks in both modes:

    light  data #00929E  ·  highlight #B0561F   on #FFFFFF
    dark   data #1FA5AD  ·  highlight #D07C42   on #161B1E

Roles, fixed across the series:
    data      the measured quantity. Never used decoratively.
    hi        the one thing the reader should look at. At most one per figure.
    soft      references, ceilings, ideal lines -- recessive, never a peer series.
    gpu0/gpu1 device identity from post 04 onward. Constant to the end.
"""
import matplotlib.pyplot as plt

THEMES = {
    "light": dict(
        data="#00929E", hi="#B0561F", ink="#14181B", mid="#454E54",
        soft="#6B757C", grid="#DDE1E4", bg="#FFFFFF", faint="#F0F2F3",
        gpu0="#00929E", gpu1="#B0561F",
    ),
    "dark": dict(
        data="#1FA5AD", hi="#D07C42", ink="#E7EAEB", mid="#AEB7BC",
        soft="#8B959B", grid="#2A3034", bg="#161B1E", faint="#1F262A",
        gpu0="#1FA5AD", gpu1="#D07C42",
    ),
}


def new_fig(t, w=7.4, h=4.6, top=0.79, left=0.115, right=0.975, bottom=0.145):
    """Explicit margins, not tight_layout: subtitles live outside the axes at
    y>1, which tight_layout does not account for."""
    fig, ax = plt.subplots(figsize=(w, h), dpi=200)
    fig.subplots_adjust(top=top, left=left, right=right, bottom=bottom)
    ax.set_facecolor(t["bg"])
    fig.patch.set_facecolor(t["bg"])
    return fig, ax


def style(ax, t, xlabel, ylabel):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(t["grid"])
        ax.spines[s].set_linewidth(1)
    ax.grid(True, which="major", axis="both", color=t["grid"], lw=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors=t["soft"], labelsize=9, length=0)
    ax.set_xlabel(xlabel, color=t["mid"], fontsize=9.5, labelpad=9)
    ax.set_ylabel(ylabel, color=t["mid"], fontsize=9.5, labelpad=9)


def title(ax, t, head, sub, pad=38, suby=1.055):
    ax.set_title(head, color=t["ink"], fontsize=13.5, fontweight="bold",
                 loc="left", pad=pad)
    ax.text(0, suby, sub, transform=ax.transAxes, color=t["soft"],
            fontsize=9.5, va="bottom")


def bare(ax, t):
    """Axes used as a drawing canvas rather than a plot."""
    ax.set_facecolor(t["bg"])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def si(v, _=None):
    """Thread counts are powers of two, so 1.04858M is noise. Round hard."""
    for d, s in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if v >= d:
            return f"{v / d:.0f}{s}"
    return f"{v:.0f}"
