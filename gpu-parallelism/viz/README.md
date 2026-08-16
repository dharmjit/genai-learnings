# Visuals

Everything here is generated. Charts read from `../results/`, so a post can
never drift from the code that produced its numbers.

| dir | tool | output | used by |
|---|---|---|---|
| `manim/` | Manim Community | MP4 → GIF | conceptual animation: warps, coalescing, tiling, sharding, collectives |
| `charts/` | matplotlib | PNG | every measured chart: roofline, bandwidth ladder, scaling curves, bubble vs theory |

## Rules

- **Never hand-enter a number.** Charts parse `../results/RESULTS.txt`. If a
  measurement changes, re-run the lab and re-render; nothing else should need
  editing.
- Export loops at 6–10 s, silent, under ~5 MB so Substack keeps them inline.
- The first frame must stand alone — email clients show it static.
- Respect the shared visual kit in `../posts/README.md`: GPU 0 / GPU 1 colours,
  the reserved "measured data" colour, one grid metaphor, one Gantt style.

## Not built yet

`charts/gantt.py` needs per-operation CUDA event timestamps that the lab does
not currently emit — today it reports only aggregate timings. Adding a CSV dump
behind `GPULAB_TRACE=1` unblocks the hero visuals for posts 04 and 07.

Proposed schema:

```
device,stream,label,start_ms,end_ms
0,compute,stage0_mb0,0.000,2.411
0,comm,xfer_mb0,2.411,3.198
1,compute,stage1_mb0,3.198,5.605
```

**Design note.** `cudaEventElapsedTime` only works between two events on the
*same* device, so two GPUs cannot share a time axis directly. Sync both
devices, record an epoch event on each, and express every measurement as
elapsed-from-that-device's-epoch. Residual skew is the barrier release time —
microseconds against millisecond bars, so invisible in the chart, but it is
the reason raw timestamps cannot simply be subtracted across devices.
