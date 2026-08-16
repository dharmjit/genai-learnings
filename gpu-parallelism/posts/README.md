# GPU Parallelism, Measured — series plan

Seven posts. Subtitle used on every one: *Learning GPU parallelism by measuring it.*

Post indices match the lab's directory names, so a reader can map any post to
the code without a lookup table.

Every post opens with a measured number that shouldn't be true, followed by a
**Key takeaway** callout — one sentence the reader keeps if they keep nothing
else, echoed again in the closing section. One takeaway per post; if a post
needs two, it is doing two jobs. See [`_TEMPLATE.md`](_TEMPLATE.md).

All seven hooks already exist in
[`../results/RESULTS.txt`](../results/RESULTS.txt).

| # | key takeaway |
|---|---|
| 01 | GPU threads are slow **because** there are so many of them — not slow *and* numerous. |
| 02 | A GPU does not read variables, it reads 128-byte lines. Whether your 32 lanes want the same line decides everything. |
| 03 | What limits you is flops per byte moved, not flops. Every rung on the ladder buys more uses of each byte. |
| 04 | The wire between two GPUs is ~13x slower than local memory. Communicate rarely, in bulk, overlapped. |
| 05 | Data parallelism communicates weights, so the cost is fixed and amortises with batch size. |
| 06 | Tensor parallelism communicates activations, so the cost scales with batch size and never amortises. |
| 07 | There is no best parallelism strategy — only a best fit for a model size, a batch size, and an interconnect. |

| # | title | hook | runs | new visuals |
|---|---|---|---|---|
| 01 | Your GPU Is Not a Fast CPU | 188 SMs, 1531 GB/s, one thread slower than a CPU core | `device_probe`, `warp_lab` | **execution hierarchy**, warp lockstep, latency hiding, 3 charts |
| 02 | The Memory Wall — Why Your Kernel Is Slow | moving the same bytes to transposed addresses costs 4× | `transpose` | **coalescing grid (hero, reused 4×)**, two-regime bars, memory hierarchy — *all built, post drafted* |
| 03 | Arithmetic Intensity — Climbing the Matmul Ladder | 0.84 → 280 TFLOP/s, 334× | `matmul` | **matmul ladder (hero)** — *built, post drafted* |
| 04 | Two Cards, One Wire | the link is 13× slower than local memory | `p2p_bandwidth`, `overlap_streams` | **bandwidth ladder (hero)**, stream overlap — *built, post drafted* |
| 05 | Data Parallelism — Copy the Model, Split the Batch | the all-reduce costs 1.45 ms at any batch size | `dp_matmul` | **flat-comm line (hero)**, efficiency curve — *built, drafted* |
| 06 | Tensor Parallelism — the Megatron Trick | change *how* you slice, move 4× less data | `tp_matmul` | **strategy comparison (hero)** — *built, drafted* |
| 07 | Pipelines, Bubbles, and When Two GPUs Are Slower Than One | a distributed transpose is 4× slower on 2 GPUs | `pp_matmul`, `alltoall_transpose`, `mlp_2gpu` | **bubble vs theory (hero)**, strategy finale — *built, drafted* |

## Status

**All seven posts drafted; every figure built and rendered in light and dark.**

Every number in every post is parsed from `../results/`, never typed by hand —
`viz/charts/*.py` and `viz/anim/*.py` are the only things that write figures,
and re-running the lab is the only way a figure changes.

Genuinely outstanding:

- **Event-trace Gantt tooling** (`GPULAB_TRACE=1`). Would let posts 04 and 07
  show real per-operation timelines instead of aggregate bars. The only new
  engineering the series still wants.
- **Bank-conflict animation** for post 02. The section explains banks in prose
  and lands the measurement; the mechanism would animate well.
- An editing pass over all seven read end to end, rather than one at a time.

## Platform constraints

Substack runs no JavaScript. Every animation ships as a **looping GIF or MP4**,
pre-rendered. Target 6–10 s, silent, one idea each, under ~5 MB so it stays
inline. Many readers see only the first frame in email — so the first frame
must be legible on its own, and every clip needs a real caption.

Anything genuinely interactive gets hosted elsewhere and linked as strictly
optional.

## Shared visual kit

Fix before drawing anything, so seven posts read as one authored series:

- one colour for GPU 0, one for GPU 1 — constant from post 04 onward
- one colour reserved for measured data, never used decoratively
- one grid metaphor for memory, reused in 02 / 03 / 05 / 06
- one Gantt style for every timeline

See [`_TEMPLATE.md`](_TEMPLATE.md) for the per-post skeleton.
