# Glossary

Every term the series uses, defined once. Linked from the footer of each post so
a reader can start anywhere. Terms are grouped by the post that introduces them,
but the list is meant to be skimmed, not read.

Numbers in parentheses are this machine's — 2× RTX PRO 6000 Blackwell — so the
abstractions stay attached to something concrete.

---

## The execution hierarchy (post 01)

**Thread** — one instance of your kernel running. The smallest unit you write
code for, and *not* the unit the hardware schedules.

**Lane** — a thread's slot inside its warp, numbered 0–31. Saying "lane 5" and
"the 6th thread of this warp" means the same thing; "lane" is used when the
warp's internal structure matters.

**Warp** — 32 threads executing together, one instruction at a time. **The real
unit of execution.** You never choose the warp size; it is 32 on every NVIDIA
GPU to date.

**Block** (or *thread block*) — a group of threads *you* choose the size of,
typically 128–1024. Threads within a block can share a scratchpad and
synchronise with each other; threads in different blocks essentially cannot. A
block runs entirely on one SM. (Our kernels mostly use 256 threads = 8 warps.)

**Grid** — all the blocks in one kernel launch.

**SM** (streaming multiprocessor) — the physical core that runs blocks. Has its
own registers, scratchpad, schedulers and execution units. (188 per GPU, up to
1536 threads resident each → 288,768 threads resident.)

**Kernel** — a function you write that runs on the GPU, launched across a grid
of blocks with `<<<blocks, threads>>>`.

**Occupancy** — resident warps as a fraction of the hardware maximum. An input
to performance, *not* a goal in itself; see post 01, where bandwidth saturates
at 33%.

**Divergence** — lanes *within one warp* taking different branches. The warp
must execute each branch in turn with the other lanes masked off, so cost scales
with the number of distinct branches taken. Branching between *different* warps
is free. (Measured: 30.1× at 32 branches; 1.05× when the same branches are moved
between warps.)

**FMA** (fused multiply-add) — `v = v*a + b` as a single instruction. The
workhorse of graphics and ML; a matrix multiply is essentially nothing else.

**Dependent chain** — a sequence where each operation needs the previous
result. Forbids overlap, so it measures *latency* rather than throughput.

**Latency vs throughput** — how fast one thing is, versus how many things per
second. CPUs optimise the first, GPUs the second, and the two trade against each
other.

**Latency hiding** — parking a warp that is waiting on memory and issuing from
another that is ready. Why a GPU needs thousands of threads to reach full
bandwidth, and why each thread can afford to be slow.

---

## Memory (posts 01–03)

**Registers** — per-thread storage, fastest tier. Plentiful but finite; using
too many per thread reduces how many threads fit on an SM.

**Shared memory** — a software-managed scratchpad private to a block (48 KiB per
block by default here). You load into it explicitly; it is not a cache.

**Bank conflict** — shared memory is split into 32 banks. If lanes in a warp hit
different addresses in the *same* bank, the accesses serialise.

**Coalescing** — a warp's 32 lanes reading one contiguous 128-byte line, which
the memory system serves as a single transaction. Miss it and you can pay up to
32×.

**L2 cache** — GPU-wide cache (**128 MiB** here, unusually large — large enough
to change which optimisations matter).

**Arithmetic intensity** — FLOPs performed per byte moved. The number that
decides whether you are compute-bound or memory-bound.

**Roofline** — a plot of achievable performance against arithmetic intensity,
showing the memory-bandwidth ceiling and the compute ceiling.

**Tensor core** — dedicated matrix-multiply hardware operating on small tiles,
typically fp16/bf16/fp8 inputs with fp32 accumulation. (281 TFLOP/s vs 73 in
fp32.)

---

## Multi-GPU (posts 04–07)

**P2P** (peer-to-peer) — one GPU reading another's memory directly. Without it,
transfers stage through host RAM. (54.6 GB/s with, 42.3 GB/s without.)

**NVLink** — NVIDIA's high-bandwidth GPU-to-GPU interconnect. **Not present on
these cards** — they talk over PCIe, which is why tensor parallelism struggles
here.

**Pinned (page-locked) memory** — host memory that can be DMA'd directly.
Pageable memory cannot, so the driver stages it through a hidden buffer (2.2×
slower).

**Stream** — an ordered queue of GPU work. Independent streams can overlap, which
is how copies hide behind compute.

**Event** — a marker recorded in a stream, used to order work across streams or
devices. Belongs to the device that was current when it was created.

**Collective** — a communication pattern involving every participant:
- **all-reduce** — every rank contributes a value, every rank gets the sum.
- **all-gather** — every rank contributes a shard, every rank gets the whole.
- **all-to-all** — every rank sends something *different* to every other rank.
  The most bandwidth-hungry of the three; it cannot be tree- or ring-optimised.

**Data parallelism (DP)** — split the *batch*; every GPU holds the full model.
Communication is fixed at the weight size, so it amortises as the batch grows.

**Tensor parallelism (TP)** — split the *weight matrices*. Communication scales
with the batch, so it never amortises. Wants NVLink.

**Pipeline parallelism (PP)** — split the model by *depth*. The least
communication of the three, at the cost of pipeline bubbles.

**Microbatch** — a slice of a batch fed through a pipeline, used to keep all
stages busy at once.

**Bubble** — idle time in a pipeline while it fills and drains. Fraction is
roughly `(P-1)/(m+P-1)` for P stages and m microbatches.

**NCCL** — NVIDIA's collective communication library. This series hand-rolls its
collectives so you can see what they do; NCCL is what you would ship.
