# CUDA multi-GPU lab — matmul & transpose on 2× RTX PRO 6000 Blackwell

A hands-on curriculum that builds from "why is my kernel 80× slower than
cuBLAS" up to "why does Megatron split W1 by column and W2 by row".

Every program **verifies its own numerics** against a single-GPU reference and
prints a benchmark table. Nothing here is a toy that only pretends to compute.

All numbers in this README were measured on the target box, not estimated.

---

## The machine

| | |
|---|---|
| GPUs | 2× NVIDIA RTX PRO 6000 Blackwell Server Edition, 95 GiB each |
| Compute capability | **sm_120** (needs CUDA ≥ 12.8) |
| SMs | 188 per GPU |
| L2 cache | **128 MiB** — unusually large, and it changes which optimisations matter |
| Shared memory | 48 KiB/block default, 100 KiB/SM |
| Measured DRAM read | **1531 GB/s** |
| Interconnect | PCIe Gen5, same NUMA node, **P2P enabled** (no NVLink) |
| Measured P2P | **54.6 GB/s** one-way, 102 GB/s bidirectional |

**The single most important ratio in this lab: local memory is ~28× faster
than the link between the two GPUs.** Everything in parts 04–08 follows from
that number.

## Build & run

```bash
export PATH=/usr/local/cuda-12.9/bin:$PATH && make -j
```

```bash
make run
```

`make run` executes the whole tour in order. Individual programs take optional
size arguments, e.g. `./bin/matmul 8192 8192 8192`.

Override the arch if you move this to another card: `make ARCH=90`.

---

## The curriculum

### 00 — `device_probe` · know your hardware
Prints SMs, shared memory, L2, achieved bandwidth, and the P2P matrix. Every
later optimisation is justified by a number here.

### 01 — `transpose` · memory movement, nothing else
Transpose does zero arithmetic, so it isolates the two rules that dominate
CUDA: **coalescing** (global memory) and **bank conflicts** (shared memory).

The same five kernels are run at two working-set sizes, and the comparison is
the lesson:

| kernel | 8192² (256 MiB, DRAM-bound) | 2048² (16 MiB, fits in L2) |
|---|---|---|
| copy (speed of light) | 1460 GB/s | 3308 GB/s |
| naive, uncoalesced writes | 886 GB/s (61%) | 804 GB/s (**24%**) |
| shared mem, 32-way bank conflict | 1470 GB/s (101%) | 2056 GB/s (**62%**) |
| shared mem + `[32][33]` padding | 1467 GB/s | 3297 GB/s (100%) |
| shared mem + swizzle, no padding | 1467 GB/s | 3297 GB/s (100%) |

**Read this carefully.** At 8192² the bank-conflict kernel is *just as fast* as
the fixed ones — the kernel is starved by DRAM and the conflict hides entirely
behind memory latency. At 2048², where the data lives in the 128 MiB L2, the
identical conflict costs 1.6×.

An optimisation is only worth what the current bottleneck lets it be worth.
Measure in the regime you actually run in.

Padding and swizzling perform identically; the swizzle gets there without
spending extra shared memory, which matters when shared memory is what caps
your occupancy.

### 02 — `matmul` · the arithmetic-intensity ladder
`C = A@B` at 4096³, each step earning its speedup:

| kernel | TFLOP/s | % of cuBLAS |
|---|---|---|
| 1 naive | 0.84 | 1.1% |
| 2 coalesced | 5.41 | 7.4% |
| 3 shared-memory tiling | 8.12 | 11.1% |
| 4 1-D register tiling (TM=8) | 26.06 | 35.5% |
| 5 2-D register tiling + `float4` | 48.40 | 66.0% |
| 6 cuBLAS SGEMM | **73.39** | 100% |
| 7 hand-written WMMA (fp16) | 51.01 | 18.2% |
| 8 cuBLAS fp16 tensor cores | **280.97** | 100% |

Two things worth pausing on:

- **Kernel 1 → 2 is a 6.4× speedup from swapping which index maps to
  `threadIdx.x`.** Identical arithmetic, identical total traffic. The only
  change is that a warp's 32 lanes now read one contiguous 128-byte line
  instead of 32 separate ones. This is the most common beginner bug in CUDA.
- **fp16 tensor cores are 3.8× faster than the best fp32 path.** This is why
  mixed precision is not optional in modern training.

The gap between the hand-written WMMA kernel (51) and cuBLAS (281) is real and
expected: closing it needs shared-memory staging, multi-stage async pipelining
and swizzled fragment layouts. The naive WMMA kernel is here to show the API
and the mental model, not to win.

### 03 — `p2p_bandwidth`, `overlap_streams` · the interconnect

| path | GB/s |
|---|---|
| H2D pageable | 26.5 |
| H2D pinned | 57.5 |
| D2D same GPU | 731 |
| GPU0→GPU1, P2P disabled (staged via host) | 42.3 |
| GPU0→GPU1, P2P enabled | 54.6 |
| P2P bidirectional aggregate | 102.2 |

Pinned memory is **2.2×** faster than pageable, because pageable memory cannot
be DMA'd — the driver stages it through an internal pinned buffer first.

`overlap_streams` turns the serial H2D → compute → D2H sequence into a
pipeline: 1.00× → 1.28× → 1.47× → **1.60×** at 8 streams. Adding the second GPU
only reaches 1.75×, because PCIe is now the shared bottleneck. Splitting
compute is easy; splitting bandwidth is not.

---

## The three parallelisms

### 04 — `dp_matmul` · DATA parallelism
Every GPU holds the **full weights** and a **different slice of the batch**.
The forward pass communicates nothing; only gradients are all-reduced.

```
compute per GPU = 4 · (B/2) · H · H     ← grows with batch
communication   = H · H · 4 bytes       ← FIXED (weights only)
```

| batch | 1-GPU ms | 2-GPU ms | comm ms | efficiency |
|---|---|---|---|---|
| 512 | 0.574 | 1.778 | 1.458 | 16% |
| 2048 | 1.960 | 2.503 | 1.445 | 39% |
| 8192 | 7.484 | 5.139 | 1.419 | 73% |
| 32768 | 32.550 | 15.775 | 1.275 | 103% |

**The `comm ms` column is flat.** That is the whole story of data parallelism:
the all-reduce moves the weights, so its cost is fixed while compute grows.
Efficiency climbs toward 100% simply by using a bigger batch.

(103% is mildly superlinear — splitting the batch shrinks each GPU's working
set, which helps cache behaviour. Not an error.)

DP's limit is **memory, not speed**: every GPU must hold the entire model.

### 05 — `tp_matmul` · TENSOR parallelism
The weights are too big for one GPU, so split the matrices themselves. The
workload is a transformer MLP: `Y = GeLU(X @ W1) @ W2`.

There are two ways to split it, and the difference *is* Megatron-LM:

| strategy | comm volume | speedup |
|---|---|---|
| 1 GPU | — | 1.00× |
| **B: column-split W1, row-split W2, all-reduce Y** | 64 MiB | **1.64×** |
| A: column-split W1, all-gather Z, replicate W2 | 256 MiB | 1.05× |

Strategy B wins on both axes: **4× less communication and no replicated
compute**. It works because splitting W1 by column and W2 by row makes the
shard boundaries line up, so what gets split is the *contraction* dimension —
and a split contraction is just a sum, i.e. an all-reduce.

The critical difference from DP: **TP communicates activations, which scale
with the batch.** A bigger batch does not amortise the cost. This is exactly
the workload NVLink exists for, and why TP is kept inside a node.

### 06 — `pp_matmul` · PIPELINE parallelism
Split the model by **depth**. GPU0 runs layers 0–3, GPU1 runs layers 4–7. One
activation crosses the link per microbatch — by far the least communication of
the three strategies.

The cost is the **bubble**: with one batch, each GPU idles half the time.
Microbatching fills the pipe; theory says the bubble is `(P-1)/(m+P-1)`.

| microbatches | ms | speedup | bubble measured | bubble predicted |
|---|---|---|---|---|
| 1 | 34.36 | 0.96× | 52% | 50% |
| 2 | 23.69 | 1.39× | 31% | 33% |
| 4 | 20.79 | 1.58× | 21% | 20% |
| 8 | **19.42** | **1.69×** | 15% | 11% |
| 16 | 20.83 | 1.58× | 21% | 6% |

Theory tracks measurement closely up to m=4, then **reality gets worse than
theory and m=16 is slower than m=8**. Past a point the microbatches are too
small to fill 188 SMs and per-launch overhead dominates. That tension — bubble
versus kernel efficiency — is precisely what real schedulers (GPipe, 1F1B,
interleaved) are tuning.

### 07 — `alltoall_transpose` · when multi-GPU *loses*
A distributed transpose is a local transpose plus an **all-to-all**: keep the
diagonal block, ship the off-diagonal one. This is the pattern behind sequence
parallelism (DeepSpeed-Ulysses), MoE token routing, and distributed FFTs.

| | ms | GB/s |
|---|---|---|
| 1 GPU, whole matrix | **0.374** | 1436 |
| 2 GPUs, distributed | 1.508 | 356 |
| — of which the exchange | 1.320 | — |

**Two GPUs are 4× slower than one.** The exchange is 88% of the runtime.
Transpose does no arithmetic, so there is nothing to hide the transfer behind.

Keep this result. The instinct that "more GPUs = faster" is wrong for
memory-bound operations on a slow link, and this is the cleanest possible
demonstration.

### 08 — `mlp_2gpu` · capstone shootout
The same 8-layer MLP forward pass under all three strategies:

| strategy | ms | speedup | comm MiB | weights/GPU MiB |
|---|---|---|---|---|
| 1 GPU | 32.94 | 1.00× | 0 | 512 |
| **DP** (split batch) | **15.07** | **2.19×** | 0 | 512 |
| TP (split weights) | 27.78 | 1.19× | 512 | 256 |
| PP (split layers, m=8) | 19.73 | 1.67× | 128 | 256 |

Read the speed column *together with* the memory column:

- **DP** is fastest and communicates nothing (in inference) — but every GPU
  stores the whole model, so it does nothing for a model that doesn't fit.
- **TP** halves weight memory but pays an all-reduce per block, and that volume
  grows with the batch. On PCIe it is the loser; on NVLink it is competitive.
- **PP** halves weight memory and moves the least data by far — the strategy of
  choice across slow links — at the cost of bubbles and scheduling complexity.

There is no best strategy, only a best fit for a given model size, batch size,
and interconnect. Real systems compose all three: TP inside a node, PP across
nodes, DP over the whole lot.

---

## Things that bit me writing this

Preserved because they're the bugs you'll hit too.

**A cross-device race that produced plausible wrong answers.** My first
all-reduce made each GPU's comm stream wait on *its own* compute stream. But in
an all-reduce, GPU0's copy **reads GPU1's buffer** — so it must also wait on
GPU1's compute. Streams on different devices are not ordered against each
other. The result was a 6% error that looked like a precision issue, and it
only appeared in one of the two strategies. See `commAfterCompute()` in
`common/collectives.cuh`.

**Events belong to the device that was current when you created them.** Create
an event under `cudaSetDevice(1)` and record it on a device-0 stream and you
get `invalid resource handle`. Waiting cross-device is fine; recording is not.

**`cudaMemcpy2DPeerAsync` does not exist.** There is `cudaMemcpyPeerAsync` and
`cudaMemcpy3DPeerAsync`, but no 2-D variant. With UVA plus peer access, use
`cudaMemcpy2DAsync` with `cudaMemcpyDefault` — it infers direction from the
pointers.

**Don't take the address of a register array.** `reinterpret_cast<float4*>(&acc[i])`
on a thread-local array forces it out of registers into local (off-chip)
memory. Build the `float4` in a temporary instead.

**ReLU must come after the all-reduce in tensor parallelism.** `ReLU(a+b) ≠
ReLU(a)+ReLU(b)`. Getting this backwards still "almost" works, which is worse
than failing outright.

**Benchmark harnesses need read-only inputs.** My first pipeline ping-ponged
activations back into the input buffer, so every timing iteration after the
first computed on garbage. The correctness check passed because it ran first.

---

## Layout

```
common/helpers.cuh       error checking, timing, row-major cuBLAS wrappers
common/collectives.cuh   P2P setup, hand-rolled all-reduce / all-gather
00_probe/                hardware capabilities
01_transpose/            coalescing, bank conflicts, padding vs swizzle
02_matmul/               6 fp32 kernels + WMMA + cuBLAS
03_multigpu_basics/      interconnect bandwidth, stream pipelining
04_data_parallel/        batch split + gradient all-reduce
05_tensor_parallel/      Megatron MLP vs naive split
06_pipeline_parallel/    microbatching and the bubble
07_distributed_transpose all-to-all
08_capstone/             all three strategies, one workload
```

The collectives are hand-rolled rather than NCCL on purpose — at P=2 an
all-reduce is just "exchange and add", and seeing that explicitly is the point.
NCCL is what you'd ship (`nccl.h` is installed on this box); it adds ring/tree
algorithms that matter at larger scale.

## Where to go next

- Rerun 01 and 02 at sizes that straddle the 128 MiB L2 and watch which
  optimisations stop mattering.
- Replace the hand-rolled collectives with NCCL and compare.
- Add a backward pass to 08 so DP has to pay its all-reduce.
- Overlap the DP gradient all-reduce of layer *L* with the backward pass of
  layer *L−1* — that's what makes real DP scale better than the numbers in 04.
- `cudaMallocAsync` + memory pools, and CUDA graphs to kill launch overhead in
  the m=16 pipeline case.
