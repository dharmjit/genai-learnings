---
series: GPU Parallelism, Measured
subtitle: Learning GPU parallelism by measuring it
post: NN
title: <post title>
hook: <the measured number that shouldn't be true>
takeaway: <the one sentence>
runs: ./bin/<binary>
---

## <The number that shouldn't be true>

Two or three sentences and one figure. No preamble, no "in this post we will".
The hook is already measured — lead with it.

> ### Key takeaway
> **<One sentence. The single thing a reader keeps if they keep nothing else.>**
> One or two more sentences of framing, then stop.

Placement is deliberate: AFTER the hook, so the surprising number still lands
first, but before anything else, so the reader carries the thesis through the
whole post. Echo the same sentence in the closing section — first thing read
and last thing read.

Exactly one takeaway per post. If you cannot compress the post to one sentence,
the post is doing two jobs.

<!-- HERO VISUAL -->

---

## Instance setup

> Identical in every post so a reader can start anywhere. Only the last two
> lines change. Keep it collapsed or skimmable — returning readers skip it in
> one scroll.

You need a Linux box with an NVIDIA GPU. Posts 01–03 run on **one** GPU; posts
04–07 need **two**. Everything here was measured on 2× RTX PRO 6000 Blackwell
(sm_120), but any pair of NVIDIA cards works — the code probes P2P at runtime
and falls back to host-staged copies.

```bash
# 1. Most rented GPU pods ship a CUDA *runtime*, not a toolkit:
#    no nvcc, no headers. Check first.
which nvcc || echo "no toolkit — run setup_pod.sh"

# 2. Get the code
git clone https://github.com/dharmjit/genai-learnings
cd genai-learnings/gpu-parallelism

# 3. Install the toolkit if step 1 came up empty.
#    (Do NOT just `apt install cuda-toolkit` — see the note below.)
sudo bash setup_pod.sh

# 4. Build
export PATH=/usr/local/cuda-12.9/bin:$PATH
make -j

# 5. This post
./bin/<binary>        # ~NN s
```

> **If `apt install cuda-toolkit-12-9` fails with "held broken packages":** the
> image pins `libcublas` to an older version than the repo's dev package
> demands. `setup_pod.sh` pins each `-dev` package to the held runtime version.
> It also warns you if `/dev/shm` is the 64 MB Docker default — which will bite
> you later with PyTorch dataloaders and NCCL.

---

## Build the intuition

The concept in plain language, carried by the hero animation. **A beginner must
be able to stop here and still have learned the main idea.**

<!-- SUPPORTING VISUAL -->

---

## Run it yourself

The command, the real output table pasted verbatim, and one sentence on what to
look at first.

```
<paste from ../results/RESULTS.txt — never retype a number>
```

---

## The part the tutorials skip

> Clearly marked so beginners can skip it guilt-free and practitioners know
> where to land. This is where the nuance, the failure modes, and the war
> stories live. It is why the series is worth reading twice.

---

## Try this

One concrete modification with a predicted outcome — change a size, disable
P2P, vary the microbatch count. Readers who run one experiment retain the post;
readers who only scroll do not.

---

## Next

One line that plants the next hook. The posts chain: 02 ends DRAM-bound, 03
asks how to stop being memory-bound, 04 asks what happens when you add a second
card.
