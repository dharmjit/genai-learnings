# genai-learnings

Hands-on notes on GPU and ML systems, written by building the thing and
measuring it. Each topic is a self-contained folder: runnable code, a
benchmark harness that verifies its own numerics, and the writing that came
out of it.

The rule for everything here: **no number gets published unless it was
measured on real hardware**, including the results that contradict the
textbook.

## Topics

| | topic | what it covers | status |
|---|---|---|---|
| 01 | [**gpu-parallelism**](gpu-parallelism/) | CUDA from a single warp to two GPUs — memory coalescing, bank conflicts, the matmul ladder, tensor cores, and data / tensor / pipeline parallelism | code complete, series in progress |

Companion writing: **GPU Parallelism, Measured** — a seven-post series.

## Layout

Every topic folder follows the same shape:

```
<topic>/
├── README.md      the technical writeup — start here
├── results/       raw measured output, the source of truth for every chart
├── viz/           chart and animation sources, generated from results/
└── posts/         drafts of the published writing
```

## License

MIT. See [LICENSE](LICENSE). Use the code however you like; if it saves you a
day of debugging, that was the point.
