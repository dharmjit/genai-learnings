// 00b - The four experiments behind post 01.
//
// Post 01 claims a GPU is not a fast CPU: it is a *throughput* machine built
// out of deliberately slow parts. These are the measurements that back it.
//
//   1  single-thread latency   one GPU thread vs one CPU core, same loop
//   2  throughput crossover    same total work, spread over more and more threads
//   3  warp divergence         cost of lanes in a warp disagreeing
//   4  latency hiding          why a GPU needs thousands of threads to go fast
//   5  memory latency          measured, then Little's Law inverted on the curve
//
// Every one of these is a number quoted in the post, so it lives in the repo
// rather than in the prose.
#include "helpers.cuh"
#include <chrono>

// ---------------------------------------------------------- experiment 1&2
// A dependent FMA chain: each op needs the previous result, so this measures
// LATENCY, not throughput. No ILP, no vectorisation, nowhere to hide.
__global__ void kChain(float *out, float a, float b, long long perThread) {
  long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  float v = 1.0f + 1e-6f * (float)tid;
  for (long long i = 0; i < perThread; ++i) v = fmaf(v, a, b);
  // Guarded store the compiler cannot prove is dead.
  if (v == 12345.678f) out[0] = v;
}

// The operands MUST be laundered through volatile. With `a` and `b` visible as
// compile-time constants the whole chain is a constant expression, and the
// host compiler folds 50M iterations down to nothing -- the first version of
// this benchmark reported 0.0 ms and a 434741x speedup, which is what a
// silently deleted loop looks like.
// Two traps here, both of which produced wrong numbers before they were fixed.
//
// 1. The operands MUST be laundered through volatile. With `a` and `b` visible
//    as compile-time constants the whole chain is a constant expression and the
//    host compiler folds 50M iterations to nothing -- the first version of this
//    reported 0.0 ms, which is what a silently deleted loop looks like.
//
// 2. Write `v*a + b`, not `fmaf(v,a,b)`. The default x86-64 target has no FMA3,
//    so `fmaf` compiles to a libm CALL and the CPU looks ~2x slower than it is.
//    The multiply-add form lets the compiler contract to a real FMA instruction
//    (needs -march=native, set in the Makefile), which is what the GPU kernel
//    actually executes. Otherwise this compares an instruction to a subroutine.
static float cpuChain(float a, float b, long long n) {
  volatile float va = a, vb = b;
  const float aa = va, bb = vb;
  float v = 1.0f;
  for (long long i = 0; i < n; ++i) v = v * aa + bb;
  volatile float sink = v;
  return (float)sink;
}

// ------------------------------------------------------------ experiment 3
// `divisor` branches inside one warp. Each lane does the SAME amount of work,
// but the warp must walk every branch its lanes collectively take, so the
// time should scale with the number of distinct branches.
__global__ void kDiverge(float *out, int divisor, int iters, float a, float b) {
  int lane = threadIdx.x & 31;
  int mine = lane % divisor;
  float v = 1.0f;
  for (int k = 0; k < divisor; ++k) {
    if (mine == k) {
      for (int i = 0; i < iters; ++i) v = fmaf(v, a, b);
    }
  }
  if (v == 12345.678f) out[0] = v;
}

// ------------------------------------------------------------ experiment 4
// Pure streaming read. Bandwidth here is a direct function of how many memory
// requests are in flight, which is a direct function of how many warps are
// resident -- that is what "latency hiding" means operationally.
__global__ void kStream(const float4 *__restrict__ in, float *out, size_t n4) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  float4 acc = make_float4(0, 0, 0, 0);
  for (; i < n4; i += stride) {
    float4 v = in[i];
    acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
  }
  if (acc.x == 1e30f) out[0] = acc.x + acc.y + acc.z + acc.w;
}

// ------------------------------------------------------------ experiment 5
// Memory latency, measured rather than assumed.
//
// A pointer chase: the NEXT address is the value you just loaded, so the
// hardware cannot prefetch, cannot pipeline, and cannot overlap. One thread,
// one outstanding load at a time -- so elapsed/steps IS the round-trip latency.
//
// The chain is a full-period LCG over a power-of-two buffer (Hull-Dobell: c
// odd, a = 1 mod 4), which visits every slot exactly once in an order no
// prefetcher will follow, without needing a host-side shuffle.
__global__ void kBuildChain(int *buf, unsigned n, unsigned a, unsigned c) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned stride = gridDim.x * blockDim.x;
  for (; i < n; i += stride) buf[i] = (int)((a * i + c) & (n - 1));
}

__global__ void kChase(const int *__restrict__ buf, int start, int steps,
                       int *out) {
  int i = start;
  for (int k = 0; k < steps; ++k) i = buf[i];
  if (i == -1) out[0] = i; // never taken; keeps the chain alive
}

int main() {
  CUDA_CHECK(cudaSetDevice(0));
  cudaDeviceProp p;
  CUDA_CHECK(cudaGetDeviceProperties(&p, 0));

  int clockKHz = 0, regsPerSM = 0;
  cudaDeviceGetAttribute(&clockKHz, cudaDevAttrClockRate, 0);
  cudaDeviceGetAttribute(&regsPerSM, cudaDevAttrMaxRegistersPerMultiprocessor, 0);
  printf("GPU: %s\n", p.name);
  printf("SMs %d · max %d threads/SM · %d threads/GPU resident · SM clock %.2f GHz\n",
         p.multiProcessorCount, p.maxThreadsPerMultiProcessor,
         p.multiProcessorCount * p.maxThreadsPerMultiProcessor,
         clockKHz / 1.0e6);

  // The price of residency. A thread is only cheap to switch to because its
  // state already lives on chip -- and that state is the chip's real budget.
  printf("\n0. THE COST OF KEEPING THREADS RESIDENT\n");
  printf("   registers / SM        %8d (32-bit) = %.0f KiB\n", regsPerSM,
         regsPerSM * 4.0 / 1024);
  printf("   registers / thread    %8.1f at full occupancy\n",
         (double)regsPerSM / p.maxThreadsPerMultiProcessor);
  printf("   register file, whole GPU %5.1f MiB across %d SMs\n",
         regsPerSM * 4.0 * p.multiProcessorCount / 1048576.0,
         p.multiProcessorCount);
  printf("   --> this is where the transistor budget went: thread state,\n"
         "       not per-thread cleverness.\n\n");

  float *sink;
  CUDA_CHECK(cudaMalloc(&sink, sizeof(float) * 4));
  const float A = 0.9999999f, B = 1e-7f;

  // ============================================ 1. single-thread latency
  {
    const long long N = 50'000'000;
    auto gpu1 = [&] { kChain<<<1, 1>>>(sink, A, B, N); };
    double gpuMs = timeKernelMs(gpu1, 3, 2000.0);

    auto t0 = std::chrono::high_resolution_clock::now();
    cpuChain(A, B, N);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpuMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    printf("1. SINGLE-THREAD LATENCY  (%lld dependent FMAs, one thread each)\n", N);
    printf("   one GPU thread   %8.1f ms   %5.2f ns/FMA\n", gpuMs, gpuMs * 1e6 / N);
    printf("   one CPU core     %8.1f ms   %5.2f ns/FMA\n", cpuMs, cpuMs * 1e6 / N);
    printf("   --> the GPU thread is %.2fx %s\n\n",
           gpuMs > cpuMs ? gpuMs / cpuMs : cpuMs / gpuMs,
           gpuMs > cpuMs ? "SLOWER" : "faster");
  }

  // ============================================ 2. throughput crossover
  {
    const long long TOTAL = 200'000'000; // same total FMAs in every row
    printf("2. THROUGHPUT CROSSOVER  (%lld FMAs total, spread wider each row)\n", TOTAL);
    printf("   %12s %12s %10s %14s\n", "threads", "per thread", "ms", "GFLOP/s");
    for (long long threads : {1LL, 32LL, 1024LL, 32768LL, 1048576LL, 8388608LL}) {
      long long per = TOTAL / threads;
      int block = threads < 256 ? (int)threads : 256;
      long long grid = (threads + block - 1) / block;
      auto run = [&] { kChain<<<(int)grid, block>>>(sink, A, B, per); };
      double ms = timeKernelMs(run, 3, 2000.0);
      printf("   %12lld %12lld %10.2f %14.1f\n", threads, per, ms,
             (double)TOTAL / (ms * 1e-3) / 1e9);
    }
    printf("   --> one thread is slow; a million of them are not.\n\n");
  }

  // ============================================ 3. warp divergence
  {
    const int ITERS = 1024, BLOCKS = 4096, BLOCK = 256;
    printf("3. WARP DIVERGENCE  (identical work per lane, %d FMAs each)\n", ITERS);
    printf("   %10s %12s %10s\n", "branches", "ms", "vs 1 branch");
    double base = 0;
    for (int d : {1, 2, 4, 8, 16, 32}) {
      auto run = [&] { kDiverge<<<BLOCKS, BLOCK>>>(sink, d, ITERS, A, B); };
      double ms = timeKernelMs(run, 3, 1500.0);
      if (d == 1) base = ms;
      printf("   %10d %12.3f %9.2fx\n", d, ms, ms / base);
    }
    printf("   --> every lane did the same work; the warp paid for all of it.\n\n");
  }

  // ============================================ 4. latency hiding
  {
    size_t bytes = 1ull << 30;
    float *buf;
    CUDA_CHECK(cudaMalloc(&buf, bytes));
    CUDA_CHECK(cudaMemset(buf, 0, bytes));
    size_t n4 = bytes / sizeof(float4);
    int maxResident = p.multiProcessorCount * p.maxThreadsPerMultiProcessor;

    printf("4. LATENCY HIDING  (1 GiB streaming read, varying threads in flight)\n");
    printf("   %12s %10s %10s %12s\n", "threads", "% of max", "ms", "GB/s");
    std::vector<std::pair<long long, double>> curve;
    for (int blocks : {1, 8, 47, 188, 376, 752, 1504, 3008}) {
      int block = 256;
      auto run = [&] { kStream<<<blocks, block>>>((const float4 *)buf, sink, n4); };
      double ms = timeKernelMs(run, 3, 1500.0);
      long long thr = (long long)blocks * block;
      double gbs = copyGBs(bytes, ms);
      curve.push_back({thr, gbs});
      printf("   %12lld %9.1f%% %10.2f %12.1f\n", thr,
             100.0 * thr / maxResident, ms, gbs);
    }
    printf("   --> the memory system is only fast if enough warps are asking.\n\n");

    // ==================================== 5. latency, and Little's Law
    // Everything above says "more requests in flight = more bandwidth".
    // Little's Law makes that quantitative -- but only if you MEASURE the
    // latency instead of picking a number that makes the arithmetic work.
    const unsigned N = 1u << 28;            // 1 GiB of int, 8x the 128 MiB L2
    const int STEPS = 200000;
    int *chain;
    CUDA_CHECK(cudaMalloc(&chain, (size_t)N * sizeof(int)));
    kBuildChain<<<1024, 256>>>(chain, N, 1664525u, 1013904223u);
    CUDA_SYNC_CHECK();
    double chaseMs = timeKernelMs([&] { kChase<<<1, 1>>>(chain, 0, STEPS, (int *)sink); },
                                  5, 1500.0);
    double latNs = chaseMs * 1e6 / STEPS;
    double peak = 0;
    for (auto &r : curve) peak = std::max(peak, r.second);

    printf("5. MEMORY LATENCY, AND LITTLE'S LAW\n");
    printf("   idle round-trip latency   %7.1f ns  (%.0f cycles at %.2f GHz)\n",
           latNs, latNs * clockKHz / 1.0e6, clockKHz / 1.0e6);
    printf("   throughput = concurrency / latency, so to sustain %.0f GB/s:\n", peak);
    printf("     %.0f GB/s x %.0f ns = %.0f KB in flight\n",
           peak, latNs, peak * latNs / 1000.0);
    printf("     at 16 B per thread that is only ~%.0f threads\n\n",
           peak * latNs / 1000.0 * 1000.0 / 16.0);
    printf("   But experiment 4 needed far more than that. Invert the law on the\n"
           "   measured curve to see the latency each row ACTUALLY experienced:\n");
    printf("   %12s %12s %14s %12s\n", "threads", "GB/s", "in flight (KB)",
           "implied lat");
    for (auto &r : curve) {
      if (r.second < 0.25 * peak) continue;           // pre-knee rows are noise
      double inflight = r.first * 16.0 / 1000.0;      // KB
      double implied = inflight * 1000.0 / r.second;  // ns
      printf("   %12lld %12.1f %14.0f %10.0f ns\n", r.first, r.second,
             inflight, implied);
    }
    printf("   --> latency is not a constant. It degrades as you approach\n"
           "       saturation, so you need more threads than the idle number\n"
           "       suggests. Little's Law gives a FLOOR, not a target.\n");
    CUDA_CHECK(cudaFree(chain));
    CUDA_CHECK(cudaFree(buf));
  }

  CUDA_CHECK(cudaFree(sink));
  return 0;
}
