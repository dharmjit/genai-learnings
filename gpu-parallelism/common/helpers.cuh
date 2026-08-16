#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <random>
#include <algorithm>

#define CUDA_CHECK(x)                                                          \
  do {                                                                         \
    cudaError_t e_ = (x);                                                      \
    if (e_ != cudaSuccess) {                                                   \
      fprintf(stderr, "CUDA %s:%d '%s' -> %s\n", __FILE__, __LINE__, #x,       \
              cudaGetErrorString(e_));                                         \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define CUBLAS_CHECK(x)                                                        \
  do {                                                                         \
    cublasStatus_t s_ = (x);                                                   \
    if (s_ != CUBLAS_STATUS_SUCCESS) {                                         \
      fprintf(stderr, "cuBLAS %s:%d '%s' -> %d\n", __FILE__, __LINE__, #x,     \
              (int)s_);                                                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// Launch-time errors are async; call this after a kernel when you care.
#define CUDA_SYNC_CHECK()                                                      \
  do {                                                                         \
    CUDA_CHECK(cudaGetLastError());                                            \
    CUDA_CHECK(cudaDeviceSynchronize());                                       \
  } while (0)

inline int ceilDiv(int a, int b) { return (a + b - 1) / b; }

// ---------------------------------------------------------------- host data

inline void fillRandom(std::vector<float> &v, unsigned seed = 1234) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> d(-1.f, 1.f);
  for (auto &x : v) x = d(rng);
}

// Relative L2 error is the right metric for GEMM: elementwise tolerances are
// meaningless once K is large and summation order differs between kernels.
inline double relL2(const std::vector<float> &got,
                    const std::vector<float> &ref) {
  double num = 0, den = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    double d = (double)got[i] - (double)ref[i];
    num += d * d;
    den += (double)ref[i] * (double)ref[i];
  }
  return std::sqrt(num / (den + 1e-30));
}

inline bool checkAndReport(const char *name, const std::vector<float> &got,
                           const std::vector<float> &ref, double tol = 1e-4) {
  double e = relL2(got, ref);
  bool ok = e <= tol;
  if (!ok) fprintf(stderr, "  [FAIL] %s relL2=%.3e (tol %.1e)\n", name, e, tol);
  return ok;
}

// ------------------------------------------------------------------ timing

struct GpuTimer {
  cudaEvent_t a, b;
  GpuTimer() {
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));
  }
  ~GpuTimer() {
    cudaEventDestroy(a);
    cudaEventDestroy(b);
  }
  void start(cudaStream_t s = 0) { CUDA_CHECK(cudaEventRecord(a, s)); }
  float stop(cudaStream_t s = 0) {
    CUDA_CHECK(cudaEventRecord(b, s));
    CUDA_CHECK(cudaEventSynchronize(b));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    return ms;
  }
};

// Median of a few timed runs. Median, not mean: one stray clock-throttle
// sample otherwise dominates the average.
//
// Iteration count adapts to a wall-clock budget, because the kernels compared
// here span three orders of magnitude -- a fixed count either gives the fast
// kernels too few samples or makes the naive ones take minutes.
template <class F>
double timeKernelMs(F &&f, int maxIters = 20, double budgetMs = 500.0) {
  f(); // warm up: first launch pays JIT/context/allocation costs
  CUDA_CHECK(cudaDeviceSynchronize());
  GpuTimer t;
  t.start();
  f();
  double first = t.stop();

  int iters = (int)(budgetMs / std::max(first, 1e-3));
  iters = std::max(1, std::min(maxIters, iters));
  std::vector<float> samples{(float)first};
  for (int i = 1; i < iters; ++i) {
    t.start();
    f();
    samples.push_back(t.stop());
  }
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

// ------------------------------------------------------------- reporting

inline double gemmTFLOPs(long long M, long long N, long long K, double ms) {
  return (2.0 * M * N * K) / (ms * 1e-3) / 1e12;
}

// Transpose moves each byte twice (one read + one write).
inline double copyGBs(size_t bytesMoved, double ms) {
  return (double)bytesMoved / (ms * 1e-3) / 1e9;
}

struct Row {
  std::string name;
  double ms;
  double metric;   // TFLOP/s or GB/s
  double pctOfRef; // vs the reference row
  bool correct;
};

inline void printTable(const char *title, const char *metricName,
                       const std::vector<Row> &rows) {
  printf("\n%s\n", title);
  printf("%-34s %10s %12s %9s %6s\n", "kernel", "ms", metricName, "vs ref",
         "ok");
  printf("%-34s %10s %12s %9s %6s\n", "----------------------------------",
         "----------", "------------", "---------", "------");
  for (const auto &r : rows)
    printf("%-34s %10.3f %12.2f %8.1f%% %6s\n", r.name.c_str(), r.ms, r.metric,
           r.pctOfRef, r.correct ? "yes" : "NO");
}

// ------------------------------------------------------- row-major gemm
//
// cuBLAS is column-major. A row-major [M x N] buffer is bit-identical to a
// column-major [N x M] one, so row-major C = A@B is computed by asking cuBLAS
// for C^T = B^T @ A^T -- i.e. pass B and A swapped, with no transpose flags.

// C[M x N] = alpha * A[M x K] @ B[K x N] + beta * C   (all row-major)
inline void rmGemm(cublasHandle_t h, int M, int N, int K, const float *A,
                   const float *B, float *C, float alpha = 1.f,
                   float beta = 0.f) {
  CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N,
                           A, K, &beta, C, N));
}

// C[M x N] = alpha * A[K x M]^T @ B[K x N] + beta * C  (all row-major).
// This is the weight-gradient shape: dW = X^T @ dY.
inline void rmGemmTN(cublasHandle_t h, int M, int N, int K, const float *A,
                     const float *B, float *C, float alpha = 1.f,
                     float beta = 0.f) {
  CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_T, N, M, K, &alpha, B, N,
                           A, M, &beta, C, N));
}

// --------------------------------------------------------------- devices

inline void printDeviceBanner() {
  int n = 0;
  CUDA_CHECK(cudaGetDeviceCount(&n));
  printf("visible GPUs: %d\n", n);
  for (int i = 0; i < n; ++i) {
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, i));
    printf("  [%d] %s  sm_%d%d  %.0f GiB  %d SMs  %.0f GB/s peak\n", i, p.name,
           p.major, p.minor, p.totalGlobalMem / 1073741824.0,
           p.multiProcessorCount,
           2.0 * p.memoryClockRate * (p.memoryBusWidth / 8) / 1.0e6);
  }
}

inline int requireTwoGpus() {
  int n = 0;
  CUDA_CHECK(cudaGetDeviceCount(&n));
  if (n < 2) {
    fprintf(stderr, "This program needs 2 GPUs; found %d.\n", n);
    exit(2);
  }
  return n;
}
