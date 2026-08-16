#pragma once
#include "helpers.cuh"

// Hand-rolled 2-GPU collectives.
//
// NCCL would be faster and is what you'd ship, but it hides the thing you are
// here to learn. Every collective below is: a peer copy plus (sometimes) an
// elementwise kernel. That is all a "ring all-reduce" is, unrolled to N=2.

// --------------------------------------------------------------------- P2P

struct P2P {
  bool enabled = false; // true => cudaMemcpyPeer moves data device->device
  // Without P2P the runtime still services cudaMemcpyPeer correctly, but it
  // stages through a host bounce buffer: two PCIe crossings instead of one.
};

inline P2P enablePeerAccess(int a, int b, bool verbose = true) {
  P2P p;
  int ab = 0, ba = 0;
  CUDA_CHECK(cudaDeviceCanAccessPeer(&ab, a, b));
  CUDA_CHECK(cudaDeviceCanAccessPeer(&ba, b, a));
  if (ab && ba) {
    CUDA_CHECK(cudaSetDevice(a));
    cudaError_t e = cudaDeviceEnablePeerAccess(b, 0);
    if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) CUDA_CHECK(e);
    CUDA_CHECK(cudaSetDevice(b));
    e = cudaDeviceEnablePeerAccess(a, 0);
    if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) CUDA_CHECK(e);
    cudaGetLastError(); // clear the benign AlreadyEnabled sticky state
    p.enabled = true;
  }
  if (verbose)
    printf("P2P %d<->%d: %s%s\n", a, b, p.enabled ? "ENABLED" : "unavailable",
           p.enabled ? "" : "  (peer copies stage through host RAM)");
  return p;
}

// ------------------------------------------------------- per-GPU handles

struct Gpu {
  int id;
  cudaStream_t compute;
  cudaStream_t comm; // separate stream so comm can overlap compute
  cublasHandle_t blas;
  cudaEvent_t ev;  // ordering *between* devices inside a collective
  cudaEvent_t evc; // handoff from this device's compute stream to its comm
};

inline void gpuInit(Gpu &g, int id) {
  g.id = id;
  CUDA_CHECK(cudaSetDevice(id));
  CUDA_CHECK(cudaStreamCreate(&g.compute));
  CUDA_CHECK(cudaStreamCreate(&g.comm));
  CUDA_CHECK(cudaEventCreateWithFlags(&g.ev, cudaEventDisableTiming));
  CUDA_CHECK(cudaEventCreateWithFlags(&g.evc, cudaEventDisableTiming));
  CUBLAS_CHECK(cublasCreate(&g.blas));
  CUBLAS_CHECK(cublasSetStream(g.blas, g.compute));
}

inline void gpuDestroy(Gpu &g) {
  CUDA_CHECK(cudaSetDevice(g.id));
  cublasDestroy(g.blas);
  cudaEventDestroy(g.evc);
  cudaEventDestroy(g.ev);
  cudaStreamDestroy(g.comm);
  cudaStreamDestroy(g.compute);
}

// Barrier: no comm stream may start until EVERY device's compute stream has
// finished.
//
// The subtle part -- and a bug I actually hit writing this -- is that it is
// not enough for GPU i's comm stream to wait on GPU i's compute stream. In an
// all-reduce, GPU 0's comm stream READS GPU 1's buffer, so it must also wait
// on GPU 1's compute. Ordering only against your own device leaves a
// cross-device race that produces plausible-looking but wrong numbers, and
// only intermittently: it depends on which GEMM happens to finish first.
inline void commAfterCompute(Gpu *g, int n) {
  for (int i = 0; i < n; ++i) {
    CUDA_CHECK(cudaSetDevice(g[i].id));
    CUDA_CHECK(cudaEventRecord(g[i].evc, g[i].compute));
  }
  for (int i = 0; i < n; ++i) {
    CUDA_CHECK(cudaSetDevice(g[i].id));
    for (int j = 0; j < n; ++j)
      CUDA_CHECK(cudaStreamWaitEvent(g[i].comm, g[j].evc, 0));
  }
}

// The reverse barrier, for when the next compute step consumes collective
// output produced on some other device's comm stream.
inline void computeAfterComm(Gpu *g, int n) {
  for (int i = 0; i < n; ++i) {
    CUDA_CHECK(cudaSetDevice(g[i].id));
    CUDA_CHECK(cudaEventRecord(g[i].evc, g[i].comm));
  }
  for (int i = 0; i < n; ++i) {
    CUDA_CHECK(cudaSetDevice(g[i].id));
    for (int j = 0; j < n; ++j)
      CUDA_CHECK(cudaStreamWaitEvent(g[i].compute, g[j].evc, 0));
  }
}

inline void syncAll(Gpu *g, int n) {
  for (int i = 0; i < n; ++i) {
    CUDA_CHECK(cudaSetDevice(g[i].id));
    CUDA_CHECK(cudaDeviceSynchronize());
  }
}

inline void waitComm(Gpu *g, int n) {
  for (int i = 0; i < n; ++i) {
    CUDA_CHECK(cudaSetDevice(g[i].id));
    CUDA_CHECK(cudaStreamSynchronize(g[i].comm));
  }
}

// ---------------------------------------------------------------- kernels

__global__ void kAddInto(float *dst, const float *src, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride) dst[i] += src[i];
}

__global__ void kScale(float *x, float s, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride) x[i] *= s;
}

// Grid-stride launch: fixed grid, kernel loops. Decouples grid size from n.
inline void launchAdd(float *dst, const float *src, size_t n, cudaStream_t s) {
  kAddInto<<<1024, 256, 0, s>>>(dst, src, n);
}

// ------------------------------------------------------------ collectives

// allReduce(sum) over 2 GPUs.
//   buf[g]     : in/out, n floats, owned by GPU g
//   scratch[g] : n floats of scratch on GPU g
//
// Volume: each GPU sends n and receives n floats. A ring all-reduce over P
// GPUs moves 2*n*(P-1)/P per rank; at P=2 that is exactly n, so this simple
// exchange-and-add is already the optimal-volume algorithm.
//
// The event dance is load-bearing: GPU0's copy READS buf[1] while GPU1's add
// WRITES buf[1]. Streams on different devices are not ordered against each
// other, so without the cross-device waits this races and silently corrupts.
inline void allReduceSum2(Gpu *g, float **buf, float **scratch, size_t n) {
  size_t bytes = n * sizeof(float);

  CUDA_CHECK(cudaSetDevice(g[0].id));
  CUDA_CHECK(cudaMemcpyPeerAsync(scratch[0], g[0].id, buf[1], g[1].id, bytes,
                                 g[0].comm));
  CUDA_CHECK(cudaEventRecord(g[0].ev, g[0].comm));

  CUDA_CHECK(cudaSetDevice(g[1].id));
  CUDA_CHECK(cudaMemcpyPeerAsync(scratch[1], g[1].id, buf[0], g[0].id, bytes,
                                 g[1].comm));
  CUDA_CHECK(cudaEventRecord(g[1].ev, g[1].comm));

  // Each side may only clobber its own buf once the peer has finished reading.
  CUDA_CHECK(cudaSetDevice(g[0].id));
  CUDA_CHECK(cudaStreamWaitEvent(g[0].comm, g[1].ev, 0));
  launchAdd(buf[0], scratch[0], n, g[0].comm);

  CUDA_CHECK(cudaSetDevice(g[1].id));
  CUDA_CHECK(cudaStreamWaitEvent(g[1].comm, g[0].ev, 0));
  launchAdd(buf[1], scratch[1], n, g[1].comm);
}

// allGather over 2 GPUs: GPU g contributes shard[g] (n floats), everyone ends
// up with the concatenated 2n-float out[g]. Half the wire traffic of an
// all-reduce and no arithmetic -- and no race, since shards are read-only.
inline void allGather2(Gpu *g, float **out, float **shard, size_t n) {
  size_t bytes = n * sizeof(float);
  for (int i = 0; i < 2; ++i) {
    int peer = 1 - i;
    CUDA_CHECK(cudaSetDevice(g[i].id));
    CUDA_CHECK(cudaMemcpyAsync(out[i] + (size_t)i * n, shard[i], bytes,
                               cudaMemcpyDeviceToDevice, g[i].comm));
    CUDA_CHECK(cudaMemcpyPeerAsync(out[i] + (size_t)peer * n, g[i].id,
                                   shard[peer], g[peer].id, bytes, g[i].comm));
  }
}
