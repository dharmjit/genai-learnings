// 03b - Streams: turning three serial phases into a pipeline.
//
// A GPU has separate copy engines and SMs (asyncEngineCount tells you how
// many DMA engines). If you do H2D -> compute -> D2H as one blocking sequence,
// two of the three units idle at all times. Chop the work into chunks on
// independent streams and the phases overlap.
//
// This is the same idea as pipeline parallelism, just inside one device: the
// bubble at the start and end is unavoidable, the middle is full.
//
// Requires pinned host memory -- pageable memory cannot be DMA'd, so
// cudaMemcpyAsync on pageable memory silently becomes synchronous.
#include "helpers.cuh"

// Deliberately compute-heavy so the kernel phase is comparable to the copies.
__global__ void kBusy(float *d, int n, int iters) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = d[i];
  for (int k = 0; k < iters; ++k) v = fmaf(v, 0.9999f, 0.0001f);
  d[i] = v;
}

int main(int argc, char **argv) {
  int nMiB = argc > 1 ? atoi(argv[1]) : 512;
  int work = argc > 2 ? atoi(argv[2]) : 256; // FMAs per element
  size_t n = ((size_t)nMiB << 20) / sizeof(float);
  size_t bytes = n * sizeof(float);
  int nGpu = 0;
  CUDA_CHECK(cudaGetDeviceCount(&nGpu));

  printf("%d MiB, %d FMA/element, %d GPUs visible\n\n", nMiB, work, nGpu);

  float *h;
  CUDA_CHECK(cudaMallocHost(&h, bytes)); // pinned: required for async copies
  for (size_t i = 0; i < n; ++i) h[i] = 1.f;

  std::vector<Row> table;

  // ---------------------------------------------------- 1 GPU, no overlap
  {
    CUDA_CHECK(cudaSetDevice(0));
    float *d;
    CUDA_CHECK(cudaMalloc(&d, bytes));
    auto run = [&] {
      CUDA_CHECK(cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice));
      kBusy<<<ceilDiv((int)n, 256), 256>>>(d, (int)n, work);
      CUDA_CHECK(cudaMemcpy(h, d, bytes, cudaMemcpyDeviceToHost));
    };
    double ms = timeKernelMs(run, 10, 800.0);
    table.push_back({"1 GPU serial (no streams)", ms, copyGBs(2 * bytes, ms), 0,
                     true});
    CUDA_CHECK(cudaFree(d));
  }

  // ---------------------------------------------- 1 GPU, S-way pipeline
  for (int S : {2, 4, 8}) {
    CUDA_CHECK(cudaSetDevice(0));
    std::vector<cudaStream_t> st(S);
    std::vector<float *> d(S);
    size_t chunk = ceilDiv((int)n, S);
    for (int i = 0; i < S; ++i) {
      CUDA_CHECK(cudaStreamCreate(&st[i]));
      CUDA_CHECK(cudaMalloc(&d[i], chunk * sizeof(float)));
    }
    auto run = [&] {
      for (int i = 0; i < S; ++i) {
        size_t off = i * chunk, len = std::min(chunk, n - off);
        if ((long long)len <= 0) continue;
        size_t b = len * sizeof(float);
        CUDA_CHECK(cudaMemcpyAsync(d[i], h + off, b, cudaMemcpyHostToDevice, st[i]));
        kBusy<<<ceilDiv((int)len, 256), 256, 0, st[i]>>>(d[i], (int)len, work);
        CUDA_CHECK(cudaMemcpyAsync(h + off, d[i], b, cudaMemcpyDeviceToHost, st[i]));
      }
      for (int i = 0; i < S; ++i) CUDA_CHECK(cudaStreamSynchronize(st[i]));
    };
    double ms = timeKernelMs(run, 10, 800.0);
    char nm[64];
    snprintf(nm, sizeof nm, "1 GPU, %d streams", S);
    table.push_back({nm, ms, copyGBs(2 * bytes, ms), 0, true});
    for (int i = 0; i < S; ++i) {
      CUDA_CHECK(cudaStreamDestroy(st[i]));
      CUDA_CHECK(cudaFree(d[i]));
    }
  }

  // ------------------------------------------- 2 GPUs, 4 streams each
  if (nGpu >= 2) {
    const int S = 4, G = 2;
    std::vector<std::vector<cudaStream_t>> st(G, std::vector<cudaStream_t>(S));
    std::vector<std::vector<float *>> d(G, std::vector<float *>(S));
    size_t half = n / G, chunk = ceilDiv((int)half, S);
    for (int g = 0; g < G; ++g) {
      CUDA_CHECK(cudaSetDevice(g));
      for (int i = 0; i < S; ++i) {
        CUDA_CHECK(cudaStreamCreate(&st[g][i]));
        CUDA_CHECK(cudaMalloc(&d[g][i], chunk * sizeof(float)));
      }
    }
    auto run = [&] {
      for (int g = 0; g < G; ++g) {
        CUDA_CHECK(cudaSetDevice(g));
        for (int i = 0; i < S; ++i) {
          size_t off = g * half + i * chunk;
          size_t len = std::min(chunk, (g * half + half) - off);
          if ((long long)len <= 0) continue;
          size_t b = len * sizeof(float);
          CUDA_CHECK(cudaMemcpyAsync(d[g][i], h + off, b,
                                     cudaMemcpyHostToDevice, st[g][i]));
          kBusy<<<ceilDiv((int)len, 256), 256, 0, st[g][i]>>>(d[g][i], (int)len, work);
          CUDA_CHECK(cudaMemcpyAsync(h + off, d[g][i], b,
                                     cudaMemcpyDeviceToHost, st[g][i]));
        }
      }
      for (int g = 0; g < G; ++g) {
        CUDA_CHECK(cudaSetDevice(g));
        for (int i = 0; i < S; ++i) CUDA_CHECK(cudaStreamSynchronize(st[g][i]));
      }
    };
    double ms = timeKernelMs(run, 10, 800.0);
    table.push_back({"2 GPUs, 4 streams each", ms, copyGBs(2 * bytes, ms), 0,
                     true});
  }

  double base = table[0].ms;
  for (auto &r : table) r.pctOfRef = 100.0 * base / r.ms;
  printTable("H2D -> compute -> D2H pipelining", "GB/s (h<->d)", table);
  printf("\n'vs ref' here is speedup over the serial version.\n"
         "Note the 2-GPU row: PCIe is now the shared bottleneck, so it does\n"
         "not double. Splitting compute is easy; splitting bandwidth is not.\n");

  CUDA_CHECK(cudaFreeHost(h));
  return 0;
}
