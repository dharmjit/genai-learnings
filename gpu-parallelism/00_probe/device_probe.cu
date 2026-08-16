// 00 - What am I actually running on?
//
// Every optimisation later is justified by a number printed here: SM count
// sets your occupancy target, shared memory per block caps your tile size,
// and the P2P matrix decides whether tensor parallelism is even a good idea.
#include "helpers.cuh"
#include "collectives.cuh"

__global__ void kTouch(float *p) { p[threadIdx.x] = threadIdx.x; }

static void probeOne(int dev) {
  cudaDeviceProp p;
  CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
  printf("\nGPU %d: %s\n", dev, p.name);
  printf("  compute capability      sm_%d%d\n", p.major, p.minor);
  printf("  SMs                     %d\n", p.multiProcessorCount);
  printf("  max threads / SM        %d\n", p.maxThreadsPerMultiProcessor);
  printf("  max threads / block     %d\n", p.maxThreadsPerBlock);
  printf("  warp size               %d\n", p.warpSize);
  printf("  registers / block       %d\n", p.regsPerBlock);
  printf("  shared mem / block      %zu KiB\n", p.sharedMemPerBlock / 1024);
  printf("  shared mem / SM         %zu KiB\n",
         p.sharedMemPerMultiprocessor / 1024);
  printf("  L2 cache                %d MiB\n", p.l2CacheSize / (1024 * 1024));
  printf("  global memory           %.1f GiB\n", p.totalGlobalMem / 1073741824.0);
  printf("  memory bus              %d-bit\n", p.memoryBusWidth);
  printf("  async engines           %d\n", p.asyncEngineCount);
  printf("  unified addressing      %s\n", p.unifiedAddressing ? "yes" : "no");
  printf("  concurrent kernels      %s\n", p.concurrentKernels ? "yes" : "no");
  printf("  ECC                     %s\n", p.ECCEnabled ? "on" : "off");
}

// Achieved bandwidth beats the spec sheet. This is a pure-read reduction over
// a buffer far larger than L2, so it measures the HBM/GDDR path honestly.
__global__ void kStreamRead(const float4 *__restrict__ in, float *out,
                            size_t n4) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  float4 acc = make_float4(0, 0, 0, 0);
  for (; i < n4; i += stride) {
    float4 v = in[i];
    acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
  }
  if (acc.x == 1e30f) out[0] = acc.x + acc.y + acc.z + acc.w; // never taken
}

static double measureReadBw(int dev, size_t bytes) {
  CUDA_CHECK(cudaSetDevice(dev));
  float *buf = nullptr, *sink = nullptr;
  CUDA_CHECK(cudaMalloc(&buf, bytes));
  CUDA_CHECK(cudaMalloc(&sink, sizeof(float)));
  CUDA_CHECK(cudaMemset(buf, 0, bytes));
  size_t n4 = bytes / sizeof(float4);
  auto run = [&] { kStreamRead<<<2048, 256>>>((const float4 *)buf, sink, n4); };
  double ms = timeKernelMs(run, 3, 10);
  CUDA_CHECK(cudaFree(buf));
  CUDA_CHECK(cudaFree(sink));
  return copyGBs(bytes, ms);
}

int main() {
  int n = 0;
  CUDA_CHECK(cudaGetDeviceCount(&n));
  printf("CUDA devices visible: %d\n", n);
  int rt = 0, drv = 0;
  CUDA_CHECK(cudaRuntimeGetVersion(&rt));
  CUDA_CHECK(cudaDriverGetVersion(&drv));
  printf("runtime %d.%d, driver supports up to %d.%d\n", rt / 1000,
         (rt % 1000) / 10, drv / 1000, (drv % 1000) / 10);

  for (int d = 0; d < n; ++d) probeOne(d);

  printf("\nachieved device-memory read bandwidth (1 GiB buffer)\n");
  for (int d = 0; d < n; ++d)
    printf("  GPU %d: %7.1f GB/s\n", d, measureReadBw(d, 1ull << 30));

  printf("\npeer access matrix (can device R read device C directly?)\n     ");
  for (int c = 0; c < n; ++c) printf("%6d", c);
  printf("\n");
  for (int r = 0; r < n; ++r) {
    printf("  %2d ", r);
    for (int c = 0; c < n; ++c) {
      if (r == c) { printf("%6s", "self"); continue; }
      int can = 0;
      CUDA_CHECK(cudaDeviceCanAccessPeer(&can, r, c));
      printf("%6s", can ? "yes" : "no");
    }
    printf("\n");
  }

  if (n >= 2) {
    printf("\n");
    enablePeerAccess(0, 1);
  }
  printf("\nprobe OK\n");
  return 0;
}
