// 03a - The interconnect, measured.
//
// Every multi-GPU strategy is a bet about the ratio between these numbers and
// the ~1500 GB/s of local device memory. Measure them before choosing.
//
// These two cards have no NVLink; they are peers across a PCIe host bridge.
// Expect roughly 1-2% of local bandwidth. That ratio is the whole reason
// tensor parallelism hurts and data parallelism does not.
#include "helpers.cuh"
#include "collectives.cuh"

static double timePeer(int dst, int src, void *pDst, void *pSrc, size_t bytes,
                       cudaStream_t s) {
  CUDA_CHECK(cudaSetDevice(src));
  auto run = [&] {
    CUDA_CHECK(cudaMemcpyPeerAsync(pDst, dst, pSrc, src, bytes, s));
  };
  auto timed = [&] { run(); CUDA_CHECK(cudaStreamSynchronize(s)); };
  return timeKernelMs(timed, 20, 300.0);
}

int main(int argc, char **argv) {
  size_t mib = argc > 1 ? atoll(argv[1]) : 256;
  size_t bytes = mib << 20;
  requireTwoGpus();
  printf("transfer size: %zu MiB\n\n", mib);

  void *d0, *d1, *d0b, *d1b;
  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cudaMalloc(&d0, bytes));
  CUDA_CHECK(cudaMalloc(&d0b, bytes));
  cudaStream_t s0;
  CUDA_CHECK(cudaStreamCreate(&s0));
  CUDA_CHECK(cudaSetDevice(1));
  CUDA_CHECK(cudaMalloc(&d1, bytes));
  CUDA_CHECK(cudaMalloc(&d1b, bytes));
  cudaStream_t s1;
  CUDA_CHECK(cudaStreamCreate(&s1));

  void *hPageable = malloc(bytes);
  void *hPinned = nullptr;
  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cudaMallocHost(&hPinned, bytes));

  std::vector<Row> t;
  auto add = [&](const char *n, double ms) {
    t.push_back({n, ms, copyGBs(bytes, ms), 0, true});
  };

  // ---- host <-> device -----------------------------------------------
  // Pageable memory cannot be DMA'd: the driver copies it into an internal
  // pinned staging buffer first. That extra memcpy is the whole difference.
  CUDA_CHECK(cudaSetDevice(0));
  add("H2D pageable", timeKernelMs([&] {
        CUDA_CHECK(cudaMemcpy(d0, hPageable, bytes, cudaMemcpyHostToDevice));
      }, 10, 300.0));
  add("H2D pinned", timeKernelMs([&] {
        CUDA_CHECK(cudaMemcpy(d0, hPinned, bytes, cudaMemcpyHostToDevice));
      }, 10, 300.0));
  add("D2H pinned", timeKernelMs([&] {
        CUDA_CHECK(cudaMemcpy(hPinned, d0, bytes, cudaMemcpyDeviceToHost));
      }, 10, 300.0));

  // ---- local device-to-device (the number to beat) --------------------
  add("D2D same GPU", timeKernelMs([&] {
        CUDA_CHECK(cudaMemcpy(d0b, d0, bytes, cudaMemcpyDeviceToDevice));
      }, 10, 300.0));

  // ---- peer, WITHOUT peer access (staged through host) ----------------
  add("GPU0->GPU1 no-P2P (staged)", timePeer(1, 0, d1, d0, bytes, s0));

  // ---- peer, WITH peer access ----------------------------------------
  P2P p = enablePeerAccess(0, 1, false);
  printf("peer access: %s\n\n", p.enabled ? "enabled" : "NOT AVAILABLE");
  if (p.enabled) {
    add("GPU0->GPU1 P2P", timePeer(1, 0, d1, d0, bytes, s0));
    add("GPU1->GPU0 P2P", timePeer(0, 1, d0, d1, bytes, s1));

    // Both directions at once. PCIe is full duplex, so a good link gives
    // close to 2x aggregate; a shared/oversubscribed one does not.
    auto bidi = [&] {
      CUDA_CHECK(cudaSetDevice(0));
      CUDA_CHECK(cudaMemcpyPeerAsync(d1, 1, d0, 0, bytes, s0));
      CUDA_CHECK(cudaSetDevice(1));
      CUDA_CHECK(cudaMemcpyPeerAsync(d0b, 0, d1b, 1, bytes, s1));
      CUDA_CHECK(cudaStreamSynchronize(s0));
      CUDA_CHECK(cudaStreamSynchronize(s1));
    };
    double ms = timeKernelMs(bidi, 20, 300.0);
    t.push_back({"P2P bidirectional (aggregate)", ms, copyGBs(2 * bytes, ms), 0,
                 true});
  }

  double local = 0, peer = 0;
  for (auto &r : t) {
    if (r.name == "D2D same GPU") local = r.metric;
    if (r.name == "GPU0->GPU1 P2P") peer = r.metric;
  }
  for (auto &r : t) r.pctOfRef = local > 0 ? 100.0 * r.metric / local : 0;
  printTable("interconnect bandwidth", "GB/s", t);
  printf("('vs ref' = percent of local same-GPU copy throughput)\n");

  if (local > 0 && peer > 0)
    printf("\nThe ratio that decides your parallelism strategy:\n"
           "  local copy / peer copy = %.0fx\n"
           "  => shipping a tensor to the other GPU costs about what it costs\n"
           "     to copy it %.0f times locally. Communicate rarely, in bulk,\n"
           "     and overlapped with compute.\n",
           local / peer, local / peer);

  free(hPageable);
  cudaFreeHost(hPinned);
  return 0;
}
