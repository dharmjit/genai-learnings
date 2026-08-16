// 06 - PIPELINE PARALLELISM
//
// Idea: split the model by DEPTH. GPU0 owns the first half of the layers,
// GPU1 the second half. An activation flows GPU0 -> GPU1 once per batch.
//
//   GPU0: layers 0..L/2-1        GPU1: layers L/2..L-1
//   comm: one activation tensor [b x H] per microbatch. Tiny.
//
// The catch is the BUBBLE. With one batch, GPU1 sits idle while GPU0 works,
// then GPU0 sits idle while GPU1 works: 50% utilisation on 2 GPUs.
//
// The fix is microbatching. Split the batch into m pieces and feed them in;
// once the pipe is full both GPUs work simultaneously.
//
//   bubble fraction = (P-1) / (m + P-1)      P = number of stages
//   for P=2:  m=1 -> 50%,  m=4 -> 20%,  m=16 -> 6%
//
// This program measures the real bubble against that formula.
//
// Pipeline parallelism has the SMALLEST communication volume of the three
// strategies (one activation per stage boundary, not per layer), which is why
// it is what you use across slow links -- between nodes, or over PCIe.
#include "helpers.cuh"
#include "collectives.cuh"

__global__ void kRelu(float *x, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t s = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += s) x[i] = fmaxf(x[i], 0.f);
}

int main(int argc, char **argv) {
  int B = argc > 1 ? atoi(argv[1]) : 8192; // total batch
  int H = argc > 2 ? atoi(argv[2]) : 4096;
  int L = argc > 3 ? atoi(argv[3]) : 8;    // layers, split evenly
  requireTwoGpus();
  enablePeerAccess(0, 1);

  int Lh = L / 2;
  printf("\n%d layers of Y = ReLU(Y @ W[%dx%d]), batch %d\n", L, H, H, B);
  printf("stage 0 = layers 0..%d on GPU0, stage 1 = layers %d..%d on GPU1\n",
         Lh - 1, Lh, L - 1);
  printf("activation handoff per microbatch: [b x %d]\n\n", H);

  std::vector<std::vector<float>> hW(L, std::vector<float>((size_t)H * H));
  for (int l = 0; l < L; ++l) {
    fillRandom(hW[l], 100 + l);
    for (auto &v : hW[l]) v *= 0.02f;
  }
  std::vector<float> hX((size_t)B * H);
  fillRandom(hX, 5);
  std::vector<float> ref((size_t)B * H), got((size_t)B * H);

  // -------------------------------------------------- single-GPU baseline
  double tSingle;
  {
    CUDA_CHECK(cudaSetDevice(0));
    cublasHandle_t bl;
    CUBLAS_CHECK(cublasCreate(&bl));
    std::vector<float *> W(L);
    float *xin, *a, *b2;
    CUDA_CHECK(cudaMalloc(&xin, (size_t)B * H * 4));
    CUDA_CHECK(cudaMalloc(&a, (size_t)B * H * 4));
    CUDA_CHECK(cudaMalloc(&b2, (size_t)B * H * 4));
    CUDA_CHECK(cudaMemcpy(xin, hX.data(), (size_t)B * H * 4, cudaMemcpyHostToDevice));
    for (int l = 0; l < L; ++l) {
      CUDA_CHECK(cudaMalloc(&W[l], (size_t)H * H * 4));
      CUDA_CHECK(cudaMemcpy(W[l], hW[l].data(), (size_t)H * H * 4,
                            cudaMemcpyHostToDevice));
    }
    // Input stays resident and read-only, exactly like the pipeline version,
    // so the comparison measures compute and not a host transfer.
    auto step = [&] {
      const float *src = xin;
      float *dst = a;
      for (int l = 0; l < L; ++l) {
        rmGemm(bl, B, H, H, src, W[l], dst);
        kRelu<<<1024, 256>>>(dst, (size_t)B * H);
        src = dst;
        dst = (dst == a) ? b2 : a;
      }
      CUDA_CHECK(cudaDeviceSynchronize());
      return src;
    };
    const float *res = step();
    CUDA_CHECK(cudaMemcpy(ref.data(), res, (size_t)B * H * 4, cudaMemcpyDeviceToHost));
    tSingle = timeKernelMs([&] { step(); }, 10, 800.0);
    cublasDestroy(bl);
  }

  Gpu g[2];
  gpuInit(g[0], 0);
  gpuInit(g[1], 1);

  // Each GPU holds only ITS half of the weights -- that is the memory win.
  std::vector<float *> W0(Lh), W1(Lh);
  for (int l = 0; l < Lh; ++l) {
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMalloc(&W0[l], (size_t)H * H * 4));
    CUDA_CHECK(cudaMemcpy(W0[l], hW[l].data(), (size_t)H * H * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaSetDevice(1));
    CUDA_CHECK(cudaMalloc(&W1[l], (size_t)H * H * 4));
    CUDA_CHECK(cudaMemcpy(W1[l], hW[Lh + l].data(), (size_t)H * H * 4,
                          cudaMemcpyHostToDevice));
  }

  printf("%8s %11s %10s %10s %12s %12s\n", "micro-", "time ms", "speedup",
         "efficy", "bubble meas", "bubble pred");
  printf("%8s %11s %10s %10s %12s %12s\n", "batches", "----------",
         "---------", "---------", "-----------", "-----------");

  for (int m : {1, 2, 4, 8, 16}) {
    if (B % m) continue;
    int mb = B / m; // rows per microbatch
    size_t actN = (size_t)mb * H;

    // Per-microbatch buffers and events. Real frameworks use a small ring
    // instead of m buffers; m is clearer and the memory is cheap here.
    // Ping-pong scratch per stage, so the microbatch INPUT is never written.
    // Without this the timing loop's second iteration would run on the
    // garbage left behind by the first.
    std::vector<float *> in0(m), s0a(m), s0b(m);
    std::vector<float *> recv1(m), s1a(m), s1b(m), out1(m);
    std::vector<cudaEvent_t> evStage0(m), evXfer(m);
    for (int k = 0; k < m; ++k) {
      CUDA_CHECK(cudaSetDevice(0));
      CUDA_CHECK(cudaMalloc(&in0[k], actN * 4));
      CUDA_CHECK(cudaMalloc(&s0a[k], actN * 4));
      CUDA_CHECK(cudaMalloc(&s0b[k], actN * 4));
      CUDA_CHECK(cudaMemcpy(in0[k], hX.data() + (size_t)k * actN, actN * 4,
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaEventCreateWithFlags(&evStage0[k], cudaEventDisableTiming));
      // Both events are RECORDED on device-0 streams, so both must be created
      // while device 0 is current -- an event belongs to the device that was
      // active at creation, and recording it elsewhere is an invalid handle.
      // Waiting on it from another device is fine; recording is not.
      CUDA_CHECK(cudaEventCreateWithFlags(&evXfer[k], cudaEventDisableTiming));
      CUDA_CHECK(cudaSetDevice(1));
      CUDA_CHECK(cudaMalloc(&recv1[k], actN * 4));
      CUDA_CHECK(cudaMalloc(&s1a[k], actN * 4));
      CUDA_CHECK(cudaMalloc(&s1b[k], actN * 4));
      out1[k] = nullptr;
    }

    auto step = [&] {
      for (int k = 0; k < m; ++k) {
        // ---- stage 0 on GPU0 -------------------------------------------
        CUDA_CHECK(cudaSetDevice(0));
        const float *src = in0[k];
        float *dst = s0a[k];
        for (int l = 0; l < Lh; ++l) {
          rmGemm(g[0].blas, mb, H, H, src, W0[l], dst);
          kRelu<<<512, 256, 0, g[0].compute>>>(dst, actN);
          src = dst;
          dst = (dst == s0a[k]) ? s0b[k] : s0a[k];
        }
        // `src` now holds this microbatch's stage-0 result.
        CUDA_CHECK(cudaEventRecord(evStage0[k], g[0].compute));

        // ---- handoff on a SEPARATE stream ------------------------------
        // This is the whole trick: the transfer of microbatch k overlaps
        // with stage-0 compute of microbatch k+1.
        CUDA_CHECK(cudaStreamWaitEvent(g[0].comm, evStage0[k], 0));
        CUDA_CHECK(cudaMemcpyPeerAsync(recv1[k], 1, src, 0, actN * 4, g[0].comm));
        CUDA_CHECK(cudaEventRecord(evXfer[k], g[0].comm));

        // ---- stage 1 on GPU1 -------------------------------------------
        CUDA_CHECK(cudaSetDevice(1));
        CUDA_CHECK(cudaStreamWaitEvent(g[1].compute, evXfer[k], 0));
        const float *s1 = recv1[k];
        float *d1 = s1a[k];
        for (int l = 0; l < Lh; ++l) {
          rmGemm(g[1].blas, mb, H, H, s1, W1[l], d1);
          kRelu<<<512, 256, 0, g[1].compute>>>(d1, actN);
          s1 = d1;
          d1 = (d1 == s1a[k]) ? s1b[k] : s1a[k];
        }
        out1[k] = (float *)s1;
      }
      syncAll(g, 2);
    };

    step();
    for (int k = 0; k < m; ++k) {
      CUDA_CHECK(cudaSetDevice(1));
      CUDA_CHECK(cudaMemcpy(got.data() + (size_t)k * actN, out1[k], actN * 4,
                            cudaMemcpyDeviceToHost));
    }
    double err = relL2(got, ref);
    double ms = timeKernelMs(step, 10, 800.0);

    // Perfect 2-GPU scaling would be tSingle/2. Whatever we lose beyond that
    // is bubble plus transfer.
    double ideal = tSingle / 2.0;
    double bubbleMeas = 100.0 * (ms - ideal) / ms;
    double bubblePred = 100.0 * 1.0 / (m + 1.0); // (P-1)/(m+P-1), P=2
    printf("%8d %11.3f %9.2fx %9.0f%% %11.0f%% %11.0f%% %s\n", m, ms,
           tSingle / ms, 100.0 * tSingle / (2.0 * ms), bubbleMeas, bubblePred,
           err < 1e-4 ? "" : " WRONG RESULT!");

    for (int k = 0; k < m; ++k) {
      CUDA_CHECK(cudaSetDevice(0));
      cudaFree(in0[k]); cudaFree(s0a[k]); cudaFree(s0b[k]);
      cudaEventDestroy(evStage0[k]);
      cudaEventDestroy(evXfer[k]);
      CUDA_CHECK(cudaSetDevice(1));
      cudaFree(recv1[k]); cudaFree(s1a[k]); cudaFree(s1b[k]);
    }
  }

  printf("\n1-GPU baseline: %.3f ms (all %d layers, whole batch)\n", tSingle, L);
  printf("\nThe measured bubble should track (P-1)/(m+P-1) = 1/(m+1) closely\n"
         "at small m and then flatten -- past a point the microbatches get so\n"
         "small that the GEMMs stop filling %d SMs and per-launch overhead\n"
         "dominates. That tension, bubble vs. kernel efficiency, is exactly\n"
         "what real schedulers (GPipe, 1F1B, interleaved) are tuning.\n", 188);

  gpuDestroy(g[0]);
  gpuDestroy(g[1]);
  return 0;
}
