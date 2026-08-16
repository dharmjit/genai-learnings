// 05 - TENSOR PARALLELISM
//
// Idea: the weight matrix itself is too big for one GPU, so SPLIT it. Every
// GPU sees the whole batch but owns only a slice of each weight matrix.
//
// The workload is a transformer MLP block:   Y = GeLU(X @ W1) @ W2
//   X  [B  x H ]      W1 [H  x 4H]      W2 [4H x H]
//
// There are two ways to split it, and the difference between them is the
// entire insight behind Megatron-LM.
//
// ---- Strategy A (naive): split W1 by columns, then re-gather ------------
//   Z_g = GeLU(X @ W1[:, g])        [B x 2H]   no comm yet
//   all-gather Z                    <-- moves B * 4H floats
//   Y   = Z @ W2                    every GPU redoes the FULL second GEMM
// Cost: comm B*4H, and the second GEMM is fully replicated (wasted flops).
//
// ---- Strategy B (Megatron): column-split W1, ROW-split W2 ---------------
//   Z_g      = GeLU(X @ W1[:, g])   [B x 2H]   no comm
//   Ypart_g  = Z_g @ W2[g, :]       [B x H ]   no comm -- shapes just work
//   all-reduce Ypart                <-- moves B * H floats
// Cost: comm B*H, and NO replicated compute.
//
// B is 4x less communication AND half the second-GEMM work. The reason it
// works is that splitting W1 by column and W2 by row makes the shard
// boundaries line up, so the contraction dimension is what gets split -- and
// a split contraction is exactly a sum, i.e. an all-reduce.
//
// Note what tensor parallelism costs: it communicates ACTIVATIONS, which
// scale with batch size. Unlike data parallelism, a bigger batch does NOT
// amortise the cost. That is why TP is reserved for within-a-node (NVLink)
// and why on PCIe-only boxes like this one it scales poorly.
#include "helpers.cuh"
#include "collectives.cuh"

__global__ void kGelu(float *x, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t s = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += s) {
    float v = x[i];
    // tanh approximation, the one actually used in GPT-family models
    float c = 0.7978845608f * (v + 0.044715f * v * v * v);
    x[i] = 0.5f * v * (1.f + tanhf(c));
  }
}

int main(int argc, char **argv) {
  int B = argc > 1 ? atoi(argv[1]) : 4096;
  int H = argc > 2 ? atoi(argv[2]) : 4096;
  int F = 4 * H;      // FFN inner dimension
  int Fs = F / 2;     // per-GPU shard of it
  requireTwoGpus();
  enablePeerAccess(0, 1);

  printf("\nMLP block: X[%dx%d] -> W1[%dx%d] -> GeLU -> W2[%dx%d] -> Y[%dx%d]\n",
         B, H, H, F, F, H, B, H);
  printf("weights: %.0f MiB total, %.0f MiB per GPU when split\n",
         2.0 * H * F * 4 / 1048576.0, (double)H * F * 4 / 1048576.0);
  printf("comm volume  strategy A (all-gather Z): %.0f MiB\n",
         (double)B * F * 4 / 1048576.0);
  printf("comm volume  strategy B (all-reduce Y): %.0f MiB\n\n",
         (double)B * H * 4 / 1048576.0);

  std::vector<float> hX((size_t)B * H), hW1((size_t)H * F), hW2((size_t)F * H);
  fillRandom(hX, 1);
  fillRandom(hW1, 2);
  fillRandom(hW2, 3);
  for (auto &v : hW1) v *= 0.02f; // keep GeLU in a sane range
  for (auto &v : hW2) v *= 0.02f;

  std::vector<float> ref((size_t)B * H), got((size_t)B * H);

  // ------------------------------------------------- single-GPU reference
  double tSingle;
  {
    CUDA_CHECK(cudaSetDevice(0));
    cublasHandle_t bl;
    CUBLAS_CHECK(cublasCreate(&bl));
    float *X, *W1, *W2, *Z, *Y;
    CUDA_CHECK(cudaMalloc(&X, (size_t)B * H * 4));
    CUDA_CHECK(cudaMalloc(&W1, (size_t)H * F * 4));
    CUDA_CHECK(cudaMalloc(&W2, (size_t)F * H * 4));
    CUDA_CHECK(cudaMalloc(&Z, (size_t)B * F * 4));
    CUDA_CHECK(cudaMalloc(&Y, (size_t)B * H * 4));
    CUDA_CHECK(cudaMemcpy(X, hX.data(), (size_t)B * H * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(W1, hW1.data(), (size_t)H * F * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(W2, hW2.data(), (size_t)F * H * 4, cudaMemcpyHostToDevice));
    auto step = [&] {
      rmGemm(bl, B, F, H, X, W1, Z);
      kGelu<<<1024, 256>>>(Z, (size_t)B * F);
      rmGemm(bl, B, H, F, Z, W2, Y);
      CUDA_CHECK(cudaDeviceSynchronize());
    };
    step();
    CUDA_CHECK(cudaMemcpy(ref.data(), Y, (size_t)B * H * 4, cudaMemcpyDeviceToHost));
    tSingle = timeKernelMs(step, 10, 600.0);
    cublasDestroy(bl);
    cudaFree(X); cudaFree(W1); cudaFree(W2); cudaFree(Z); cudaFree(Y);
  }

  Gpu g[2];
  gpuInit(g[0], 0);
  gpuInit(g[1], 1);

  // Shard the weights. W1 by COLUMN: GPU g gets columns [g*Fs, (g+1)*Fs).
  // W2 by ROW: GPU g gets rows [g*Fs, (g+1)*Fs). Same index range -- that is
  // the alignment that makes strategy B work.
  std::vector<float> w1s((size_t)H * Fs), w2s((size_t)Fs * H);
  float *X[2], *W1s[2], *W2s[2], *Zs[2], *Yp[2], *scratch[2], *Zfull[2];
  for (int i = 0; i < 2; ++i) {
    for (int r = 0; r < H; ++r)
      for (int c = 0; c < Fs; ++c)
        w1s[(size_t)r * Fs + c] = hW1[(size_t)r * F + i * Fs + c];
    for (size_t r = 0; r < (size_t)Fs; ++r)
      for (int c = 0; c < H; ++c)
        w2s[r * H + c] = hW2[((size_t)i * Fs + r) * H + c];

    CUDA_CHECK(cudaSetDevice(g[i].id));
    CUDA_CHECK(cudaMalloc(&X[i], (size_t)B * H * 4));
    CUDA_CHECK(cudaMalloc(&W1s[i], (size_t)H * Fs * 4));
    CUDA_CHECK(cudaMalloc(&W2s[i], (size_t)Fs * H * 4));
    CUDA_CHECK(cudaMalloc(&Zs[i], (size_t)B * Fs * 4));
    CUDA_CHECK(cudaMalloc(&Yp[i], (size_t)B * H * 4));
    CUDA_CHECK(cudaMalloc(&scratch[i], (size_t)B * H * 4));
    CUDA_CHECK(cudaMemcpy(X[i], hX.data(), (size_t)B * H * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(W1s[i], w1s.data(), (size_t)H * Fs * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(W2s[i], w2s.data(), (size_t)Fs * H * 4, cudaMemcpyHostToDevice));
  }

  std::vector<Row> table;
  table.push_back({"1 GPU (no parallelism)", tSingle, 0, 100.0, true});

  // ----------------------------------------- strategy B: Megatron split
  {
    auto step = [&] {
      for (int i = 0; i < 2; ++i) {
        CUDA_CHECK(cudaSetDevice(g[i].id));
        rmGemm(g[i].blas, B, Fs, H, X[i], W1s[i], Zs[i]);
        kGelu<<<1024, 256, 0, g[i].compute>>>(Zs[i], (size_t)B * Fs);
        rmGemm(g[i].blas, B, H, Fs, Zs[i], W2s[i], Yp[i]);
      }
      commAfterCompute(g, 2);
      allReduceSum2(g, Yp, scratch, (size_t)B * H);
      waitComm(g, 2);
    };
    step();
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMemcpy(got.data(), Yp[0], (size_t)B * H * 4, cudaMemcpyDeviceToHost));
    bool ok = checkAndReport("TP-B", got, ref, 1e-4);
    double ms = timeKernelMs(step, 10, 600.0);
    table.push_back({"TP-B Megatron (all-reduce Y)", ms, 0, 0, ok});
  }

  // --------------------------------- strategy A: all-gather the activations
  {
    for (int i = 0; i < 2; ++i) {
      CUDA_CHECK(cudaSetDevice(g[i].id));
      CUDA_CHECK(cudaMalloc(&Zfull[i], (size_t)B * F * 4));
    }
    // Each GPU needs the whole W2 for the replicated second GEMM -- note the
    // memory cost: strategy A does not actually shard W2 at all.
    float *W2full[2];
    for (int i = 0; i < 2; ++i) {
      CUDA_CHECK(cudaSetDevice(g[i].id));
      CUDA_CHECK(cudaMalloc(&W2full[i], (size_t)F * H * 4));
      CUDA_CHECK(cudaMemcpy(W2full[i], hW2.data(), (size_t)F * H * 4,
                            cudaMemcpyHostToDevice));
    }

    auto step = [&] {
      for (int i = 0; i < 2; ++i) {
        CUDA_CHECK(cudaSetDevice(g[i].id));
        rmGemm(g[i].blas, B, Fs, H, X[i], W1s[i], Zs[i]);
        kGelu<<<1024, 256, 0, g[i].compute>>>(Zs[i], (size_t)B * Fs);
      }
      commAfterCompute(g, 2);
      // NOTE: this is a strided gather -- Z is [B x F] row-major but each
      // shard supplies a [B x Fs] column block, so it is B separate rows, not
      // one contiguous run. Communication cost is not just about volume.
      // There is no cudaMemcpy2DPeerAsync in the runtime API. With unified
      // addressing plus peer access, cudaMemcpyDefault infers the direction
      // from the pointers and handles the cross-device case.
      for (int i = 0; i < 2; ++i) {
        CUDA_CHECK(cudaSetDevice(g[i].id));
        for (int src = 0; src < 2; ++src)
          CUDA_CHECK(cudaMemcpy2DAsync(Zfull[i] + (size_t)src * Fs, F * 4,
                                       Zs[src], Fs * 4, Fs * 4, B,
                                       cudaMemcpyDefault, g[i].comm));
      }
      waitComm(g, 2);
      computeAfterComm(g, 2);
      for (int i = 0; i < 2; ++i) { // replicated full-width GEMM
        CUDA_CHECK(cudaSetDevice(g[i].id));
        rmGemm(g[i].blas, B, H, F, Zfull[i], W2full[i], Yp[i]);
      }
      syncAll(g, 2);
    };
    step();
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMemcpy(got.data(), Yp[0], (size_t)B * H * 4, cudaMemcpyDeviceToHost));
    bool ok = checkAndReport("TP-A", got, ref, 1e-4);
    double ms = timeKernelMs(step, 10, 600.0);
    table.push_back({"TP-A naive (all-gather Z)", ms, 0, 0, ok});
  }

  for (auto &r : table) {
    r.metric = 2.0 * 2.0 * B * H * F / (r.ms * 1e-3) / 1e12; // 2 GEMMs
    r.pctOfRef = 100.0 * tSingle / r.ms;
  }
  printTable("tensor parallelism strategies", "TFLOP/s", table);
  printf("('vs ref' = speedup over 1 GPU; 200%% would be perfect scaling)\n");
  printf("\nStrategy B beats A on both axes. But look at the speedup column:\n"
         "over PCIe, even the good strategy struggles to reach 2x, because\n"
         "the all-reduce moves activations that scale with the batch. This is\n"
         "precisely the workload NVLink exists for.\n");

  gpuDestroy(g[0]);
  gpuDestroy(g[1]);
  return 0;
}
