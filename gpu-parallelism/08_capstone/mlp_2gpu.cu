// 08 - CAPSTONE: the same workload under all three strategies.
//
// Workload: forward pass of an L-layer MLP stack, Y = ReLU(Y @ W_l), on a
// batch of B. Identical math in every variant; only the decomposition differs.
//
//   strategy   what is split    per-GPU weights   communication
//   ---------  ---------------  ----------------  ---------------------------
//   1 GPU      nothing          L*H*H             none
//   DP         the batch        L*H*H  (full!)    NONE in forward
//   TP         each weight      L*H*H / 2         one all-reduce per 2 layers
//   PP         the layer stack  L*H*H / 2         one activation per microbatch
//
// The table at the end is the point of this whole lab: there is no best
// strategy, only a best fit for a given model size, batch size, and
// interconnect. Read the comm column together with the memory column.
//
// Note DP does zero communication here because this is INFERENCE. In training
// it pays a fixed all-reduce of the weights (see 04), which is why DP remains
// the default: that cost amortises as the batch grows, and TP's does not.
#include "helpers.cuh"
#include "collectives.cuh"

__global__ void kRelu(float *x, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t s = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += s) x[i] = fmaxf(x[i], 0.f);
}

int main(int argc, char **argv) {
  int B = argc > 1 ? atoi(argv[1]) : 8192;
  int H = argc > 2 ? atoi(argv[2]) : 4096;
  int L = argc > 3 ? atoi(argv[3]) : 8;
  if (L % 2) { fprintf(stderr, "L must be even\n"); return 2; }
  requireTwoGpus();
  enablePeerAccess(0, 1);

  int Hs = H / 2, Lh = L / 2;
  size_t actN = (size_t)B * H, wN = (size_t)H * H;
  printf("\n%d layers, H=%d, batch=%d\n", L, H, B);
  printf("total weights %.0f MiB, activations %.0f MiB\n\n",
         L * wN * 4 / 1048576.0, actN * 4 / 1048576.0);

  std::vector<std::vector<float>> hW(L, std::vector<float>(wN));
  for (int l = 0; l < L; ++l) {
    fillRandom(hW[l], 200 + l);
    for (auto &v : hW[l]) v *= 0.02f;
  }
  std::vector<float> hX(actN), ref(actN), got(actN);
  fillRandom(hX, 9);

  struct Res { const char *name; double ms; double commMiB; double memMiB; bool ok; };
  std::vector<Res> res;

  Gpu g[2];
  gpuInit(g[0], 0);
  gpuInit(g[1], 1);

  // ------------------------------------------------------------ 1 GPU
  double t1;
  {
    CUDA_CHECK(cudaSetDevice(0));
    std::vector<float *> W(L);
    float *xin, *a, *b;
    CUDA_CHECK(cudaMalloc(&xin, actN * 4));
    CUDA_CHECK(cudaMalloc(&a, actN * 4));
    CUDA_CHECK(cudaMalloc(&b, actN * 4));
    CUDA_CHECK(cudaMemcpy(xin, hX.data(), actN * 4, cudaMemcpyHostToDevice));
    for (int l = 0; l < L; ++l) {
      CUDA_CHECK(cudaMalloc(&W[l], wN * 4));
      CUDA_CHECK(cudaMemcpy(W[l], hW[l].data(), wN * 4, cudaMemcpyHostToDevice));
    }
    auto step = [&] {
      const float *src = xin;
      float *dst = a;
      for (int l = 0; l < L; ++l) {
        rmGemm(g[0].blas, B, H, H, src, W[l], dst);
        kRelu<<<1024, 256, 0, g[0].compute>>>(dst, actN);
        src = dst;
        dst = (dst == a) ? b : a;
      }
      CUDA_CHECK(cudaStreamSynchronize(g[0].compute));
      return src;
    };
    const float *r = step();
    CUDA_CHECK(cudaMemcpy(ref.data(), r, actN * 4, cudaMemcpyDeviceToHost));
    t1 = timeKernelMs([&] { step(); }, 10, 800.0);
    res.push_back({"1 GPU", t1, 0, L * wN * 4 / 1048576.0, true});
    for (auto w : W) cudaFree(w);
    cudaFree(xin); cudaFree(a); cudaFree(b);
  }

  // --------------------------------------------------- DATA PARALLEL
  // Split the batch. Forward inference needs no communication at all.
  {
    int b = B / 2;
    size_t aN = (size_t)b * H;
    std::vector<std::vector<float *>> W(2, std::vector<float *>(L));
    float *xin[2], *p[2], *q[2];
    for (int i = 0; i < 2; ++i) {
      CUDA_CHECK(cudaSetDevice(g[i].id));
      CUDA_CHECK(cudaMalloc(&xin[i], aN * 4));
      CUDA_CHECK(cudaMalloc(&p[i], aN * 4));
      CUDA_CHECK(cudaMalloc(&q[i], aN * 4));
      CUDA_CHECK(cudaMemcpy(xin[i], hX.data() + (size_t)i * aN, aN * 4,
                            cudaMemcpyHostToDevice));
      for (int l = 0; l < L; ++l) { // FULL weight replica on each GPU
        CUDA_CHECK(cudaMalloc(&W[i][l], wN * 4));
        CUDA_CHECK(cudaMemcpy(W[i][l], hW[l].data(), wN * 4, cudaMemcpyHostToDevice));
      }
    }
    const float *out[2];
    auto step = [&] {
      for (int i = 0; i < 2; ++i) {
        CUDA_CHECK(cudaSetDevice(g[i].id));
        const float *src = xin[i];
        float *dst = p[i];
        for (int l = 0; l < L; ++l) {
          rmGemm(g[i].blas, b, H, H, src, W[i][l], dst);
          kRelu<<<512, 256, 0, g[i].compute>>>(dst, aN);
          src = dst;
          dst = (dst == p[i]) ? q[i] : p[i];
        }
        out[i] = src;
      }
      syncAll(g, 2);
    };
    step();
    for (int i = 0; i < 2; ++i) {
      CUDA_CHECK(cudaSetDevice(g[i].id));
      CUDA_CHECK(cudaMemcpy(got.data() + (size_t)i * aN, out[i], aN * 4,
                            cudaMemcpyDeviceToHost));
    }
    bool ok = relL2(got, ref) < 1e-4;
    double ms = timeKernelMs(step, 10, 800.0);
    res.push_back({"DP (split batch)", ms, 0, L * wN * 4 / 1048576.0, ok});
    for (int i = 0; i < 2; ++i) {
      CUDA_CHECK(cudaSetDevice(g[i].id));
      for (auto w : W[i]) cudaFree(w);
      cudaFree(xin[i]); cudaFree(p[i]); cudaFree(q[i]);
    }
  }

  // ------------------------------------------------- TENSOR PARALLEL
  // Layers are paired into Megatron MLP blocks: even layer column-split
  // (no comm), odd layer row-split (all-reduce the partial sums).
  {
    std::vector<std::vector<float *>> Wa(2, std::vector<float *>(Lh)); // [H x Hs]
    std::vector<std::vector<float *>> Wb(2, std::vector<float *>(Lh)); // [Hs x H]
    float *x[2], *z[2], *y[2], *scratch[2];
    std::vector<float> was((size_t)H * Hs), wbs((size_t)Hs * H);
    for (int i = 0; i < 2; ++i) {
      CUDA_CHECK(cudaSetDevice(g[i].id));
      CUDA_CHECK(cudaMalloc(&x[i], actN * 4));
      CUDA_CHECK(cudaMalloc(&z[i], (size_t)B * Hs * 4));
      CUDA_CHECK(cudaMalloc(&y[i], actN * 4));
      CUDA_CHECK(cudaMalloc(&scratch[i], actN * 4));
      CUDA_CHECK(cudaMemcpy(x[i], hX.data(), actN * 4, cudaMemcpyHostToDevice));
      for (int blk = 0; blk < Lh; ++blk) {
        for (int r = 0; r < H; ++r) // column shard of the even layer
          for (int c = 0; c < Hs; ++c)
            was[(size_t)r * Hs + c] = hW[2 * blk][(size_t)r * H + i * Hs + c];
        for (int r = 0; r < Hs; ++r) // row shard of the odd layer
          for (int c = 0; c < H; ++c)
            wbs[(size_t)r * H + c] = hW[2 * blk + 1][((size_t)i * Hs + r) * H + c];
        CUDA_CHECK(cudaMalloc(&Wa[i][blk], (size_t)H * Hs * 4));
        CUDA_CHECK(cudaMalloc(&Wb[i][blk], (size_t)Hs * H * 4));
        CUDA_CHECK(cudaMemcpy(Wa[i][blk], was.data(), (size_t)H * Hs * 4,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(Wb[i][blk], wbs.data(), (size_t)Hs * H * 4,
                              cudaMemcpyHostToDevice));
      }
    }
    auto step = [&] {
      for (int blk = 0; blk < Lh; ++blk) {
        for (int i = 0; i < 2; ++i) {
          CUDA_CHECK(cudaSetDevice(g[i].id));
          const float *in = (blk == 0) ? x[i] : y[i];
          rmGemm(g[i].blas, B, Hs, H, in, Wa[i][blk], z[i]);
          kRelu<<<512, 256, 0, g[i].compute>>>(z[i], (size_t)B * Hs);
          rmGemm(g[i].blas, B, H, Hs, z[i], Wb[i][blk], y[i]);
        }
        commAfterCompute(g, 2);
        allReduceSum2(g, y, scratch, actN);
        computeAfterComm(g, 2);
        // The second ReLU must come AFTER the all-reduce: ReLU of a partial
        // sum is not the partial sum of ReLUs. Getting this backwards is a
        // classic tensor-parallel bug that still "almost" works.
        for (int i = 0; i < 2; ++i) {
          CUDA_CHECK(cudaSetDevice(g[i].id));
          kRelu<<<512, 256, 0, g[i].compute>>>(y[i], actN);
        }
      }
      syncAll(g, 2);
    };
    step();
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMemcpy(got.data(), y[0], actN * 4, cudaMemcpyDeviceToHost));
    bool ok = relL2(got, ref) < 1e-4;
    double ms = timeKernelMs(step, 10, 800.0);
    res.push_back({"TP (split weights)", ms, Lh * actN * 4 / 1048576.0,
                   L * wN * 4 / 2 / 1048576.0, ok});
    for (int i = 0; i < 2; ++i) {
      CUDA_CHECK(cudaSetDevice(g[i].id));
      for (int blk = 0; blk < Lh; ++blk) { cudaFree(Wa[i][blk]); cudaFree(Wb[i][blk]); }
      cudaFree(x[i]); cudaFree(z[i]); cudaFree(y[i]); cudaFree(scratch[i]);
    }
  }

  // ----------------------------------------------- PIPELINE PARALLEL
  {
    const int m = 8;
    int mb = B / m;
    size_t aN = (size_t)mb * H;
    std::vector<float *> W0(Lh), W1(Lh);
    for (int l = 0; l < Lh; ++l) {
      CUDA_CHECK(cudaSetDevice(0));
      CUDA_CHECK(cudaMalloc(&W0[l], wN * 4));
      CUDA_CHECK(cudaMemcpy(W0[l], hW[l].data(), wN * 4, cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaSetDevice(1));
      CUDA_CHECK(cudaMalloc(&W1[l], wN * 4));
      CUDA_CHECK(cudaMemcpy(W1[l], hW[Lh + l].data(), wN * 4, cudaMemcpyHostToDevice));
    }
    std::vector<float *> in0(m), s0a(m), s0b(m), rc(m), s1a(m), s1b(m), out(m);
    std::vector<cudaEvent_t> e0(m), e1(m);
    for (int k = 0; k < m; ++k) {
      CUDA_CHECK(cudaSetDevice(0));
      CUDA_CHECK(cudaMalloc(&in0[k], aN * 4));
      CUDA_CHECK(cudaMalloc(&s0a[k], aN * 4));
      CUDA_CHECK(cudaMalloc(&s0b[k], aN * 4));
      CUDA_CHECK(cudaMemcpy(in0[k], hX.data() + (size_t)k * aN, aN * 4,
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaEventCreateWithFlags(&e0[k], cudaEventDisableTiming));
      CUDA_CHECK(cudaEventCreateWithFlags(&e1[k], cudaEventDisableTiming));
      CUDA_CHECK(cudaSetDevice(1));
      CUDA_CHECK(cudaMalloc(&rc[k], aN * 4));
      CUDA_CHECK(cudaMalloc(&s1a[k], aN * 4));
      CUDA_CHECK(cudaMalloc(&s1b[k], aN * 4));
    }
    auto step = [&] {
      for (int k = 0; k < m; ++k) {
        CUDA_CHECK(cudaSetDevice(0));
        const float *src = in0[k];
        float *dst = s0a[k];
        for (int l = 0; l < Lh; ++l) {
          rmGemm(g[0].blas, mb, H, H, src, W0[l], dst);
          kRelu<<<256, 256, 0, g[0].compute>>>(dst, aN);
          src = dst;
          dst = (dst == s0a[k]) ? s0b[k] : s0a[k];
        }
        CUDA_CHECK(cudaEventRecord(e0[k], g[0].compute));
        CUDA_CHECK(cudaStreamWaitEvent(g[0].comm, e0[k], 0));
        CUDA_CHECK(cudaMemcpyPeerAsync(rc[k], 1, src, 0, aN * 4, g[0].comm));
        CUDA_CHECK(cudaEventRecord(e1[k], g[0].comm));

        CUDA_CHECK(cudaSetDevice(1));
        CUDA_CHECK(cudaStreamWaitEvent(g[1].compute, e1[k], 0));
        const float *s = rc[k];
        float *d = s1a[k];
        for (int l = 0; l < Lh; ++l) {
          rmGemm(g[1].blas, mb, H, H, s, W1[l], d);
          kRelu<<<256, 256, 0, g[1].compute>>>(d, aN);
          s = d;
          d = (d == s1a[k]) ? s1b[k] : s1a[k];
        }
        out[k] = (float *)s;
      }
      syncAll(g, 2);
    };
    step();
    for (int k = 0; k < m; ++k) {
      CUDA_CHECK(cudaSetDevice(1));
      CUDA_CHECK(cudaMemcpy(got.data() + (size_t)k * aN, out[k], aN * 4,
                            cudaMemcpyDeviceToHost));
    }
    bool ok = relL2(got, ref) < 1e-4;
    double ms = timeKernelMs(step, 10, 800.0);
    res.push_back({"PP (split layers, m=8)", ms, m * aN * 4 / 1048576.0,
                   L * wN * 4 / 2 / 1048576.0, ok});
  }

  printf("%-26s %10s %9s %12s %12s %5s\n", "strategy", "ms", "speedup",
         "comm MiB", "weights MiB", "ok");
  printf("%-26s %10s %9s %12s %12s %5s\n", "--------------------------",
         "---------", "--------", "-----------", "-----------", "-----");
  for (auto &r : res)
    printf("%-26s %10.3f %8.2fx %12.0f %12.0f %5s\n", r.name, r.ms,
           t1 / r.ms, r.commMiB, r.memMiB, r.ok ? "yes" : "NO");

  printf("\nHow to read this:\n"
         "  DP is fastest and needs no comm -- but every GPU stores the whole\n"
         "     model, so it does nothing for a model that does not fit.\n"
         "  TP halves the weight memory but pays an all-reduce per block, and\n"
         "     that volume grows with the batch. Over PCIe it is the loser;\n"
         "     over NVLink (~10x this link) it would be competitive.\n"
         "  PP also halves weight memory and moves the LEAST data by far,\n"
         "     which is why it is the strategy of choice across slow links --\n"
         "     at the cost of pipeline bubbles and scheduling complexity.\n"
         "\nReal systems compose all three: TP inside a node, PP across\n"
         "nodes, DP over the whole lot.\n");

  gpuDestroy(g[0]);
  gpuDestroy(g[1]);
  return 0;
}
