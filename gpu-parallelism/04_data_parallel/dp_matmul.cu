// 04 - DATA PARALLELISM
//
// Idea: every GPU holds a FULL copy of the weights and processes a DIFFERENT
// slice of the batch. Nothing is communicated during the forward pass at all.
// Only the gradients must be summed at the end of the step.
//
//   GPU0: Y0 = X0 @ W      dW0 = X0^T @ dY0    (X0 = first half of batch)
//   GPU1: Y1 = X1 @ W      dW1 = X1^T @ dY1
//   all-reduce: dW = dW0 + dW1     <-- the ONLY communication
//
// The economics:
//   compute per GPU  = 4 * (B/2) * H * H flops     (grows with batch)
//   communication    = H * H * 4 bytes             (FIXED -- weights only)
//
// So the comm/compute ratio falls as the batch grows. That is why data
// parallelism is the default strategy and why "use a bigger batch" is the
// standard fix for poor multi-GPU scaling.
//
// Its limit is memory, not speed: every GPU must hold the whole model.
#include "helpers.cuh"
#include "collectives.cuh"

int main(int argc, char **argv) {
  int H = argc > 1 ? atoi(argv[1]) : 4096;
  requireTwoGpus();
  enablePeerAccess(0, 1);

  printf("\nhidden size H=%d, weights W[%dx%d] = %.0f MiB replicated per GPU\n",
         H, H, H, (double)H * H * 4 / 1048576.0);
  printf("all-reduce volume is FIXED at %.0f MiB regardless of batch size\n\n",
         (double)H * H * 4 / 1048576.0);

  Gpu g[2];
  gpuInit(g[0], 0);
  gpuInit(g[1], 1);

  size_t wN = (size_t)H * H;
  std::vector<float> hW(wN);
  fillRandom(hW, 7);
  for (auto &v : hW) v *= 0.02f;

  printf("%9s %11s %11s %11s %10s %9s %11s\n", "batch", "1-GPU ms", "2-GPU ms",
         "no-comm ms", "comm ms", "speedup", "efficiency");
  printf("%9s %11s %11s %11s %10s %9s %11s\n", "--------", "----------",
         "----------", "----------", "---------", "--------", "----------");

  for (int B : {512, 2048, 8192, 32768}) {
    int b = B / 2;
    std::vector<float> hX((size_t)B * H);
    fillRandom(hX, (unsigned)B);
    std::vector<float> refdW(wN), gotdW(wN);

    // ------------------------------------------- single-GPU full batch
    double t1;
    {
      CUDA_CHECK(cudaSetDevice(0));
      float *X, *W, *Y, *dW;
      CUDA_CHECK(cudaMalloc(&X, (size_t)B * H * 4));
      CUDA_CHECK(cudaMalloc(&Y, (size_t)B * H * 4));
      CUDA_CHECK(cudaMalloc(&W, wN * 4));
      CUDA_CHECK(cudaMalloc(&dW, wN * 4));
      CUDA_CHECK(cudaMemcpy(X, hX.data(), (size_t)B * H * 4, cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(W, hW.data(), wN * 4, cudaMemcpyHostToDevice));
      auto step = [&] {
        rmGemm(g[0].blas, B, H, H, X, W, Y);    // forward
        rmGemmTN(g[0].blas, H, H, B, X, Y, dW); // weight gradient
        CUDA_CHECK(cudaStreamSynchronize(g[0].compute));
      };
      step();
      CUDA_CHECK(cudaMemcpy(refdW.data(), dW, wN * 4, cudaMemcpyDeviceToHost));
      t1 = timeKernelMs(step, 10, 600.0);
      cudaFree(X); cudaFree(Y); cudaFree(W); cudaFree(dW);
    }

    // ------------------------------------------------- data parallel
    float *X[2], *W[2], *Y[2], *dW[2], *scratch[2];
    for (int i = 0; i < 2; ++i) {
      CUDA_CHECK(cudaSetDevice(g[i].id));
      CUDA_CHECK(cudaMalloc(&X[i], (size_t)b * H * 4));
      CUDA_CHECK(cudaMalloc(&Y[i], (size_t)b * H * 4));
      CUDA_CHECK(cudaMalloc(&W[i], wN * 4));
      CUDA_CHECK(cudaMalloc(&dW[i], wN * 4));
      CUDA_CHECK(cudaMalloc(&scratch[i], wN * 4));
      // GPU i takes batch rows [i*b, (i+1)*b)
      CUDA_CHECK(cudaMemcpy(X[i], hX.data() + (size_t)i * b * H,
                            (size_t)b * H * 4, cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(W[i], hW.data(), wN * 4, cudaMemcpyHostToDevice));
    }

    auto step = [&](bool doAllReduce) {
      for (int i = 0; i < 2; ++i) {
        CUDA_CHECK(cudaSetDevice(g[i].id));
        rmGemm(g[i].blas, b, H, H, X[i], W[i], Y[i]);
        rmGemmTN(g[i].blas, H, H, b, X[i], Y[i], dW[i]);
      }
      if (doAllReduce) {
        commAfterCompute(g, 2);
        allReduceSum2(g, dW, scratch, wN);
        waitComm(g, 2);
      } else {
        syncAll(g, 2);
      }
    };

    double tNo = timeKernelMs([&] { step(false); }, 10, 600.0);
    step(true);
    // The summed gradient must equal the full-batch gradient computed on one
    // GPU -- a real reference, not just "both GPUs agree with each other".
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMemcpy(gotdW.data(), dW[0], wN * 4, cudaMemcpyDeviceToHost));
    double err = relL2(gotdW, refdW);
    double tAll = timeKernelMs([&] { step(true); }, 10, 600.0);

    printf("%9d %11.3f %11.3f %11.3f %10.3f %8.2fx %10.0f%% %s\n", B, t1, tAll,
           tNo, tAll - tNo, t1 / tAll, 100.0 * t1 / (2.0 * tAll),
           err < 1e-5 ? "" : "  GRADIENT MISMATCH!");

    for (int i = 0; i < 2; ++i) {
      CUDA_CHECK(cudaSetDevice(g[i].id));
      cudaFree(X[i]); cudaFree(Y[i]); cudaFree(W[i]);
      cudaFree(dW[i]); cudaFree(scratch[i]);
    }
  }

  printf("\nWatch the 'comm ms' column: nearly constant, because the\n"
         "all-reduce moves the WEIGHTS, not the data. As the batch grows that\n"
         "fixed cost is amortised over more compute and efficiency climbs\n"
         "toward 100%%. Small batches cannot hide the interconnect.\n");
  printf("\nContrast with tensor parallelism (05), where the comm volume\n"
         "scales WITH the batch and therefore never amortises.\n");

  gpuDestroy(g[0]);
  gpuDestroy(g[1]);
  return 0;
}
