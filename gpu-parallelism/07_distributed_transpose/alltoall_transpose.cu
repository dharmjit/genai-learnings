// 07 - DISTRIBUTED TRANSPOSE = local transpose + all-to-all
//
// A matrix A[N x N] is row-block distributed: GPU0 owns the top half, GPU1
// the bottom half. We want A^T distributed the same way.
//
// Think of A as four N/2 x N/2 blocks:
//
//        cols 0..N/2   cols N/2..N
//  GPU0 [    A00    |     A01     ]
//  GPU1 [    A10    |     A11     ]
//
// A^T's top half is columns 0..N/2 of A, transposed:
//
//  GPU0 needs [ A00^T | A10^T ]      <- A10 lives on GPU1
//  GPU1 needs [ A01^T | A11^T ]      <- A01 lives on GPU0
//
// So: transpose the DIAGONAL block locally and keep it; transpose the
// OFF-DIAGONAL block and send it to the peer. Each GPU sends exactly one
// block. With P GPUs this generalises to an all-to-all, where every rank
// sends a distinct block to every other rank.
//
// Why this matters: all-to-all is the communication pattern behind sequence
// parallelism (DeepSpeed-Ulysses), Mixture-of-Experts token routing, and
// distributed FFTs. It is the most bandwidth-hungry collective there is --
// unlike an all-reduce it cannot be tree- or ring-optimised, because every
// byte has exactly one unique destination.
#include "helpers.cuh"
#include "collectives.cuh"

constexpr int TILE = 32, BROWS = 8;

// The bank-conflict-free transpose from lesson 01, applied to a sub-block.
__global__ void kTransposeBlock(float *__restrict__ out, int ldOut,
                                const float *__restrict__ in, int ldIn,
                                int rows, int cols) {
  __shared__ float tile[TILE][TILE + 1];
  int x = blockIdx.x * TILE + threadIdx.x;
  int y = blockIdx.y * TILE + threadIdx.y;
  for (int j = 0; j < TILE; j += BROWS)
    if (x < cols && y + j < rows)
      tile[threadIdx.y + j][threadIdx.x] = in[(size_t)(y + j) * ldIn + x];
  __syncthreads();
  x = blockIdx.y * TILE + threadIdx.x;
  y = blockIdx.x * TILE + threadIdx.y;
  for (int j = 0; j < TILE; j += BROWS)
    if (x < rows && y + j < cols)
      out[(size_t)(y + j) * ldOut + x] = tile[threadIdx.x][threadIdx.y + j];
}

int main(int argc, char **argv) {
  int N = argc > 1 ? atoi(argv[1]) : 8192;
  if (N % (2 * TILE)) { fprintf(stderr, "N must be a multiple of 64\n"); return 2; }
  requireTwoGpus();
  P2P p = enablePeerAccess(0, 1);

  int h = N / 2; // rows per GPU, and block dimension
  size_t halfN = (size_t)h * N, blockN = (size_t)h * h;
  printf("\nA[%d x %d] row-block distributed over 2 GPUs\n", N, N);
  printf("each GPU owns %.0f MiB; the all-to-all moves %.0f MiB per GPU\n\n",
         halfN * 4 / 1048576.0, blockN * 4 / 1048576.0);

  std::vector<float> hA((size_t)N * N), hRef((size_t)N * N), hGot((size_t)N * N);
  fillRandom(hA, 11);
  for (int r = 0; r < N; ++r)
    for (int c = 0; c < N; ++c) hRef[(size_t)c * N + r] = hA[(size_t)r * N + c];

  Gpu g[2];
  gpuInit(g[0], 0);
  gpuInit(g[1], 1);

  float *A[2], *AT[2], *sendbuf[2];
  for (int i = 0; i < 2; ++i) {
    CUDA_CHECK(cudaSetDevice(g[i].id));
    CUDA_CHECK(cudaMalloc(&A[i], halfN * 4));
    CUDA_CHECK(cudaMalloc(&AT[i], halfN * 4));
    CUDA_CHECK(cudaMalloc(&sendbuf[i], blockN * 4));
    CUDA_CHECK(cudaMemcpy(A[i], hA.data() + (size_t)i * halfN, halfN * 4,
                          cudaMemcpyHostToDevice));
  }

  dim3 block(TILE, BROWS), grid(ceilDiv(h, TILE), ceilDiv(h, TILE));

  // Each GPU's output row-block AT[i] is [h x N], holding two h x h blocks:
  //   AT[i] column-block j  =  (A[j] column-block i)^T
  // j == i is local; j != i must come from the peer.
  auto step = [&] {
    for (int i = 0; i < 2; ++i) {
      CUDA_CHECK(cudaSetDevice(g[i].id));
      // diagonal block: transpose straight into place
      kTransposeBlock<<<grid, block, 0, g[i].compute>>>(
          AT[i] + (size_t)i * h, N, A[i] + (size_t)i * h, N, h, h);
      // off-diagonal block: transpose into a contiguous staging buffer
      kTransposeBlock<<<grid, block, 0, g[i].compute>>>(
          sendbuf[i], h, A[i] + (size_t)(1 - i) * h, N, h, h);
    }
    commAfterCompute(g, 2);
    // Exchange. Destination is a column-block of a [h x N] matrix, so it is
    // strided: h rows of h floats, with a row pitch of N floats.
    for (int i = 0; i < 2; ++i) {
      int peer = 1 - i;
      CUDA_CHECK(cudaSetDevice(g[i].id));
      CUDA_CHECK(cudaMemcpy2DAsync(AT[i] + (size_t)peer * h, (size_t)N * 4,
                                   sendbuf[peer], (size_t)h * 4, (size_t)h * 4,
                                   h, cudaMemcpyDefault, g[i].comm));
    }
    waitComm(g, 2);
  };

  step();
  for (int i = 0; i < 2; ++i) {
    CUDA_CHECK(cudaSetDevice(g[i].id));
    CUDA_CHECK(cudaMemcpy(hGot.data() + (size_t)i * halfN, AT[i], halfN * 4,
                          cudaMemcpyDeviceToHost));
  }
  bool ok = relL2(hGot, hRef) == 0.0;
  printf("distributed transpose correct: %s\n", ok ? "yes" : "NO");

  double msDist = timeKernelMs(step, 10, 600.0);

  // Single-GPU transpose of the whole matrix, for reference.
  double msSingle;
  {
    CUDA_CHECK(cudaSetDevice(0));
    float *full, *fullT;
    CUDA_CHECK(cudaMalloc(&full, (size_t)N * N * 4));
    CUDA_CHECK(cudaMalloc(&fullT, (size_t)N * N * 4));
    CUDA_CHECK(cudaMemcpy(full, hA.data(), (size_t)N * N * 4, cudaMemcpyHostToDevice));
    dim3 gg(ceilDiv(N, TILE), ceilDiv(N, TILE));
    auto s = [&] {
      kTransposeBlock<<<gg, block>>>(fullT, N, full, N, N, N);
      CUDA_CHECK(cudaDeviceSynchronize());
    };
    msSingle = timeKernelMs(s, 10, 600.0);
    cudaFree(full); cudaFree(fullT);
  }

  // Estimate the pure-exchange cost by running the same transfers alone.
  double msCommOnly;
  {
    auto c = [&] {
      for (int i = 0; i < 2; ++i) {
        int peer = 1 - i;
        CUDA_CHECK(cudaSetDevice(g[i].id));
        CUDA_CHECK(cudaMemcpy2DAsync(AT[i] + (size_t)peer * h, (size_t)N * 4,
                                     sendbuf[peer], (size_t)h * 4,
                                     (size_t)h * 4, h, cudaMemcpyDefault,
                                     g[i].comm));
      }
      waitComm(g, 2);
    };
    msCommOnly = timeKernelMs(c, 10, 600.0);
  }

  std::vector<Row> t;
  t.push_back({"1 GPU, whole matrix", msSingle,
               copyGBs(2ull * N * N * 4, msSingle), 100.0, true});
  t.push_back({"2 GPUs, distributed", msDist,
               copyGBs(2ull * N * N * 4, msDist), 100.0 * msSingle / msDist, ok});
  t.push_back({"  of which: all-to-all", msCommOnly,
               copyGBs(2ull * blockN * 4, msCommOnly), 0, true});
  printTable("distributed transpose", "GB/s", t);

  printf("\nThe exchange alone is %.0f%% of the distributed runtime. Transpose\n"
         "does no arithmetic, so there is nothing to hide the transfer\n"
         "behind -- this is the worst case for multi-GPU, and a good reminder\n"
         "that splitting a memory-bound op across a slow link can LOSE.\n",
         100.0 * msCommOnly / msDist);
  printf("P2P was %s for this run.\n", p.enabled ? "enabled" : "NOT available");

  gpuDestroy(g[0]);
  gpuDestroy(g[1]);
  return 0;
}
