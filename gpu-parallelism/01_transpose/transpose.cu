// 01 - Transpose: a pure memory-movement problem.
//
// Transpose does zero arithmetic, so the only thing that matters is how you
// touch DRAM and shared memory. That makes it the cleanest possible lesson in
// the two rules that dominate CUDA performance:
//
//   1. GLOBAL memory wants a warp's 32 lanes to hit ONE contiguous 128B line
//      (coalescing). Miss this and you pay up to 32x.
//   2. SHARED memory is 32 banks of 4B, striped by address. If two lanes in a
//      warp hit different rows of the SAME bank, the access serialises.
//
// The `copy` kernel is the speed-of-light: it moves exactly the same bytes
// without transposing. Any transpose kernel is judged against it.
#include "helpers.cuh"

constexpr int TILE = 32;
constexpr int BROWS = 8; // each thread handles TILE/BROWS = 4 elements

// ---------------------------------------------------------------- kernel 0
// Upper bound. Reads coalesced, writes coalesced, no transpose at all.
__global__ void kCopy(float *__restrict__ out, const float *__restrict__ in,
                      int rows, int cols) {
  int x = blockIdx.x * TILE + threadIdx.x;
  int y = blockIdx.y * TILE + threadIdx.y;
  for (int j = 0; j < TILE; j += BROWS)
    if (x < cols && y + j < rows)
      out[(y + j) * cols + x] = in[(y + j) * cols + x];
}

// ---------------------------------------------------------------- kernel 1
// Naive. Reads are coalesced; writes stride by `rows` floats, so each lane
// lands in a different cache line -> 32 separate memory transactions.
__global__ void kNaive(float *__restrict__ out, const float *__restrict__ in,
                       int rows, int cols) {
  int x = blockIdx.x * TILE + threadIdx.x;
  int y = blockIdx.y * TILE + threadIdx.y;
  for (int j = 0; j < TILE; j += BROWS)
    if (x < cols && y + j < rows) out[x * rows + (y + j)] = in[(y + j) * cols + x];
}

// ---------------------------------------------------------------- kernel 2
// Stage through shared memory so BOTH global accesses are coalesced. But the
// tile is [32][32]: shared address = r*32+c, and bank = addr % 32 = c. In the
// read-back loop every lane uses the same column -> 32-way bank conflict.
__global__ void kSharedConflict(float *__restrict__ out,
                                const float *__restrict__ in, int rows,
                                int cols) {
  __shared__ float tile[TILE][TILE];
  int x = blockIdx.x * TILE + threadIdx.x;
  int y = blockIdx.y * TILE + threadIdx.y;
  for (int j = 0; j < TILE; j += BROWS)
    if (x < cols && y + j < rows)
      tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * cols + x];
  __syncthreads();
  // Swap which block-coordinate feeds x vs y: that is the actual transpose.
  x = blockIdx.y * TILE + threadIdx.x;
  y = blockIdx.x * TILE + threadIdx.y;
  for (int j = 0; j < TILE; j += BROWS)
    if (x < rows && y + j < cols)
      out[(y + j) * rows + x] = tile[threadIdx.x][threadIdx.y + j];
}

// ---------------------------------------------------------------- kernel 3
// Pad the row to 33 floats. Now bank = (r*33 + c) % 32 = (r + c) % 32, so a
// fixed column across 32 rows spreads over all 32 banks. Costs 32 extra floats
// of shared memory per tile.
__global__ void kSharedPadded(float *__restrict__ out,
                              const float *__restrict__ in, int rows,
                              int cols) {
  __shared__ float tile[TILE][TILE + 1];
  int x = blockIdx.x * TILE + threadIdx.x;
  int y = blockIdx.y * TILE + threadIdx.y;
  for (int j = 0; j < TILE; j += BROWS)
    if (x < cols && y + j < rows)
      tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * cols + x];
  __syncthreads();
  x = blockIdx.y * TILE + threadIdx.x;
  y = blockIdx.x * TILE + threadIdx.y;
  for (int j = 0; j < TILE; j += BROWS)
    if (x < rows && y + j < cols)
      out[(y + j) * rows + x] = tile[threadIdx.x][threadIdx.y + j];
}

// ---------------------------------------------------------------- kernel 4
// Same conflict-free bank mapping, zero wasted shared memory: rotate each row
// by its row index. Logical (r,c) lives at physical [r][(c + r) % 32].
// Shared memory is what caps occupancy in real kernels, so "free" matters.
__global__ void kSharedSwizzled(float *__restrict__ out,
                                const float *__restrict__ in, int rows,
                                int cols) {
  __shared__ float tile[TILE][TILE];
  int x = blockIdx.x * TILE + threadIdx.x;
  int y = blockIdx.y * TILE + threadIdx.y;
  for (int j = 0; j < TILE; j += BROWS) {
    int r = threadIdx.y + j, c = threadIdx.x;
    if (x < cols && y + j < rows)
      tile[r][(c + r) & (TILE - 1)] = in[(y + j) * cols + x];
  }
  __syncthreads();
  x = blockIdx.y * TILE + threadIdx.x;
  y = blockIdx.x * TILE + threadIdx.y;
  for (int j = 0; j < TILE; j += BROWS) {
    int r = threadIdx.x, c = threadIdx.y + j;
    if (x < rows && y + j < cols)
      out[(y + j) * rows + x] = tile[r][(c + r) & (TILE - 1)];
  }
}

// ------------------------------------------------------------------- host

static void refTranspose(std::vector<float> &out, const std::vector<float> &in,
                         int rows, int cols) {
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < cols; ++c) out[(size_t)c * rows + r] = in[(size_t)r * cols + c];
}

static void runAll(int rows, int cols) {
  size_t n = (size_t)rows * cols, bytes = n * sizeof(float);

  std::vector<float> hIn(n), hRef(n), hOut(n);
  fillRandom(hIn);
  refTranspose(hRef, hIn, rows, cols);

  float *dIn, *dOut;
  CUDA_CHECK(cudaMalloc(&dIn, bytes));
  CUDA_CHECK(cudaMalloc(&dOut, bytes));
  CUDA_CHECK(cudaMemcpy(dIn, hIn.data(), bytes, cudaMemcpyHostToDevice));

  dim3 block(TILE, BROWS);
  dim3 grid(ceilDiv(cols, TILE), ceilDiv(rows, TILE));

  cublasHandle_t blas;
  CUBLAS_CHECK(cublasCreate(&blas));

  std::vector<Row> table;
  double refMs = 0;

  auto bench = [&](const char *name, auto &&launch, bool isTranspose) {
    CUDA_CHECK(cudaMemset(dOut, 0, bytes));
    double ms = timeKernelMs(launch);
    CUDA_SYNC_CHECK();
    CUDA_CHECK(cudaMemcpy(hOut.data(), dOut, bytes, cudaMemcpyDeviceToHost));
    bool ok = isTranspose ? checkAndReport(name, hOut, hRef, 0.0)
                          : checkAndReport(name, hOut, hIn, 0.0);
    if (refMs == 0) refMs = ms;
    // 2x bytes: every element is read once and written once.
    table.push_back({name, ms, copyGBs(2 * bytes, ms), 100.0 * refMs / ms, ok});
  };

  bench("copy (speed of light, no T)",
        [&] { kCopy<<<grid, block>>>(dOut, dIn, rows, cols); }, false);
  bench("naive (uncoalesced writes)",
        [&] { kNaive<<<grid, block>>>(dOut, dIn, rows, cols); }, true);
  bench("shared, 32-way bank conflict",
        [&] { kSharedConflict<<<grid, block>>>(dOut, dIn, rows, cols); }, true);
  bench("shared + padding [32][33]",
        [&] { kSharedPadded<<<grid, block>>>(dOut, dIn, rows, cols); }, true);
  bench("shared + swizzle (no padding)",
        [&] { kSharedSwizzled<<<grid, block>>>(dOut, dIn, rows, cols); }, true);

  // cuBLAS geam. Row-major [rows x cols] is column-major [cols x rows], so the
  // output we want is the column-major transpose with leading dim `rows`.
  const float alpha = 1.f, beta = 0.f;
  bench("cublasSgeam", [&] {
    CUBLAS_CHECK(cublasSgeam(blas, CUBLAS_OP_T, CUBLAS_OP_N, rows, cols, &alpha,
                             dIn, cols, &beta, dOut, rows, dOut, rows));
  }, true);

  char title[128];
  snprintf(title, sizeof title, "transpose %d x %d  (%.0f MiB/buffer, %s)",
           rows, cols, bytes / 1048576.0,
           bytes * 2 < 128u * 1024 * 1024 ? "FITS in 128 MiB L2"
                                          : "exceeds L2 -> DRAM bound");
  printTable(title, "GB/s", table);

  cublasDestroy(blas);
  CUDA_CHECK(cudaFree(dIn));
  CUDA_CHECK(cudaFree(dOut));
}

int main(int argc, char **argv) {
  if (argc > 2) { runAll(atoi(argv[1]), atoi(argv[2])); return 0; }

  // Two regimes on purpose. Blackwell's L2 is 128 MiB -- big enough that a
  // 2048^2 working set never touches DRAM. Bank conflicts are invisible when
  // DRAM is the bottleneck and expensive when it is not; you have to measure
  // in the regime you actually run in.
  runAll(8192, 8192);
  runAll(2048, 2048);
  printf("\nSame kernels, two working-set sizes. Compare the conflict row\n"
         "against padding/swizzle in each table: the optimisation only pays\n"
         "when the kernel is not already starved by DRAM.\n");
  return 0;
}
