// 02 - Matmul: the arithmetic-intensity ladder.
//
// C[MxN] = A[MxK] @ B[KxN], all row-major.
//
// A naive matmul does 2*M*N*K flops but issues 2*M*N*K loads -> 0.25 flop per
// byte. The GPU can do ~60 flop per byte. So the entire optimisation story is
// REUSE: get each loaded byte used more times before you throw it away.
//
//   naive          1 load  per FMA   (reuse 1x)     memory bound, badly
//   coalesced      same traffic, but the right lanes ask for the right bytes
//   shared tiling  BK loads per BK FMAs             reuse = tile width
//   1D reg tiling  each thread owns TM outputs      reuse = TM
//   2D reg tiling  each thread owns TM*TN outputs   reuse = 2*TM*TN/(TM+TN)
//   tensor cores   the FMA itself gets 8-16x cheaper
#include "helpers.cuh"
#include <mma.h>
#include <cuda_fp16.h>

// ---------------------------------------------------------------- kernel 1
// One thread per output. threadIdx.x maps to the ROW, so consecutive lanes in
// a warp read down a column of B: 32 lanes, 32 different cache lines.
__global__ void kNaive(int M, int N, int K, const float *A, const float *B,
                       float *C) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < M && col < N) {
    float acc = 0.f;
    for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = acc;
  }
}

// ---------------------------------------------------------------- kernel 2
// Identical arithmetic, identical total traffic. The ONLY change is that
// consecutive threadIdx.x now maps to consecutive COLUMNS, so a warp reads one
// contiguous 128B line of B. Free speedup, and the most common beginner bug.
template <int BS>
__global__ void kCoalesced(int M, int N, int K, const float *A, const float *B,
                           float *C) {
  int row = blockIdx.x * BS + (threadIdx.x / BS);
  int col = blockIdx.y * BS + (threadIdx.x % BS);
  if (row < M && col < N) {
    float acc = 0.f;
    for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = acc;
  }
}

// ---------------------------------------------------------------- kernel 3
// Classic shared-memory tiling. Load a BS x BS tile of A and of B once, then
// every thread in the block reuses them BS times.
template <int BS>
__global__ void kSharedTiled(int M, int N, int K, const float *A,
                             const float *B, float *C) {
  __shared__ float As[BS][BS];
  __shared__ float Bs[BS][BS];
  int tRow = threadIdx.x / BS, tCol = threadIdx.x % BS;
  int row = blockIdx.x * BS + tRow, col = blockIdx.y * BS + tCol;

  float acc = 0.f;
  for (int t = 0; t < K; t += BS) {
    As[tRow][tCol] = (row < M && t + tCol < K) ? A[row * K + t + tCol] : 0.f;
    Bs[tRow][tCol] = (t + tRow < K && col < N) ? B[(t + tRow) * N + col] : 0.f;
    __syncthreads();
    for (int k = 0; k < BS; ++k) acc += As[tRow][k] * Bs[k][tCol];
    __syncthreads(); // must not overwrite the tile while peers still read it
  }
  if (row < M && col < N) C[row * N + col] = acc;
}

// ---------------------------------------------------------------- kernel 4
// 1D register tiling: each thread computes TM outputs stacked in a column.
// The B value loaded into a register is reused TM times without touching
// shared memory again. This is the single biggest step on the ladder.
template <int BM, int BN, int BK, int TM>
__global__ void kReg1D(int M, int N, int K, const float *A, const float *B,
                       float *C) {
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  A += blockIdx.y * BM * K;
  B += blockIdx.x * BN;
  C += blockIdx.y * BM * N + blockIdx.x * BN;

  int tCol = threadIdx.x % BN, tRow = threadIdx.x / BN;
  int innerRowA = threadIdx.x / BK, innerColA = threadIdx.x % BK;
  int innerRowB = threadIdx.x / BN, innerColB = threadIdx.x % BN;

  float acc[TM] = {0.f};
  for (int bk = 0; bk < K; bk += BK) {
    As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
    Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
    __syncthreads();
    A += BK;
    B += BK * N;
    for (int d = 0; d < BK; ++d) {
      float bv = Bs[d * BN + tCol]; // loaded once, used TM times
      for (int m = 0; m < TM; ++m) acc[m] += As[(tRow * TM + m) * BK + d] * bv;
    }
    __syncthreads();
  }
  for (int m = 0; m < TM; ++m) C[(tRow * TM + m) * N + tCol] = acc[m];
}

// ---------------------------------------------------------------- kernel 5
// 2D register tiling + float4 vectorised loads. Each thread owns a TM x TN
// patch, so TM+TN shared loads feed TM*TN FMAs. A is staged TRANSPOSED in
// shared memory so the inner loop reads it contiguously.
template <int BM, int BN, int BK, int TM, int TN>
__global__ void kReg2D(int M, int N, int K, const float *A, const float *B,
                       float *C) {
  __shared__ float As[BK * BM]; // transposed: As[k][m]
  __shared__ float Bs[BK * BN];

  A += blockIdx.y * BM * K;
  B += blockIdx.x * BN;
  C += blockIdx.y * BM * N + blockIdx.x * BN;

  const int tCol = threadIdx.x % (BN / TN);
  const int tRow = threadIdx.x / (BN / TN);
  const int innerRowA = threadIdx.x / (BK / 4);
  const int innerColA = threadIdx.x % (BK / 4);
  const int innerRowB = threadIdx.x / (BN / 4);
  const int innerColB = threadIdx.x % (BN / 4);

  float acc[TM * TN] = {0.f};
  float regM[TM], regN[TN];

  for (int bk = 0; bk < K; bk += BK) {
    float4 a = *reinterpret_cast<const float4 *>(&A[innerRowA * K + innerColA * 4]);
    As[(innerColA * 4 + 0) * BM + innerRowA] = a.x;
    As[(innerColA * 4 + 1) * BM + innerRowA] = a.y;
    As[(innerColA * 4 + 2) * BM + innerRowA] = a.z;
    As[(innerColA * 4 + 3) * BM + innerRowA] = a.w;
    *reinterpret_cast<float4 *>(&Bs[innerRowB * BN + innerColB * 4]) =
        *reinterpret_cast<const float4 *>(&B[innerRowB * N + innerColB * 4]);
    __syncthreads();
    A += BK;
    B += BK * N;

    for (int d = 0; d < BK; ++d) {
      for (int m = 0; m < TM; ++m) regM[m] = As[d * BM + tRow * TM + m];
      for (int nn = 0; nn < TN; ++nn) regN[nn] = Bs[d * BN + tCol * TN + nn];
      for (int m = 0; m < TM; ++m)
        for (int nn = 0; nn < TN; ++nn) acc[m * TN + nn] += regM[m] * regN[nn];
    }
    __syncthreads();
  }
  // Build the float4 in a temporary rather than casting &acc[...]: taking the
  // address of a register array forces it into local (off-chip) memory.
  for (int m = 0; m < TM; ++m)
    for (int nn = 0; nn < TN; nn += 4) {
      float4 v;
      v.x = acc[m * TN + nn + 0];
      v.y = acc[m * TN + nn + 1];
      v.z = acc[m * TN + nn + 2];
      v.w = acc[m * TN + nn + 3];
      *reinterpret_cast<float4 *>(&C[(tRow * TM + m) * N + tCol * TN + nn]) = v;
    }
}

// ---------------------------------------------------------------- kernel 6
// Tensor cores via WMMA. One warp cooperatively computes a 16x16 output tile.
// Inputs are fp16, accumulation is fp32. The fragment layout is opaque on
// purpose -- you are handing the hardware a matrix, not a loop.
using namespace nvcuda;
constexpr int WM = 16, WN = 16, WK = 16;

__global__ void kWmma(int M, int N, int K, const __half *A, const __half *B,
                      float *C) {
  int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  int warpN = blockIdx.y * blockDim.y + threadIdx.y;

  wmma::fragment<wmma::matrix_a, WM, WN, WK, __half, wmma::row_major> aFrag;
  wmma::fragment<wmma::matrix_b, WM, WN, WK, __half, wmma::row_major> bFrag;
  wmma::fragment<wmma::accumulator, WM, WN, WK, float> cFrag;
  wmma::fill_fragment(cFrag, 0.f);

  if (warpM * WM >= M || warpN * WN >= N) return;
  for (int k = 0; k < K; k += WK) {
    wmma::load_matrix_sync(aFrag, A + warpM * WM * K + k, K);
    wmma::load_matrix_sync(bFrag, B + k * N + warpN * WN, N);
    wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);
  }
  wmma::store_matrix_sync(C + warpM * WM * N + warpN * WN, cFrag, N,
                          wmma::mem_row_major);
}

__global__ void kToHalf(__half *out, const float *in, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t s = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += s) out[i] = __float2half(in[i]);
}

// ------------------------------------------------------------------- host

int main(int argc, char **argv) {
  int M = argc > 1 ? atoi(argv[1]) : 4096;
  int N = argc > 2 ? atoi(argv[2]) : 4096;
  int K = argc > 3 ? atoi(argv[3]) : 4096;
  if (M % 128 || N % 128 || K % 128) {
    fprintf(stderr, "sizes must be multiples of 128 (tiling assumes it)\n");
    return 2;
  }
  printf("C[%dx%d] = A[%dx%d] @ B[%dx%d]   %.1f GFLOP per call\n", M, N, M, K,
         K, N, 2.0 * M * N * K / 1e9);

  size_t sA = (size_t)M * K, sB = (size_t)K * N, sC = (size_t)M * N;
  std::vector<float> hA(sA), hB(sB), hRef(sC), hC(sC);
  fillRandom(hA, 1);
  fillRandom(hB, 2);

  float *dA, *dB, *dC;
  CUDA_CHECK(cudaMalloc(&dA, sA * 4));
  CUDA_CHECK(cudaMalloc(&dB, sB * 4));
  CUDA_CHECK(cudaMalloc(&dC, sC * 4));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), sA * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), sB * 4, cudaMemcpyHostToDevice));

  cublasHandle_t blas;
  CUBLAS_CHECK(cublasCreate(&blas));
  const float alpha = 1.f, beta = 0.f;

  // Ground truth from cuBLAS. Row-major C = A@B is column-major C^T = B^T@A^T,
  // which is just a no-transpose sgemm with the operands swapped.
  auto cublasRun = [&] {
    CUBLAS_CHECK(cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                             dB, N, dA, K, &beta, dC, N));
  };
  cublasRun();
  CUDA_SYNC_CHECK();
  CUDA_CHECK(cudaMemcpy(hRef.data(), dC, sC * 4, cudaMemcpyDeviceToHost));

  std::vector<Row> table;
  double refMs = 0;
  auto bench = [&](const char *name, auto &&launch, double tol) {
    CUDA_CHECK(cudaMemset(dC, 0, sC * 4));
    launch();
    CUDA_SYNC_CHECK();
    CUDA_CHECK(cudaMemcpy(hC.data(), dC, sC * 4, cudaMemcpyDeviceToHost));
    bool ok = checkAndReport(name, hC, hRef, tol);
    double ms = timeKernelMs(launch);
    table.push_back({name, ms, gemmTFLOPs(M, N, K, ms), 0, ok});
  };

  {
    dim3 b(32, 32), g(ceilDiv(M, 32), ceilDiv(N, 32));
    bench("1 naive (uncoalesced)",
          [&] { kNaive<<<g, b>>>(M, N, K, dA, dB, dC); }, 1e-4);
  }
  {
    dim3 g(ceilDiv(M, 32), ceilDiv(N, 32));
    bench("2 coalesced",
          [&] { kCoalesced<32><<<g, 32 * 32>>>(M, N, K, dA, dB, dC); }, 1e-4);
  }
  {
    dim3 g(ceilDiv(M, 32), ceilDiv(N, 32));
    bench("3 shared tiling 32x32",
          [&] { kSharedTiled<32><<<g, 32 * 32>>>(M, N, K, dA, dB, dC); }, 1e-4);
  }
  {
    dim3 g(ceilDiv(N, 64), ceilDiv(M, 64));
    bench("4 1D reg tiling (TM=8)",
          [&] { kReg1D<64, 64, 8, 8><<<g, 512>>>(M, N, K, dA, dB, dC); }, 1e-4);
  }
  {
    dim3 g(ceilDiv(N, 128), ceilDiv(M, 128));
    bench("5 2D reg tiling + float4",
          [&] { kReg2D<128, 128, 8, 8, 8><<<g, 256>>>(M, N, K, dA, dB, dC); },
          1e-4);
  }
  bench("6 cuBLAS SGEMM (fp32 ref)", cublasRun, 1e-6);

  for (auto &r : table)
    if (r.name[0] == '6') refMs = r.ms;
  for (auto &r : table) r.pctOfRef = 100.0 * refMs / r.ms;
  printTable("fp32 matmul ladder", "TFLOP/s", table);

  // ---- fp16 / tensor cores, judged against a tensor-core cuBLAS call ----
  __half *hA16, *hB16;
  CUDA_CHECK(cudaMalloc(&hA16, sA * 2));
  CUDA_CHECK(cudaMalloc(&hB16, sB * 2));
  kToHalf<<<1024, 256>>>(hA16, dA, sA);
  kToHalf<<<1024, 256>>>(hB16, dB, sB);
  CUDA_SYNC_CHECK();

  std::vector<float> hRef16(sC);
  auto cublasHalf = [&] {
    CUBLAS_CHECK(cublasGemmEx(blas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                              hB16, CUDA_R_16F, N, hA16, CUDA_R_16F, K, &beta,
                              dC, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  };
  CUDA_CHECK(cudaMemset(dC, 0, sC * 4));
  cublasHalf();
  CUDA_SYNC_CHECK();
  CUDA_CHECK(cudaMemcpy(hRef16.data(), dC, sC * 4, cudaMemcpyDeviceToHost));

  std::vector<Row> t16;
  double ref16 = 0;
  auto bench16 = [&](const char *name, auto &&launch) {
    CUDA_CHECK(cudaMemset(dC, 0, sC * 4));
    launch();
    CUDA_SYNC_CHECK();
    CUDA_CHECK(cudaMemcpy(hC.data(), dC, sC * 4, cudaMemcpyDeviceToHost));
    bool ok = checkAndReport(name, hC, hRef16, 2e-2);
    double ms = timeKernelMs(launch);
    t16.push_back({name, ms, gemmTFLOPs(M, N, K, ms), 0, ok});
  };

  {
    // 128 threads = 4 warps in x, each doing a 16-wide row strip.
    dim3 b(128, 4);
    dim3 g(ceilDiv(M, WM * 4), ceilDiv(N, WN * 4));
    bench16("7 WMMA 16x16x16 (fp16)",
            [&] { kWmma<<<g, b>>>(M, N, K, hA16, hB16, dC); });
  }
  bench16("8 cuBLAS fp16 TC (ref)", cublasHalf);
  for (auto &r : t16)
    if (r.name[0] == '8') ref16 = r.ms;
  for (auto &r : t16) r.pctOfRef = 100.0 * ref16 / r.ms;
  printTable("fp16 tensor-core matmul (vs fp32 above)", "TFLOP/s", t16);

  cublasDestroy(blas);
  cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(hA16); cudaFree(hB16);
  return 0;
}
