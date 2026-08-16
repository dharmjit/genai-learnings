// The CPU half of experiment 1 in post 01, as a standalone program so you can
// run it on whatever machine you are reading this on.
//
//   cc -O3 -march=native -o cpu_chain cpu_chain.c && ./cpu_chain
//
// -march=native matters: without FMA in the target ISA the multiply-add is
// two instructions (or a libm call), and you measure the wrong thing.
//
// It times the identical dependent FMA chain that warp_lab.cu runs on one GPU
// thread. Dependent means each op waits for the previous result, so this is a
// pure latency measurement: no instruction-level parallelism, no vectorisation.
//
// Note the volatile laundering of a and b. Without it the operands are
// compile-time constants, the entire loop is a constant expression, and the
// compiler deletes it -- which is exactly the bug the first version of this
// benchmark shipped with.
#include <stdio.h>
#include <time.h>

int main(void) {
  const long long N = 50000000;
  volatile float va = 0.9999999f, vb = 1e-7f;
  const float a = va, b = vb;

  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);
  float v = 1.0f;
  for (long long i = 0; i < N; ++i) v = v * a + b;
  clock_gettime(CLOCK_MONOTONIC, &t1);
  volatile float sink = v;
  (void)sink;

  double ms = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
  printf("one CPU core: %lld dependent FMAs in %.1f ms  =  %.2f ns/FMA\n", N, ms,
         ms * 1e6 / (double)N);
  return 0;
}
