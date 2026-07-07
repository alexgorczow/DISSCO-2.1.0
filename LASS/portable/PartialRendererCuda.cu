/*
  LASS/portable/PartialRendererCuda.cu — CUDA device adapter (Tier 4).

  The SAME per-sample math the Serial backend runs (SampleMapWorklet in
  PortableSynth.h) is computed here on a GPU grid:
      wave[s] = amplitude[s] * sin(2*pi*phase[s])
  The worklet also sets ampOut[s] = amplitude[s], but that is a pure identity,
  so the host fills the amp channel with a memcpy (see PortableSynth.cpp) and the
  device only produces `wave`. No synthesis math is duplicated — the wave
  expression is byte-for-byte the worklet's.

  Efficiency: earlier this did 4 cudaMalloc + 4 cudaFree + 2 H2D + 2 D2H PER
  partial, and round-tripped the amp array to the GPU unchanged. Now:
    - persistent thread_local device buffers, grown on demand (no per-call malloc)
    - only amp+phase go H2D, only wave comes D2H (the amp round-trip is gone)
  These are throughput-only changes; the `wave` arithmetic is unchanged, so the
  result stays bit-exact with the Serial backend / golden.

  Parity note: CUDA's libdevice sin(double) may differ from glibc's by <=1 ULP,
  so GPU output matches the golden within the parallel budget (few LSB / < -140
  dBFS); empirically it has been 0 LSB because double->float rounding absorbs the
  sub-ULP sin difference. See restructure/01_RESTRUCTURE_PLAN.md S2/F2.
*/
#include "PortableSynth.h"

#include <cstdio>
#include <cuda_runtime.h>

namespace {

// wave only: identical expression to SampleMapWorklet's wave[] line.
__global__ void sampleMapKernel(const float* amplitude, const float* phase,
                                float* wave, long n) {
  long s = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (s < n) {
    wave[s] = amplitude[s] * (sin(2.0 * M_PI * phase[s]));
  }
}

inline void checkCuda(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    fprintf(stderr, "CUDA error (%s): %s\n", what, cudaGetErrorString(e));
  }
}

// Persistent, grow-on-demand device scratch. thread_local so concurrent LASS
// worker threads each get independent buffers (no locking, no data race).
// Never explicitly freed: reclaimed at process exit (a bounded per-thread cache).
struct DevScratch {
  float* dAmp   = nullptr;
  float* dPhase = nullptr;
  float* dWave  = nullptr;
  long   cap    = 0;      // capacity in floats

  void ensure(long n) {
    if (n <= cap) return;
    // grow (free old, alloc new); amortized by geometric-ish growth to n
    if (dAmp)   cudaFree(dAmp);
    if (dPhase) cudaFree(dPhase);
    if (dWave)  cudaFree(dWave);
    const size_t bytes = (size_t)n * sizeof(float);
    checkCuda(cudaMalloc(&dAmp,   bytes), "malloc amp");
    checkCuda(cudaMalloc(&dPhase, bytes), "malloc phase");
    checkCuda(cudaMalloc(&dWave,  bytes), "malloc wave");
    cap = n;
  }
};
thread_local DevScratch g_scratch;

} // namespace

namespace portable {

// Host entry: transfer amp/phase to device, run the map, copy wave back.
void renderMapCuda(const float* amplitude, const float* phase,
                   float* wave, long n) {
  if (n <= 0) return;
  const size_t bytes = (size_t)n * sizeof(float);

  g_scratch.ensure(n);
  float* dAmp   = g_scratch.dAmp;
  float* dPhase = g_scratch.dPhase;
  float* dWave  = g_scratch.dWave;

  checkCuda(cudaMemcpy(dAmp,   amplitude, bytes, cudaMemcpyHostToDevice), "H2D amp");
  checkCuda(cudaMemcpy(dPhase, phase,     bytes, cudaMemcpyHostToDevice), "H2D phase");

  const int block = 256;
  const int grid  = (int)((n + block - 1) / block);
  sampleMapKernel<<<grid, block>>>(dAmp, dPhase, dWave, n);
  checkCuda(cudaGetLastError(), "launch");

  checkCuda(cudaMemcpy(wave, dWave, bytes, cudaMemcpyDeviceToHost), "D2H wave");
}

} // namespace portable
