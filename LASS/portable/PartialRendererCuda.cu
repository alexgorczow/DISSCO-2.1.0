/*
  LASS/portable/PartialRendererCuda.cu — CUDA device adapter (Tier 4).

  This is the payoff of the restructuring: the SAME SampleMapWorklet body that
  the Serial backend runs (defined inline in PortableSynth.h) is compiled here by
  nvcc as device code (LASS_EXEC -> __host__ __device__) and dispatched over a
  GPU grid. No synthesis math is duplicated — only the dispatch differs.

  Parity note: CUDA's libdevice sin(double) may differ from glibc's by <=1 ULP,
  so GPU output matches the golden within the parallel budget (few LSB / < -140
  dBFS), not bit-exactly. See restructure/01_RESTRUCTURE_PLAN.md S2/F2.
*/
#include "PortableSynth.h"

#include <cstdio>
#include <cuda_runtime.h>

namespace {

__global__ void sampleMapKernel(const float* amplitude, const float* phase,
                                float* wave, float* ampOut, long n) {
  long s = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (s < n) {
    portable::SampleMapWorklet worklet;   // the write-once worklet, on device
    worklet((std::size_t)s, amplitude, phase, wave, ampOut);
  }
}

inline void checkCuda(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    fprintf(stderr, "CUDA error (%s): %s\n", what, cudaGetErrorString(e));
  }
}

} // namespace

namespace portable {

// Host entry: transfer amp/phase to device, run the map, copy wave/amp back.
void renderMapCuda(const float* amplitude, const float* phase,
                   float* wave, float* ampOut, long n) {
  if (n <= 0) return;
  const size_t bytes = (size_t)n * sizeof(float);

  float *dAmp = nullptr, *dPhase = nullptr, *dWave = nullptr, *dAmpOut = nullptr;
  checkCuda(cudaMalloc(&dAmp,    bytes), "malloc amp");
  checkCuda(cudaMalloc(&dPhase,  bytes), "malloc phase");
  checkCuda(cudaMalloc(&dWave,   bytes), "malloc wave");
  checkCuda(cudaMalloc(&dAmpOut, bytes), "malloc ampOut");

  checkCuda(cudaMemcpy(dAmp,   amplitude, bytes, cudaMemcpyHostToDevice), "H2D amp");
  checkCuda(cudaMemcpy(dPhase, phase,     bytes, cudaMemcpyHostToDevice), "H2D phase");

  const int block = 256;
  const int grid  = (int)((n + block - 1) / block);
  sampleMapKernel<<<grid, block>>>(dAmp, dPhase, dWave, dAmpOut, n);
  checkCuda(cudaGetLastError(), "launch");
  checkCuda(cudaDeviceSynchronize(), "sync");

  checkCuda(cudaMemcpy(wave,   dWave,   bytes, cudaMemcpyDeviceToHost), "D2H wave");
  checkCuda(cudaMemcpy(ampOut, dAmpOut, bytes, cudaMemcpyDeviceToHost), "D2H amp");

  cudaFree(dAmp); cudaFree(dPhase); cudaFree(dWave); cudaFree(dAmpOut);
}

} // namespace portable
