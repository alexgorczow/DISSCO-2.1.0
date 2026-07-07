/*
  LASS/portable/CompositeCuda.cu — deterministic GPU score composite.
  See CompositeCuda.h for the contract and 06_DETERMINISTIC_COMPOSITE.md for
  the design + bit-exactness argument (pure IEEE-754 adds, stream-ordered).
*/
#include "CompositeCuda.h"

#include "../src/MultiTrack.h"
#include "../src/Track.h"
#include "../src/SoundSample.h"

#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

namespace {

inline void checkCuda(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    fprintf(stderr, "CUDA error (composite %s): %s\n", what,
            cudaGetErrorString(e));
  }
}

// score[off+i] += src[i]. One thread per sample; each output position is
// touched exactly once per launch, and launches on the (single) default stream
// execute in launch order -- so per-sample add order == commit order.
__global__ void addKernel(float* dst, const float* src, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] += src[i];
}

// Device-side score state (composite thread only; see header contract).
struct DeviceScore {
  std::vector<float*> wave;   // one per track
  std::vector<float*> amp;    // one per track
  long len = 0;               // current logical length (samples)
  float* stage = nullptr;     // persistent H2D staging buffer
  long stageCap = 0;

  float* stageFor(const float* src, long n) {
    if (n > stageCap) {
      if (stage) cudaFree(stage);
      checkCuda(cudaMalloc(&stage, (size_t)n * sizeof(float)), "stage alloc");
      stageCap = n;
    }
    checkCuda(cudaMemcpy(stage, src, (size_t)n * sizeof(float),
                         cudaMemcpyHostToDevice), "H2D");
    return stage;
  }
};
DeviceScore g_score;

// Allocate a zeroed buffer of newLen, copy the exact oldLen prefix, free old.
float* growBuffer(float* oldBuf, long oldLen, long newLen) {
  float* nb = nullptr;
  checkCuda(cudaMalloc(&nb, (size_t)newLen * sizeof(float)), "grow alloc");
  checkCuda(cudaMemset(nb, 0, (size_t)newLen * sizeof(float)), "grow zero");
  if (oldBuf) {
    if (oldLen > 0)
      checkCuda(cudaMemcpy(nb, oldBuf, (size_t)oldLen * sizeof(float),
                           cudaMemcpyDeviceToDevice), "grow copy");
    cudaFree(oldBuf);
  }
  return nb;
}

} // namespace

namespace portable {

void compositeCudaEnsure(int numTracks, long numSamples) {
  if ((long)g_score.wave.size() == 0) {
    g_score.wave.assign(numTracks, nullptr);
    g_score.amp.assign(numTracks, nullptr);
  }
  if (numSamples <= g_score.len) return;
  for (int t = 0; t < numTracks; ++t) {
    g_score.wave[t] = growBuffer(g_score.wave[t], g_score.len, numSamples);
    g_score.amp[t]  = growBuffer(g_score.amp[t],  g_score.len, numSamples);
  }
  g_score.len = numSamples;
}

void compositeCudaAdd(int track, bool kindAmp, const float* src, long n,
                      long offset) {
  if (track < 0 || track >= (int)g_score.wave.size()) return;
  // Mirror SoundSample::composite's clip-to-buffer semantics exactly.
  if (offset >= g_score.len) return;
  if (offset + n > g_score.len) n = g_score.len - offset;
  if (n <= 0) return;

  float* dst = (kindAmp ? g_score.amp[track] : g_score.wave[track]) + offset;
  float* d_src = g_score.stageFor(src, n);
  const int block = 256;
  const long grid = (n + block - 1) / block;
  addKernel<<<(unsigned)grid, block>>>(dst, d_src, n);
  checkCuda(cudaGetLastError(), "launch");
  // NOTE: the sync H2D of the NEXT commit orders after this kernel on the
  // default stream, so reusing one staging buffer is safe.
}

MultiTrack* compositeCudaFetch(int numChannels, unsigned int samplingRate) {
  if (g_score.len == 0 || g_score.wave.empty()) return nullptr;
  checkCuda(cudaDeviceSynchronize(), "fetch sync");

  MultiTrack* mt = new MultiTrack(numChannels, g_score.len, samplingRate);
  for (int t = 0; t < numChannels; ++t) {
    Track* tr = mt->get(t);
    checkCuda(cudaMemcpy(tr->getWave().getData(), g_score.wave[t],
                         (size_t)g_score.len * sizeof(float),
                         cudaMemcpyDeviceToHost), "D2H wave");
    checkCuda(cudaMemcpy(tr->getAmp().getData(), g_score.amp[t],
                         (size_t)g_score.len * sizeof(float),
                         cudaMemcpyDeviceToHost), "D2H amp");
    cudaFree(g_score.wave[t]);
    cudaFree(g_score.amp[t]);
  }
  g_score.wave.clear();
  g_score.amp.clear();
  g_score.len = 0;
  if (g_score.stage) { cudaFree(g_score.stage); g_score.stage = nullptr; g_score.stageCap = 0; }
  return mt;
}

} // namespace portable
