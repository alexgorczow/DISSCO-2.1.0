/*
  Microbenchmark: raw throughput of the SampleMapWorklet (sample = amp*sin(2pi*phase))
  on the Serial (CPU) backend vs the CUDA backend (including H2D/D2H transfer).

  This isolates the part of Partial::render that the portable layer accelerates.
  It intentionally does NOT measure reverb/loudness, which dominate whole-song
  render time (see restructure/02_RESULTS.md). Run: make bench && ./bench_map
*/
#include "PortableSynth.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>

using clk = std::chrono::high_resolution_clock;
static double ms(clk::time_point a, clk::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

int main(int argc, char** argv) {
  // Sizes: 1s, 30s, 300s of audio at 44.1 kHz (single partial worth of samples).
  std::vector<long> sizes = {44100L, 44100L * 30, 44100L * 300};
  if (argc > 1) sizes = { atol(argv[1]) };

  portable::SampleMapWorklet worklet;
  printf("%-12s %14s %14s %10s %12s\n",
         "N", "CPU map (ms)", "GPU map (ms)", "speedup", "CPU Msmpl/s");

  for (long n : sizes) {
    std::vector<float> amp(n), phase(n), waveCpu(n), ampCpu(n), waveGpu(n), ampGpu(n);
    for (long i = 0; i < n; ++i) {
      amp[i]   = 0.5f + 0.5f * (float)(i % 1000) / 1000.0f;
      phase[i] = (float)((i * 440.0 / 44100.0) - (long)(i * 440.0 / 44100.0));
    }

    // warm up + CPU timing (3 reps, take best)
    double cpuBest = 1e30;
    for (int r = 0; r < 3; ++r) {
      auto t0 = clk::now();
      for (long s = 0; s < n; ++s)
        worklet((std::size_t)s, amp.data(), phase.data(), waveCpu.data(), ampCpu.data());
      auto t1 = clk::now();
      cpuBest = std::min(cpuBest, ms(t0, t1));
    }

    // GPU timing (transfer + kernel; persistent buffers after the 1st rep, so
    // min-of-reps reflects the realistic amortized per-call cost).
    double gpuBest = 1e30;
    for (int r = 0; r < 3; ++r) {
      auto t0 = clk::now();
      portable::renderMapCuda(amp.data(), phase.data(), waveGpu.data(), n);
      auto t1 = clk::now();
      gpuBest = std::min(gpuBest, ms(t0, t1));
    }
    (void)ampGpu;

    // parity spot-check
    double maxAbs = 0;
    for (long s = 0; s < n; ++s)
      maxAbs = std::max(maxAbs, (double)std::fabs(waveCpu[s] - waveGpu[s]));

    printf("%-12ld %14.3f %14.3f %9.2fx %12.1f   (map parity maxAbs=%.2e)\n",
           n, cpuBest, gpuBest, cpuBest / gpuBest, n / (cpuBest * 1000.0), maxAbs);
  }
  return 0;
}
