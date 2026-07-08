/*
  bench_scan.cu — gpu-fast feasibility spike (restructure/07, Tier 3).

  Measures the TWO risky recurrences of a budget-relaxed GPU-fused pipeline,
  for both SPEED and ACCUMULATED ERROR vs the exact CPU float reference:

  T1  Reverb unit (6 LPComb + AllPass, the exact LASS math and REV parameters).
      - LPComb: y[n] = s[n-D];  s[n] = x[n] + g*L[n];  L[n] = y[n] + a*L[n-1]
        (a = lowpass feedback gain). Every-sample chain L is the sequential
        obstacle. GPU: process D-sample blocks sequentially (w-inputs of block
        k depend only on L of block k-1), affine parallel scan inside a block:
            w[n] = x[n-D] + g*L[n-D]   (== the comb OUTPUT y[n])
            L[n] = a*L[n-1] + w[n]
        The scan REASSOCIATES float adds -> the error we are here to measure.
      - AllPass: y[n] = -g*x[n] + (1-g^2)*(x[n-D] + g*y[n-D]) couples only at
        lag D -> D independent residue-class chains; one thread per class
        replays the EXACT CPU float op order (expected bit-exact).
      Error = GPU float pipeline vs CPU float sequential, same input.

  T2  Synth phase accumulator: CPU does  p += f/sr (float); pmod(p)  per
      sample; sample = A*sin(2*pi*p) (double sin, float store).
      GPU-fast would do a DOUBLE prefix-sum of the float increments + wrap +
      float sinf. The double scan is *more* accurate than the float chain, so
      the diff vs CPU measures the CPU chain's own accumulated float noise --
      a CONTRACT finding (raw sample-diff is the wrong yardstick for phase),
      quantified here alongside CPU-float-chain vs CPU-double-chain.

  Build:  nvcc -O3 -std=c++14 bench_scan.cu -o bench_scan
  Run:    ./bench_scan [seconds=4] [reps=3]
*/
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <chrono>

using clk = std::chrono::high_resolution_clock;
static double msec(clk::time_point a, clk::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

// ---------------------------------------------------------------------------
// Exact LASS reverb parameters (Reverb::ConstructorCommon + REV_Medium spread).
static const int   NCOMB = 6;
static const float COMB_G[NCOMB] = {0.46f, 0.48f, 0.50f, 0.52f, 0.53f, 0.55f};
static const float COMB_DELAY_S[NCOMB] = {0.050f, 0.056f, 0.061f, 0.068f, 0.072f, 0.078f};
// REV_Medium (tutorial): lp = 0.05 + spread*(0.95 - comb_g), spread = 0.5
static const float HILOW_SPREAD = 0.5f;
static const float AP_GAIN = 0.1f;      // tutorial <AllPass>0.1
static const float AP_DELAY_S = 0.5f;   // tutorial <Delay>0.5
static const float MIX = 0.5f;          // constant percentReverb stand-in
static const unsigned SR = 44100;

// ---------------------------------------------------------------------------
// CPU reference: verbatim float math / op order of LPCombFilter, LowPassFilter,
// AllPassFilter, Reverb::do_reverb (hist queues modeled as ring buffers).
struct CpuComb {
  float g, a; int D; std::vector<float> s; int idx; float L;
  CpuComb(float g_, float a_, int D_) : g(g_), a(a_), D(D_), s(D_, 0.0f), idx(0), L(0.0f) {}
  inline float step(float x) {
    float y = s[idx];                 // x_hist.dequeue()
    L = y + (a * L);                  // lpf.do_filter(y)
    s[idx] = x + (g * L);             // x_hist.enqueue(x + g*lpf)
    idx = (idx + 1) % D;
    return y;
  }
};
struct CpuAllPass {
  float g, g2; int D; std::vector<float> xh, yh; int idx;
  CpuAllPass(float g_, int D_) : g(g_), g2(g_ * g_), D(D_), xh(D_, 0.0f), yh(D_, 0.0f), idx(0) {}
  inline float step(float x) {
    float y = -g * x + (1.0f - g2) * (xh[idx] + g * yh[idx]);
    xh[idx] = x; yh[idx] = y;
    idx = (idx + 1) % D;
    return y;
  }
};

static void cpuReverb(const std::vector<float>& x, std::vector<float>& out) {
  long N = (long)x.size();
  std::vector<CpuComb> combs;
  for (int k = 0; k < NCOMB; ++k) {
    float lp = 0.05f + HILOW_SPREAD * (0.95f - COMB_G[k]);
    combs.push_back(CpuComb(COMB_G[k], lp, (int)(SR * COMB_DELAY_S[k])));
  }
  CpuAllPass ap(AP_GAIN, (int)((double)AP_DELAY_S * SR));
  for (long n = 0; n < N; ++n) {
    float y = combs[0].step(x[n]);            // Reverb::do_reverb op order
    y += combs[1].step(x[n]);
    y += combs[2].step(x[n]);
    y += combs[3].step(x[n]);
    y += combs[4].step(x[n]);
    y += combs[5].step(x[n]);
    y /= (float)NCOMB;
    y = ap.step(y);
    out[n] = (MIX * y) + ((1.0f - MIX) * x[n]);
  }
}

// ---------------------------------------------------------------------------
// GPU comb: blocked affine scan.   pair (A,B) means  L -> A*L + B
struct Affine { float A, B; };
struct ComposeF {
  __host__ __device__ Affine operator()(const Affine& p, const Affine& q) const {
    Affine r; r.A = q.A * p.A; r.B = q.A * p.B + q.B; return r;   // apply p, then q
  }
};
struct AffineD { double A, B; };
struct ComposeD {
  __host__ __device__ AffineD operator()(const AffineD& p, const AffineD& q) const {
    AffineD r; r.A = q.A * p.A; r.B = q.A * p.B + q.B; return r;
  }
};

__global__ void combW(const float* x, const float* L, float* w, float* y,
                      long n0, long len, long N, int D, float g) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= len) return;
  long n = n0 + i;
  long m = n - D;
  float wv = (m >= 0) ? (x[m] + g * L[m]) : 0.0f;
  w[i] = wv;
  y[n] = wv;                       // comb output y[n] == w[n]
}
__global__ void packAffineF(const float* w, Affine* e, long len, float a) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) { e[i].A = a; e[i].B = w[i]; }
}
__global__ void unpackLF(const Affine* s, float* L, long n0, long len, float carry) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) L[n0 + i] = s[i].A * carry + s[i].B;
}
__global__ void packAffineD(const float* w, AffineD* e, long len, double a) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) { e[i].A = a; e[i].B = (double)w[i]; }
}
__global__ void unpackLD(const AffineD* s, float* L, long n0, long len, double carry) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) L[n0 + i] = (float)(s[i].A * carry + s[i].B);
}

// AllPass: one thread per residue class, replaying exact CPU float op order.
__global__ void allpassKernel(const float* x, float* y, long N, int D, float g) {
  long c = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= D) return;
  float g2 = g * g;
  float xprev = 0.0f, yprev = 0.0f;          // zero-filled history
  for (long n = c; n < N; n += D) {
    float yv = -g * x[n] + (1.0f - g2) * (xprev + g * yprev);
    xprev = x[n]; yprev = yv;
    y[n] = yv;
  }
}
__global__ void sumMix(const float* y0, const float* y1, const float* y2,
                       const float* y3, const float* y4, const float* y5,
                       const float* x, float* out, long N, float mix) {
  long n = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  float y = y0[n];                            // exact CPU accumulation order
  y += y1[n]; y += y2[n]; y += y3[n]; y += y4[n]; y += y5[n];
  y /= (float)NCOMB;
  out[n] = y;                                 // allpass applied afterwards
}
__global__ void mixDry(const float* y, const float* x, float* out, long N, float mix) {
  long n = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) out[n] = (mix * y[n]) + ((1.0f - mix) * x[n]);
}

// Runs the full 6-comb + allpass + mix unit on the device.
// scanDouble: carry the L-scan in double (error-floor variant).
static void gpuReverb(const std::vector<float>& xh, std::vector<float>& outh,
                      bool scanDouble, double* wallMs) {
  long N = (long)xh.size();
  thrust::device_vector<float> x(xh.begin(), xh.end());
  thrust::device_vector<float> ysum(N), ap(N), out(N);
  std::vector<thrust::device_vector<float>*> ycomb(NCOMB), Lbuf(NCOMB);
  for (int k = 0; k < NCOMB; ++k) {
    ycomb[k] = new thrust::device_vector<float>(N);
    Lbuf[k]  = new thrust::device_vector<float>(N);
  }
  int maxD = 0;
  for (int k = 0; k < NCOMB; ++k) maxD = std::max(maxD, (int)(SR * COMB_DELAY_S[k]));
  thrust::device_vector<float> w(maxD);
  thrust::device_vector<Affine> ef(maxD);
  thrust::device_vector<AffineD> ed(maxD);

  cudaDeviceSynchronize();
  auto t0 = clk::now();
  const int TB = 256;
  for (int k = 0; k < NCOMB; ++k) {
    int D = (int)(SR * COMB_DELAY_S[k]);
    float g = COMB_G[k];
    float a = 0.05f + HILOW_SPREAD * (0.95f - COMB_G[k]);
    float carryF = 0.0f; double carryD = 0.0;
    for (long n0 = 0; n0 < N; n0 += D) {
      long len = std::min((long)D, N - n0);
      long gb = (len + TB - 1) / TB;
      combW<<<gb, TB>>>(thrust::raw_pointer_cast(x.data()),
                        thrust::raw_pointer_cast(Lbuf[k]->data()),
                        thrust::raw_pointer_cast(w.data()),
                        thrust::raw_pointer_cast(ycomb[k]->data()),
                        n0, len, N, D, g);
      if (!scanDouble) {
        packAffineF<<<gb, TB>>>(thrust::raw_pointer_cast(w.data()),
                                thrust::raw_pointer_cast(ef.data()), len, a);
        thrust::inclusive_scan(thrust::device, ef.begin(), ef.begin() + len,
                               ef.begin(), ComposeF());
        unpackLF<<<gb, TB>>>(thrust::raw_pointer_cast(ef.data()),
                             thrust::raw_pointer_cast(Lbuf[k]->data()),
                             n0, len, carryF);
        Affine last = ef[len - 1];
        carryF = last.A * carryF + last.B;
      } else {
        packAffineD<<<gb, TB>>>(thrust::raw_pointer_cast(w.data()),
                                thrust::raw_pointer_cast(ed.data()), len, (double)a);
        thrust::inclusive_scan(thrust::device, ed.begin(), ed.begin() + len,
                               ed.begin(), ComposeD());
        unpackLD<<<gb, TB>>>(thrust::raw_pointer_cast(ed.data()),
                             thrust::raw_pointer_cast(Lbuf[k]->data()),
                             n0, len, carryD);
        AffineD last = ed[len - 1];
        carryD = last.A * carryD + last.B;
      }
    }
  }
  long gbN = (N + TB - 1) / TB;
  sumMix<<<gbN, TB>>>(thrust::raw_pointer_cast(ycomb[0]->data()),
                      thrust::raw_pointer_cast(ycomb[1]->data()),
                      thrust::raw_pointer_cast(ycomb[2]->data()),
                      thrust::raw_pointer_cast(ycomb[3]->data()),
                      thrust::raw_pointer_cast(ycomb[4]->data()),
                      thrust::raw_pointer_cast(ycomb[5]->data()),
                      thrust::raw_pointer_cast(x.data()),
                      thrust::raw_pointer_cast(ysum.data()), N, MIX);
  int Dap = (int)((double)AP_DELAY_S * SR);
  long gbA = (Dap + TB - 1) / TB;
  allpassKernel<<<gbA, TB>>>(thrust::raw_pointer_cast(ysum.data()),
                             thrust::raw_pointer_cast(ap.data()), N, Dap, AP_GAIN);
  mixDry<<<gbN, TB>>>(thrust::raw_pointer_cast(ap.data()),
                      thrust::raw_pointer_cast(x.data()),
                      thrust::raw_pointer_cast(out.data()), N, MIX);
  cudaDeviceSynchronize();
  *wallMs = msec(t0, clk::now());

  thrust::copy(out.begin(), out.end(), outh.begin());
  for (int k = 0; k < NCOMB; ++k) { delete ycomb[k]; delete Lbuf[k]; }
}

// ---------------------------------------------------------------------------
// Error metrics vs a float reference, reported in 24-bit LSB and dBFS.
static void reportDiff(const char* tag, const std::vector<float>& ref,
                       const std::vector<float>& test) {
  long N = (long)ref.size();
  double maxd = 0, sq = 0;
  long   nDiff = 0;
  for (long i = 0; i < N; ++i) {
    double d = fabs((double)ref[i] - (double)test[i]);
    if (d > 0) nDiff++;
    if (d > maxd) maxd = d;
    sq += d * d;
  }
  double rms = sqrt(sq / N);
  double lsb = 8388608.0;   // 2^23, full-scale 24-bit
  printf("  %-34s max %.3g (%.2f LSB24)  RMS %.3g (%s%.1f dBFS)  diff %ld/%ld\n",
         tag, maxd, maxd * lsb, rms, rms > 0 ? "" : "<",
         rms > 0 ? 20.0 * log10(rms) : -999.0, nDiff, N);
}

// ---------------------------------------------------------------------------
// T2: phase accumulator.
__global__ void wrapSin(const double* prefix, float* out, long N, float amp) {
  long n = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  double p = prefix[n];
  p = p - floor(p);                       // wrap to [0,1)
  out[n] = amp * sinf((float)(2.0 * M_PI * p));
}

int main(int argc, char** argv) {
  double seconds = (argc > 1) ? atof(argv[1]) : 4.0;
  int    reps    = (argc > 2) ? atoi(argv[2]) : 3;
  long   N       = (long)(seconds * SR);
  printf("gpu-fast spike: N=%ld (%.1fs @ %u Hz), reps=%d\n\n", N, seconds, SR, reps);

  // realistic input: 3 sines + decay, amp ~0.3 (typical rendered partial sum)
  std::vector<float> x(N);
  for (long n = 0; n < N; ++n) {
    double t = (double)n / SR;
    x[n] = (float)(0.3 * exp(-t / 20.0) *
                   (sin(2 * M_PI * 220.0 * t) + 0.5 * sin(2 * M_PI * 553.0 * t) +
                    0.25 * sin(2 * M_PI * 1319.0 * t)));
  }

  // ---------------- T1: reverb unit ----------------
  printf("T1  reverb unit (6 LPComb + AllPass, REV_Medium params)\n");
  std::vector<float> refOut(N), gpuF(N), gpuD(N);
  double cpuMs = 1e30, gf = 1e30, gd = 1e30, w;
  for (int r = 0; r < reps; ++r) {
    auto t0 = clk::now();
    cpuReverb(x, refOut);
    cpuMs = std::min(cpuMs, msec(t0, clk::now()));
  }
  gpuReverb(x, gpuF, false, &w);                     // warmup incl. context
  for (int r = 0; r < reps; ++r) { gpuReverb(x, gpuF, false, &w); gf = std::min(gf, w); }
  for (int r = 0; r < reps; ++r) { gpuReverb(x, gpuD, true,  &w); gd = std::min(gd, w); }

  printf("  CPU sequential (1 core): %8.2f ms  (%.1f Msmpl/s)\n", cpuMs, N / cpuMs / 1e3);
  printf("  GPU float-scan         : %8.2f ms  (%.1f Msmpl/s)  speedup %.2fx\n",
         gf, N / gf / 1e3, cpuMs / gf);
  printf("  GPU double-scan        : %8.2f ms  (%.1f Msmpl/s)  speedup %.2fx\n",
         gd, N / gd / 1e3, cpuMs / gd);
  reportDiff("float-scan vs CPU float", refOut, gpuF);
  reportDiff("double-scan vs CPU float", refOut, gpuD);

  // ---------------- T2: phase accumulator ----------------
  printf("\nT2  phase accumulator + sine (f=440 Hz, A=0.5)\n");
  float fInc = 440.0f / (float)SR;                    // CPU adds this float
  float A = 0.5f;
  std::vector<float> sCpuF(N), sCpuD(N), sGpu(N);
  {                                                    // CPU float chain (verbatim)
    auto t0 = clk::now();
    float p = 0.0f;
    for (long n = 0; n < N; ++n) {
      p = p + fInc;
      while (p > 1.0f) p -= 1.0f;                      // Partial::pmod
      sCpuF[n] = A * (float)sin(2.0 * M_PI * (double)p);
    }
    printf("  CPU float chain        : %8.2f ms\n", msec(t0, clk::now()));
  }
  {                                                    // CPU double chain (context)
    double p = 0.0;
    for (long n = 0; n < N; ++n) {
      p += (double)fInc;
      while (p > 1.0) p -= 1.0;
      sCpuD[n] = A * (float)sin(2.0 * M_PI * p);
    }
  }
  {                                                    // GPU double prefix + sinf
    thrust::device_vector<double> inc(N, (double)fInc), pre(N);
    thrust::device_vector<float> outd(N);
    cudaDeviceSynchronize();
    auto t0 = clk::now();
    thrust::inclusive_scan(thrust::device, inc.begin(), inc.end(), pre.begin());
    long TB = 256, gb = (N + TB - 1) / TB;
    wrapSin<<<(unsigned)gb, (unsigned)TB>>>(thrust::raw_pointer_cast(pre.data()),
                                            thrust::raw_pointer_cast(outd.data()), N, A);
    cudaDeviceSynchronize();
    printf("  GPU double-scan + sinf : %8.2f ms\n", msec(t0, clk::now()));
    thrust::copy(outd.begin(), outd.end(), sGpu.begin());
  }
  reportDiff("GPU(double+sinf) vs CPU float", sCpuF, sGpu);
  reportDiff("CPU double vs CPU float chain", sCpuF, sCpuD);
  printf("  (if the two lines above are the same magnitude, the 'error' is the\n"
         "   CPU float chain's own accumulated noise, not a GPU defect)\n");
  return 0;
}
