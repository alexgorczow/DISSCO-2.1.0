/*
  bench_batch.cu — gpu-fast step 2: BATCHED single-kernel reverb scan.

  The spike (bench_scan.cu) proved the accuracy gate (-165 dBFS, length-
  independent) but its naive per-block thrust implementation was launch/sync
  bound (0.04x). This prototype measures the real architecture:

    - one CUDA block per (sound, comb): the D-sample block-sequential loop runs
      INSIDE the kernel (__syncthreads between D-blocks, no host round trips);
      the previous D-block's lowpass state L lives in shared memory
      (max D = 3439 floats = 13.8 KB <= 48 KB).
    - affine chunk-scan: within a D-block, threads scan L[n] = a*L[n-1] + w[n]
      in blockDim-sized chunks (Hillis-Steele on (A,B) pairs in shared memory,
      carry across chunks). Reassociation error class == spike's (measured).
    - allpass: one thread per (sound, residue class) — exact CPU float op order.
    - the whole batch of sounds runs in ONE kernel launch per stage.

  Compares against the exact CPU float reference (same math as bench_scan.cu)
  on a batch of nSounds, reporting error + throughput incl/excl PCIe transfer.

  Build: nvcc -O3 -std=c++17 bench_batch.cu -o bench_batch
  Run:   ./bench_batch [nSounds=64] [seconds=4] [reps=3]
*/
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <chrono>
#include <algorithm>

using clk = std::chrono::high_resolution_clock;
static double msec(clk::time_point a, clk::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}
#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

static const int   NCOMB = 6;
static const float COMB_G[NCOMB] = {0.46f, 0.48f, 0.50f, 0.52f, 0.53f, 0.55f};
static const float COMB_DELAY_S[NCOMB] = {0.050f, 0.056f, 0.061f, 0.068f, 0.072f, 0.078f};
static const float HILOW_SPREAD = 0.5f;
static const float AP_GAIN = 0.1f;
static const float AP_DELAY_S = 0.5f;
static const float MIX = 0.5f;
static const unsigned SR = 44100;

// ---------------------------------------------------------------------------
// CPU reference (identical to bench_scan.cu; verbatim LASS float math).
struct CpuComb {
  float g, a; int D; std::vector<float> s; int idx; float L;
  CpuComb(float g_, float a_, int D_) : g(g_), a(a_), D(D_), s(D_, 0.0f), idx(0), L(0.0f) {}
  inline float step(float x) {
    float y = s[idx];
    L = y + (a * L);
    s[idx] = x + (g * L);
    idx = (idx + 1) % D;
    return y;
  }
};
struct CpuAllPass {
  float g, g2; int D; std::vector<float> xh, yh; int idx;
  CpuAllPass(float g_, int D_) : g(g_), g2(g_*g_), D(D_), xh(D_,0.0f), yh(D_,0.0f), idx(0) {}
  inline float step(float x) {
    float y = -g * x + (1.0f - g2) * (xh[idx] + g * yh[idx]);
    xh[idx] = x; yh[idx] = y;
    idx = (idx + 1) % D;
    return y;
  }
};
static void cpuReverbOne(const float* x, float* out, long N) {
  std::vector<CpuComb> combs;
  for (int k = 0; k < NCOMB; ++k) {
    float lp = 0.05f + HILOW_SPREAD * (0.95f - COMB_G[k]);
    combs.emplace_back(COMB_G[k], lp, (int)(SR * COMB_DELAY_S[k]));
  }
  CpuAllPass ap(AP_GAIN, (int)((double)AP_DELAY_S * SR));
  for (long n = 0; n < N; ++n) {
    float y = combs[0].step(x[n]);
    y += combs[1].step(x[n]); y += combs[2].step(x[n]);
    y += combs[3].step(x[n]); y += combs[4].step(x[n]);
    y += combs[5].step(x[n]);
    y /= (float)NCOMB;
    y = ap.step(y);
    out[n] = (MIX * y) + ((1.0f - MIX) * x[n]);
  }
}

// ---------------------------------------------------------------------------
// GPU batched comb kernel: one block per (sound, comb).
// Shared memory layout: Lprev[D] (previous D-block's L), pairs A[]/B[] for the
// chunk scan (blockDim each), plus chunk carry.
__global__ void combBatchKernel(const float* __restrict__ x,   // [nSounds*N]
                                float* __restrict__ ycomb,     // [nSounds*NCOMB*N]
                                long N, const int* Ds, const float* gs,
                                const float* as, int maxD) {
  int sound = blockIdx.x / NCOMB;
  int comb  = blockIdx.x % NCOMB;
  int D = Ds[comb];
  float g = gs[comb], a = as[comb];
  const float* xs = x + (long)sound * N;
  float* ys = ycomb + ((long)sound * NCOMB + comb) * N;

  extern __shared__ float sh[];
  float* Lprev = sh;                 // [maxD] — L of the previous D-block
  float* Lcur  = sh + maxD;          // [maxD] — L being built for this D-block
  float* SA    = sh + 2 * maxD;                  // [blockDim] scan pairs
  float* SB    = sh + 2 * maxD + blockDim.x;     // [blockDim]

  // zero-init "previous" L (history starts at silence)
  for (int i = threadIdx.x; i < D; i += blockDim.x) Lprev[i] = 0.0f;
  __syncthreads();

  float carry = 0.0f;                // L at the end of the previous chunk
  for (long n0 = 0; n0 < N; n0 += D) {
    int len = (int)min((long)D, N - n0);
    // chunked affine scan over this D-block
    for (int c0 = 0; c0 < len; c0 += blockDim.x) {
      int clen = min((int)blockDim.x, len - c0);
      int i = threadIdx.x;
      float w = 0.0f;
      if (i < clen) {
        long n = n0 + c0 + i;
        long m = n - D;
        // w[n] = x[n-D] + g*L[n-D]; L[n-D] is in Lprev (same offset in-block)
        w = (m >= 0) ? (xs[m] + g * Lprev[c0 + i]) : 0.0f;
        ys[n] = w;                   // comb output y[n] == w[n]
        SA[i] = a; SB[i] = w;
      }
      __syncthreads();
      // Hillis-Steele inclusive scan on affine pairs (apply left-to-right)
      for (int off = 1; off < clen; off <<= 1) {
        float pA = 0.f, pB = 0.f; bool has = false;
        if (i < clen && i >= off) { pA = SA[i - off]; pB = SB[i - off]; has = true; }
        __syncthreads();
        if (has) { SB[i] = SA[i] * pB + SB[i]; SA[i] = SA[i] * pA; }
        __syncthreads();
      }
      if (i < clen) Lcur[c0 + i] = SA[i] * carry + SB[i];
      __syncthreads();
      if (i == 0) {
        int last = clen - 1;
        carry = SA[last] * carry + SB[last];
        SB[0] = carry;               // broadcast slot
      }
      __syncthreads();
      carry = SB[0];
      __syncthreads();
    }
    // Lcur becomes Lprev for the next D-block
    for (int i = threadIdx.x; i < len; i += blockDim.x) Lprev[i] = Lcur[i];
    __syncthreads();
  }
}

// sum 6 comb outputs (CPU accumulation order) then /6
__global__ void sumKernel(const float* __restrict__ ycomb, float* __restrict__ ysum,
                          long N, int nSounds) {
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long total = (long)nSounds * N;
  if (idx >= total) return;
  long sound = idx / N, n = idx % N;
  const float* base = ycomb + (long)sound * NCOMB * N;
  float y = base[n];
  y += base[N + n]; y += base[2*N + n]; y += base[3*N + n];
  y += base[4*N + n]; y += base[5*N + n];
  ysum[idx] = y / (float)NCOMB;
}

// allpass: one thread per (sound, residue class); exact CPU float op order.
__global__ void allpassBatchKernel(const float* __restrict__ in, float* __restrict__ out,
                                   long N, int nSounds, int D, float g) {
  long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long total = (long)nSounds * D;
  if (t >= total) return;
  long sound = t / D, c = t % D;
  const float* xs = in + (long)sound * N;
  float* ys = out + (long)sound * N;
  float g2 = g * g, xprev = 0.0f, yprev = 0.0f;
  for (long n = c; n < N; n += D) {
    float y = -g * xs[n] + (1.0f - g2) * (xprev + g * yprev);
    xprev = xs[n]; yprev = y;
    ys[n] = y;
  }
}
__global__ void mixKernel(const float* __restrict__ y, const float* __restrict__ x,
                          float* __restrict__ out, long total, float mix) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < total) out[i] = (mix * y[i]) + ((1.0f - mix) * x[i]);
}

int main(int argc, char** argv) {
  int    nSounds = (argc > 1) ? atoi(argv[1]) : 64;
  double seconds = (argc > 2) ? atof(argv[2]) : 4.0;
  int    reps    = (argc > 3) ? atoi(argv[3]) : 3;
  long   N       = (long)(seconds * SR);
  long   total   = (long)nSounds * N;
  printf("batched reverb scan: %d sounds x %.1fs (%ld samples total)\n\n",
         nSounds, seconds, total);

  // batch of distinct inputs
  std::vector<float> x(total);
  for (int s = 0; s < nSounds; ++s)
    for (long n = 0; n < N; ++n) {
      double t = (double)n / SR;
      x[(long)s*N + n] = (float)(0.3 * exp(-t/20.0) *
        (sin(2*M_PI*(180.0 + 7.0*s)*t) + 0.5*sin(2*M_PI*(490.0 + 11.0*s)*t)));
    }

  // ---- CPU reference (1 core), timed on the whole batch ----
  std::vector<float> ref(total);
  double cpuMs = 1e30;
  for (int r = 0; r < std::max(1, reps/2); ++r) {
    auto t0 = clk::now();
    for (int s = 0; s < nSounds; ++s)
      cpuReverbOne(&x[(long)s*N], &ref[(long)s*N], N);
    cpuMs = std::min(cpuMs, msec(t0, clk::now()));
  }
  printf("CPU 1-core (whole batch): %9.1f ms  (%.1f Msmpl/s)\n",
         cpuMs, total / cpuMs / 1e3);

  // ---- GPU ----
  int Ds[NCOMB]; float as[NCOMB];
  int maxD = 0;
  for (int k = 0; k < NCOMB; ++k) {
    Ds[k] = (int)(SR * COMB_DELAY_S[k]);
    as[k] = 0.05f + HILOW_SPREAD * (0.95f - COMB_G[k]);
    maxD = std::max(maxD, Ds[k]);
  }
  int *dDs; float *dGs, *dAs;
  CK(cudaMalloc(&dDs, sizeof(Ds)));  CK(cudaMemcpy(dDs, Ds, sizeof(Ds), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&dGs, sizeof(COMB_G))); CK(cudaMemcpy(dGs, COMB_G, sizeof(COMB_G), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&dAs, sizeof(as)));  CK(cudaMemcpy(dAs, as, sizeof(as), cudaMemcpyHostToDevice));

  float *dx, *dycomb, *dysum, *dap, *dout;
  CK(cudaMalloc(&dx,     total * sizeof(float)));
  CK(cudaMalloc(&dycomb, (long)nSounds * NCOMB * N * sizeof(float)));
  CK(cudaMalloc(&dysum,  total * sizeof(float)));
  CK(cudaMalloc(&dap,    total * sizeof(float)));
  CK(cudaMalloc(&dout,   total * sizeof(float)));

  const int TB = 256;
  size_t shBytes = (2 * (size_t)maxD + 2 * TB) * sizeof(float);
  CK(cudaFuncSetAttribute(combBatchKernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                          (int)shBytes));
  int Dap = (int)((double)AP_DELAY_S * SR);

  std::vector<float> gpuOut(total);

  auto runGpu = [&](double* kernelMs, double* e2eMs) {
    auto tAll = clk::now();
    CK(cudaMemcpy(dx, x.data(), total * sizeof(float), cudaMemcpyHostToDevice));
    auto tK = clk::now();
    combBatchKernel<<<nSounds * NCOMB, TB, shBytes>>>(dx, dycomb, N, dDs, dGs, dAs, maxD);
    CK(cudaGetLastError());
    long gbT = (total + TB - 1) / TB;
    sumKernel<<<(unsigned)gbT, TB>>>(dycomb, dysum, N, nSounds);
    long apThreads = (long)nSounds * Dap;
    allpassBatchKernel<<<(unsigned)((apThreads + TB - 1) / TB), TB>>>(dysum, dap, N, nSounds, Dap, AP_GAIN);
    mixKernel<<<(unsigned)gbT, TB>>>(dap, dx, dout, total, MIX);
    CK(cudaDeviceSynchronize());
    *kernelMs = msec(tK, clk::now());
    CK(cudaMemcpy(gpuOut.data(), dout, total * sizeof(float), cudaMemcpyDeviceToHost));
    *e2eMs = msec(tAll, clk::now());
  };

  double kMs = 1e30, eMs = 1e30, k1, e1;
  runGpu(&k1, &e1);                          // warmup (context)
  for (int r = 0; r < reps; ++r) { runGpu(&k1, &e1); kMs = std::min(kMs,k1); eMs = std::min(eMs,e1); }

  printf("GPU kernels only        : %9.1f ms  (%.1f Msmpl/s)  %6.1fx vs 1-core\n",
         kMs, total / kMs / 1e3, cpuMs / kMs);
  printf("GPU end-to-end (H2D+D2H): %9.1f ms  (%.1f Msmpl/s)  %6.1fx vs 1-core\n",
         eMs, total / eMs / 1e3, cpuMs / eMs);

  // error vs CPU float reference
  double maxd = 0, sq = 0;
  for (long i = 0; i < total; ++i) {
    double d = fabs((double)ref[i] - (double)gpuOut[i]);
    if (d > maxd) maxd = d; sq += d * d;
  }
  double rms = sqrt(sq / total);
  printf("error vs CPU float      : max %.3g (%.2f LSB24)  RMS %.3g (%.1f dBFS)\n",
         maxd, maxd * 8388608.0, rms, rms > 0 ? 20.0 * log10(rms) : -999.0);
  return 0;
}
