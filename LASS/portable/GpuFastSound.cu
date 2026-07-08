/*
  LASS/portable/GpuFastSound.cu — gpu-fast v1 fused device loudness + synth.
  See GpuFastSound.h for the contract and restructure/07_GPU_FAST_SPIKE.md for
  the measured basis. Compiled by nvcc with -DHAVE_CUDA (premake).

  Semantics mirrored (budget-relaxed, float transcendentals per contract):
    - Loudness::calculate 24-band model INCLUDING the reference quirk where the
      "find max gamma" loop writes the running max into bandGamma[0] while
      maxGamma stays 0 (Loudness.cpp: bandGamma[(int)maxGamma] = ...).
    - Partial::render / PortableSynth pre-pass: tremolo & vibrato use the phase
      EXCLUSIVE of the current rate sample; the carrier phase is INCLUSIVE, and
      phases are wrapped to [0,1) (CPU pmod leaves (0,1]; sin is 1-periodic so
      only exact-integer phases differ — inside budget).
    - Placeholder partial spatialization: every channel = partial/nCh, summed
      over partials in index order (deterministic adds).
  Host keeps: dynamic-variable iteration (RLE-compressed streams), and the
  whole filter/reverb/Pan/composite tail — unchanged code paths.
*/
#include "GpuFastSound.h"
#include "PortableSynth.h"       // canRenderPortably

#include "../src/Sound.h"
#include "../src/Partial.h"
#include "../src/Track.h"
#include "../src/MultiTrack.h"
#include "../src/SoundSample.h"
#include "../src/DynamicVariable.h"
#include "../src/Constant.h"
#include "../src/Envelope.h"
#include "../src/Iterator.h"
#include "../src/Loudness.h"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/execution_policy.h>

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <mutex>
#include "../../restructure/profiling/StageProfiler.h"

namespace {

inline bool ck(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    fprintf(stderr, "gpu-fast CUDA error (%s): %s -- falling back to CPU\n",
            what, cudaGetErrorString(e));
    return false;
  }
  return true;
}
#define CKF(x, w) do { if (!ck((x), (w))) return nullptr; } while (0)

// ---------------------------------------------------------------------------
// Device-side critical-band constants (precomputed from Loudness::BANDS).
__constant__ float cLB[24], cUB[24], cE1[24], cE2[24], cFF[24], cOFF[24], cSLOPE[24];

// The 8 per-partial parameter streams, RLE-expanded to dense [P][N].
enum Param { P_WAVESHAPE = 0, P_TREMAMP, P_TREMRATE, P_VIBAMP, P_VIBRATE,
             P_PHASEOFF, P_FREQBASE, P_FREQLOUD, NPARAM };

struct Run { float v; int len; };

// key iterator for scan_by_key: element i belongs to partial i/N
struct KeyOfIndex {
  long N;
  __host__ __device__ long operator()(long i) const { return i / N; }
};

// expand runs -> dense: one thread per sample, binary search its run.
__global__ void expandKernel(const float* __restrict__ vals,
                             const int* __restrict__ starts,  // run start sample
                             int nRuns, float* __restrict__ dense, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int lo = 0, hi = nRuns - 1;
  while (lo < hi) {                       // last start <= i
    int mid = (lo + hi + 1) >> 1;
    if (starts[mid] <= i) lo = mid; else hi = mid - 1;
  }
  dense[i] = vals[lo];
}

// One envelope segment for on-device evaluation (see Envelope::DeviceSegment).
struct GfSeg { long start; long steps; int type; float vFrom, vTo; };

// evaluate an envelope stream from its segment table: one thread per sample,
// binary search the segment, closed-form value. Past the last emitted sample
// the value HOLDS (matching EnvelopeIterator's past-end behavior).
__global__ void evalSegKernel(const GfSeg* __restrict__ segs, int nSegs,
                              float* __restrict__ dense, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const GfSeg& last = segs[nSegs - 1];
  long total = last.start + last.steps;
  long s = (i < total) ? i : (total - 1);          // hold last value past end
  int lo = 0, hi = nSegs - 1;
  while (lo < hi) {
    int mid = (lo + hi + 1) >> 1;
    if (segs[mid].start <= s) lo = mid; else hi = mid - 1;
  }
  const GfSeg& g = segs[lo];
  long j = s - g.start;
  float v;
  if (g.type == 1) {                                // EXPONENTIAL, j in [1,steps]
    float y1 = (g.vFrom == 0.0f) ? 0.0001f : g.vFrom;
    float y2 = (g.vTo   == 0.0f) ? 0.0001f : g.vTo;
    float alpha = (y1 > y2) ? -3.0f : 3.0f;
    float I = (float)(j + 1) / (float)g.steps;
    v = y1 + (y2 - y1) * ((1.0f - powf(2.718282f, I * alpha)) /
                          (1.0f - powf(2.718282f, alpha)));
  } else {                                          // LINEAR, j in [0,steps)
    v = g.vFrom + (float)j * ((g.vTo - g.vFrom) / (float)g.steps);
  }
  dense[i] = v;
}

// Loudness model: one thread per sample; loops partials (<=64) and 24 bands.
__global__ void loudnessKernel(const float* __restrict__ freqLoud,  // [P][N]
                               const float* __restrict__ maxWS,     // [P]
                               const float* __restrict__ relAmp,    // [P]
                               float maxAmp, float loudParam,
                               float* __restrict__ loudScalar,      // [P][N]
                               int P, long N) {
  long s = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= N) return;

  int   band[64];
  float bandSum[24];
#pragma unroll
  for (int b = 0; b < 24; ++b) bandSum[b] = 0.0f;

  for (int p = 0; p < P; ++p) {
    float f = freqLoud[(long)p * N + s];
    int b;
    if (f < cLB[0]) b = 0;
    else if (f > cUB[23]) b = 23;
    else { b = 23; for (int i = 0; i < 24; ++i) if (f < cUB[i]) { b = i; break; } }
    band[p] = b;
    bandSum[b] += powf(maxWS[p] / maxAmp, cE1[b]);
  }

  float gamma[24];
#pragma unroll
  for (int b = 0; b < 24; ++b) gamma[b] = powf(bandSum[b], cE2[b]);

  // Reference quirk: the max is written INTO gamma[0]; "maxGamma" stays 0.
  for (int i = 0; i < 24; ++i) if (gamma[i] > gamma[0]) gamma[0] = gamma[i];
  float gammaTotal = 0.0f;
  for (int i = 1; i < 24; ++i) gammaTotal += gamma[i] * cFF[i];
  float numerator = loudParam / (gamma[0] + gammaTotal);

  for (int p = 0; p < P; ++p) {
    int b = band[p];
    float Ls = (maxWS[p] / maxAmp) * numerator;
    float Lp = logf(Ls) / logf(2.0f) * 10.0f + 40.0f;
    float L  = cOFF[b] + cSLOPE[b] * Lp;
    float A  = powf(10.0f, -((120.0f - L) / 20.0f));
    loudScalar[(long)p * N + s] = (A / maxWS[p]) * relAmp[p];
  }
}

__global__ void buildIncKernel(const float* __restrict__ rate, double* __restrict__ inc,
                               long total, float sr) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < total) inc[i] = (double)(rate[i] / sr);
}
// frequency[n] = freqBase * (1 + vibAmp*sinf(2pi*wrap(vibPhaseExcl)))
__global__ void freqKernel(const float* __restrict__ freqBase,
                           const float* __restrict__ vibAmp,
                           const double* __restrict__ vibExcl,
                           double* __restrict__ freqInc, long total, float sr) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  double vp = vibExcl[i]; vp -= floor(vp);
  float vib = vibAmp[i] * sinf((float)(2.0 * M_PI * vp));
  float f = freqBase[i] * (1.0f + vib);
  freqInc[i] = (double)(f / sr);
}
// wave/amp: amp = loud*ws*(1+tremAmp*sinf(2pi*wrap(tremExcl)));
//           wave = amp*sinf(2pi*(wrap(freqIncl)+phaseOff))
__global__ void synthKernel(const float* __restrict__ loud,
                            const float* __restrict__ ws,
                            const float* __restrict__ tremAmp,
                            const double* __restrict__ tremExcl,
                            const double* __restrict__ freqIncl,
                            const float* __restrict__ phaseOff,
                            float* __restrict__ wave, float* __restrict__ amp,
                            long total) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  double tp = tremExcl[i]; tp -= floor(tp);
  float trem = tremAmp[i] * sinf((float)(2.0 * M_PI * tp));
  float a = loud[i] * ws[i] * (1.0f + trem);
  double fp = freqIncl[i]; fp -= floor(fp);
  float phase = (float)fp + phaseOff[i];
  wave[i] = a * sinf((float)(2.0 * M_PI * (double)phase));
  amp[i] = a;
}
// deterministic in-order partial sum with placeholder 1/nCh scaling
__global__ void sumKernel(const float* __restrict__ wave, const float* __restrict__ amp,
                          float* __restrict__ monoW, float* __restrict__ monoA,
                          int P, long N, long sampleCount, float invCh) {
  long s = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= sampleCount) return;
  float w = 0.0f, a = 0.0f;
  if (s < N)
    for (int p = 0; p < P; ++p) {
      w += wave[(long)p * N + s] * invCh;
      a += amp[(long)p * N + s] * invCh;
    }
  monoW[s] = w; monoA[s] = a;
}

// ---------------------------------------------------------------------------
// Persistent device arena (grown on demand); one sound in flight at a time.
struct Arena {
  float*  dense = nullptr;   long denseCap = 0;   // NPARAM+3 dense float arrays
  double* scans = nullptr;   long scansCap = 0;   // 3 double arrays
  float*  runsV = nullptr;   int* runsS = nullptr; int runsCap = 0;
  GfSeg*  segs = nullptr;    int segsCap = 0;
  float*  mono = nullptr;    long monoCap = 0;
  float*  scal = nullptr;    int scalCap = 0;     // maxWS + relAmp
  bool bandsUp = false;

  bool ensure(long PN, long sampleCount, int nRuns, int P) {
    if (PN * (NPARAM + 3) > denseCap) {
      if (dense) cudaFree(dense);
      if (!ck(cudaMalloc(&dense, PN * (NPARAM + 3) * sizeof(float)), "dense")) return false;
      denseCap = PN * (NPARAM + 3);
    }
    if (PN * 3 > scansCap) {
      if (scans) cudaFree(scans);
      if (!ck(cudaMalloc(&scans, PN * 3 * sizeof(double)), "scans")) return false;
      scansCap = PN * 3;
    }
    (void)0; if (nRuns > runsCap) {
      if (runsV) cudaFree(runsV); if (runsS) cudaFree(runsS);
      if (!ck(cudaMalloc(&runsV, nRuns * sizeof(float)), "runsV")) return false;
      if (!ck(cudaMalloc(&runsS, nRuns * sizeof(int)), "runsS")) return false;
      runsCap = nRuns;
    }
    if (sampleCount * 2 > monoCap) {
      if (mono) cudaFree(mono);
      if (!ck(cudaMalloc(&mono, sampleCount * 2 * sizeof(float)), "mono")) return false;
      monoCap = sampleCount * 2;
    }
    if (P * 2 > scalCap) {
      if (scal) cudaFree(scal);
      if (!ck(cudaMalloc(&scal, P * 2 * sizeof(float)), "scal")) return false;
      scalCap = P * 2;
    }
    return true;
  }
};
Arena g_arena;
std::mutex g_mutex;

// host-side RLE of one iterated parameter stream
static void rleCompress(const std::vector<float>& src, std::vector<float>& vals,
                        std::vector<int>& starts) {
  vals.clear(); starts.clear();
  if (src.empty()) return;
  float v = src[0]; starts.push_back(0); vals.push_back(v);
  for (long i = 1; i < (long)src.size(); ++i)
    if (src[i] != v) { v = src[i]; vals.push_back(v); starts.push_back((int)i); }
}

} // namespace

namespace portable {

bool gpuFastEnabled() {
  static const char* pl = getenv("LASS_PIPELINE");
  static const bool on = pl && strcmp(pl, "gpu-fast") == 0;
  return on;
}

MultiTrack* renderSoundGpuFast(Sound& snd, int numChannels, long sampleCount,
                               float duration, unsigned int samplingRate) {
  if (!gpuFastEnabled()) return nullptr;
  // Eligibility gates (fall back to the unchanged CPU path)
  if (getenv("LASS_LOUDNESS_RATE")) return nullptr;      // preview knob owns loudness
  if (samplingRate != 44100) return nullptr;             // loudness runs at 44100
  if (snd.getParam(LOUDNESS) < 0) return nullptr;        // loudness disabled
  if (snd.getParam(DETUNE_FUNDAMENTAL) == 1.0) return nullptr;   // detune envelopes
  int P = snd.size();
  long N = (long)(duration * (m_time_type)samplingRate);
  if (P < 1 || P > 64 || N < 1 || numChannels < 1) return nullptr;
  for (int p = 0; p < P; ++p)
    if (!canRenderPortably(snd.get(p))) return nullptr;  // transients/random/reverb

  // ---- host: iterate all dynamic variables into RLE streams ----
  PROFILE_SCOPE(prof::GF_PREPASS);
  typedef Iterator<m_value_type> ValIter;
  std::vector<std::vector<float> > runsV(P * NPARAM);
  std::vector<std::vector<int> >   runsS(P * NPARAM);
  std::vector<std::vector<Envelope::DeviceSegment> > segLists(P * NPARAM);
  std::vector<float> hMaxWS(P), hRelAmp(P);
  std::vector<float> buf(N);
  float maxAmp = 0.0f;

  for (int p = 0; p < P; ++p) {
    Partial& part = snd.get(p);
    // mirror PortableSynth.cpp / Partial::render setup
    part.getParam(FREQUENCY).setDuration(duration);
    part.getParam(WAVE_SHAPE).setDuration(duration);
    part.getParam(TREMOLO_AMP).setDuration(duration);
    part.getParam(TREMOLO_RATE).setDuration(duration);
    part.getParam(VIBRATO_AMP).setDuration(duration);
    part.getParam(VIBRATO_RATE).setDuration(duration);
    part.getParam(PHASE).setDuration(duration);
    part.getParam(FREQ_ENV).setDuration(duration);
    part.getParam(DETUNING_ENV).setDuration(duration);
    part.getParam(FREQUENCY).setSamplingRate(samplingRate);
    part.getParam(WAVE_SHAPE).setSamplingRate(samplingRate);
    part.getParam(TREMOLO_AMP).setSamplingRate(samplingRate);
    part.getParam(TREMOLO_RATE).setSamplingRate(samplingRate);
    part.getParam(VIBRATO_AMP).setSamplingRate(samplingRate);
    part.getParam(VIBRATO_RATE).setSamplingRate(samplingRate);
    part.getParam(PHASE).setSamplingRate(samplingRate);
    part.getParam(FREQ_ENV).setSamplingRate(samplingRate);
    part.getParam(DETUNING_ENV).setSamplingRate(samplingRate);

    DynamicVariable* frequency_env = part.getParam(FREQUENCY).clone();
    frequency_env->setDuration(duration);
    DynamicVariable* freq_env = part.getParam(FREQ_ENV).clone();
    freq_env->setDuration(duration);
    DynamicVariable* detuning_env = part.getParam(DETUNING_ENV).clone();
    detuning_env->setDuration(duration);

    /* Constant-DV shortcut: a Constant's iterator yields getValue() N times,
       so its stream is one run -- emit it without the N-step iteration. This
       collapses most streams (tremolo/vibrato/phase/frequency are typically
       Constants); only real envelopes still iterate. Identical values, so the
       compressed stream -- and the device result -- is unchanged. */
    #define CONST_OR_ITER(DVEXPR, SLOT)                                     \
      { DynamicVariable& dv_ = (DVEXPR);                                    \
        Envelope* e_ = dynamic_cast<Envelope*>(&dv_);                       \
        if (Constant* c_ = dynamic_cast<Constant*>(&dv_)) {                 \
          runsV[p*NPARAM+SLOT].assign(1, (float)c_->getValue());            \
          runsS[p*NPARAM+SLOT].assign(1, 0);                                \
        } else if (e_ && e_->exportDeviceSegments(segLists[p*NPARAM+SLOT])) {\
          /* evaluated on-device from the segment table */                  \
        } else {                                                            \
          ValIter it_ = dv_.valueIterator();                                \
          for (long n_ = 0; n_ < N; ++n_) buf[n_] = it_.next();             \
          rleCompress(buf, runsV[p*NPARAM+SLOT], runsS[p*NPARAM+SLOT]);     \
        } }

    CONST_OR_ITER(part.getParam(WAVE_SHAPE),   P_WAVESHAPE);
    CONST_OR_ITER(part.getParam(TREMOLO_AMP),  P_TREMAMP);
    CONST_OR_ITER(part.getParam(TREMOLO_RATE), P_TREMRATE);
    CONST_OR_ITER(part.getParam(VIBRATO_AMP),  P_VIBAMP);
    CONST_OR_ITER(part.getParam(VIBRATO_RATE), P_VIBRATE);
    CONST_OR_ITER(part.getParam(PHASE),        P_PHASEOFF);
    CONST_OR_ITER(part.getParam(FREQUENCY),    P_FREQLOUD);   // loudness freq
    #undef CONST_OR_ITER

    // freqBase = frequency * freq_env * detuning: all-constant => one run
    Constant* cf = dynamic_cast<Constant*>(frequency_env);
    Constant* ce = dynamic_cast<Constant*>(freq_env);
    Constant* cd = dynamic_cast<Constant*>(detuning_env);
    if (cf && ce && cd) {
      float v = (float)(cf->getValue() * ce->getValue() * cd->getValue());
      runsV[p*NPARAM+P_FREQBASE].assign(1, v);
      runsS[p*NPARAM+P_FREQBASE].assign(1, 0);
    } else {
      ValIter fq = frequency_env->valueIterator();
      ValIter fe = freq_env->valueIterator();
      ValIter de = detuning_env->valueIterator();
      for (long n = 0; n < N; ++n) buf[n] = fq.next() * fe.next() * de.next();
      rleCompress(buf, runsV[p*NPARAM+P_FREQBASE], runsS[p*NPARAM+P_FREQBASE]);
    }
    delete frequency_env; delete freq_env; delete detuning_env;

    hMaxWS[p]  = part.getParam(WAVE_SHAPE).getMaxValue();
    hRelAmp[p] = part.getParam(RELATIVE_AMPLITUDE);
    float thisAmp = hMaxWS[p] * hRelAmp[p];
    if (thisAmp > maxAmp) maxAmp = thisAmp;
  }
  if (maxAmp <= 0.0f) return nullptr;

  // ---- device ----
  PROFILE_SCOPE(prof::GF_GPU);
  std::lock_guard<std::mutex> lock(g_mutex);
  long PN = (long)P * N;
  int maxRuns = 1;
  for (auto& v : runsV) maxRuns = std::max(maxRuns, (int)v.size());
  if (!g_arena.ensure(PN, sampleCount, maxRuns, P)) return nullptr;

  if (!g_arena.bandsUp) {
    const double (*B)[7] = Loudness::bandsTable();
    float lb[24], ub[24], e1[24], e2[24], ff[24], off[24], sl[24];
    for (int b = 0; b < 24; ++b) {
      lb[b] = (float)B[b][0]; ub[b] = (float)B[b][1];
      e1[b] = (float)(1.0 / log10(2.0) / B[b][6]);
      e2[b] = (float)(log10(2.0) / B[b][6]);
      ff[b] = (float)B[b][4]; off[b] = (float)B[b][5]; sl[b] = (float)B[b][6];
    }
    CKF(cudaMemcpyToSymbol(cLB, lb, sizeof(lb)), "sym");
    CKF(cudaMemcpyToSymbol(cUB, ub, sizeof(ub)), "sym");
    CKF(cudaMemcpyToSymbol(cE1, e1, sizeof(e1)), "sym");
    CKF(cudaMemcpyToSymbol(cE2, e2, sizeof(e2)), "sym");
    CKF(cudaMemcpyToSymbol(cFF, ff, sizeof(ff)), "sym");
    CKF(cudaMemcpyToSymbol(cOFF, off, sizeof(off)), "sym");
    CKF(cudaMemcpyToSymbol(cSLOPE, sl, sizeof(sl)), "sym");
    g_arena.bandsUp = true;
  }

  const int TB = 256;
  long gbN = (N + TB - 1) / TB;
  float* d = g_arena.dense;                    // [NPARAM dense] + loud + wave + amp
  float* dLoud = d + PN * NPARAM;
  float* dWave = d + PN * (NPARAM + 1);
  float* dAmp  = d + PN * (NPARAM + 2);
  double* dS1 = g_arena.scans;                 // trem excl
  double* dS2 = g_arena.scans + PN;            // vib excl -> freq incl
  double* dS3 = g_arena.scans + 2 * PN;        // scratch increments

  // expand all streams — dense layout [param][partial][N]: param k's block is
  // d + k*PN, partial p at offset p*N inside it. Elementwise kernels then index
  // every param with the same i in [0, PN); loudness gets a contiguous [P][N].
  // All P*NPARAM run streams are packed into ONE staging upload (two memcpys
  // total) instead of two tiny synchronous copies per stream — the per-sound
  // GPU section is inside a global mutex, so its latency is what serializes
  // worker threads at high thread counts.
  {
    std::vector<float> packV; std::vector<int> packS; std::vector<int> off(P*NPARAM+1, 0);
    for (int k = 0; k < NPARAM; ++k)
      for (int p = 0; p < P; ++p) {
        int idx = k*P + p;
        auto& vv = runsV[p*NPARAM+k]; auto& ss = runsS[p*NPARAM+k];
        off[idx+1] = off[idx] + (int)vv.size();
        packV.insert(packV.end(), vv.begin(), vv.end());
        packS.insert(packS.end(), ss.begin(), ss.end());
      }
    int totRuns = off[P*NPARAM];
    static const bool dbg = getenv("LASS_GPUFAST_DEBUG") != nullptr;
    static bool dbgOnce = false;
    if (dbg && !dbgOnce) {
      dbgOnce = true;
      static const char* pn[NPARAM] = {"waveShape","tremAmp","tremRate","vibAmp",
                                       "vibRate","phaseOff","freqBase","freqLoud"};
      fprintf(stderr, "gpu-fast[dbg] P=%d N=%ld totRuns=%d\n", P, N, totRuns);
      for (int k = 0; k < NPARAM; ++k)
        fprintf(stderr, "  %-10s p0_runs=%d\n", pn[k], off[k*P+1]-off[k*P]);
    }
    if (totRuns > g_arena.runsCap) {
      if (g_arena.runsV) cudaFree(g_arena.runsV);
      if (g_arena.runsS) cudaFree(g_arena.runsS);
      CKF(cudaMalloc(&g_arena.runsV, totRuns*sizeof(float)), "runsV");
      CKF(cudaMalloc(&g_arena.runsS, totRuns*sizeof(int)), "runsS");
      g_arena.runsCap = totRuns;
    }
    CKF(cudaMemcpy(g_arena.runsV, packV.data(), totRuns*sizeof(float), cudaMemcpyHostToDevice), "runsV");
    CKF(cudaMemcpy(g_arena.runsS, packS.data(), totRuns*sizeof(int), cudaMemcpyHostToDevice), "runsS");

    // pack envelope segment tables (device-evaluated streams)
    std::vector<GfSeg> hSegs; std::vector<int> soff(P*NPARAM+1, 0);
    for (int k = 0; k < NPARAM; ++k)
      for (int p = 0; p < P; ++p) {
        int idx = k*P + p;
        auto& sl = segLists[p*NPARAM+k];
        soff[idx+1] = soff[idx] + (int)sl.size();
        long start = 0;
        for (auto& es : sl) {
          GfSeg g;
          g.start = start;
          if (es.steps <= 0) { g.steps = 1; g.type = 0; g.vFrom = es.vFrom; g.vTo = es.vFrom; }
          else { g.steps = es.steps; g.type = es.type; g.vFrom = es.vFrom; g.vTo = es.vTo; }
          start += g.steps;
          hSegs.push_back(g);
        }
      }
    if (!hSegs.empty()) {
      if ((int)hSegs.size() > g_arena.segsCap) {
        if (g_arena.segs) cudaFree(g_arena.segs);
        CKF(cudaMalloc(&g_arena.segs, hSegs.size()*sizeof(GfSeg)), "segs");
        g_arena.segsCap = (int)hSegs.size();
      }
      CKF(cudaMemcpy(g_arena.segs, hSegs.data(), hSegs.size()*sizeof(GfSeg),
                     cudaMemcpyHostToDevice), "segs");
    }

    for (int k = 0; k < NPARAM; ++k)
      for (int p = 0; p < P; ++p) {
        int idx = k*P + p;
        int nSeg = soff[idx+1] - soff[idx];
        if (nSeg > 0)
          evalSegKernel<<<(unsigned)gbN, TB>>>(g_arena.segs + soff[idx], nSeg,
                                               d + (long)k*PN + (long)p*N, N);
        else
          expandKernel<<<(unsigned)gbN, TB>>>(g_arena.runsV + off[idx], g_arena.runsS + off[idx],
                                              off[idx+1]-off[idx], d + (long)k*PN + (long)p*N, N);
      }
  }
  CKF(cudaMemcpy(g_arena.scal, hMaxWS.data(), P*sizeof(float), cudaMemcpyHostToDevice), "scal");
  CKF(cudaMemcpy(g_arena.scal + P, hRelAmp.data(), P*sizeof(float), cudaMemcpyHostToDevice), "scal");

  // ---- loudness map ----
  loudnessKernel<<<(unsigned)gbN, TB>>>(d + (long)P_FREQLOUD*PN, g_arena.scal,
                                        g_arena.scal + P, maxAmp,
                                        (float)snd.getParam(LOUDNESS),
                                        dLoud, P, N);
  CKF(cudaGetLastError(), "loudness");

  // ---- phase scans (double), batched: ONE scan_by_key per phase type over
  // all partials (key = i/N via a transform iterator; no key array in VRAM)
  // instead of P separate device scans (72 launches+syncs per sound in v1). ----
  long gbPN = (PN + TB - 1) / TB;
  auto keys = thrust::make_transform_iterator(
      thrust::make_counting_iterator<long>(0), KeyOfIndex{N});
  // tremolo: EXCLUSIVE prefix of tremRate/sr, per partial
  buildIncKernel<<<(unsigned)gbPN, TB>>>(d + (long)P_TREMRATE*PN, dS3, PN, (float)samplingRate);
  thrust::exclusive_scan_by_key(thrust::device, keys, keys + PN,
      thrust::device_pointer_cast(dS3), thrust::device_pointer_cast(dS1), 0.0);
  // vibrato: EXCLUSIVE prefix of vibRate/sr -> then frequency increments
  buildIncKernel<<<(unsigned)gbPN, TB>>>(d + (long)P_VIBRATE*PN, dS3, PN, (float)samplingRate);
  thrust::exclusive_scan_by_key(thrust::device, keys, keys + PN,
      thrust::device_pointer_cast(dS3), thrust::device_pointer_cast(dS2), 0.0);
  freqKernel<<<(unsigned)gbPN, TB>>>(d + (long)P_FREQBASE*PN, d + (long)P_VIBAMP*PN,
                                     dS2, dS3, PN, (float)samplingRate);
  // carrier: INCLUSIVE prefix of frequency/sr
  thrust::inclusive_scan_by_key(thrust::device, keys, keys + PN,
      thrust::device_pointer_cast(dS3), thrust::device_pointer_cast(dS2));

  // ---- synth + deterministic partial sum ----
  synthKernel<<<(unsigned)gbPN, TB>>>(dLoud, d + (long)P_WAVESHAPE*PN,
                                      d + (long)P_TREMAMP*PN, dS1, dS2,
                                      d + (long)P_PHASEOFF*PN, dWave, dAmp, PN);
  CKF(cudaGetLastError(), "synth");
  long gbSC = (sampleCount + TB - 1) / TB;
  sumKernel<<<(unsigned)gbSC, TB>>>(dWave, dAmp, g_arena.mono,
                                    g_arena.mono + sampleCount, P, N, sampleCount,
                                    1.0f / (float)numChannels);
  CKF(cudaGetLastError(), "sum");

  // ---- one D2H, assemble placeholder-spatialized MultiTrack ----
  std::vector<float> monoW(sampleCount), monoA(sampleCount);
  CKF(cudaMemcpy(monoW.data(), g_arena.mono, sampleCount*sizeof(float),
                 cudaMemcpyDeviceToHost), "D2H w");
  CKF(cudaMemcpy(monoA.data(), g_arena.mono + sampleCount, sampleCount*sizeof(float),
                 cudaMemcpyDeviceToHost), "D2H a");

  MultiTrack* mt = new MultiTrack(numChannels, sampleCount, samplingRate);
  for (int c = 0; c < numChannels; ++c) {
    memcpy(mt->get(c)->getWave().getData(), monoW.data(), sampleCount*sizeof(float));
    memcpy(mt->get(c)->getAmp().getData(),  monoA.data(), sampleCount*sizeof(float));
  }
  return mt;
}

} // namespace portable
