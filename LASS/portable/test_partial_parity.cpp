/*
  Tier 1 checkpoint C1: prove portable::renderPartial (Serial) reproduces the
  original Partial::render() bit-for-bit for the no-transient sine path.

  For each test partial we render two identical Partials — one via the original
  Partial::render(), one via portable::renderPartial() — and compare every
  sample of every channel of the resulting MultiTracks.
*/
#include "PortableSynth.h"

#include "../src/Partial.h"
#include "../src/MultiTrack.h"
#include "../src/Track.h"
#include "../src/SoundSample.h"
#include "../src/InterpolatorTypes.h"

#include <cstdio>
#include <cmath>
#include <cstring>
#include <functional>

static const int   kChannels = 2;
static const float kDuration = 1.0f;
static const unsigned int kRate = 44100;

struct DiffResult {
  long   totalSamples = 0;
  long   diffSamples  = 0;
  double maxAbs       = 0.0;
  bool   bitExact     = true;   // exact IEEE-754 equality of every float
  long   firstDiff    = -1;
};

static DiffResult compareMultiTrack(MultiTrack* A, MultiTrack* B) {
  DiffResult r;
  int chA = A->size(), chB = B->size();
  if (chA != chB) { r.bitExact = false; r.firstDiff = 0; return r; }
  for (int c = 0; c < chA; ++c) {
    SoundSample& wa = A->get(c)->getWave();
    SoundSample& wb = B->get(c)->getWave();
    long na = wa.getSampleCount(), nb = wb.getSampleCount();
    long n = na < nb ? na : nb;
    if (na != nb) { r.bitExact = false; }
    for (long s = 0; s < n; ++s) {
      float va = wa[s], vb = wb[s];
      r.totalSamples++;
      // bit-exact test on the raw float representation
      if (std::memcmp(&va, &vb, sizeof(float)) != 0) {
        r.bitExact = false;
        r.diffSamples++;
        double d = std::fabs((double)va - (double)vb);
        if (d > r.maxAbs) r.maxAbs = d;
        if (r.firstDiff < 0) r.firstDiff = s;
      }
    }
  }
  return r;
}

static int runCase(const char* name, std::function<Partial*()> make) {
  Partial* p1 = make();
  Partial* p2 = make();
  long sampleCount = (long)(kDuration * (float)kRate);

  bool portable_ok = portable::canRenderPortably(*p2);

  MultiTrack* mtOrig = p1->render(kChannels, sampleCount, kDuration, kRate);
  MultiTrack* mtPort = portable::renderPartial(
      *p2, kChannels, sampleCount, kDuration, kRate, portable::Backend::Serial);

  DiffResult r = compareMultiTrack(mtOrig, mtPort);

  printf("[%-22s] portable=%s  samples=%ld  diffs=%ld  maxAbs=%.3e  %s\n",
         name, portable_ok ? "yes" : "FALLBACK", r.totalSamples, r.diffSamples,
         r.maxAbs, r.bitExact ? "BIT-EXACT" : "MISMATCH");
  if (!r.bitExact)
    printf("    first diff at sample %ld\n", r.firstDiff);

  delete mtOrig; delete mtPort; delete p1; delete p2;
  return r.bitExact ? 0 : 1;
}

int main() {
  int fails = 0;

  // Case 1: default partial — pure 440 Hz sine, amplitude 1 (Constant params).
  fails += runCase("default_sine_440", []() {
    Partial* p = new Partial();
    return p;
  });

  // Case 2: constant frequency at an arbitrary value.
  fails += runCase("const_freq_333", []() {
    Partial* p = new Partial();
    p->setParam(FREQUENCY, 333.0);
    return p;
  });

  // Case 3: LINEAR frequency glissando 220->440 (exercises value+=delta scan).
  fails += runCase("linear_gliss_220_440", []() {
    Partial* p = new Partial();
    LinearInterpolator freq;
    freq.addEntry(0.0, 220.0);
    freq.addEntry(1.0, 440.0);
    p->setParam(FREQUENCY, freq);
    return p;
  });

  // Case 4: LINEAR wave_shape envelope 0->1->0 (amplitude scan) + linear gliss.
  fails += runCase("linear_shape_and_gliss", []() {
    Partial* p = new Partial();
    LinearInterpolator freq;
    freq.addEntry(0.0, 200.0);
    freq.addEntry(1.0, 600.0);
    p->setParam(FREQUENCY, freq);
    LinearInterpolator shape;
    shape.addEntry(0.0, 0.0);
    shape.addEntry(0.5, 1.0);
    shape.addEntry(1.0, 0.0);
    p->setParam(WAVE_SHAPE, shape);
    return p;
  });

  printf("\n%s (%d case(s) failed)\n", fails == 0 ? "C1 PASS: all bit-exact" : "C1 FAIL", fails);
  return fails == 0 ? 0 : 1;
}
