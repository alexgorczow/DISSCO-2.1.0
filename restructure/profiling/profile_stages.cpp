/*
  Stage-level profiler for the LASS synthesis pipeline.

  Constructs a representative Sound and times each stage in isolation so we can
  rank what to parallelize:
    - Loudness::calculate  (full-rate 24-band psychoacoustic map)
    - Partial synthesis    (per-partial additive render)
    - Reverb               (comb + all-pass IIR)

  Build: see restructure/profiling/Makefile   Run: ./profile_stages
*/
#include "../../LASS/src/Sound.h"
#include "../../LASS/src/Partial.h"
#include "../../LASS/src/Loudness.h"
#include "../../LASS/src/Reverb.h"
#include "../../LASS/src/MultiTrack.h"
#include "../../LASS/src/Track.h"
#include "../../LASS/src/SoundSample.h"

#include <cstdio>
#include <chrono>
#include <vector>

using clk = std::chrono::high_resolution_clock;
static double ms(clk::time_point a, clk::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

static Sound* makeSound(int numPartials, double baseFreq, float duration) {
  Sound* s = new Sound(numPartials, baseFreq);
  s->setParam(DURATION, duration);
  s->setParam(START_TIME, 0.0);
  s->setParam(LOUDNESS, 44100.0);       // enable loudness (as CMOD does)
  s->setParam(LOUDNESS_RATE, 44100.0);
  return s;
}

int main(int argc, char** argv) {
  const unsigned int SR = 44100;
  const int CH = 2;
  int partials = (argc > 1) ? atoi(argv[1]) : 6;
  float dur    = (argc > 2) ? atof(argv[2]) : 2.0f;
  long N = (long)(dur * SR);

  printf("Representative sound: %d partials, %.1fs (%ld samples), %d ch, %u Hz\n\n",
         partials, dur, N, CH, SR);

  // ---- Loudness ----
  double tLoud = 0;
  {
    Sound* s = makeSound(partials, 220.0, dur);
    auto t0 = clk::now();
    Loudness::calculate(*s);
    auto t1 = clk::now();
    tLoud = ms(t0, t1);
    delete s;
  }

  // ---- Partial synthesis (all partials, no reverb) ----
  double tSynth = 0;
  {
    Sound* s = makeSound(partials, 220.0, dur);
    Loudness::calculate(*s);            // synthesis reads loudness scalars
    auto t0 = clk::now();
    // render each partial the way Sound::render does
    for (int p = 0; p < partials; ++p) {
      Partial part;
      part.setParam(FREQUENCY, 220.0 * (p + 1));
      MultiTrack* mt = part.render(CH, N, dur, SR);
      delete mt;
    }
    auto t1 = clk::now();
    tSynth = ms(t0, t1);
    delete s;
  }

  // ---- Reverb on a composited MultiTrack (warm; min of several reps) ----
  // NOTE: with HAVE_CUDA the reverb runs on the GPU. The FIRST cuda call in the
  // process pays ~300-500ms of context init, so we warm up and take the min.
  double tRev = 1e30;
  {
    auto makeMT = [&]() {
      MultiTrack* mt = new MultiTrack(CH, N, SR);
      for (int c = 0; c < CH; ++c) {
        SoundSample& w = mt->get(c)->getWave();
        for (long i = 0; i < N; ++i) w[i] = 0.1f * (float)((i % 441) - 220) / 220.0f;
      }
      return mt;
    };
    Reverb rev(0.5f, SR);
    // warmup (also initializes CUDA context if present)
    { MultiTrack* mt = makeMT(); delete &rev.do_reverb_MultiTrack(*mt); delete mt; }
    for (int r = 0; r < 5; ++r) {
      MultiTrack* mt = makeMT();
      auto t0 = clk::now();
      MultiTrack& out = rev.do_reverb_MultiTrack(*mt);
      auto t1 = clk::now();
      tRev = std::min(tRev, ms(t0, t1));
      delete &out; delete mt;
    }
  }

  double total = tLoud + tSynth + tRev;
  printf("%-22s %10s %8s\n", "stage", "time (ms)", "share");
  printf("%-22s %10.2f %7.1f%%\n", "Loudness::calculate", tLoud, 100*tLoud/total);
  printf("%-22s %10.2f %7.1f%%\n", "Partial synthesis",   tSynth, 100*tSynth/total);
  printf("%-22s %10.2f %7.1f%%\n", "Reverb (comb+allpass)", tRev, 100*tRev/total);
  printf("%-22s %10.2f %7.1f%%\n", "TOTAL (per sound)",    total, 100.0);
  return 0;
}
