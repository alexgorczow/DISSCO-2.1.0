#ifndef DISSCO_STAGE_PROFILER_H
#define DISSCO_STAGE_PROFILER_H
//============================================================================//
//  StageProfiler.h  --  opt-in, thread-safe stage timing for DISSCO's render
//                       pipeline. Part of the restructure/ benchmarking work.
//
//  Enable at runtime:   DISSCO_PROFILE=1 cmod project.dissco
//  Optional JSON dump:  DISSCO_PROFILE_OUT=/path/stages.json
//
//  When DISSCO_PROFILE is unset/0, every PROFILE_SCOPE is a single boolean
//  check (no clock read, no atomics), so the default build is unaffected and
//  the audio output stays byte-for-byte identical. This header is self
//  contained (no link dependency): include it and call prof::report() once.
//
//  Two kinds of stage are reported:
//    * WALL phases   (TOTAL/PARSE/EVENT_BUILD/RENDER_JOIN/WRITE/FINAL_REVERB/
//                     CLIP) are entered once from the main/composite thread and
//                     do not overlap each other, so accumulated == wall clock.
//    * AGGREGATE     (SOUND_RENDER/LOUDNESS/PARTIAL_SYNTH/SOUND_REVERB/
//                     SPATIALIZE/COMPOSITE) run on concurrent worker threads,
//                     so their totals are summed CPU time across threads and
//                     may exceed wall clock -- which is exactly what exposes
//                     where the compute goes.
//============================================================================//

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace prof {

enum Stage {
  TOTAL = 0,      // whole Piece run                                  (wall)
  PARSE,          // XML parse + config read                         (wall)
  EVENT_BUILD,    // CMOD stochastic event tree build                (wall)
  RENDER_JOIN,    // doneCMOD: join workers + final reverb + clip    (wall)
  WRITE,          // AuWriter::write                                 (wall)
  SOUND_RENDER,   // Sound::render, whole                       (aggregate)
  LOUDNESS,       // Loudness::calculate                        (aggregate)
  PARTIAL_SYNTH,  // partial synthesis + per-sound composite    (aggregate)
  SOUND_REVERB,   // per-Sound reverb                           (aggregate)
  SPATIALIZE,     // per-Sound spatialize                       (aggregate)
  GF_PREPASS,     // gpu-fast host DV iteration + RLE           (aggregate)
  GF_GPU,         // gpu-fast device section (mutex-held)       (aggregate)
  COMPOSITE,      // Score composite drain                         (thread)
  FINAL_REVERB,   // score-level reverb in joinThreadsAndMix         (wall)
  CLIP,           // clipping management                             (wall)
  NUM_STAGES
};

inline const char* stageName(int s) {
  static const char* names[NUM_STAGES] = {
    "TOTAL (piece)",
    "parse + config",
    "event-tree build",
    "render + join",
    "write AIFF",
    "Sound::render (all)",
    "loudness",
    "partial synth",
    "sound reverb",
    "spatialize",
    "gpu-fast pre-pass",
    "gpu-fast device",
    "composite drain",
    "final reverb",
    "clip management",
  };
  return (s >= 0 && s < NUM_STAGES) ? names[s] : "?";
}

// true == wall-clock phase (non-overlapping); false == aggregate over threads
inline bool isWall(int s) {
  switch (s) {
    case TOTAL: case PARSE: case EVENT_BUILD: case RENDER_JOIN:
    case WRITE: case FINAL_REVERB: case CLIP:
      return true;
    default:
      return false;
  }
}

struct Registry {
  std::atomic<uint64_t> ns[NUM_STAGES];
  std::atomic<uint64_t> calls[NUM_STAGES];
  bool enabled;

  Registry() {
    for (int i = 0; i < NUM_STAGES; ++i) { ns[i] = 0; calls[i] = 0; }
    const char* e = std::getenv("DISSCO_PROFILE");
    enabled = e && e[0] && !(e[0] == '0' && e[1] == '\0');
  }
  void add(int s, uint64_t d) {
    ns[s].fetch_add(d, std::memory_order_relaxed);
    calls[s].fetch_add(1, std::memory_order_relaxed);
  }
};

inline Registry& registry() { static Registry r; return r; }
inline bool enabled() { return registry().enabled; }

// RAII: times the enclosing block into `stage` when profiling is enabled.
struct Scope {
  int stage;
  bool on;
  std::chrono::high_resolution_clock::time_point t0;
  explicit Scope(int s) : stage(s), on(registry().enabled) {
    if (on) t0 = std::chrono::high_resolution_clock::now();
  }
  ~Scope() {
    if (on) {
      auto t1 = std::chrono::high_resolution_clock::now();
      registry().add(stage,
        (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0)
          .count());
    }
  }
  Scope(const Scope&) = delete;
  Scope& operator=(const Scope&) = delete;
};

// Print a human table to stderr and, if DISSCO_PROFILE_OUT is set, a JSON file.
// No-op when profiling is disabled. Call once, at the end of main().
inline void report() {
  Registry& r = registry();
  if (!r.enabled) return;

  const double NS = 1e6;  // ns -> ms
  double total_ms = r.ns[TOTAL].load() / NS;
  if (total_ms <= 0) total_ms = 1e-9;

  // denominator for the aggregate render sub-stages' share
  double agg_sum = 0;
  for (int s = LOUDNESS; s <= GF_GPU; ++s) agg_sum += r.ns[s].load() / NS;
  if (agg_sum <= 0) agg_sum = 1e-9;

  std::fprintf(stderr,
    "\n================= DISSCO stage profile =================\n");
  std::fprintf(stderr, "%-22s %11s %8s %11s %8s\n",
               "stage", "total(ms)", "calls", "mean(ms)", "share");
  std::fprintf(stderr,
    "-------------------------------------------------------\n");
  for (int s = 0; s < NUM_STAGES; ++s) {
    uint64_t c = r.calls[s].load();
    if (c == 0 && s != TOTAL) continue;
    double ms = r.ns[s].load() / NS;
    double mean = c ? ms / c : 0.0;
    double denom = isWall(s) ? total_ms : agg_sum;
    double share = 100.0 * ms / denom;
    // indent aggregate sub-stages under Sound::render for readability
    const char* prefix = (s >= LOUDNESS && s <= GF_GPU) ? "  " : "";
    char label[48];
    std::snprintf(label, sizeof(label), "%s%s", prefix, stageName(s));
    std::fprintf(stderr, "%-22s %11.2f %8llu %11.3f %7.1f%%\n",
                 label, ms, (unsigned long long)c, mean, share);
  }
  std::fprintf(stderr,
    "-------------------------------------------------------\n");
  std::fprintf(stderr,
    "wall phases share = %% of TOTAL; aggregate share = %% of summed\n"
    "render sub-stages (loudness+synth+reverb+spatialize).\n");
  std::fprintf(stderr,
    "=======================================================\n\n");

  const char* out = std::getenv("DISSCO_PROFILE_OUT");
  if (out && out[0]) {
    FILE* f = std::fopen(out, "w");
    if (f) {
      std::fprintf(f, "{\n  \"total_ms\": %.4f,\n  \"stages\": {\n", total_ms);
      bool first = true;
      for (int s = 0; s < NUM_STAGES; ++s) {
        double ms = r.ns[s].load() / NS;
        uint64_t c = r.calls[s].load();
        if (!first) std::fprintf(f, ",\n");
        first = false;
        // machine-friendly key: strip spaces/parens
        std::fprintf(f,
          "    \"%s\": {\"ms\": %.4f, \"calls\": %llu, \"wall\": %s}",
          stageName(s), ms, (unsigned long long)c, isWall(s) ? "true" : "false");
      }
      std::fprintf(f, "\n  }\n}\n");
      std::fclose(f);
    }
  }
}

}  // namespace prof

#define PROFILE_CONCAT_(a, b) a##b
#define PROFILE_CONCAT(a, b) PROFILE_CONCAT_(a, b)
// Times the enclosing block into `stage` (a prof::Stage) via an RAII guard.
#define PROFILE_SCOPE(stage) \
  prof::Scope PROFILE_CONCAT(_prof_scope_, __LINE__)(stage)

#endif  // DISSCO_STAGE_PROFILER_H
