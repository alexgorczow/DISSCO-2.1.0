# DISSCO benchmarking pipeline

A comprehensive, opt-in benchmarking pipeline that breaks a full `cmod` render
into per-stage timings so bottlenecks are obvious. Three layers:

| Layer | File | What it answers |
|---|---|---|
| **Stage profiler** | `StageProfiler.h` | Where does time go *inside one render*? |
| **Sweep driver** | `benchmark.py` | How does it scale across threads? Where does it plateau? |
| **Micro-profiler** | `profile_stages.cpp` | Isolated cost of one synthetic Sound's stages |

Everything lives under `restructure/profiling/` and is **opt-in** — with the
profiler disabled (the default), audio output is byte-for-byte unchanged
(verified: md5 identical to the pre-instrumentation golden).

---

## 1. Stage profiler (`StageProfiler.h`)

A header-only, thread-safe timer compiled into `cmod`. It instruments the real
pipeline at these points:

```
TOTAL (piece)                                    ── Main.cpp
├─ parse + config      XML parse                 ── piece-experimental.cpp
├─ event-tree build    CMOD stochastic build     ── piece-experimental.cpp
├─ render + join       LASS render + mix + clip  ── piece-experimental.cpp
│   ├─ Sound::render (per sound, worker threads) ── Sound.cpp
│   │   ├─ loudness         24-band psychoacoustic map
│   │   ├─ partial synth    additive sine + composite
│   │   ├─ sound reverb     comb + all-pass (GPU if HAVE_CUDA)
│   │   └─ spatialize       pan across channels
│   ├─ composite drain  Score::compositeRenderedSounds ── Score.cpp
│   ├─ final reverb     score-level reverb            ── Score.cpp
│   └─ clip             clipping management           ── Score.cpp
└─ write AIFF          AuWriter::write               ── piece-experimental.cpp
```

**Wall phases** (parse/build/render+join/write/final reverb/clip) run once on the
main/composite thread and don't overlap, so their time == wall clock.
**Aggregate sub-stages** (loudness/synth/reverb/spatialize) run on concurrent
worker threads, so their totals are *summed CPU time across threads* — which is
exactly what reveals compute distribution and thread contention.

Run any render with timing:

```bash
DISSCO_PROFILE=1 cmod /abs/path/project.dissco          # table -> stderr
DISSCO_PROFILE=1 DISSCO_PROFILE_OUT=stages.json cmod ... # + machine-readable JSON
```

Disabled (env unset) the scopes are a single boolean check — no clock read, no
atomics, no effect on output.

## 2. Sweep driver (`benchmark.py`)

Runs a render across a matrix of thread counts × repetitions, pins a fixed seed,
rewrites `<NumberOfThreads>` in a scratch copy (never touches your project file),
records each output's md5, and emits `results.csv`, `results.json`, and a
self-contained `report.html`.

```bash
python3 restructure/profiling/benchmark.py Tutorial_small.dissco \
    --threads "1 2 4 8 20" --reps 2 --seed 42 \
    --out restructure/profiling/bench_runs/sweep_small
```

Console output ranks the sub-stages and prints the scaling curve; the HTML
report visualizes the compute mix and the per-thread stage stack.

## 3. Micro-profiler (`profile_stages.cpp`)

Times Loudness / Partial synthesis / Reverb for a single *synthetic* Sound in
isolation (warm CUDA). Useful for A/B'ing a kernel change without a whole song.
Build via the profiling `Makefile`, run `./profile_stages [partials] [seconds]`.

---

## Findings (Tutorial_small, 30 s, 50 sounds, RTX 4050, seed 42)

- **Compute mix (1 thread):** sound reverb **44%**, loudness **27%**,
  spatialize **16%**, partial synth **13%**. Reverb is the #1 cost on real
  reverb-heavy pieces (the synthetic micro-profiler over-weighted loudness).
- **Thread scaling plateaus at ~2.5×** (8 t = 2.54×; 20 t *regresses* to 2.45×).
- **The plateau's mechanism, quantified:** aggregate reverb CPU balloons
  **5.5 s → 69.8 s (12.7×)** from 1→20 threads while wall clock stays flat —
  all worker threads block in `do_reverb_MultiTrack` contending on the single
  serialized GPU. Adding cores just adds contenders.
- **Determinism:** 1 thread + fixed seed = bit-exact (identical md5 across
  reps); multi-thread = non-deterministic md5 (FP composite order), inaudible.

**Where acceleration lives:** break the serial GPU-reverb section (batch per
Sound / a fast parallel CPU reverb), then loudness. See
[`../02_RESULTS.md`](../02_RESULTS.md) and the project memory.
