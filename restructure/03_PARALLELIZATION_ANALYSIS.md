# DISSCO Parallelization & Efficiency Analysis

*Serious, measurement-driven analysis of what can be parallelized/accelerated in
DISSCO, what was done, and the parity guarantees. Machine: i7-13705H (20 logical
cores) + RTX 4050, g++ 12.2, nvcc 13.0. Written 2026-07-07.*

> **Headline:** DISSCO's slowness was mostly a **software-efficiency** problem,
> not a lack of hardware. One O(1) fix to `Envelope::getValue` cut a
> reverb-heavy render **4×** single-threaded (bit-exact), and switching from the
> serialized (and incorrect) GPU reverb to the now-fast CPU reverb made the
> default **correct and 2.5× faster at 20 threads**, scaling 5.2× vs the old
> 2.5× plateau. Net vs the true single-core original: **40.8 s → 1.9 s = 21×.**

## 1. Method: measure before touching anything

- **gprof** on a CPU render ranked self-time by function.
- **Stage profiler** (`restructure/profiling/`, `DISSCO_PROFILE=1`) attributes
  wall + per-thread-aggregate time to Loudness / synthesis / reverb / spatialize
  / composite / clip.
- **Thread sweeps** + reverb-on/off + CPU-vs-GPU-reverb A/Bs isolate each cost.
- **Parity** is checked bit-exactly (single-thread + fixed seed ⇒ deterministic).

## 2. What the profiles revealed

1. **`Envelope::getValue` was catastrophically inefficient.** gprof:
   `ExponentialInterpolatorIterator::next()` = **42% of self-time, 2.35 billion
   calls**. `getValue` rebuilt an interpolator and *stepped it up to 100 times to
   read one point*, on every one of its 43 M calls (per-sample in the reverb).
2. **The reverb dominated wall clock** (≈53% at 20 threads) — but because of #1
   and because the CUDA reverb **serializes on the single GPU** (aggregate reverb
   CPU balloons ~12.7× from 1→20 threads while wall stays flat).
3. **The CUDA reverb is not the reference algorithm** — it diverges from the CPU
   reverb by **~−59 dBFS (70% of samples differ)**. The shipped default (CUDA)
   was already *wrong* relative to DISSCO's own CPU algorithm.
4. Thread scaling **plateaued at ~2.5×** regardless of core count — a serial
   section (GPU-reverb serialization; secondarily the single composite thread).

## 3. Per-stage parallelization map (measured)

Per-sound aggregate CPU share after the getValue fix (20 threads):
**loudness 32%, reverb 32%, spatialize 20%, synthesis 16%**, composite drain 0.6%.

| Stage | Structure | Parallel? | Status / lever | Parity |
|---|---|---|---|---|
| **Envelope::getValue** | was O(100) iterate-to-point per call | — | **FIXED: O(1) memoized table** (`1d7035d`) | bit-exact |
| **Reverb** (comb+allpass IIR) | linear-recurrence scan; 6 combs independent | across sounds ✓ | **CPU default** (fast+correct+scales) vs GPU opt-in (`a8590fe`); further: SIMD the filter step, or a *correct* GPU scan | CPU bit-exact; GPU −59 dBFS |
| **Loudness** | per-sample **map** (24 bands × partials, `pow`) at 44.1 kHz | across sounds ✓; across samples (todo) | biggest remaining: **rate reduction** (historically 10 Hz, now 44100 — ~4400× oversampled) as opt-in fast mode; or SIMD/within-sound OpenMP | rate-reduce = tolerance; SIMD = bit-exact |
| **Synthesis** (`Partial::render`) | map + phase scan | across sounds ✓; portable worklet CPU/GPU | done (Tier 1–4): `LASS_PORTABLE_BACKEND` | bit-exact (serial) |
| **Spatialize** (Pan/MultiPan) | per-sample × per-channel **map** | across sounds ✓ | SIMD; hoist constant-pan scale out of loop | bit-exact possible |
| **Composite / mixing** | elementwise add | serial (1 thread) | parallel reduction tree ⇒ also makes multi-thread **deterministic** | bit-exact by fixed order |
| **Clipping** (scale/anticlip) | reduce (max) + map | trivially | negligible (0.1%) | bit-exact |
| **CMOD composition** | stochastic tree / sieves / Markov | poorly (irregular) | leave on CPU (negligible: 0.5%) | n/a |

## 4. What was implemented (all committed, all parity-checked)

| Change | Effect | Parity |
|---|---|---|
| `getValue` O(1) memoization | reverb-heavy render **40.8 s → 10.2 s @1t (4.0×)** | bit-exact vs true original |
| Reverb per-sample alloc hoist | removes millions of heap allocs | bit-exact (self-referential no-op) |
| `constructAmp` off-by-one fix | fixes latent OOB read past buffer | bit-exact (amp channel unused for audio) |
| Reverb backend select (CPU default) | **correct** output + scales 5.2× (vs GPU 2.5× & −59 dBFS) | CPU = true original; GPU via `LASS_REVERB=gpu` |
| Portable synth worklet (Tier 1–4) | CPU/GPU portable synthesis kernel | bit-exact (serial), within-budget (CUDA) |

**Speedup summary (30 s tutorial, correct output):**

| baseline | new default | speedup |
|---|---|---|
| true original, 1 core (40.8 s) | 1.9 s @20t | **21×** |
| old GPU default, 1 thread (11.8 s) | 1.9 s @20t | 6.2× |
| old GPU default, 20 threads (4.7 s) | 1.9 s @20t | 2.5× (**and now correct**) |
| old CPU reverb, 1 thread (40.8 s) | 10.2 s @1t | 4.0× (bit-exact) |

## 5. The remaining path to 10× over the *multi-threaded* default

At 20 threads the 50-sound tutorial is **load-imbalance bound**. Measured
scaling of the correct default (CPU reverb): 1→4→8→20 threads = 1.0×→3.0×→4.0×→
**~5.2×**, then flat — because 50 variable-length sounds over 20 cores leaves the
critical thread with the few longest sounds. This is *coarse-grained* (per-Sound)
parallelism hitting Amdahl on sound-count, not a compute wall; pieces with more
sounds scale closer to core count. Beyond that, the ranked levers are:

1. **Loudness rate reduction** (opt-in `LASS_LOUDNESS_RATE`, default off): loudness
   is 32% and evaluated at 44.1 kHz. **MEASURED and refuted as a free lunch** —
   dropping to 4410 Hz gave only **1.46×** while introducing **−19.7 dBFS RMS**
   error (audible); 100 Hz → 1.6×, −18 dBFS. The per-sample `LOUDNESS_SCALAR` is
   *not* slowly varying (it couples `maxAmp`/gamma across partials and interacts
   with anticlip). Kept only as a fast-**preview** knob, not an optimization.
   The real loudness win is bit-exact **SIMD / within-sound parallelism**, not
   subsampling.
2. **Within-sound parallelism / SIMD** — *investigated and measured to have no
   bit-exact upside here* (see §7). The per-sample loops are the wrong shape for
   SIMD: memory-bound copies (spatialize, reverb output) don't speed up under
   vectorization; the compute-bound loops need *vector transcendentals*
   (`sin`/`pow`) that change FP results; and the reverb is a sequential IIR
   recurrence. Compiler auto-vectorization (`-O3 -march=native`) gave ~4% only
   because of FMA contraction, which breaks parity (−88 dBFS); with
   `-ffp-contract=off` it is bit-exact but **no faster** than `-O2`.
3. **Deterministic parallel composite** — removes the last serial section *and*
   makes multi-thread output bit-reproducible (today it drifts ≤3 LSB).
4. **A *correct* GPU reverb** (matching the CPU algorithm within budget) with
   buffer reuse + streams to avoid the malloc/serialize overhead — for very
   large pieces where one GPU can still beat 20 cores.

## 6. Parity methodology (how we know we still match the original)

- **Reference = the CPU reverb algorithm** (the GPU reverb is a measured
  approximation and is *not* the reference).
- **Single-thread + fixed seed ⇒ bit-exact reproducible**, so every check is an
  md5/`cmp`, not a tolerance.
- `restructure/harness/parity_regression.sh` renders several seeds and asserts:
  determinism, `portable-serial == default`, and the CPU-reverb reference md5;
  it also reports the GPU divergence as a number.
- Multi-thread / GPU paths are bounded by the original's own run-to-run budget
  (≤ few LSB / < −140 dBFS), which is inaudible and which the original already
  spends via FP summation order.

## 7. Two follow-up investigations (2026-07-07)

### 7a. Within-Sound SIMD — measured dead end for bit-exact speedup
Compiler auto-vectorization of the whole LASS library (CPU-reverb default,
30 s tutorial, seed 42):

| flags | 1-thread | parity vs golden (12d2ff21) |
|---|---|---|
| `-O2` (default) | 9.90 s | bit-exact |
| `-O3 -march=native` | 9.77 s (~1%) | **DIFF: −88.6 dBFS RMS** (FMA contraction) |
| `-O3 -march=native -ffp-contract=off -fno-fast-math` | 10.08 s | **bit-exact but slower** |

Why: the hot per-sample loops are (a) memory-bound copies (spatialize; reverb
output) that SIMD cannot accelerate, (b) transcendental-bound (`sin` in
synthesis, `pow` in loudness) which only vectorize via `libmvec`/`-ffast-math`
that change the FP result, or (c) a sequential IIR recurrence (reverb feedback).
None yield a bit-exact win. The real remaining levers stay coarse-grained
(more sounds → more per-Sound threads) plus the already-landed algorithmic fix.

### 7b. `7_final.dissco` segfault — root-caused and fixed
`7_final.dissco` crashed for every seed. Root cause: the file is **malformed
XML** (mismatched tag ~line 3164); xerces recovers into a DOM where the Mid
event "m4" is not a reachable palette sibling, so it is never registered. A
child then references "m4", and `Utilities::getEventElement` did
`return it->second` on a `std::map::find()` result that was `end()` —
**undefined behavior**, segfaulting deep in the `Event` constructor. Fixed
(`getEventElement` returns NULL on not-found; the caller reports which event
referenced which missing child and aborts). Now it prints a clear diagnostic
and exits 1; valid pieces are unaffected (tutorial still bit-exact). Rendering
7_final itself would require repairing its malformed XML (user data).

### 7c. Multi-threaded valgrind (memcheck + helgrind)
- **memcheck, 4 threads:** 0 bytes lost, 464,251 allocs = frees, **0 errors** (after
  the Partial rule-of-three leak fix, which also holds single-threaded).
- **helgrind, 4 threads:** found real data races on `Score` scalars shared across
  the add/worker/composite threads without consistent locking — `scoreEndTime`
  (written under `mutexSoundVector`, read lock-free by the composite thread) and
  the `doneGettingSoundObjects` / `workerThreadsAllJoined` coordination flags.
  Made them `std::atomic` (bit-exact; benign on x86 but UB, and a portability/
  optimization hazard). Score-member race contexts 6 → 2 (the residual 2 are
  helgrind approximate-stack artifacts around `pthread_create` happens-before in
  the Score constructor). The rest are benign `std::cout` interleaving from
  worker-thread logging. No races in the render path, the `Envelope::getValue`
  cache, or the RNG.
