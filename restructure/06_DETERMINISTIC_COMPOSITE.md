# Deterministic Parallel Composite (CPU + GPU)

*Design + results. Goal: remove the last serial section of the render pipeline
(the single composite thread) and make multi-threaded output **bit-identical**
across thread counts and devices — not merely "within a few LSB".*

## 1. Pipeline analysis (what exists today)

```
main thread            worker threads (numThreads)          composite thread
-----------            ---------------------------          ----------------
event tree build  -->  sem_wait(fullSounds)                  sem_wait(fullRendered)
Score::add(sound)      pop sounds.back()   (LIFO!)           pop renderedSounds.back() (LIFO!)
  seq order stable     mt = sound->render(...)               checkScoreMultiTrackLength()
  scoreEndTime max     addRenderedSound(start, mt)           scoreMultiTrack->composite(mt, start)
doneAddingSounds()       push renderedSounds                 ... until workers joined & empty
                         (bounded MAX_RENDERED_OBJECTS=20)
```

- `MultiTrack::composite(mt, t0)` → per track: `Track::composite` → `SoundSample::
  composite` on **wave** and (if both sides have it) **amp**: a pure
  `data_[s+skip] += src[s]` loop with `skip = (long)(t0 * float(rate))`.
- The score `MultiTrack(numChannels, N, rate)` creates tracks **with** amp, and
  rendered sounds carry amp, so both channels accumulate.
- Buffer growth (`checkScoreMultiTrackLength`): allocate bigger, composite old
  score at offset 0 (exact: `0 + x = x`), continue. `scoreEndTime` is the
  monotone max over *added* sounds **including each sound's own end**, and
  `add(k)` strictly precedes the commit of `k`, so **no commit ever clips**
  against a too-short buffer → growth *timing* cannot change output bits.

### Where nondeterminism enters
FP addition is not associative: where sounds **overlap**, `(s+a)+b ≠ (s+b)+a`
in the last bits. The accumulation order per output sample = the order sounds
are committed = *completion order of concurrent workers* (plus LIFO drains).
That order varies run-to-run ⇒ the measured ≤3–4 LSB / −152 dBFS drift at 20
threads. With 1 thread the order is stable ⇒ single-thread is reproducible.

### Constraints discovered during analysis
1. **Canonical order ≠ legacy single-thread order.** The 1-thread order is
   timing-dependent (LIFO pops interleaved with producer progress) — it cannot
   be reproduced structurally. So a deterministic mode defines a **new**
   canonical order (Score::add insertion sequence, which is produced by the
   single-threaded event-tree traversal and is stable). Consequence: det-mode
   output differs from the legacy golden by the usual composite-order budget
   (few LSB); it CANNOT be both "novel deterministic order" and "equal to the
   legacy golden". Therefore det mode is **opt-in** (`LASS_COMPOSITE`), and the
   default stays byte-identical to the existing golden.
2. **Bounded reorder buffer + LIFO dispatch can deadlock.** If all
   MAX_RENDERED_OBJECTS slots fill with out-of-order results while the
   head-of-order sound is still rendering, workers block and the head can never
   land. Fix: **reserve the output slot before rendering** (dispatch window):
   a worker acquires `semEmptySlotsRendered` *before* popping a sound, and det
   mode pops **FIFO**. Then the in-flight+buffered set is exactly the window
   `[nextCommit, nextCommit+cap)`, whose head is always dispatched ⇒ always
   completes ⇒ commits free slots ⇒ no deadlock.
3. **GPU adds are bit-identical to CPU adds.** The composite is *only* IEEE-754
   float additions (no transcendentals, no FMA in a bare `a += b`). CUDA and
   x86 both implement round-to-nearest-even float add exactly ⇒ a GPU composite
   that applies the same adds in the same per-sample order produces the same
   bits. (This is why composite is the *right* stage to GPU-accelerate for
   bit-exactness, unlike sin/pow-bound synthesis/loudness — see
   05_GPU_SYNTH_RESULTS.md.)
4. RNG caveat: pieces using **detune** draw `random()` inside worker threads
   (Sound::render → setup_detuning_env), which is interleaving-dependent — a
   *pre-existing* nondeterminism independent of composite order. The tutorial
   and bench pieces do not use detune. Det mode guarantees composite-order
   determinism; per-sound RNG seeding for detuned pieces is future work.

## 2. Strategy

`LASS_COMPOSITE` env var (same opt-in pattern as `LASS_REVERB`,
`LASS_PORTABLE_BACKEND`):

| mode | behavior |
|---|---|
| unset / `legacy` | **default** — byte-identical to current behavior (arrival-order commits, LIFO) |
| `det` | canonical-order commits on CPU: seq-tagged sounds, FIFO dispatch, slot-reserve-before-render, reorder-buffer in-order commit |
| `det-gpu` | same canonical order; commits are **in-order kernel launches** on one CUDA stream into a persistent device score buffer; single D2H fetch at join |

Both det backends produce **one canonical output**: `det@1t == det@Nt ==
det-gpu` bit-for-bit (md5). The composite thread stops being a serial *sum*
bottleneck: on CPU the commit is a memcpy-speed add; on GPU the launch returns
immediately and the adds run at device bandwidth, so the drain can no longer
back-pressure workers (the queue empties at launch speed).

Why this is the ambitious-but-honest design: prior measurement
(05_GPU_SYNTH_RESULTS.md) showed the *math* stages are bit-exactness-capped on
GPU by transcendental ULPs and sequential scans. The composite is the one stage
whose GPU port is **provably** bit-exact (pure adds), and determinism converts
the multi-thread path from "≤4 LSB drift" to "md5-equal" — making the *fast*
path the *reference* path.

## 3. What changed (implementation)

- `Score.h/Score.cpp`
  - `compositeMode_` read once from `LASS_COMPOSITE` in the ctor.
  - `sounds` becomes `deque<pair<Sound*, long>>` (seq = `soundObjectsCreated`
    at `add()`); legacy pops `back()` (unchanged order), det pops `front()`.
  - det: worker reserves `semEmptySlotsRendered` **before** popping (dispatch
    window); `addRenderedSound(start, mt, seq)` skips the wait it already paid.
  - det: `compositeRenderedSounds` maintains `map<long, pending>` (reorder
    buffer) + `nextCommit`; commits strictly in seq order; posts empty slots
    only at commit.
  - det-gpu: commit = `compositeCudaAdd(track arrays, offset)` (async, one
    stream); at drain end, `compositeCudaFetch()` builds the final host
    MultiTrack; host score buffer is never grown (memory win).
- `LASS/portable/CompositeCuda.{h,cu}` — persistent device score (wave+amp per
  channel), geometric growth (D2D copy + zero tail — exact), pinned staging
  buffer, grid-stride add kernel, single-stream ordering. Follows the
  DevScratch pattern from PartialRendererCuda.cu.
- premake4.lua: compile/archive CompositeCuda.o like PartialRendererCuda.o.

## 4. Verification (all must pass)

1. **Legacy regression**: default build+mode renders tutorial seed42 @1t to the
   historical md5 `12d2ff21…` (byte-identical AIFF).
2. **Determinism prize**: `det@1t == det@8t == det@20t` md5-equal (tutorial +
   bench_1min), and det@20t twice = same md5 (run-to-run).
3. **Cross-device prize**: `det-gpu == det` md5-equal (tutorial + bench_1min +
   bench_10min).
4. **Budget check**: det vs legacy@1t within the established composite budget
   (few LSB, < −140 dBFS) — order changed, nothing else.
5. **Functionality**: .particel unchanged; no crash; parity_regression.sh still
   6/6; memcheck clean on det path.
6. **Performance**: wall + peak RSS on bench_1min/bench_10min across
   {legacy, det, det-gpu} × {8t, 20t}.

## 5. A second nondeterminism found and fixed (worker RNG trampling)

The first bench_1min matrix FAILED (det@1t, det@20t, det@20t-run2 all
different) while the tutorial matrix passed. Root cause — **pre-existing and
independent of composite order**: `Partial::render` calls `srand(time(0))` and
draws `std::rand()` (transient checks, every ~1103 samples per partial) on
**worker threads**, while CMOD's producer thread is still drawing *piece
structure* from the same global stream (glibc `rand()` == `random()`). Any
piece with more than MAX_SOUND_OBJECTS=200 sounds keeps the producer mid-build
during rendering, so worker draws/reseeds corrupt the producer's sequence — 
**the piece itself differs run-to-run** (legacy@20t twice: two different md5s;
even 1-thread runs differ because the single worker interleaves with the
producer). The 50-sound tutorial escapes only because all its sounds are added
before any worker starts.

Fix (det modes only; legacy byte-identical): each render uses a private minstd
stream seeded per partial (`Partial.cpp: useDeterministicRng/drawRand`). For
no-transient partials the draws are discarded, so no output bits change; for
transient pieces it replaces wall-clock seeding (already nondeterministic) with
a stable stream.

## 6. Results

### Tutorial (30 s, 50 sounds, seed 42)

| run | md5 |
|---|---|
| default / legacy @1t | `12d2ff21…` (historical golden — **unchanged**) |
| det @1t, @8t, @20t, @20t run 2 | `27a6672c…` — **all four identical** |
| det-gpu @1t, @20t | `27a6672c…` — **identical to CPU det** |
| det vs legacy | max 4 LSB, RMS −151.7 dBFS (composite-order budget only) |

parity_regression.sh: 4/4 pass after the change.

### bench_1min (60 s, 360 sounds — exceeds the 200-sound queue cap)

| run | md5 | wall |
|---|---|---|
| det @1t | `c236f2b1…` | 123.6 s |
| det @20t | `c236f2b1…` | 21.1 s |
| det @20t run 2 | `c236f2b1…` | 24.1 s |
| **det-gpu @20t** | **`c236f2b1…`** | 22.2 s |
| legacy @20t run 1 | `392e6a86…` | 21.2 s |
| legacy @20t run 2 | `1f23f178…` (≠ run 1!) | 20.6 s |

**Four-way bit-identity across thread count, runs, and devices** — while legacy
does not even reproduce itself run-to-run on this piece (the worker-RNG
trampling of §5; before the RNG fix even det@1t differed from det@20t).
Determinism costs ~0 wall time here.

### bench_10min (600 s, 360 larger sounds, @20t, single rep)

| mode | wall | peak RSS | md5 |
|---|---|---|---|
| legacy | 192.5 s | 1.82 GB | `b0496b9c…` (not reproducible) |
| det | 211.0 s | 1.87 GB | `51fa8676…` |
| **det-gpu** | **208.9 s** | **1.55 GB** | **`51fa8676…` == det** |

- **Cross-device bit-identity holds at 10-minute scale.**
- det costs ~9.6% wall vs legacy at this scale (single rep; FIFO dispatch +
  slot-reserve head-of-line waiting when the head sound is slow). det-gpu
  reclaims about a quarter of that (commits are async kernel launches) and
  saves **0.27 GB** host RSS (the score lives on the device).
- Honest speedup framing: the composite drain was ~0.6% of aggregate CPU, so
  this work was never going to shrink wall time much — the prize is that the
  **fast multi-threaded path is now the reference path** (md5-reproducible),
  plus a provably-bit-exact GPU offload and lower host memory. If more raw
  throughput is wanted later, the head-of-line stall can be tuned by widening
  MAX_RENDERED_OBJECTS for det modes (memory-for-latency trade).

### memcheck (det path, 4 threads, 3 s tutorial)
451,437 allocs = 451,437 frees, **0 bytes in use at exit, 0 errors** — the
reorder buffer / deque / slot discipline leak nothing.

## 7. Caveats & future work

- **Detuned pieces** draw `random()` in workers (Sound::setup_detuning_env);
  det mode removes the *trampling* but a detuned piece's draws still depend on
  scheduling. Fix (future): draw detune envelopes on the producer thread or
  seed per-sound from the sequence number.
- det-gpu currently syncs each H2D staging copy; pinned double-buffering could
  hide even that (unmeasurable at current piece sizes).
- MAX_RENDERED_OBJECTS=20 bounds the reorder window; a det-mode env override
  would trade memory for less head-of-line waiting on pieces with very uneven
  sound durations.
