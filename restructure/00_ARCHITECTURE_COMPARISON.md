# DISSCO ↔ Viskores: Architecture Comparison

*Part of the DISSCO hardware-acceleration restructuring effort. Companion to
[`01_RESTRUCTURE_PLAN.md`](01_RESTRUCTURE_PLAN.md). Written 2026-07-06.*

This document is the "know your two codebases" prerequisite for the
restructuring. It is descriptive, not prescriptive; design decisions live in the
plan.

---

## 1. What each project *is*

| | **DISSCO / LASS** | **Viskores** |
|---|---|---|
| Domain | Algorithmic music composition + additive sound synthesis | Scientific-visualization algorithms for many-core / GPU |
| Core deliverable | `cmod` renders a `.dissco` (XML) to a 24-bit AIFF | A C++ library of data-parallel filters over mesh/field data |
| Scale | ~199 source files, single build target family | ~2300 source files, template-heavy header library |
| Parallel model | `pthread` pool at the *Sound* granularity (LASS/src/Score.cpp) + a one-off hand-written CUDA reverb (LASS/CUDA/FilterGPU.cu) | Portable *worklet* dispatch over pluggable **device adapters** (Serial / TBB / OpenMP / CUDA / Kokkos) |
| Portability abstraction | **None** — CUDA code is bespoke, duplicated, and separate from the CPU path | **First-class** — one worklet body compiles to every backend |
| Precision | `float` samples; transcendentals computed in `double` then rounded | Templated on value type |

The single most important structural difference: **Viskores separates *what to
compute* (a worklet) from *where it runs* (a device adapter), and DISSCO does
not.** DISSCO's CPU synthesis loop and its CUDA reverb are two independent
implementations of overlapping ideas. That duplication is exactly the failure
mode Viskores was built to eliminate.

---

## 2. Viskores: the four transferable concepts

Distilled from `viskores/cont/{ArrayHandle,Invoker,DeviceAdapter}.h`,
`viskores/worklet/WorkletMapField.h`, and `examples/hello_worklet/HelloWorklet.cxx`.

1. **`ArrayHandle<T>` — a device-agnostic array.**
   Owns data in the *control* (host) environment and lends read/write *portals*
   to the *execution* (device) environment. Host↔device transfer is lazy and
   implicit. The algorithm never names a device pointer.

2. **Worklet — a per-element functor.**
   A `struct` deriving from a worklet base (e.g. `WorkletMapField`) with:
   - a `ControlSignature` declaring abstract inputs/outputs (`FieldIn`, `FieldOut`),
   - an `ExecutionSignature` mapping them to `operator()` arguments,
   - a `VISKORES_EXEC operator()` — the body, compiled for *both* host and
     device (`__host__ __device__` under CUDA).
   The body is written **once**.

3. **`Invoker` / `DeviceAdapterAlgorithm` — the dispatch layer.**
   `Invoke(worklet, inArray, outArray)` schedules the worklet across the active
   device adapter. The adapter supplies the primitives every backend must
   implement: `Schedule` (parallel-for), `ScanExclusive`/`ScanInclusive`
   (prefix sum), `Reduce`, `Sort`, `Copy`. Selecting a backend is a runtime
   flag (`--viskores-device=cuda`); the worklet is untouched.

4. **`Filter` — the coarse-grained operation.**
   Consumes a `DataSet`, resolves field types, invokes one or more worklets,
   and packages the result. This is the "user-facing verb" layer.

The pattern in one line:
> *Independent per-element work becomes a worklet; sequential dependencies
> become scans/reduces; both are dispatched by a device adapter so the same
> source runs Serial or CUDA.*

---

## 3. DISSCO / LASS: the synthesis pipeline

Data flow for `cmod project.dissco`:

```
.dissco (XML)
   │  CMOD/src/piece-experimental.cpp  (deterministic, seeded)
   ▼
Event tree (stochastic composition)  ── CMOD/src/Event.cpp
   │  builds
   ▼
LASS Score  ── holds vector<Sound*>              LASS/src/Score.cpp
   │  pthread producer/consumer pool renders Sounds concurrently
   ▼
Sound::render      ── N Partials → composite → filter → reverb → spatialize
   │                                                     LASS/src/Sound.cpp
   ▼
Partial::render    ── THE HOT LOOP: per-sample additive synthesis
   │                                                   LASS/src/Partial.cpp
   ▼
MultiTrack composite → score reverb → clip → AuWriter::write → 24-bit AIFF
                                                LASS/src/{Score,AuWriter}.cpp
```

### 3.1 The hot loop (`Partial::render`, LASS/src/Partial.cpp:216)

For each of `numSamplesToRender` samples it computes:

```
sample[s] = amplitude[s] * sin(2π · phase[s])
```

where, per sample, it advances a set of iterators and accumulators. Classifying
each by its **data-dependency structure** is the whole game:

| Quantity | Structure | Parallelizable? |
|---|---|---|
| envelope values (freq, wave_shape, loudness, tremolo/vibrato amp & rate, phase offset, detune) | `Constant` → identity; `Linear/Exp/Spline` → `value += delta` **scan** | yes (map or prefix-scan) |
| `tremolo_phase`, `vibrato_phase`, `freq_phase` | running sum `phase = pmod(phase + rate/sr)` — **scan** | yes (prefix-scan + fract) |
| `amplitude`, `frequency` combine | pure **map** of the above | yes |
| `sin` sample | pure **map** | yes (embarrassingly) |
| amp/freq **transients** | per-sample **RNG + state machine** | **no** (sequential, non-deterministic) |

**Key finding:** with transients disabled (`AMPTRANS_*`/`FREQTRANS_*` amp = 0,
the default), the loop contains *no* sequential dependency that is not a prefix
scan, and consumes *no* RNG output that affects the result. It is a textbook
map+scan pipeline — precisely Viskores' wheelhouse.

### 3.2 Buffers

`SoundSample` (LASS/src/SoundSample.h) wraps a `vector<float>` with `getData()`
raw access and a `composite(other, startTime)` add. `Track` = wave + amp
`SoundSample`; `MultiTrack` = per-channel `Track`s. This is DISSCO's
`ArrayHandle` analog — but host-only, with no execution-environment portal and
no lazy transfer.

### 3.3 Existing parallelism & its ceiling

- **CPU:** `Score` renders Sounds on a `pthread` pool (LASS/src/Score.cpp:103).
  Measured speedup on the 30 s tutorial: 12 s (1 thread) → 7 s (64 threads),
  only **~1.7×** — the single composite thread and per-Sound granularity cap it.
- **GPU:** `do_reverb_SoundSample_GPU` (LASS/CUDA/FilterGPU.cu) is a standalone
  CUDA reverb. It is *not* wired into the portability story: it duplicates the
  filter math, hard-codes launch config, and shares no abstraction with the CPU
  path. It is a proof the hardware works here, not a design to extend.

---

## 4. The mapping (why Viskores' shape fits DISSCO)

| Viskores concept | DISSCO today | Restructuring target |
|---|---|---|
| `ArrayHandle<T>` | `SoundSample`/`vector<float>` (host only) | a thin device-agnostic buffer with host + device residency |
| Worklet `operator()` | body of the `Partial::render` loop | a `PartialSynthWorklet` written once, `__host__ __device__` |
| `Invoker` + device adapter | `pthread` pool *ad hoc*; CUDA reverb *ad hoc* | a `Dispatch`/adapter seam: `Serial` and `Cuda` backends |
| `ScanExclusive` | inline `phase += …` accumulation | explicit prefix-scan for phase & linear envelopes |
| `Filter` | `Sound::render` / `Reverb` | unchanged orchestration calling the new dispatch |

**Non-goal:** we are *not* vendoring Viskores into DISSCO. Viskores is a
5-million-line mesh/field toolkit; DISSCO synthesizes 1-D audio. We are
importing Viskores' *architecture* (worklet + device adapter + scan), realized
as a small, purpose-built layer, not its code.

---

## 5. Consequences for verification (measured, not assumed)

Established empirically on this machine (see plan §6 and the parity harness):

- Single-thread + fixed seed ⇒ **bit-exact** run-to-run (0 LSB).
- 64-thread ⇒ **≤ 3 LSB / RMS −152 dBFS** run-to-run drift, purely from FP
  summation-order in the threaded composite. Bit-exact reproduction of the
  original is therefore impossible *for the original itself* under threading.

This dictates the parity contract in the plan: **serial backend must match the
single-thread golden bit-exactly; parallel/GPU backends must stay within the
original's own multi-thread budget.**
