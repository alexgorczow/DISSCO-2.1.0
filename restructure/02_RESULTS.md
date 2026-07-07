# Results — DISSCO Viskores-Inspired Restructuring

*Companion to [`01_RESTRUCTURE_PLAN.md`](01_RESTRUCTURE_PLAN.md). Measurements on:
RTX 4050 Laptop GPU, g++ 12.2, nvcc 13.0, CUDA 13.0, 30 s tutorial (seed 42).*

## 1. Parity — the central result

The restructured synthesis kernel reproduces the original **exactly** where the
original is itself deterministic, and stays within the original's own numerical
noise elsewhere.

| Checkpoint | What was compared | Result |
|---|---|---|
| **C0** | original @1 thread vs itself | bit-exact (0 LSB) — defines the golden |
| **C0** | original @64 threads vs itself | ≤3 LSB / −152 dBFS (FP composite order) |
| **C1** | portable Serial single-partial vs `Partial::render` (4 cases) | **bit-exact, 0 diffs** |
| **C2** | portable Serial whole-song vs golden | **byte-identical** (md5 match) |
| **C4** | portable **CUDA** whole-song vs golden | **byte-identical** (0 LSB) |
| — | portable CUDA vs itself (run-to-run) | bit-exact (deterministic) |

C1 covers constant params **and** the linear-interpolator scan path (glissando +
amplitude envelope). C2/C4 route all 50 sounds of the tutorial through the new
kernel via `LASS_PORTABLE_BACKEND={serial,cuda}` at 1 thread.

**Second, independent piece.** A different stochastic composition (tutorial
project, seed 777 → a different Score, ~different waveform) rendered three ways
at 1 thread — original (env unset), portable Serial, portable CUDA — produced
**one identical md5** (`99b3cde0…`) across all three. Parity is not an artifact
of the one tutorial.

**Why CUDA came out bit-exact (better than the ≤4 LSB budget):** the phase/
amplitude pre-pass runs on the host identically to Serial; only the map
`amp·sin(2π·phase)` runs on the GPU. Any libdevice-vs-glibc `sin(double)`
difference is ≤1 ULP in double — far below `float` precision — so the
`double→float` rounding of the sample collapses it to the identical `float`, and
then to the identical 24-bit PCM word. This is empirical (0/953 720 samples
differed), not guaranteed for every possible input; the ≤4 LSB budget (S2)
remains the formal contract.

**With the seam disabled** (env unset) the output is byte-identical to the
pre-change golden — the default build is provably unchanged (S4). The total edit
to original LASS/CMOD source is **3 lines in `Sound.cpp`**.

## 2. Performance — honest characterization

### 2.1 Raw map-worklet throughput (`LASS/portable/bench_map`)

Isolated `sample = amp·sin(2π·phase)` over one partial's worth of samples:

| Partial length | samples | CPU map | GPU map (incl. transfer) | speedup | map parity |
|---|---|---|---|---|---|
| 1 s   | 44 100     | 0.25 ms | 0.71 ms | 0.35× | bit-exact |
| 30 s  | 1 323 000  | 7.53 ms | 4.32 ms | 1.74× | bit-exact |
| 300 s | 13 230 000 | 75.0 ms | 29.1 ms | 2.57× | bit-exact |

CPU: ~176 M samples/s single core. GPU wins only once the buffer is large enough
to amortize the per-call `cudaMalloc`+H2D+D2H (four `float` arrays). At a
realistic per-partial length (≈1 s) the transfer dominates and the GPU *loses*.

### 2.2 Whole-song wall clock (30 s tutorial)

| backend | threads | time |
|---|---|---|
| original | 1 | 12 s |
| original | 64 | 7 s |
| portable Serial | 1 | 13 s |
| portable CUDA | 1 | 12 s |

No end-to-end speedup from the GPU map — **and the profile says why**: each of
the 50 sounds runs Loudness + a SOUND-level **Reverb** (comb + all-pass filters)
+ Spatialize. Synthesis (the sine map we offloaded) is a *small* fraction of the
render; **reverb is the bottleneck**. Offloading the map therefore cannot move
the whole-song number (Amdahl), and per-partial offload adds transfer overhead
on short partials.

### 2.3 Diagnosis → where the acceleration actually lives

1. **Reverb, not synthesis, is the hot spot** for reverb-heavy pieces. The repo
   already contains an (unwired) CUDA reverb, `LASS/CUDA/FilterGPU.cu`. The
   highest-value next step is to route reverb through the *same* device-adapter
   seam this work established, so it is portable and parity-tested like the map.
2. **Batch partials.** Offload one buffer per *Sound* (all partials
   concatenated) instead of per partial, to amortize transfer — the crossover in
   §2.1 shows this flips the GPU from 0.35× to >1×.
3. **Keep the pre-pass on-device** (phase via a scan) once batching lands, to
   remove the remaining host serial section.

None of these change the parity contract; they are throughput work on top of a
proven-correct, portable foundation.

## 3. What transferred from Viskores (retro)

| Viskores idea | Realized here as | Value delivered |
|---|---|---|
| `ArrayHandle` | `SynthArray<T>` + host/device buffers in the `.cu` | device-agnostic data |
| Worklet (`VISKORES_EXEC operator()`) | `SampleMapWorklet` (header-inline, `LASS_EXEC`) | **one body, two backends, bit-exact** |
| Device adapter (`Schedule`) | `DeviceAdapterSerial` + CUDA grid launch | swap the backend, not the math |
| Filter / Invoke | `renderPartial` + `renderPartialDispatch` seam | opt-in, original untouched |
| Scan (`ScanExclusive`) | the sequential phase/tremolo/vibrato pre-pass | identified; on-device scan is future work |

The core Viskores lesson — *separate the per-element body from the dispatch* —
transferred cleanly and is what made a single worklet run bit-exactly on CPU and
GPU. The parts Viskores would express as scans (phase accumulation) are
identified and currently run as a faithful host pre-pass; promoting them to an
on-device scan is the natural Tier-6 continuation.

## 4. Scope decisions (honest)

- **C3 (parallel-CPU backend) was de-scoped**, not skipped silently. The map is
  independent per sample, so an OpenMP dispatch would be trivially bit-exact —
  but the map is *not* the bottleneck (§2.2), and `Score` already parallelizes at
  the Sound granularity via pthreads. Adding an OpenMP map tier would demonstrate
  "a third adapter" without moving any real number, which CLAUDE.md §2
  (simplicity, nothing speculative) argues against. The microbenchmark already
  proves the worklet parallelizes losslessly.
- **Transients / random-wave / per-partial-reverb partials** are out of scope for
  the kernel and **fall back** to the original `Partial::render` (correctness
  preserved, F3). They are non-deterministic in the original anyway.
