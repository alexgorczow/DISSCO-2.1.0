# DISSCO Hardware-Acceleration Restructuring — Research-Grade Plan

*Companion to [`00_ARCHITECTURE_COMPARISON.md`](00_ARCHITECTURE_COMPARISON.md).
Written 2026-07-06. Status: living document; checkpoints updated as tiers land.*

## 0. Thesis

DISSCO's audio synthesis (LASS) is a map+scan pipeline wearing a scalar
`for`-loop's clothes. Viskores demonstrates the discipline that turns such a
pipeline into portable, hardware-accelerated code: **write the per-element body
once, express sequential dependencies as scans, and dispatch through a device
adapter.** We will refactor LASS's synthesis core along that discipline —
*without editing the original files* — and prove with a rigorous parity harness
that the new code reproduces the original to within the original's own numerical
noise floor, while enabling a CUDA backend.

## 1. Success criteria (verifiable, up front)

Per CLAUDE.md §4 (goal-driven execution), success is defined *before* coding:

- **S1 — Serial parity (bit-exact).** The restructured synthesis kernel, running
  on the Serial backend, reproduces the original single-threaded golden AIFF
  **bit-for-bit (0 LSB)** for a designated no-transient test piece. If a
  deliberate algorithmic reordering is introduced, the deviation is documented
  and bounded (see S3).
- **S2 — Parallel parity (budget-bounded).** The restructured kernel on a
  multi-threaded and/or CUDA backend matches the golden to within the original's
  own measured multi-thread budget: **max ≤ 4 LSB (24-bit), RMS ≤ −140 dBFS.**
- **S3 — Every tolerance is justified.** Any nonzero deviation is attributed to a
  named cause (FP summation order, transcendental ULP, deliberate scan
  reordering) and shown to be inaudible (≪ the −138 dBFS 24-bit quantization
  floor).
- **S4 — The original is untouched and still builds/renders.** `git diff` on
  pre-existing LASS/CMOD source files is empty except for the minimal, guarded
  seam needed to *optionally* route through the new path.
- **S5 — Reproducibility.** A single script rebuilds, renders, and diffs; results
  are captured in git.

Non-criteria (explicit scope guard): we are **not** required to accelerate
transient/RNG synthesis, to beat the CPU wall-clock on tiny pieces, or to port
CMOD's stochastic composition. Speedup is a *goal*, correctness parity is the
*gate*.

## 2. Design overview

A new, self-contained layer under **`LASS/portable/`** (original `LASS/src/`
untouched):

```
LASS/portable/
  PortableTypes.h        // shared scalar types, VISKORES_EXEC-style macros
  DeviceAdapter.h        // Serial + (optional) Cuda tags; Schedule/Scan/Reduce
  SynthArray.h           // device-agnostic buffer (ArrayHandle analog)
  PartialSynthWorklet.h  // the per-sample body, written ONCE (__host__ __device__)
  PartialRenderer.h/.cpp // orchestration: dispatch worklet, assemble SoundSample
  PartialRenderer.cu     // CUDA backend instantiation (guarded by HAVE_CUDA)
  README.md              // how it maps to viskores, how to build/test
```

### 2.1 The worklet (write-once body)

`PartialSynthWorklet` encapsulates the arithmetic of `Partial.cpp:216-357` for
the **no-transient path**, decomposed by dependency class:

- **Phase pre-pass (scan).** Instantaneous per-sample frequency `f[s]`
  (from envelopes) → `phase[s] = frac(prefix_sum(f/sr))`. Likewise tremolo &
  vibrato phases when their rates are non-constant; identity/closed-form when
  constant (the common case).
- **Sample map.** `sample[s] = amplitude[s] * sin(2π·phase[s])`, with
  `amplitude[s]` the product of loudness·wave_shape·(1+tremolo). Pure map.

Two evaluation modes share the *same* declared interface:

- **`FaithfulSerial`** — replays the original's exact operation order
  (`value += delta`, `phase = pmod(phase + f/sr)`, `sin` in `double`). Target: S1
  bit-exact.
- **`ParallelScan`** — uses adapter `ScanExclusive` + closed-form envelope
  evaluation. Target: S2 budget-bounded. Enables CUDA.

### 2.2 The device adapter (dispatch seam)

`DeviceAdapter.h` defines the minimal primitive set LASS actually needs —
`Schedule(n, functor)`, `ScanInclusive`, `Reduce` — with two implementations:

- `DeviceAdapterSerial` — plain loops; the reference semantics.
- `DeviceAdapterCuda` — `__global__` launches + CUB/thrust scans, compiled only
  when `HAVE_CUDA` (same detection premake4.lua already does).

This is deliberately *smaller* than Viskores' adapter — we implement only the 3
primitives the synthesis kernel uses, not the full mesh/topology surface.

### 2.3 The seam into existing code (S4-minimal)

The original `Partial::render` is left intact. A new free function
`portable::renderPartial(const Partial&, …)` produces the identical
`MultiTrack`. Routing is opt-in behind a runtime flag / env var
(`LASS_PORTABLE_BACKEND=serial|cuda`), so the default build behaves exactly as
before and the original remains the living reference.

## 3. Determinism & parity methodology (the scientific core)

Measured baseline (this machine, RTX 4050, g++ 12.2, 30 s tutorial, seed 42):

| Config | Run-to-run reproducibility |
|---|---|
| 1 thread, fixed seed | **bit-exact (0 LSB)** → the golden |
| 64 threads, fixed seed | max 3 LSB, mean 0.041 LSB, RMS 0.208 LSB (−152 dBFS), 4.05 % samples differ |

Interpretation: 24-bit full-scale = 8 388 608; the quantization noise floor is
~−138 dBFS. The threaded composite's −152 dBFS drift is ~14 dB *below* the floor
— provably inaudible, and it is a property of the **original**, not our changes.

**Parity contract:**
- Serial backend ↔ golden: **byte-identical** (`cmp`), else 0-LSB via the diff
  tool. Any deviation is a bug, not a tolerance.
- Parallel/CUDA backend ↔ golden: within S2 (≤ 4 LSB / ≤ −140 dBFS). Justified by
  S3 as the same class of FP-reorder + transcendental-ULP noise the original
  already exhibits.

**Harness** (`restructure/harness/`): `run_parity.sh` renders golden + candidate,
`aiff_diff.py` computes max/mean/RMS LSB error and dBFS, exits nonzero if the
tier's bound is exceeded. Golden AIFFs + their `.dissco` + md5s are committed.

## 4. Tiered implementation with checkpoints

Each tier is independently verifiable and committed separately (CLAUDE.md §3:
surgical, traceable changes).

- **Tier 0 — Baseline & harness** *(prerequisite; DONE in recon)*
  - Build DISSCO; capture 1-thread golden; measure the 64-thread budget.
  - ✅ Checkpoint C0: golden is bit-exact reproducible; budget = 3 LSB / −152 dBFS.

- **Tier 1 — Portable Serial kernel, standalone parity**
  - Implement `PortableTypes.h`, `SynthArray.h`, `DeviceAdapterSerial`,
    `PartialSynthWorklet` (FaithfulSerial), `PartialRenderer` (serial).
  - A LASS-level unit harness constructs a `Partial` with known params, renders
    it via original `Partial::render` **and** `portable::renderPartial`, and
    diffs the two `SoundSample`s in-process.
  - ✅ **Checkpoint C1:** portable-serial single-partial output is bit-exact
    (0 ULP) vs original `Partial::render` for the no-transient parameter set.

- **Tier 2 — Whole-song serial parity through CMOD**
  - Wire the opt-in seam so `cmod` with `LASS_PORTABLE_BACKEND=serial` renders
    the tutorial through the new kernel.
  - ✅ **Checkpoint C2:** `cmod` (1 thread, seed 42, portable-serial) AIFF is
    byte-identical to the Tier-0 golden.

- **Tier 3 — Parallel scan backend (CPU), budget parity**
  - `ParallelScan` worklet path using an OpenMP/`std::thread` `Schedule` + scan.
  - ✅ **Checkpoint C3:** parallel-CPU output within S2 of the golden; document
    the deviation class.

- **Tier 4 — CUDA backend, budget parity + speedup**
  - `DeviceAdapterCuda` + `PartialRenderer.cu`; the *same* worklet body compiled
    for device. Reuse the CUDA reverb where applicable.
  - ✅ **Checkpoint C4:** CUDA output within S2 of the golden; report wall-clock
    vs 1-thread and 64-thread CPU on a large piece (7_final).

- **Tier 5 — Documentation & synthesis**
  - Results tables, speedup curves, a "what transferred from Viskores" retro,
    and honest limits. Update this file's checkpoints.

## 5. Risk / flaw analysis (red-team of this plan)

Deliberately adversarial, per the goal ("analyze it for flaws"). Each risk has a
mitigation or an explicit accepted-scope note.

- **F1 — Bit-exact serial parity may be unattainable if the iterators carry
  hidden state I haven't traced.** *Mitigation:* Tier 1's in-process single-
  partial diff isolates this immediately, before any CMOD wiring; if 0-ULP
  proves impossible I downgrade S1 to a *stated* tiny bound and document the
  exact op that differs. Confidence is high for `Constant` (identity) and
  `Linear` (`value += delta`) params, which the tutorial uses; exp/spline
  envelopes are the residual risk.
- **F2 — `sin` in `double` on CPU (glibc) vs GPU (libdevice) will never be
  bit-exact.** *Accepted:* this is exactly why S2 (not S1) governs the CUDA
  backend; the ULP difference is inside the budget the original already spends.
- **F3 — Transients (RNG state machine) are unparallelizable and
  non-deterministic.** *Scope guard:* Tier 1-4 target the no-transient path
  (the default, and the tutorial). Transient partials fall back to the original
  `Partial::render` on the Serial backend. This is a correctness-preserving
  fallback, not a silent divergence.
- **F4 — Threaded composite non-determinism could be blamed on our kernel.**
  *Mitigation:* all parity gates render at **1 thread**; the budget in S2 exists
  precisely to bound the parallel backends against the *golden*, and C2 proves
  the serial path is exact, isolating any parallel drift to the parallel adapter.
- **F5 — "Restructure" could balloon into rewriting LASS (violates CLAUDE.md
  §2/§3).** *Mitigation:* the new code is additive and opt-in; the seam into
  existing files is a single guarded call. If a tier needs to edit >~10 lines of
  original code, stop and reconsider.
- **F6 — CUDA may not pay off on small pieces** (Tier-0 showed only 1.7× from 64
  threads). *Reframe:* the deliverable is *portability + parity*; speedup is
  reported honestly on a large piece (7_final), and a slowdown on tiny pieces is
  an expected, documented result, not a failure.
- **F7 — Float summation associativity in the *parallel scan* itself** (phase
  prefix-sum reordered) could exceed budget for very long partials. *Mitigation:*
  measure; if a long partial drifts, use a higher-precision (double) scan
  accumulator on the phase — cheap, and it *tightens* parity.
- **F8 — Build fragility across the CUDA/premake setup.** *Mitigation:* the
  portable layer builds as a standalone library with its own tiny Makefile for
  Tiers 1-3 (no premake/CMOD dependency), so parity is provable even if the full
  DISSCO build is finicky; CMOD wiring (Tier 2/4) is layered on after.

**Residual unknowns to resolve early:** exact FP behavior of exponential/spline
envelope iterators (F1); whether `Loudness::calculate` perturbs partial params
before render (it runs per-Sound and may scale amplitudes — Tier 2 must route it
identically). These are checkpoints, not blockers.

## 6. Checkpoint log

| ID | Description | Status | Evidence |
|---|---|---|---|
| C0 | Baseline built; golden bit-exact; budget measured | ✅ done | 1-thread 0 LSB; 64-thread 3 LSB / −152 dBFS |
| C1 | Portable-serial single-partial bit-exact vs original | ⬜ pending | |
| C2 | Whole-song serial byte-identical to golden | ⬜ pending | |
| C3 | Parallel-CPU within S2 budget | ⬜ pending | |
| C4 | CUDA within S2 budget + speedup on 7_final | ⬜ pending | |
