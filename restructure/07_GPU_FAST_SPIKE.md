# gpu-fast feasibility spike (Tier 3)

*De-risking the budget-relaxed GPU-fused pipeline before building it. The two
scan-parallelized recurrences are the make-or-break: reverb's IIR chain (44% of
render) and the synth phase accumulator. `LASS/portable/bench_scan.cu` measures
both for speed AND accumulated error vs the exact CPU float reference (verbatim
LASS math: LPComb/LowPass/AllPass recurrences, REV_Medium parameters).
Machine: RTX 4050 laptop, CUDA 13.0, i7-13705H. 2026-07-08.*

## Results

### T1 — reverb unit (6 LPComb + AllPass), GPU blocked affine scan

| | 4 s | 30 s |
|---|---|---|
| float-scan vs CPU float: max | **0.50 LSB24** | **0.50 LSB24** |
| float-scan vs CPU float: RMS | −161.9 dBFS | **−165.4 dBFS** |
| CPU 1-core raw math | 2.65 ms (66 Msmpl/s) | 22.3 ms (60 Msmpl/s) |
| GPU naive blocked-thrust | 65 ms (0.04×) | 591 ms (0.04×) |

**Accuracy: GO, with ~25 dB of margin.** Scan reassociation error is
*length-independent* (bounded by the lowpass memory horizon, `a≈0.25–0.30` →
~25-sample influence), max half an LSB at 24 bits. The reverb budget condition
for the 5–15× projection **holds**.

**Speed: the naive implementation is launch/sync-bound, not compute-bound.**
Per-block thrust scans + a host-side carry read per block ⇒ thousands of
launches/syncs. The raw arithmetic is trivial (10 MB traffic/sound-4s ≈ 55 µs
at device bandwidth). A production design must: batch all sounds × 6 combs
concurrently, use single-kernel device-carry scans (decoupled-lookback /
cub::DeviceScan style), and keep carries on-device. The spike deliberately did
not build that — it answers the accuracy question, which was the gate.

### T2 — phase accumulator + sine (440 Hz, A=0.5)

| | 4 s | 30 s |
|---|---|---|
| GPU (double-scan + sinf) vs CPU float chain | −59.2 dBFS | **−41.7 dBFS** |
| CPU **double** chain vs CPU float chain | −59.2 dBFS | **−41.7 dBFS** (identical) |
| GPU vs CPU-double chain | ~0.3 LSB24 | ~0.3 LSB24 |
| speed | 4.2× | **11.7×** |

**Contract finding (the important one):** the "error" of the GPU synth is
byte-for-byte the CPU float chain's own accumulated rounding noise — the GPU
double-scan matches a double-precision sequential chain to ~0.3 LSB, while the
CPU float chain drifts from mathematical truth by −59 dBFS at 4 s and −42 dBFS
at 30 s (2% of amplitude!), growing with duration. Two consequences:

1. A "≤ −140 dBFS vs CPU reference" contract is **unsatisfiable for gpu-fast
   synthesis on long sounds** — not because the GPU is inaccurate but because
   the reference itself is noisy. (The existing bit-exact modes sidestep this
   by *reproducing* the float chain exactly.)
2. The correct gpu-fast quality contract is:
   - **determinism**: md5-stable run-to-run per (piece, seed, GPU, CUDA) — the
     det-composite machinery already provides the ordering discipline;
   - **accuracy**: within budget of the **double-precision reference** (where
     the GPU lands at sub-LSB), i.e. *strictly closer to mathematical truth
     than the CPU chain it replaces*;
   - perceptual note: phase drift of this kind is a sub-µHz frequency
     perturbation — inaudible by construction, which is why 20 years of DISSCO
     output with this drift never sounded wrong.

### Bonus finding — the reverb stage is mostly NOT filter math

Full LASS CPU reverb stage (profile_stages, 2 s stereo sound): **28.6 ms**.
Raw filter arithmetic for the same samples (spike): **~5.3 ms** → **5.4× gap**.
~80% of the reverb stage is overhead *around* the recurrence: per-sample
`Envelope::getValue` calls (O(1)-memoized but still a call+branches per
sample), `constructAmp`'s windowed max, allocations/copies. A **bit-exact CPU
cleanup** of this overhead could recover a large part of what gpu-fast targets
(reverb = 44% of render), at a fraction of the complexity — and it raises the
bar gpu-fast must clear to be worth it.

## Step 2 results — batched single-kernel prototype (`bench_batch.cu`)

One CUDA block per (sound × comb), D-block loop inside the kernel
(`__syncthreads`, zero host round-trips), previous-block L in shared memory,
chunked Hillis-Steele affine scan; allpass = one thread per (sound, residue
class) replaying exact CPU float order; whole batch = one launch per stage.

| batch | CPU 1-core | GPU kernels | GPU e2e (PCIe) | error |
|---|---|---|---|---|
| 16 × 4 s | 23.1 ms | 3.2 ms (**7.2×**) | 5.4 ms (4.3×) | 1.00 LSB24 / −161 dBFS |
| 64 × 4 s | 92.0 ms | 13.0 ms (**7.1×**) | 22.0 ms (4.2×) | 1.00 LSB24 / −163 dBFS |
| 360 × 4 s | 515 ms | 60.8 ms (**8.5×**, 1044 Msmpl/s) | 106 ms (4.8×) | 1.50 LSB24 / −163 dBFS |

Replaces the naive 0.04× with the real number. Honest read:
- Error is stable at scale (−163 dBFS, max 1.5 LSB) — accuracy holds in the
  production architecture too.
- Reverb-on-GPU ≈ **8.5× one core ≈ rough parity with the whole 20-core pool**
  on raw reverb math. PCIe transfers halve it — which is precisely the argument
  for the **fused** pipeline: synth produces the buffers on-device, composite
  consumes them on-device, so the reverb pays no transfer at all.
- The gpu-fast end-to-end case therefore rests on the map-shaped stages
  (synth ~30–60×, loudness ~20–50× projected; phase already measured 12×)
  with reverb riding along at parity but transfer-free, plus the CPU cores
  freed for CMOD event building. On server GPUs (Delta A100s) every one of
  these numbers scales up ~5–8×.

## Step 3 results — gpu-fast v1 SHIPPED (`LASS_PIPELINE=gpu-fast`)

`LASS/portable/GpuFastSound.{h,cu}` + a 3-line seam in `Sound::render`: fused
device **loudness + synthesis** per eligible Sound (the two dominant stages).
Host iterates dynamic variables once into RLE streams (one packed upload);
device expands, runs the 24-band loudness map (float transcendentals, incl. the
reference's `bandGamma[0]` max-write quirk), tremolo/vibrato/carrier **double**
prefix scans, `sinf` synthesis, deterministic in-order partial sum; one D2H;
the filter/reverb/Pan/composite tail is unchanged code. Ineligible sounds
(transients, random wave, detune envelopes, per-partial reverb, non-44.1k,
loudness off) fall back to the untouched CPU path.

| check | result |
|---|---|
| default build+mode | `12d2ff21` — **byte-identical, untouched** |
| determinism (gpu-fast + det composite) | tutorial `9c5b50cd`, bench_1min `e7e04184` — **identical across 1t/8t/20t and run-to-run**, 0 fallbacks |
| accuracy vs bit-exact reference — tutorial (short sounds) | max 4 LSB, RMS **−151.7 dBFS** (composite-budget class) |
| accuracy vs bit-exact reference — bench_1min (4 s sounds) | RMS **−67.6 dBFS** = the CPU float chains' own phase drift (spike T2: −59 dBFS @4 s); the GPU tracks the *double* reference at ~0.3 LSB — reported, not asserted, per the restated contract |
| speed — tutorial | **1.50× @1t**, 0.95× @20t (50 light 10-partial sounds; GPU sections serialize on a mutex) |
| speed — bench_1min (24-partial sounds) | **1.35× @1t, 1.19× @20t** (111.5→82.6 s; 25.0→21.1 s) |

## Step 4 results — gpu-fast v3: **5–15× target HIT**

v1 profiling showed the host dynamic-variable iteration was 69% of gpu-fast
time (213 ms/sound) and the device section 23% (72 ms, inflated by degenerate
RLE uploads — envelope streams change every sample, so "compression" shipped
67 MB/sound). Three fixes, landing together:

1. **Device-side envelope evaluation** (the real rung-1): new
   `Envelope::exportDeviceSegments()` exports the exact per-segment step
   structure the iterator would walk; `evalSegKernel` computes closed forms on
   the GPU (linear `vFrom + j·delta`; the exponential interpolator's
   `y1+(y2−y1)(1−e^{αj/steps})/(1−e^α)` formula with its 0→0.0001 substitutions
   and α=±3, in `powf`). Host iteration eliminated for envelope streams;
   uploads shrink from megabytes to a few segment descriptors.
2. **Constant-DV shortcut**: `Constant` streams emit one RLE run without the
   176k-step iteration (phase/frequency/detuning collapse).
3. **Batched scans**: 3 `thrust::scan_by_key` calls (transform-iterator keys)
   replace 72 per-partial device scans per sound.

| bench_1min (360 sounds, 24 partials) | CPU (det) | gpu-fast v3 | speedup |
|---|---|---|---|
| @1t | 111.5 s | **10.9 s** | **10.2×** |
| @20t | 19.7 s | **2.7 s** | **7.4×** |

- per-sound: pre-pass 213→7.4 ms, device 72→7.3 ms; reverb (31%) and
  spatialize (29%) are now the dominant CPU stages — the next frontier.
- determinism: `9127d8b1` identical across 1t/20t/run-to-run (tutorial:
  `c41910aa`, also 3-way identical). **gpu-fast goldens version with the
  implementation** — v3 md5s supersede v1's (closed-form float envelopes).
- accuracy vs bit-exact reference: tutorial −77.1 dBFS (float `powf` envelope
  class), bench −67.2 dBFS (unchanged: the CPU reference's own phase drift
  dominates). Defaults untouched: 12/12 regression, tutorial `12d2ff21`.
- build gotcha for posterity: premake only re-archives `liblass.a` when a
  `.cpp` changes — after editing only `.cu` files, `rm lib/liblass.a` first
  (the "v2 had no effect" mystery was a stale archive).

### v1 limitations → the remaining speed ladder
*(status 2026-07-08 / v3: #1 **done** — see Step 4 above; #2 largely mooted by
the 10× smaller device section, residual only on light-sound pieces; #3 and #4
remain open — #3 is now the dominant on-node lever.)*
1. ✅ Host still iterates every dynamic variable per sample (the old pre-pass
   cost) — moving envelope evaluation on-device is the next big step.
   **Done in v3.**
2. ◐ One global GPU mutex (one sound in flight): per-thread streams/arenas or
   sound batching would un-serialize the 20-worker case. **Largely mooted by
   v3; fold residual into #3's batching point.**
3. Reverb still CPU per-sound; the step-2 batched kernels want a Score-level
   batching point to pay off. **Open — #1 remaining lever (reverb 31% +
   spatialize 29% of residual).**
4. Loudness kernel is `float` (contract-compliant); a double variant would
   pull bench-piece accuracy toward the tutorial's −151 dBFS at some FP64 cost.
   **Open.**

## Step 5 — stream pool + dup-reverb (the "batching point" rungs)

Two changes landed together (`LASS_GPUFAST_STREAMS`, default 2):
1. **Arena+stream pool** replaces the global GPU mutex: N arenas, each with its
   own CUDA stream; all copies/kernels/thrust scans stream-scoped. Per-sound
   math unchanged ⇒ gpu-fast md5s BIT-UNCHANGED (tutorial `c41910aa`,
   bench `9127d8b1`).
2. **Reverb-once-duplicate**: gpu-fast builds every channel as a memcpy of the
   same mono mix and `do_reverb_MultiTrack` resets filters between tracks, so
   channel reverbs are bit-identical — compute one, deep-copy (exact).

Same-day A/B @20t: **7_final 1.05× → 1.89×** (8.5→4.5 s, ≈65× real-time);
bench_1min → **8.62×** (2.9 s). 3 streams measured slightly worse than 2 on
this 4 GB GPU (VRAM pressure) — default stays 2.

### Discovery during verification: 7_final composition instability
The A/B md5s flagged it; bisection proved it PRE-EXISTS (HEAD reproduces) and
is COMPOSITION-side: fixed seed, identical 6425-draw Random stream (LD_PRELOAD
spy), yet different frequencies chosen when heap layout shifts (load /
LD_PRELOAD / ASLR all flip it; ≥3 stable-ish outcomes). Some CMOD build
decision consumes an allocation address — root cause open (suspects: pointer-
keyed container iteration in the Select/CURRENT_CHILD_NUM/Markov chain).
Tutorial + bench det goldens are stress-verified stable; 7_final's golden is
withdrawn in the manifest. This is a CMOD correctness bug worth its own hunt.

## Verdict & revised plan

1. **Accuracy gate: PASSED** (reverb scan −165 dBFS, length-independent).
   The 5–15× projection survives on the accuracy axis.
2. **Quality contract restated** (see T2): determinism md5 per config +
   sub-LSB vs double reference. Diff-vs-float-reference is the wrong yardstick
   and would misreport the GPU as "wrong" when it is more accurate.
3. **Next steps, reordered by information-per-effort:**
   a. Quantify + fix the bit-exact reverb-stage overhead (5.4× gap) — cheap,
      bit-exact, immediately lowers total render time AND sharpens the
      baseline gpu-fast must beat.
   b. Prototype the batched single-kernel scan (all sounds × combs, device
      carries) to replace the 0.04× naive number with a real one.
   c. Only then commit to the full fused pipeline architecture.
