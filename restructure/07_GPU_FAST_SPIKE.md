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
