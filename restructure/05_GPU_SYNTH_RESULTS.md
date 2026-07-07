# GPU Synthesis — What a Bit-Exact CUDA Implementation Can (and Can't) Do

*Measurement-driven follow-up to the "is 100× feasible?" question. Machine:
i7-13705H (20 logical cores) + RTX 4050 Laptop (2560 CUDA cores, 6 GB, **96-bit
/ ~192 GB/s** bus, weak FP64). g++ 12.2, nvcc 13.0. Written 2026-07-07.*

> **Headline:** Under a **bit-exact** constraint, GPU acceleration of DISSCO is
> structurally capped. The only stage that parallelizes bit-exactly is the sine
> **map**; the stages that dominate cost (phase/tremolo/vibrato pre-pass,
> loudness, reverb comb) are **sequential scans** that cannot be reproduced
> bit-for-bit in parallel. We made the CUDA map **2–4× faster (bit-exact)** by
> removing wasteful transfer/alloc, but **end-to-end it is Amdahl-capped to
> ~break-even** because the sine map is a minority of even a synthesis-dominated
> render. **100× is not reachable bit-exactly.**

## 1. The efficiency fix (bit-exact, committed)

The pre-existing `renderMapCuda` was a textbook transfer-bound anti-pattern:
4× `cudaMalloc`+`cudaFree` **per partial**, and it round-tripped the amplitude
array to the GPU and back **unchanged** (the worklet's `ampOut[s]=amplitude[s]`
is an identity). Fixed with zero math change (so bit-exact):

- **persistent `thread_local` device buffers**, grown on demand (no per-call malloc;
  `thread_local` ⇒ concurrent LASS workers don't race, no lock);
- **dropped the redundant amp round-trip** — the host fills the amp channel with a
  `memcpy` (identity), so only amp+phase go H2D and only wave comes D2H.

### Isolated map throughput (`bench_map`, incl. transfer)

| buffer | before | **after** | parity |
|---|---|---|---|
| 1 s (44.1 k)   | 0.35× (GPU *lost*) | **1.34×** | 0 LSB |
| 30 s (1.32 M)  | 1.74× | **3.37×** | 0 LSB |
| 300 s (13.2 M) | 2.57× | **4.05×** | 0 LSB |

The realistic ~1 s partial flipped from **losing** to **1.34×**, all bit-exact.

## 2. The ceiling: pure on-device kernel (no transfer)

Pure sine-map kernel, buffers already resident (the fusion ceiling):

```
13.2 M samples in 4.34 ms = 3047 Msample/s = 17.7× vs CPU single core (172 Msample/s)
```

**~18× is the bit-exact synthesis ceiling on this GPU**, and it is **FP64-bound,
not bandwidth-bound**: 3047 Msample/s × 12 B/sample ≈ **37 GB/s**, only ~1/5 of the
192 GB/s the bus can sustain. The limiter is the **double-precision `sin`** (needed
to match glibc bit-for-bit); consumer GPUs are weak at FP64. A `float sinf` map
would be far faster but would **not** be bit-exact.

## 3. Why end-to-end doesn't move (Amdahl)

Whole-song wall, seed-fixed, 1 thread, 3 reps (min), **all md5-identical**:

| piece | cpu default | portable serial | portable **cuda** |
|---|---|---|---|
| tutorial (50 snd × ~6 part) | 9.96 s | 9.47 s | ~9.9 s |
| bench_ab (6 snd × 24 part)  | 4.71 s | **4.50 s** | 4.94 s (**slower**) |

Even on `bench_ab`, which is **synthesis-dominated** —

| stage | share |
|---|---|
| partial synth | **65 %** |
| loudness | 28 % |
| sound reverb | 4 % |
| spatialize | 2 % |

— CUDA still loses, because the GPU only accelerates the **sine map**, and the map
is a *minority of "partial synth."* The bulk of synthesis is the **host sequential
pre-pass** that builds `phase[]`/`amp[]`: three phase accumulators
(`freq_phase = pmod(freq_phase + f/sr)`, tremolo, vibrato) — mod-1 recurrences
whose floating-point result is **order-dependent** and therefore not bit-exactly
parallelizable (a parallel prefix-sum grows the accumulator large and loses the
precision the sequential mod-1 keeps). CUDA pays transfer on top of the *same*
pre-pass, so it is ~break-even to slightly slower.

`portable serial` being a hair faster than the original (bit-exact) is a
free by-product of materializing the map into a tight vectorizable loop.

## 4. The other stages, for the record

- **Loudness (28 %)** — parallel across time samples in principle, but uses
  `pow`/`log` in **double**; libdevice≠glibc at the ULP level and the result feeds
  a double interpolator → **not bit-exact**. Off the table under the constraint.
- **Reverb comb (the 44 %/32 % bottleneck)** — `L[n] = w[n-D] + coef·L[n-1]`; the
  low-pass in the feedback chains **every consecutive sample** → a true sequential
  scan. Bit-exact parallelism exists only **across** the 6 combs × channels ×
  sounds, which the 20-core CPU pool already exploits (and now beats the GPU).
  The all-pass alone *is* residue-class parallel (feedback only from `n-D`), but it
  is 1 of 5 sub-filters. This is why the shipped `FilterGPU.cu` had to approximate
  and drifted −59 dBFS. **Bit-exact GPU reverb is a dead end for speedup.**

## 5. Verdict on 100×

- **Bit-exact:** no. Ceiling ≈ **18× on-device for the one parallel stage**, ≈4×
  with transfer, ≈1× end-to-end. The cost is dominated by sequential scans
  (pre-pass, loudness, reverb) that are unparallelizable *without changing the
  bits*, and the one map that parallelizes is FP64-`sin`-bound.
- **100× would require** (a) accepting the original's own **≤ few-LSB budget** so
  the scans (phase, loudness, reverb) can go parallel on-device with `float`
  transcendentals, (b) a **fully fused** on-device pipeline (one H2D, one D2H) on a
  **large** piece, and (c) realistically a **wider-bus desktop GPU** (700–1000 GB/s)
  — on this 96-bit laptop part even the tolerance path lands ~40–70×, not 100×.

## 6. What changed

`LASS/portable/PartialRendererCuda.cu`, `PortableSynth.h/.cpp`, `bench_map.cpp`.
Default path untouched (env unset ⇒ original `Partial::render`). Verified:
`test_partial_parity` 0 LSB; whole-song md5 identical to golden on two pieces
(`12d2ff21…`, `c34e46…`); `parity_regression.sh` green.
