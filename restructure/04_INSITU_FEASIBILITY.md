# In-Situ Rendering — Feasibility Investigation

Status: investigation only (no code changes). Date: 2026-07-07.
Scope: is "in-situ rendering" feasible for DISSCO/LASS, and is it worth doing?

---

## 0. What "in-situ rendering" could mean here

The term is loaded — it comes from the HPC/visualization world (viskores/VTK-m,
Catalyst, Ascent), which is exactly the lineage this restructure draws on. In that
world *in-situ* means: **process/emit results where the data already lives, as it is
produced, instead of staging everything to a big buffer/disk and post-processing.**
The motivation is to avoid the I/O + memory blow-up of the "compute-all-then-dump"
model.

Mapped onto DISSCO, that yields three distinct readings. They are NOT the same
project and have very different value:

- **(A) In-situ *audio* emit (streaming).** Emit final PCM incrementally as time
  windows finish, instead of buffering the whole piece then writing once. Payoff:
  bounded memory, progressive/low-latency output. This is the literal in-situ analog.
- **(B) In-situ *pipeline* (on-device / in-block post-processing).** Keep each
  rendered block's data *in place* (on the GPU / in the worker) and run the
  downstream stages — loudness, reverb, spatialize, composite, encode — there,
  instead of copying every Sound's `MultiTrack` back to a single CPU composite
  thread. This is the viskores-native reading and it targets the measured
  bottleneck. See [[dissco-perf-findings]].
- **(C) In-situ *visualization*.** Produce visual/analytic artifacts (spectrograms,
  per-partial amplitude fields, spatial trajectories, loudness heatmaps) *during*
  synthesis, from the in-memory buffers, rather than re-analyzing the AIFF later.
  This is the literal viskores use case, but for an audio tool it is greenfield.

Verdict up front: **(B) is feasible and aligned with the perf goal; (A) is partially
feasible but blocked from bit-parity by the default clipping mode; (C) is technically
feasible but low-value / speculative.** Details below.

---

## 1. The current pipeline is the exact opposite of in-situ

DISSCO today is a textbook "stage-everything-then-write" batch renderer:

1. Worker threads render each `Sound` to its own `MultiTrack`
   ([Score.cpp:104](../LASS/src/Score.cpp#L104)).
2. A **single** composite thread mixes every rendered `Sound` into one whole-piece
   `scoreMultiTrack` at its start-time offset
   ([Score.cpp:218](../LASS/src/Score.cpp#L218)).
3. After all threads join, a **global reverb pass** rebuilds the entire buffer
   ([Score.cpp:274-282](../LASS/src/Score.cpp#L274-L282)).
4. A **global clipping pass** runs over the whole buffer
   ([Score.cpp:286-288](../LASS/src/Score.cpp#L286-L288)).
5. `AuWriter::write` iterates all channels and writes to AIFF — it already chunks at
   16K frames ([AuWriter.cpp:178-194](../LASS/src/AuWriter.cpp#L178-L194)), but only
   *after* steps 2-4 have finished the whole piece.

So the write is already streamed; **everything upstream of it is not.** The whole
piece lives in RAM (twice, during reverb) before a single final sample is emitted.

**Memory cost of the batch model** (`m_sample_type = float`, Track = wave+amp = 2
buffers/track, [Types.h:41](../LASS/src/Types.h#L41), [Track.h:132-133](../LASS/src/Track.h#L132-L133)):

| piece            | `scoreMultiTrack` | peak (reverb makes a 2nd copy) |
|------------------|------------------:|-------------------------------:|
| stereo, 5 min    |            212 MB |                         423 MB |
| stereo, 10 min   |            423 MB |                         847 MB |
| 8-ch, 10 min     |          1 693 MB |                       3 387 MB |
| 16-ch, 10 min    |          3 387 MB |                       6 774 MB |

Plus every in-flight per-`Sound` `MultiTrack`. For big multichannel pieces this is
the real motivation for in-situ (A).

---

## 2. Reading (A): streaming audio emit — the blocker is CHANNEL_ANTICLIP

To emit a finished time window `[t0, t1)` and free it, three things must hold. Two
are workable; one is a hard blocker under the default config.

### 2a. Out-of-order, arbitrary-offset composite — *workable*
Sounds finish in any order and composite at arbitrary start times. To finalize window
`[t0, t1)` you need *all sounds that overlap it* to be done. This needs a
**completion watermark**: render/emit in start-time order and track "all sounds
starting before T are composited," then flush `[.., T)`. Sounds are already added in
score order; this is a bookkeeping layer over the existing producer/consumer, not a
rewrite. Effort: moderate.

### 2b. Global reverb — *workable*
The CPU reverb (the correct, default path since commit a8590fe) is a **forward,
per-sample IIR**: `out[i] = do_reverb(in[i], i/N, percentReverb)` over comb+all-pass
delay lines with running state ([Reverb.cpp:440-442](../LASS/src/Reverb.cpp#L440-L442)).
A forward IIR is inherently streamable — feed it the composited stream in time order
and carry the delay-line state across windows. The only coupling is the `i/N`
time-normalization for the reverb-percent envelope, which needs the final length `N`
up front; DISSCO already tracks that as `scoreEndTime`. So reverb can move from a
batch pass to a streaming filter with the same output. Effort: moderate; parity is
provable sample-by-sample.

### 2c. Global normalization (CHANNEL_ANTICLIP) — **the hard blocker**
CMOD sets the clipping mode to **`CHANNEL_ANTICLIP`**
([Utilities.cpp:60](../CMOD/src/Utilities.cpp#L60)), not `NONE`. `CHANNEL_ANTICLIP`
scans the **entire** composited signal for the peak, then compresses `[-6dB, peak]`
into `[-6dB, 0)` ([Score.cpp:556-637](../LASS/src/Score.cpp#L556-L637)). **Every
output sample's final value depends on the whole-piece peak**, which is unknown until
the last sample is computed. This is a genuine two-pass, whole-signal dependency —
fundamentally incompatible with single-pass streaming emit.

Consequences (honest):
- You **cannot** stream bit-identical default output in one pass. Options, each with a
  cost:
  - **Two-pass streaming**: pass 1 finds the peak while spilling the composited signal
    to a scratch file; pass 2 re-reads, compresses, emits. Preserves parity, trades
    RAM for disk — still O(whole piece) staged, just not in RAM. Gets the memory win,
    loses the "progressive output" win.
  - **Behavior change**: switch to a pointwise mode (`CLIP`/`NONE`, already
    streamable) or a look-ahead limiter. Single-pass and low-memory, but **not
    bit-parity** with today's output — a musical/authorial decision, not just an
    engineering one.
- Only when the mode is `NONE` or `CLIP` is true single-pass streaming both
  low-memory *and* parity-exact.

**Reading (A) verdict:** feasible as a *two-pass, low-RAM* renderer with full parity
(watermark composite + streaming reverb + spill-to-disk peak pass), or as a
*single-pass progressive* renderer only if the project accepts a non-`CHANNEL_ANTICLIP`
normalization. It reduces peak memory and enables progressive output; it does **not**
speed up throughput (that is reverb/loudness/composite-bound — see [[dissco-perf-findings]]).

---

## 3. Reading (B): in-situ pipeline (on-device post-processing) — best aligned

The measured ceiling is **not** synthesis; it is the **single composite thread +
per-Sound GPU-reverb serialization** (thread scaling plateaus ~2.5×; aggregate reverb
CPU balloons 5.5s→69.8s from 1→20 threads while wall stays flat — [[dissco-perf-findings]]).
The batch model forces every rendered `Sound` back through one CPU funnel.

An in-situ pipeline keeps each block's data *where it was produced* and runs the
downstream stages there:
- The portable layer already exposes per-partial buffers as device-agnostic
  `SynthArray`/`ArrayHandle` with a `renderPartialDispatch` seam
  ([[dissco-viskores-restructure]]). Loudness, reverb, spatialize, and composite are
  all per-sample maps/scans — expressible as the same worklet pattern.
- Doing loudness+reverb+spatialize+composite **on-device per Sound** (or per batch of
  Sounds) amortizes the H2D/D2H transfer that currently dominates the GPU reverb path,
  and moves the composite off the single serialized CPU thread onto a device
  reduction. This is precisely the "NEXT LEVER" already recorded in
  [[dissco-viskores-restructure]]: *route reverb through the device-adapter seam and
  batch per-Sound.*

**Reading (B) verdict:** feasible, and the only one of the three that attacks the
actual performance goal. It is the natural continuation of the existing restructure.
Parity is tractable because each stage is already proven bit-exact-capable through the
worklet seam (C1/C4 results). Effort: high, but incremental — one stage at a time
behind the existing opt-in env flag.

---

## 4. Reading (C): in-situ visualization — feasible, low priority

DISSCO has **no** data-visualization consumer today: no FFT/spectrogram, no plotting,
no image/SVG output (grep across CMOD/LASS finds none; the only "visual" output is
LilyPond score notation via particel, which is unrelated). viskores *can* render
(it ships a rendering module), and the portable layer already surfaces the exact
buffers you'd visualize. So coupling live visuals to synthesis is technically
feasible.

But: it solves no stated problem (the goal is *acceleration with parity*), it adds a
heavy new dependency surface, and its audience (a live "what is the synthesizer
doing" display, or synthesis-debugging heatmaps) is speculative. **Feasible but not
recommended** unless a concrete need appears — e.g., debugging per-partial energy or
spatial trajectories, where a quick offline Python analysis of the AIFF is anyway
cheaper than an in-situ pipeline.

---

## 5. Recommendation

1. **If "in-situ rendering" means performance** (most likely, given the goal and the
   viskores framing): pursue **reading (B)**. It is the already-identified next lever,
   it targets the real bottleneck, and it reuses the proven worklet/parity machinery.
   Start by routing the reverb through the device-adapter seam and batching per-Sound
   to kill the transfer thrash + composite serialization.
2. **If it means memory / progressive output**: reading (A) is feasible as a
   **two-pass low-RAM renderer** with full parity (watermark composite + streaming
   reverb + spill-to-disk peak). True single-pass progressive output requires
   abandoning `CHANNEL_ANTICLIP` — a musical decision to raise with the author, not a
   silent change.
3. **Visualization (C)**: park it unless a concrete debugging/authoring need is named.

### Minimal experiment to de-risk (A)
Add an opt-in "streaming composite" path behind an env flag that (i) renders in
start-time order, (ii) maintains a completion watermark, (iii) runs the reverb as a
streaming filter, and (iv) writes finished windows via the existing chunked
`AuWriter`. Gate it to `cmm_ ∈ {NONE, CLIP}` and diff against the batch path with
`restructure/harness/aiff_diff.py` — expect bit-exact. That single prototype proves or
kills the streaming claim without touching the default path.

See [[dissco-viskores-restructure]], [[dissco-perf-findings]], [[dissco-determinism]].
