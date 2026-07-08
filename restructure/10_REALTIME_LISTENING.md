# Real-Time Listening (streaming render) — design & results

*Goal (2026-07-08, from the project owner): render faster than playback so a
listener — ultimately in Jupyter — hears the piece while it renders. This is
the actual end-goal the DISSCO_hpc fork's notebooks approximate
([`09`](09_HPC_FORK_ANALYSIS.md) §2.4); multi-node is a throughput backend for
when one node can't keep up, not the core mechanism.*

## 1. Why this is now primarily a *streaming* problem, not a speed problem

Measured margins on this laptop (same-session A/B, [`07`](07_GPU_FAST_SPIKE.md)):

| engine, bench_1min (60 s piece) | wall | × real-time |
|---|---|---|
| gpu-fast v3 @20t | 2.7 s | **22×** |
| gpu-fast v3 @1t | 10.9 s | 5.5× |
| CPU det @20t | 19.7 s | 3.0× |

Every configuration already renders faster than playback. What is missing is
*progressive emission*: today all audio appears only after the render, the
score-level reverb, and the **global** `CHANNEL_ANTICLIP` pass — the same
global pass that [`04`](04_INSITU_FEASIBILITY.md) identified as the streaming
blocker and that the fork works around with per-slice anticlip (gain
discontinuities at slice boundaries; [`09`](09_HPC_FORK_ANALYSIS.md) §4.2 —
confirmed structurally in this recon: nothing in their merge glue applies a
global pass).

## 2. The frontier-flush design

In the deterministic composite ([`06`](06_DETERMINISTIC_COMPOSITE.md)), the
composite thread commits rendered sounds in insertion order, and every sound
writes only forward from its start (`data[start + i] += src[i]`). Therefore:

> **A time window [a, b) of the score is FINAL as soon as every sound with
> startTime < b has been committed.**

Since det commits in sequence order, the uncommitted set is exactly
`seq ∈ [nextCommitSeq, added)`, so the *frontier* is
`min(startTime over that range)` — an O(pending≤~220) scan per commit. Flushing
starts once CMOD has finished adding sounds (`doneGettingSoundObjects`; on the
bench piece composition completes in ~5 s, which bounds time-to-first-audio),
after which the frontier only advances.

**Emission (`LASS_STREAM=<path>`, requires `LASS_COMPOSITE=det`):** after each
commit, full windows (`LASS_STREAM_WINDOW` s, default 0.5) below the frontier
are appended to `<path>` as a tiny header (`DSTR`, rate, channels) + raw
float32 interleaved frames, hard-clamped to ±1. At drain end the remaining
tail is flushed. A Python/Jupyter client tails the file and plays windows as
they arrive.

## 3. The two-output contract (what the stream is and is not)

| output | content | guarantee |
|---|---|---|
| stream (`LASS_STREAM`) | pre-(score-reverb), pre-anticlip mix, clamped to ±1 | deterministic per config; PREVIEW quality — where the final file gets globally compressed, the preview may clip (tutorial peak: +1.56 dB) |
| final AIFF | unchanged pipeline (global reverb + `CHANNEL_ANTICLIP`) | **byte-identical to non-streaming render** — streaming reads the committed prefix and never mutates state |

This is the honest resolution of the anticlip problem: the global pass is
*unstreamable by definition* (it needs the future); the preview admits that
instead of faking it per-slice. Pieces with score-level `reverbObj` also miss
that reverb in the preview (none of the goldens use one; documented).

## 4. Relationship to the HPC track

Same seam, larger scale: the fork's slice-players poll per-slice files; our
client tails one deterministic stream. When a piece/config drops below 1×
real-time, [`09`](09_HPC_FORK_ANALYSIS.md) M1–M3 (rank seam + deterministic
merge) slot *behind* this interface — nodes fill disjoint time regions and the
frontier rule generalizes (window final when all ranks' sounds below b are
merged). Nothing in this design needs rework for that step.

## 5. Results (2026-07-08, first implementation)

Implementation: `LASS_STREAM=<path>` (+`LASS_STREAM_WINDOW`, default 0.5 s) in
`Score` (det CPU composite required; det-gpu falls outside v1 since the score
is device-resident — a windowed range-fetch is the follow-up). Client:
`restructure/realtime/listen.py` (measurement everywhere; `--play` via
sounddevice; `tail_windows()` importable for Jupyter autoplay cells).

| check | result |
|---|---|
| authoritative AIFF with streaming ON | **byte-identical** (tutorial `c41910aa`, gpu-fast det golden) |
| stream completeness | frames == final AIFF exactly (bench_10min: 24,300,514 = 551.03 s) |
| stream determinism | 2 runs byte-identical (`bab2f984…`) @20t |
| default paths | parity regression green |
| time-to-first-audio (tutorial @20t, gpu-fast) | **3.3 s** from process launch |
| sustained margin (bench_10min @20t, gpu-fast) | **29→36× real-time** while streaming |

### Real-piece validation + Jupyter client (2026-07-09)

`7_final.dissco` — the only real composition in the repo, previously
un-renderable (the palette segfault, then one corrupt closing tag
`0.5/Size>`; repaired with a one-character fix, original backed up) — now:

| check | result |
|---|---|
| renders | 300 s piece: **8.7 s @20t (det) = 34× real-time**; 48.7 s @1t |
| det determinism on real music | `a1c41b39…` identical: 2 runs AND 1t vs 20t |
| gpu-fast eligibility | **0/302 sounds fall back** — but only 1.05× faster: the piece is reverb/spatialize-bound (REV_Simple per sound), the exact "reverb-heavy repertoire" case of [`09`](09_HPC_FORK_ANALYSIS.md); use plain det here |
| gpu-fast vs det accuracy | −61.6 dBFS RMS (phase-drift class, as documented) |
| live streaming | full piece streamed in 8.4 s wall, **31–32× real-time margin**, AIFF byte-identical |

`RealtimeListen.ipynb` (this directory) is the end-user client: configure a
piece, run one cell, audio plays in chunked autoplay widgets while the render
runs (fork-style UX on the deterministic stream). Executed headless end-to-end
(TTFA 2.77 s on the tutorial); browser playback needs a real audio device.
`parity_regression.sh` now asserts the streaming contract (AIFF unchanged +
stream byte-identical across runs) — 18 checks total.

The composition phase bounds TTFA (flushing starts at `doneAddingSounds`);
pieces with slow event trees would benefit from a chronological-frontier
relaxation (flush below the earliest *possible* future start) — future work,
as is the det-gpu range-fetch and the M1–M3 multi-node backend behind the same
stream interface.
