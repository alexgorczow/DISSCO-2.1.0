# DISSCO_hpc Fork Analysis — and the Best-of-Both-Worlds Merge Plan

*Written 2026-07-08 from a survey of `/home/alexg/projects/DISSCO_hpc`
(GitHub fork, "V8exp" core). Companion to
[`08_MODERNIZATION_ROADMAP.md`](08_MODERNIZATION_ROADMAP.md) — this document
adds the multi-node track. Claims marked ⚠ are inferred from code reading and
need a verification run before being relied on.*

---

## 1. What DISSCO_hpc is

A fork of a **pre-2.1.0** DISSCO whose contribution is **multi-node scale-out
plus an interactive front-end**, built almost entirely *around* the core
rather than in it:

| Piece | What it is |
|---|---|
| `DISSCO2_V8exp/` | the forked core (CMOD ~12.8k lines vs 2.1.0's ~19.2k; LASS heavily diverged — Score.cpp differs from stock 2.1.0 by ~1150 lines) |
| `delta_multinode.slurm`, `delta_jupyter.slurm`, `delta_test_single.slurm` | SLURM templates for NCSA Delta (4 nodes × 32 CPUs pattern) |
| `RUN_NOTEBOOK_V6_master/` | Jupyter notebooks: setup → run → **live audio playback** |
| `get_mpirank_runcmd.pl`, `do_paste01.py`, `bundler2.pl`, ffmpeg | rank launch + post-hoc audio merge glue |
| `*_GUIDE.md` (Delta, HPC, Jupyter, XML, libraries) | genuinely good deployment/user docs — better operability docs than our repo has |
| `MERGING_V2.1.0_ANALYSIS.md` | a prior analysis of merging 2.1.0 features *into* the fork (est. 2–4 weeks) |

**Folklore correction:** despite the name and the `DISSCO_MPI.odt` heritage,
there is **no MPI in the code** (no `mpi.h`, no `MPI_Init`) and **no CUDA**.
Parallelism is process-level: independent `cmod` runs per node, "file-based
MPI simulation" via CLI rank/size arguments.

## 2. How the multi-node architecture actually works

### 2.1 The rank seam
`cmod <path> <rank> <size> <maponly> <dobrew>` (Main.cpp). Globals
`global_rank`/`global_size` thread through CMOD.

### 2.2 The partition rule — and its determinism trick (the fork's best idea)
`Bottom.cpp:214–219`: **every rank runs the full composition**, drawing the
full RNG sequence identically; a sound is *rendered* only when
`sound_count % size == rank`. The author's comment states the reason
explicitly: per-sound RNG draws depend on which sounds are processed, so
skipping composition work would desynchronize the stochastic sequence —
processing everything and rendering an interleaved subset keeps every rank's
composition **bit-identical to a single-node run**. This is the same
contract-first instinct as our det-composite work, discovered independently.
(The trailing comment — "maybe remove this if to make it repeatable?" — shows
the author was still unsure; the trick as implemented is the right call.)

### 2.3 Map-only mode
`maponly` runs CMOD composition without synthesis — used by the notebooks for
fast iteration. Side effect we care about: it produces the full sound list
(start, duration, per-sound parameters) **for free**, i.e. a per-sound cost
estimator before any rendering starts.

### 2.4 Time-slices + interactive mode
`Score.cpp` in the fork also has *time*-slices (`slices_done`,
`wait4slice`): sounds render in temporal windows so notebooks can play audio
while later material still synthesizes. Two distinct "slice" meanings —
rank-slicing (across nodes) and time-slicing (for interactivity).

### 2.5 The merge step
Per-rank audio is combined post-hoc by scripts (`do_paste01.py`,
`bundler2.pl`, ffmpeg). This is the architecture's weak joint — see §4.2.

## 3. Comparison with the restructured 2.1.0

| Axis | DISSCO_hpc (V8exp) | DISSCO-2.1.0 restructure |
|---|---|---|
| Parallel philosophy | scale **out** — farm sounds across nodes, merge after | scale **deep** — pthread pool + GPU within one process |
| Measured/expected multiplier | ~N nodes (near-linear until merge; ⚠ unmeasured by us) | **21×** per node vs true 1-core original ([`02`](02_RESULTS.md)); synthesis-heavy pieces a further **7–10×** on a GPU node (v3) |
| Core codebase | pre-2.1.0, diverged; no Markov, no notation export, no ModParser | 2.1.0 + additive portable layer |
| Perf fixes | none — no `getValue` memoization (our 4×), leaks unfixed | 4× bit-exact; ~1.6 GB leaks fixed |
| Determinism | composition: **solved** (§2.2 trick); render+merge: ⚠ doubtful (§4.2) | md5-stable across thread counts ([`06`](06_DETERMINISTIC_COMPOSITE.md)) |
| GPU | none | opt-in portable/CUDA + **gpu-fast v3: bench_1min 10.2×@1t / 7.4×@20t, md5-deterministic** ([`07`](07_GPU_FAST_SPIKE.md) §Step 4) |
| Operability (SLURM, guides, notebooks) | **ahead of us** | 2-line README; developer-facing harness only |
| Eng hygiene | vendored xerces-3.1.4, precompiled binaries in repo, web-upload commits | goldens, parity harness, decision docs |

**The headline:** each fork holds exactly the multiplier the other lacks.
Their node farming multiplies whatever a node can do; our restructure makes a
node 4–21× faster. Neither subsumes the other.

## 4. Discoveries & red flags

### 4.1 The per-node engine is the slow one
Every node-hour DISSCO_hpc rents runs the pre-memoization, leaky core: on a
reverb-heavy piece that is up to **~21× slower per node** than the
restructured engine at 20 threads (and ~4× slower even single-threaded,
bit-exact). Scale-out multiplies an inefficient base.

### 4.2 Render + merge determinism/parity is unproven, and two hazards are visible ⚠
1. **FP order in the paste.** Mixing per-rank AIFFs post-hoc reorders the
   floating-point composite sums — exactly the ≤3 LSB class of drift the
   det-composite work eliminated in-process. Changing rank count likely
   changes output bits.
2. **CHANNEL_ANTICLIP is nonlinear and global.** Per
   [`04_INSITU_FEASIBILITY.md`](04_INSITU_FEASIBILITY.md), anticlip is a pass
   over the *whole* composite. If each rank anticlips its partial mix before
   the paste (⚠ unverified — depends on what the scripts feed ffmpeg), then
   `merge(anticlip(slices)) ≠ anticlip(merge(slices))` **structurally**, not
   just at the LSB level. Slice-count-dependent output would follow.
3. Per-rank AIFFs are themselves rendered with the fork's threaded pool —
   pre-det-composite, so each rank adds its own ≤3 LSB thread jitter.

Verification is cheap: render a small piece at size=1 and size=4, merge,
byte-compare. Do this before trusting any parity claim either way.

### 4.3 Static round-robin will show tail imbalance at node granularity
We measured per-Sound tail starvation capping thread scaling at ~5.2×/20t
([`03`](03_PARALLELIZATION_ANALYSIS.md)). `sound_count % size` is the same
granularity at node scope: one long sound can idle three nodes. At node
prices, imbalance costs node-hours, not idle cores. (§6 M3 has the fix — and
map-only mode already provides the cost estimates it needs.)

### 4.4 Hygiene
Precompiled binaries and vendored xerces-c-3.1.4 in-tree; git history includes
"Add files via upload" (GitHub web UI). The *code* should be treated as an
artifact to harvest ideas and thin layers from — not a base to build on.

## 5. Assets worth harvesting (regardless of merge direction)

1. **The rank-seam design** (§2.2) — small, deterministic-by-construction at
   the composition level, proven on Delta and Expanse.
2. **SLURM templates + deployment guides** — directly reusable with path/
   binary updates; our repo has nothing equivalent.
3. **Jupyter notebook front-end** — setup/run/playback; a real UX for
   composers, which nothing in our tree offers.
4. **Map-only mode** — both as a user feature (fast iteration) and as the
   free per-sound cost model for load balancing (§6 M3).
5. **Time-sliced interactive rendering** — future in-situ/preview feature;
   note it intersects the anticlip constraint from [`04`](04_INSITU_FEASIBILITY.md).

## 6. Best of both worlds — the merge plan

**Direction decision (reverses `MERGING_V2.1.0_ANALYSIS.md`):** port the
**thin HPC layer onto the restructured 2.1.0**, not 2.1.0 features onto the
fork. Rationale: the HPC-specific code is small and peripheral (Main.cpp args,
one `%`-rule at the Bottom→Sound seam, external glue), while the core
improvements (memoization, leak fixes, det-composite, portable/gpu-fast) are
large, entangled, and hard to back-port onto a diverged pre-2.1.0 core.

Phases, each with an acceptance criterion in the house style:

- **M1 — rank seam in 2.1.0.** Add rank/size (env vars or args, consistent
  with the `LASS_*` surface / the contract tracker of 08 §2.6), implement the
  full-composition + `sound_count % size == rank` render rule, port map-only
  mode. → verify: size=1 output byte-identical to current default golden;
  each rank's particel/composition log identical across ranks.
- **M2 — deterministic merge.** Replace script/ffmpeg pasting with a `cmod`
  merge mode: read per-rank raw (pre-anticlip ⚠) slices, composite with the
  det-composite ordering discipline, run CHANNEL_ANTICLIP **once, globally**,
  write final AIFF. → verify: **md5 identical for size ∈ {1, 2, 4}** — the
  slice-count-independence contract, added to `parity_regression.sh`.
- **M3 — load balance.** Swap round-robin for longest-processing-time
  assignment using map-only duration×partial-count estimates. Note this is
  free under the §2.2 trick: assignment changes only the *render set*, never
  the RNG sequence, so determinism survives any partition. → verify: M2
  md5 contract still holds under LPT partition; node-utilization spread
  measured on the 30-min bench piece.
- **M4 — SLURM + notebooks on the new binary.** Update templates for the
  2.1.0 binary and Delta's current modules; GPU-node variant with
  `LASS_PIPELINE=gpu-fast` optional per 07's ladder. Notebooks: keep or adapt
  the `INFO,...` stdout contract their parsers scrape (⚠ check before
  changing log lines). → verify: single-node and 4-node Delta runs of the
  bench piece complete and satisfy the M2 contract.
- **M5 — docs fold-in.** Merge their deployment guides into ours (Tier 0.2 /
  3.5 of 08); retire the fork with a pointer.

**Expected multiplier (honest arithmetic):** per-node 21× (reverb-heavy,
20t) × ~4 nodes ≈ **~80× vs the true single-core original** — reaching the
"100×" neighborhood that [`05`](05_GPU_SYNTH_RESULTS.md) proved unreachable
on-node — via the boring end of the parallelism spectrum. *(Conservative
post-v3: synthesis-heavy repertoire on GPU nodes stacks gpu-fast's measured
7.4×@20t on top of the node count — see [`07`](07_GPU_FAST_SPIKE.md) §Step 4.)* Synthesis-heavy
pieces on GPU nodes (gpu-fast ladder complete) stack further. Costs to
subtract: merge step (serial, but cheap — one pass), tail imbalance (M3
mitigates), CMOD composition repeated per rank (super-linear in sound count
per the bench-piece findings — fine at 360 sounds, measure before assuming at
10k).

## 7. What NOT to port

- **The V8exp core itself** — pre-2.1.0, diverged, superseded.
- **`dobrew` re-run loop and collaborative mode** — niche; revisit only on a
  user request.
- **Vendored xerces / precompiled binaries** — use system deps per
  [`08`](08_MODERNIZATION_ROADMAP.md) Tier 1.2.
- **Script-based audio pasting** — replaced by M2 precisely because it is
  where parity dies.

## 8. Where this slots into the roadmap

The merge track is **parallel to Tier 2** of
[`08_MODERNIZATION_ROADMAP.md`](08_MODERNIZATION_ROADMAP.md): M1–M2 depend on
nothing in Tier 2. *(Written pre-v3: "deliver more end-to-end speedup than the
entire gpu-fast ladder" — since `83d5a05` the ladder has DELIVERED 7.4×@20t on
synthesis-heavy pieces, so the claim now holds only for reverb-heavy
repertoire, where gpu-fast leaves reverb on the CPU. The deeper point stands:
node count MULTIPLIES whatever the node does, including v3's gains.)* M4's
GPU-node variant is where the two tracks meet. Suggested
sequencing if multi-node matters to the user base: 08 Tier 0–1 → **M1–M2** →
08 Tier 2 and M3–M5 as demand dictates.

Open questions before M-work starts:
1. Does the fork's sliced output actually differ across rank counts? (§4.2 —
   one afternoon to test, settles the urgency of M2.)
2. Where exactly do their scripts apply anticlip/normalization? (Read
   `bundler2.pl`/ffmpeg invocations end-to-end.)
3. Is multi-node demand real for current users, or was the fork built for a
   one-off Expanse/Delta campaign? (Determines whether this track outranks
   Tier 2 at all.)
