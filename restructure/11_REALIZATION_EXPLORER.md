# Realization Explorer — mapping the possibility space of a stochastic piece

*Design sketch, 2026-07-09. Working name: **Seedscape** (open to better).
Status: NOT started; targeted as the fall-2026 build. Companion docs:
[`09`](09_HPC_FORK_ANALYSIS.md) (whose dormant M-track this replaces as "the
HPC aspect"), [`10`](10_REALTIME_LISTENING.md) (the streaming client this
extends). Numbers marked ~est are laptop-derived estimates pending the P0
baseline run on Delta.*

---

## 1. The idea

DISSCO is a *stochastic* composition system: one `.dissco` + one seed = one
**realization**; a different seed = a different, equally "authentic"
realization of the same piece. Historically a composer heard a handful of
realizations, because renders were slow. The acceleration work made a
realization cost ~10 s ([`10`](10_REALTIME_LISTENING.md): 7_final, 300 s,
~9–14 s @20t on a laptop), and a Delta allocation (~10k A100 GPU-hours) is
available. That makes a previously unaskable question askable:

> **What does the realization space of a piece look like?** How much does it
> vary, along which perceptual axes, are there distinct families of outcomes,
> which realizations are outliers, and can we predict outcome character from
> the composition's own parameters?

This is HPC in service of the music: the cluster as a *creative instrument*
(explore, characterize, audition), not as a render accelerator. It needs
**none** of the rank-slicing machinery of [`09`](09_HPC_FORK_ANALYSIS.md) —
every realization is an independent whole-piece render in a SLURM job array.

Evidence the group already wants a tiny version of this: stock cmod's
interactive "how many times do you want to run (1–10)" loop — multiple
realizations per seed session is an existing workflow, currently capped at 10
and unanalyzed.

## 2. Pipeline overview

```
                        ┌─ A. GENERATE (SLURM job array; CPU-heavy) ──────────┐
 .dissco + seeds 1..N ─▶│ cmod, LASS_COMPOSITE=det, seed injected             │
                        │ → FLAC audio + .particel + md5 manifest             │
                        └──────────────┬──────────────────────────────────────┘
                                       ▼
                        ┌─ B. DESCRIBE ────────────────────────────────────────┐
                        │ tier 1 (CPU): DSP descriptors — loudness/RMS curves, │
                        │   spectral centroid/flux/rolloff, onset density,     │
                        │   stereo width, dynamics range                       │
                        │ tier 2 (A100): neural embeddings — CLAP/MERT/OpenL3, │
                        │   windowed (≈10 s) + aggregated per realization      │
                        │ tier 3 (free): composition-native features parsed    │
                        │   from .particel (sound count, density, per-sound    │
                        │   duration/partial distributions)                    │
                        └──────────────┬──────────────────────────────────────┘
                                       ▼
                        ┌─ C. ANALYZE ─────────────────────────────────────────┐
                        │ UMAP/PCA map of embedding space; HDBSCAN clusters;   │
                        │ medoids + outliers; descriptor-distribution stats;   │
                        │ variance decomposition (how seed-sensitive is the    │
                        │ piece, per descriptor?)                              │
                        └──────────────┬──────────────────────────────────────┘
                                       ▼
                        ┌─ D. MODEL (A100) ────────────────────────────────────┐
                        │ v1: params → descriptors regression (tier-3 → tier-1/2│
                        │   targets; report R² honestly)                        │
                        │ v2: preference model from composer labels via active │
                        │   learning (audition medoids → label → retrain)      │
                        └──────────────┬──────────────────────────────────────┘
                                       ▼
                        ┌─ E. EXPLORE (laptop) ────────────────────────────────┐
                        │ notebook client (extends RealtimeListen.ipynb):      │
                        │ 2-D realization map → click → stream audio;          │
                        │ filter by descriptors; "more like this one"          │
                        └──────────────────────────────────────────────────────┘
```

## 3. What already exists (why this is buildable in a semester)

| Asset | Provides |
|---|---|
| det composite ([`06`](06_DETERMINISTIC_COMPOSITE.md)) | every realization md5-reproducible → manifest is both dedup and integrity check; any rendering bug is detectable by re-render |
| harness seed-injection pattern | the job-array render script is 80% written (`parity_regression.sh` render() function) |
| `RealtimeListen.ipynb` ([`10`](10_REALTIME_LISTENING.md)) | the audition client — streaming playback at 31× real-time margin |
| `.particel` output | tier-3 features for free: the composition's own parameter record per realization |
| Delta guides + SLURM templates (`DISSCO_GPU_Delta_Guide.md`, HPC fork §5) | build + submission mechanics on Delta |
| gpu-fast v3 ([`07`](07_GPU_FAST_SPIKE.md)) | GPU rendering for synthesis-heavy corpus pieces (see honesty note §5.1) |

## 4. Sizing (~est; P0 replaces these with measurements)

- **Render cost:** 7_final ≈ 9–14 s @20t laptop. Per-Sound parallelism
  saturates ~20 threads, so Delta CPU throughput mode = several concurrent
  renders per node (e.g., 6–8 renders × 16t on a 128-thread Milan node)
  → ~est **0.3–0.5 realizations/s/node** → 10k realizations ≈ a few
  node-hours. Generation is *cheap*.
- **Storage:** 300 s stereo 24-bit ≈ 79 MB AIFF, ~40–50 MB FLAC. 10k ≈
  ~0.5 TB, 100k ≈ ~5 TB. Policy: keep descriptors + embeddings for **all**;
  keep audio only for the representative subset (medoids, outliers, random
  sample) + re-render on demand — determinism makes audio *reconstructable
  from the manifest*, which is the storage escape hatch.
- **A100 hours:** embeddings ~est ≤ 2 h per 100k realizations (30 windows ×
  100k, batched); model training negligible at this data size; P0 baseline
  ~10 h. **The allocation is not the constraint** — storage and composer
  label time are.

## 5. Honest caveats

1. **Rendering barely uses the GPU for 7_final-class pieces** (reverb/
   spatialize-bound; gpu-fast measured 1.05× there). GPU hours pay in tiers
   B/D (embeddings, training) and in rendering only for synthesis-heavy
   corpus pieces. The doc says this plainly so the allocation story stays
   credible.
2. **The realization space might be boring** — the piece may be robust to
   seed (small perceptual variance). That is a *finding*, not a failure:
   "how seed-sensitive is a DISSCO piece, quantified per descriptor" is
   publishable either way, and composers want to know it.
3. **Corpus is thin**: 7_final is the only real piece in-repo. The bench
   family adds synthetic synthesis-heavy coverage, but the project gets much
   stronger with 2–3 more real pieces — a natural fall-semester coordination
   hook with the group (and with Sever's catalog).
4. **Labels are scarce**: preference learning (D-v2) needs composer
   listening time. Design for few labels: active learning over medoids, not
   random sampling. D-v1 (params → descriptors) needs no labels at all.
5. **Seed is opaque**: one seed feeds all stochastic draws; we characterize
   the *output* space, not a clean per-parameter sensitivity. Per-parameter
   ablations (freeze seed, vary one input) are a natural v2 experiment.

## 6. Phases with acceptance criteria

- **P0 — Delta baseline (~10 GPU-hrs, 1 afternoon).** Build the branch on
  Delta; render 7_final (CPU throughput mode + one gpu-fast run on A100);
  verify md5 determinism *on Delta* vs the laptop manifest (same seed —
  glibc/hardware differences may legitimately break cross-machine md5;
  measure, don't assume); record per-realization cost + storage.
  → verify: cost table replaces every ~est in §4.
- **P1 — Generation at 1k seeds.** Job-array script from the harness
  render(); FLAC + .particel + manifest; spot re-render 1% for integrity.
  → verify: 1k realizations, manifest-complete, re-render md5-stable.
- **P2 — Descriptors + first map.** Tier-1 DSP + tier-3 particel features;
  UMAP + HDBSCAN; notebook explorer v0 (map → click → stream medoids).
  → verify: a composer (or Alex) auditions 5 medoids + 5 outliers and the
  map's neighborhoods are perceptually coherent (informal but real).
- **P3 — Embeddings + scale + model v1.** CLAP/MERT on A100; scale 10k–100k;
  params → descriptor regression with honest R² and error analysis.
  → verify: held-out R² reported per descriptor; map quality vs tier-1-only
  compared.
- **P4 — Preference loop + writeup.** Active-learning labeling with composer;
  optional ICMC / ISMIR-LBD / NCSA-poster writeup.
  → verify: model ranks a held-out labeled set better than descriptor
  heuristics; draft exists.

Each phase is independently demo-able; stop-anywhere is a feature.

## 7. Non-goals

- No rank-slicing / deterministic-merge machinery (09 stays dormant).
- No generative model of audio (we analyze DISSCO's outputs; we don't
  replace DISSCO).
- No new GUI — the notebook client is the interface; LASSIE integration is
  the modernization track's call, later, if ever.
