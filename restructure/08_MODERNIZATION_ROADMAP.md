# DISSCO Modernization Roadmap

*Compiled 2026-07-08. Consolidates (a) the unmined viskores concepts from the
architecture comparison ([`00`](00_ARCHITECTURE_COMPARISON.md)), (b) the
gpu-fast v1 speed ladder ([`07`](07_GPU_FAST_SPIKE.md) §"v1 limitations"), and
(c) a software-engineering survey of the repo (tests / CI / docs / licensing,
2026-07-08). Nothing here is started. Items are ranked in tiers of importance;
within a tier, order is by value-per-effort.*

*See also [`09_HPC_FORK_ANALYSIS.md`](09_HPC_FORK_ANALYSIS.md) — the
multi-node (DISSCO_hpc merge) track, which runs parallel to Tier 2 and may
outrank it for reverb-heavy repertoire.*

> **Status sync (2026-07-08, gpu-fast v3 `83d5a05` — landed after this doc was
> compiled):** the 5–15× target was HIT (bench_1min 10.2× @1t / 7.4× @20t,
> [`07`](07_GPU_FAST_SPIKE.md) §Step 4). Consequences for Tier 2: **2.1 is
> done** (device envelope evaluation), **2.3 is largely mooted** (device
> section 72→7.3 ms/sound; @20t is now a win on real pieces), **2.4 is
> partially done** (`b8360cc`), 2.2 exists ad-hoc, and **2.5 (+ spatialize)
> is now the dominant remaining on-node lever** (reverb 31% + spatialize 29%
> of gpu-fast residual per-sound CPU). Per-item notes inline below.

---

## Survey findings the ranking is based on

Measured / observed, not assumed:

- **No CI of any kind.** No `.github/`, no GitLab CI. The parity harness
  (`restructure/harness/parity_regression.sh`, PASS/FAIL + exit codes) and the
  worklet unit test (`LASS/portable/test_partial_parity`) exist but nothing
  runs them automatically. Historical cost of this gap: the GPU-reverb default
  shipped **audibly wrong (−59 dBFS)** for years with no golden comparison to
  catch it.
- **Zero unit tests in the legacy core** (LASS/CMOD/LASSIE, ~80k lines,
  ~20 years). All test infrastructure is from the 2026 restructure.
- **README.md is 2 lines.** No top-level LICENSE (per-file GPL-2+ headers
  exist), no CHANGELOG; versioning is done by renaming the folder.
- **User docs are binary Office files from ~2005** (`Documents/*.doc`,
  `*.odt`) — unversionable, undiffable.
- **Build: Premake4 (dead project) + C++11**, nvcc bolted on via a separate
  `LASS/portable/Makefile`. Clean build emits 27 warnings, **2 of which are
  real latent bugs** (below).
- **836 raw `new`, zero smart pointers** in the core; leak-hunting has been
  manual valgrind archaeology (`valgrind_full.log`).
- gpu-fast v1 measured limits ([`07`](07_GPU_FAST_SPIKE.md) §Step 3): host
  pre-pass still iterates every dynamic variable per sample; one global GPU
  mutex (0.95× **regression** at 20 threads on the tutorial); reverb still
  CPU per-sound while the batched prototype hit 8.5×/1044 Msmpl/s.
  **[Superseded by v3 (`83d5a05`): pre-pass 213→7.4 ms/sound (device envelope
  eval), device section 72→7.3 ms, bench_1min 10.2×@1t / 7.4×@20t. The mutex
  penalty persists only on light-sound pieces (tutorial 0.95×). Reverb-on-CPU
  remains, now 31% of the residual.]**
- Reverb stage is **80% overhead around the filter math** (28.6 ms stage vs
  5.3 ms raw arithmetic = 5.4× gap) — a bit-exact CPU win waiting.
  **[Partially claimed (`b8360cc`): getValue sweep-resume + loop hoists +
  queue wrap → 16.7 ms (1.71×, bit-exact). Remaining gap ≈3.2×
  (constructAmp, allocations, Track plumbing).]**

## Summary table

| # | Item | Effort | Depends on |
|---|------|--------|------------|
| **Tier 0 — hygiene** | | | |
| 0.1 | Top-level LICENSE | XS | — |
| 0.2 | Real README | XS | — |
| 0.3 | CHANGELOG with honest release framing | XS | — |
| 0.4 | Fix the two real warning bugs | S | — |
| **Tier 1 — foundations** | | | |
| 1.1 | CI running the existing harness | S | 0.x nice-to-have |
| 1.2 | CMake + C++17 migration | M | — |
| 1.3 | Unit tests below whole-song level | M | — (doctest works on C++11) |
| **Tier 2 — performance architecture** | | | |
| 2.1 | ✅ **DONE** (`83d5a05`) Implicit envelope arrays (device-side eval) | ~~M–L~~ | — |
| 2.2 | ◐ Scan-with-operator device primitive (ad-hoc exists: `scan_by_key`, bench_batch) | M | — |
| 2.3 | ◐ largely mooted by v3 (device 7.3 ms/sound); residual = light-piece mutex | S | fold into 2.5 |
| 2.4 | ◐ partial (`b8360cc`, 1.71×); remaining ≈3.2× gap | S–M | 1.3 (tests first) |
| 2.5 | Score-level reverb batching (fuse step-2 kernel) — **now the #1 on-node lever** (+ spatialize 29%) | L | 2.2 |
| 2.6 | Contract-aware runtime device tracker | S–M | — |
| **Tier 3 — robustness & breadth** | | | |
| 3.1 | Device-parametrized test matrix | S | 1.1, 1.3 |
| 3.2 | Sanitizer CI leg (ASan/LSan) | S | 1.1 |
| 3.3 | OpenMP device adapter (within-sound maps) | M | 2.6 helps |
| 3.4 | float/double worklet type-list instantiation | S | 1.2 |
| 3.5 | User docs out of .doc/.odt into markdown | S–M | — |
| **Tier 4 — opportunistic** | | | |
| 4.1 | Performance-regression goldens | S | 1.1 |
| 4.2 | Doxygen build + publish | S | 1.1 |
| 4.3 | Kokkos backend | L | only if non-NVIDIA hardware appears |

Effort: XS ≤ 1 h · S ≤ 1 day · M ≤ 1 week · L = multi-week.

---

## Tier 0 — hygiene (hours; no architecture; do in one PR)

### 0.1 Top-level LICENSE
Per-file headers already declare GPL-2+ (e.g. `LASS/src/Partial.cpp`,
© 2005 Sever Tipei). Add the matching top-level `LICENSE` file. Five minutes;
legal clarity forever.

### 0.2 Real README
Replace the 2-line README with ~15 honest lines: what DISSCO is, build
one-liner (`make config=release`), the env-var/contract table
(`LASS_PORTABLE_BACKEND`, `LASS_PIPELINE`, `LASS_REVERB`, `LASS_COMPOSITE`),
links to `restructure/` docs and the tutorial.

### 0.3 CHANGELOG
`CHANGELOG.md` for the current version bump with the accurate framing:
headline = **faster (4× single-thread bit-exact, 21× vs true single-core
original), leak-free (~1.6 GB fixed), deterministic multi-threaded renders**;
GPU pipelines filed as *experimental opt-in* (gpu-fast v1: 1.2–1.5×). Do not
oversell the GPU story — the measurements don't support more.

### 0.4 Fix the two real warning bugs
From the 27-warning clean-build survey:
- `LASSIE/src/IEvent.h:548` — `virtual bool deletePartial(...)` has **no
  return statement** (UB if the value is used; emits 13 of the 27 warnings).
- `LASS/src/InterpolatorIterator.cpp:195–197` — **use-after-free** (pointer
  read after `delete`).

Optional cosmetics while there (separate commits): `default:` in
`BiQuadFilter`'s switch, `<ext/hash_map>` → `<unordered_map>` in
`StandardHeaders.h`, drop `-std=c++11` from the one C file. These three are
noise-only; skip if churn-averse.

---

## Tier 1 — foundations (the enablers; highest value)

### 1.1 CI running the existing harness ← single biggest gap
GitHub Actions workflow, CPU-only build, running what already exists:
`parity_regression.sh` serial checks (determinism, `portable-serial ==
default` bit-exact, det-composite md5-stable across 1t/8t) +
`test_partial_parity`. GPU legs stay manual/self-hosted. Converts the goldens
from "artifacts of one machine" into a regression gate. **Justification is in
this project's own history:** −59 dBFS shipped for years because nothing
compared output to a golden.

### 1.2 CMake + C++17 migration
Replace Premake4 (7 targets in `premake4.lua` → generated `make/*.make`).
Wins: one build graph compiling the shared worklet headers under g++ *and*
nvcc; `-DLASS_ENABLE_CUDA=OFF` configure-time backend selection (viskores
pattern) instead of link-time accident; `if constexpr`/`std::variant` in the
portable layer; CTest integration for 1.1/1.3; retires the dead-upstream
build system and the C++11 freeze. Legacy source stays untouched — this is a
build change, not a refactor.

### 1.3 Unit tests below the whole-song level
Everything today verifies at md5-of-the-whole-AIFF granularity — a great gate,
brutal for diagnosis (mismatch ⇒ manual bisection, as the leak/parity work
experienced). Adopt viskores' per-worklet test pattern via a single-header
framework (doctest; works under C++11, so this does **not** wait on 1.2).
First targets = where bugs are known to live: `Envelope::getValue`
memoization, the reverb unit (LPComb/AllPass vs reference), the loudness map,
`BiQuadFilter` (site of the maybe-uninitialized warnings), and regression
tests pinning the two Tier-0.4 fixes.

---

## Tier 2 — performance architecture (the gpu-fast ladder, ordered)

*Collectively these are what stands between gpu-fast v1 (1.2–1.5×) and the
projected 5–15× ([`07`](07_GPU_FAST_SPIKE.md) verdict); 2.4 is the bit-exact
CPU win that raises the bar gpu-fast must clear.*
**[2026-07-08: the 5–15× was reached (v3, `83d5a05`) via 2.1 + scan batching +
a Constant shortcut. The items below are re-annotated; what remains pushes
toward ~20×/node.]**

### 2.1 Implicit envelope arrays — viskores "fancy ArrayHandle" pattern
**[DONE in v3 (`83d5a05`): `Envelope::exportDeviceSegments()` + `evalSegKernel`
closed forms; pre-pass 29× down. Remaining refinement (optional): truly
implicit portals evaluated inside consuming kernels without materializing the
dense [P][N] arrays — a VRAM/bandwidth optimization, not a speed rung.]**
Attacks **v1 limitation #1** (host iterates every dynamic variable per
sample — the dominant remaining cost). Envelopes are a few piecewise segments;
their expansion is 44.1k samples/s. Give `SynthArray` an implicit portal
(`ArrayHandleTransform` analog): upload segment tables (tiny), evaluate
`value(s)` on-device. The existing RLE streams are a hand-rolled
`ArrayHandleRunLength` — this formalizes and generalizes them, killing both
the host pre-pass and most upload bandwidth.

### 2.2 Scan-with-custom-operator as a device-adapter primitive
Viskores' `DeviceAdapterAlgorithm::ScanInclusive(in, out, binaryOp)`. Today
every scan is bespoke (`bench_scan.cu`, `bench_batch.cu` Hillis-Steele, the
gpu-fast double phase scans). Define it once per backend — cub
`DeviceScan`/decoupled-lookback on CUDA, plain loop on Serial — with the
affine reverb operator as the first client. Prereq for 2.5; every future
recurrence (loudness accumulation, envelope integration) rides it.

### 2.3 Un-serialize GPU submission — **CUDA practice, not viskores**
**[Largely mooted by v3: the mutex-held section shrank 10× (72→7.3 ms/sound),
turning @20t into a 7.4× win on bench pieces. The 0.95× tutorial regression
(many light sounds) stands; fold the residual fix into 2.5's batching point
rather than building streams/arenas standalone.]**
Attacks **v1 limitation #2**: one global GPU mutex ⇒ one sound in flight ⇒
0.95× at 20 threads (a regression). Fix = per-worker CUDA streams + a
partitioned VRAM arena, or Score-level sound batching (which 2.5 wants
anyway). Noted explicitly: viskores is *not* the exemplar here — its own
concurrent-submission story is historically weak.

### 2.4 Bit-exact CPU reverb-stage overhead cleanup
**[Partial (`b8360cc`): stage 28.6→16.7 ms (1.71×), render 1.20× @1t,
bit-exact (md5 12d2ff21, 12/12). Remaining: constructAmp, allocations,
Track plumbing — ≈3.2× gap left.]**
The 5.4× gap: full reverb stage 28.6 ms vs 5.3 ms raw filter math — ~80% is
per-sample `Envelope::getValue` calls, `constructAmp` windowed max,
allocations/copies. A bit-exact hoist/cleanup recovers a large slice of the
44%/32% reverb share **on the default path, for every user, no GPU** — and
honestly re-baselines what gpu-fast must beat. Do after 1.3 so tests pin
bit-exactness.

### 2.5 Score-level reverb batching
**[Post-v3 this is the #1 remaining on-node lever: reverb is 31% and
spatialize 29% of gpu-fast residual per-sound CPU — consider extending the
batching point to cover spatialize's per-sample map too.]**
Fuse the step-2 batched kernel (measured **8.5× one core / 1044 Msmpl/s,
−163 dBFS** at 360×4 s) into the composite so reverb pays no PCIe transfer
(transfers currently halve it to 4.8×). Needs a Score-level batching point +
2.2's primitive + 2.3's concurrency. This is **v1 limitation #3**.

### 2.6 Contract-aware runtime device tracker — viskores `RuntimeDeviceTracker`/`TryExecute`
Four env vars (`LASS_PORTABLE_BACKEND`, `LASS_PIPELINE`, `LASS_REVERB`,
`LASS_COMPOSITE`) + scattered NULL-fallbacks → one policy object. Each backend
registers the **quality contract** it satisfies (bit-exact / ≤3 LSB composite
budget / gpu-fast double-reference — the contract taxonomy of
[`06`](06_DETERMINISTIC_COMPOSITE.md) and [`07`](07_GPU_FAST_SPIKE.md) §T2);
the user requests a contract, the tracker walks the fallback chain and reports
per-run which device rendered what (formalizing the existing fallback counts).

---

## Tier 3 — robustness & breadth

### 3.1 Device-parametrized test matrix
Viskores runs each unit test across every enabled adapter. Here: each worklet
gets one parity test executed per available backend **against its declared
contract** (serial → 0 LSB; cuda → composite budget; gpu-fast → sub-LSB vs
double reference). New backends inherit the whole matrix for free.

### 3.2 Sanitizer CI leg
ASan/LSan build target, run weekly if per-commit is too slow. Institutionalizes
the Spatializer/Partial leak hunt: the next 1.5 GB leak becomes a CI failure,
not valgrind archaeology. Also the honest home for the known-deferred
**Partial Rule-of-Three leak** (tricky; currently unfixed by choice).

### 3.3 OpenMP device adapter
Within-sound parallelism for the *map* stages of the same worklet, under the
gpu-fast (tolerance) contract — bit-exact within-sound SIMD was the measured
dead end ([`03`](03_PARALLELIZATION_ANALYSIS.md) §7a). Fills idle cores where
per-Sound granularity starves the pool (the ~5.2×-at-20-threads tail plateau).
Also the natural second client proving the adapter seam is real.

### 3.4 float/double worklet type-list instantiation
Compile each worklet body over `{float, double}` (viskores type-list pattern,
lightweight version) so the bit-exact double-`sin` and gpu-fast float variants
come from one source. Already done informally in gpu-fast (double scans +
`sinf`); formalize when 1.2 lands. Also covers v1 limitation #4 (a double
loudness variant pulling bench accuracy toward −151 dBFS).

### 3.5 User docs out of binary formats
Convert `Documents/DISSCOmanual.doc` and the tutorials (`.odt`/`.pdf`) to
markdown in-repo. Mechanical; makes the actual product documentation
diffable, reviewable, and renderable on GitHub.

---

## Tier 4 — opportunistic / future

### 4.1 Performance-regression goldens
Track render time per benchmark piece (`restructure/profiling/pieces/`) per
commit — even manually run. Catches performance regressions the md5 goldens
can't see.

### 4.2 Doxygen build + publish
`Doxyfile` exists; core comment density (~30%) is usable raw material. Wire
into CI, publish to Pages. Low priority.

### 4.3 Kokkos backend
Portability insurance for AMD/Intel hardware. **Not now**: Delta A100s and
DeltaAI are NVIDIA. File under "if the hardware appears."

---

## Explicit non-goals (decided against, with reasons)

- **Vendoring viskores.** Already ruled out
  ([`00`](00_ARCHITECTURE_COMPARISON.md) §4): 5M-line mesh/field toolkit vs
  1-D audio; we import the architecture, not the code.
- **`DataSet`/field model & the Filter ecosystem.** `Sound::render` is already
  the filter verb; a dataset abstraction is pure overhead here.
- **Dynamic types / `UncertainArrayHandle` / deep template dispatch.** Two
  value types exist. 3.4 is the entire useful subset.
- **Repo-wide clang-format.** Reformatting ~80k legacy lines destroys
  `git blame` for zero behavioral gain. Format new (`LASS/portable/`,
  `restructure/`) code only; likewise `-Wall -Wextra` on new code only.
- **Wholesale smart-pointer refactor of the legacy core.** 836 raw `new`
  sites; the additive-layer strategy (leave `LASS/src` untouched, harden with
  sanitizers per 3.2) is the deliberate alternative.
- **Full Kitware process ceremony** (CDash dashboards, MR robots,
  changelog-fragment tooling, deprecation macros). Right for a staffed
  toolkit; dead process for a ~1-person academic project. The cheap 80% is
  Tiers 0–1.

---

## Suggested first move (when implementation starts)

One small PR: **Tier 0 (0.1–0.4) + 1.1**. No architecture, no risk, and the
project permanently gains a license, an honest front page, and a regression
gate on everything the restructure already proved.
