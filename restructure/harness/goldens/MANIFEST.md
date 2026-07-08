# Golden Reference Manifest

Golden AIFFs are **not** committed (they are large, derived artifacts). Instead we
commit the deterministic input and the md5 + metrics needed to regenerate and
verify them. The single-thread + fixed-seed contract makes regeneration
bit-exact (see `../../01_RESTRUCTURE_PLAN.md` §3).

## Regenerate + verify

```bash
# from repo root, after `make cmod config=release`
bash restructure/harness/run_parity.sh render ./cmod \
     restructure/harness/goldens/tutorial_seed42_1thread.dissco /tmp/g
md5sum /tmp/g/SoundFiles/*.aiff     # must equal the md5 below
```

## Entries

**Reference algorithm = the CPU reverb** (the true original). As of the reverb
backend change, the default build uses the correct+fast CPU reverb; the old CUDA
reverb (a ~−59 dBFS approximation) is opt-in via `LASS_REVERB=gpu`.

Default (CPU reverb) golden md5s — the correct references:

| project | seed | threads | md5 (CPU reverb, default) | md5 (`LASS_REVERB=gpu`) |
|---|---|---|---|---|
| Tutorial.dissco | 42 | 1 | `12d2ff21332c22453b51e883c58c8177` | `6cdcc107206c9eeec203302efdddf375` |
| Tutorial.dissco | 777 | 1 | `d45d383b60cb6db6ccdf7be8a85cf389` | `99b3cde0f00f566faaf2ecac85efc586` |

`portable-serial` synthesis (`LASS_PORTABLE_BACKEND=serial`) is bit-exact with
the default at every seed (verified by `../parity_regression.sh`).

## Deterministic-composite goldens (`LASS_COMPOSITE=det` / `det-gpu`)

Unlike the legacy entries above (reproducible only at **1 thread**), these
goldens are **thread-count-, run-, and device-independent**: det@1t == det@Nt
== det-gpu, md5-equal (see `../../06_DETERMINISTIC_COMPOSITE.md`). They can
therefore be verified with a fast multi-threaded render — the cheap path IS
the reference path. det output differs from legacy by design (canonical
insertion order vs. timing-dependent arrival order): measured max 4 LSB,
RMS −151.7 dBFS on the tutorial — the composite-order budget, nothing else.

| project | seed | threads | md5 (`LASS_COMPOSITE=det` or `det-gpu`) |
|---|---|---|---|
| Tutorial.dissco | 42 | **any** | `27a6672c9e5d0594ce6f39bea358edf5` |
| ../../profiling/pieces/bench_1min.dissco | 8675309 (in file) | **any** | `c236f2b1ba7cc540c3c7eae5f5a07554` |
| ../../profiling/pieces/bench_10min.dissco | 8675309 (in file) | **any** | `51fa8676d8a174c03b1a4e73a0b7bb95` |
| 7_final.dissco (real piece, post-repair) | 42 | **any** | `a1c41b392332e9f0d4ac58665022838f` |

Verified matrices (2026-07-07): tutorial det@{1,8,20}t + det@20t-run2 +
det-gpu@{1,20}t all equal; bench_1min det@{1,20}t + run2 + det-gpu@20t all
equal; bench_10min det@20t == det-gpu@20t.

**Caution for legacy-mode goldens on large pieces:** pieces with more than
MAX_SOUND_OBJECTS=200 sounds are NOT reproducible in legacy mode at any thread
count (worker `srand(time(0))`/`rand()` trampled the producer's RNG stream —
bench_1min legacy@20t gave a different md5 per run). Only det modes yield
stable goldens for such pieces; do not record legacy md5s for them.

## Measured determinism budget (machine: RTX 4050, g++ 12.2, 2026-07-06)

Original DISSCO, same seed, run-to-run:

| threads | max \|diff\| | mean \|diff\| | RMS | % samples differ | reproducible? |
|---|---|---|---|---|---|
| 1  | 0 LSB | 0.000 | 0.000 LSB | 0.00 % | **bit-exact** |
| 64 | 3 LSB | 0.041 | 0.208 LSB (−152 dBFS) | 4.05 % | no (FP composite order) |

24-bit full-scale = 8 388 608; quantization floor ≈ −138 dBFS. The 64-thread
drift is ~14 dB below the floor → inaudible, and is a property of the *original*.
