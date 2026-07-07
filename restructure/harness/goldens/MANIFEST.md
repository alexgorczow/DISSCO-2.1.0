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

## Measured determinism budget (machine: RTX 4050, g++ 12.2, 2026-07-06)

Original DISSCO, same seed, run-to-run:

| threads | max \|diff\| | mean \|diff\| | RMS | % samples differ | reproducible? |
|---|---|---|---|---|---|
| 1  | 0 LSB | 0.000 | 0.000 LSB | 0.00 % | **bit-exact** |
| 64 | 3 LSB | 0.041 | 0.208 LSB (−152 dBFS) | 4.05 % | no (FP composite order) |

24-bit full-scale = 8 388 608; quantization floor ≈ −138 dBFS. The 64-thread
drift is ~14 dB below the floor → inaudible, and is a property of the *original*.
