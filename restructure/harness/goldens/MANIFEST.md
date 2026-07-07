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

| project | seed | threads | duration | format | md5 | backends verified identical |
|---|---|---|---|---|---|---|
| tutorial_seed42_1thread.dissco | 42 | 1 | 30 s | 24-bit AIFF stereo 44.1 kHz | `6cdcc107206c9eeec203302efdddf375` | original, serial, cuda |
| Tutorial.dissco (seed 777, 1 thread) | 777 | 1 | 30 s | 24-bit AIFF stereo 44.1 kHz | `99b3cde0f00f566faaf2ecac85efc586` | original, serial, cuda |

## Measured determinism budget (machine: RTX 4050, g++ 12.2, 2026-07-06)

Original DISSCO, same seed, run-to-run:

| threads | max \|diff\| | mean \|diff\| | RMS | % samples differ | reproducible? |
|---|---|---|---|---|---|
| 1  | 0 LSB | 0.000 | 0.000 LSB | 0.00 % | **bit-exact** |
| 64 | 3 LSB | 0.041 | 0.208 LSB (−152 dBFS) | 4.05 % | no (FP composite order) |

24-bit full-scale = 8 388 608; quantization floor ≈ −138 dBFS. The 64-thread
drift is ~14 dB below the floor → inaudible, and is a property of the *original*.
