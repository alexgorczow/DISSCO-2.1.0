# LASS/portable — device-agnostic synthesis kernel

A viskores-inspired restructuring of LASS's additive-synthesis hot loop. It is
**additive and opt-in**: the original `LASS/src` is untouched except a 3-line
seam in `Sound.cpp`, and the default build behaves exactly as before.

See [`restructure/`](../../restructure/) for the full plan, architecture
comparison, and results.

## Concepts (↔ viskores)

| file | role | viskores analog |
|---|---|---|
| `PortableSynth.h` | `SynthArray<T>`, `SampleMapWorklet` (header-inline), API | ArrayHandle + Worklet |
| `PortableSynth.cpp` | `renderPartial` (materialize → pre-pass → dispatch), Serial adapter, `renderPartialDispatch` seam | Filter + Serial device adapter |
| `PartialRendererCuda.cu` | CUDA device adapter: the *same* worklet as a `__global__` | CUDA device adapter |

The per-sample body `sample = amplitude · sin(2π·phase)` is written **once**
(`SampleMapWorklet::operator()`, marked `LASS_EXEC`) and compiled for both host
(g++) and device (nvcc).

## Using it (via cmod)

```bash
make lass config=release && make cmod config=release   # from repo root
LASS_PORTABLE_BACKEND=serial  cmod project.dissco   # portable CPU kernel
LASS_PORTABLE_BACKEND=cuda    cmod project.dissco   # GPU kernel
#   (unset)                                          # original path, unchanged
```

Partials using amplitude/frequency transients, a random wave type, or a
per-partial reverb automatically fall back to the original `Partial::render`.

## Tests / benchmarks (standalone, needs `lib/liblass.a`)

```bash
cd LASS/portable
make run     # test_partial_parity: C1 bit-exact single-partial parity
make bench   # bench_map: CPU vs GPU map throughput
```

## Parity contract

- Serial backend → **bit-exact** with the original single-thread golden.
- CUDA backend → within the original's own multi-thread budget (≤4 LSB /
  −140 dBFS); empirically bit-exact on the tutorial.

## Status / next

Correct + portable foundation is complete (checkpoints C1/C2/C4). The measured
bottleneck for reverb-heavy pieces is **reverb**, not synthesis; the next
throughput step is routing `LASS/CUDA/FilterGPU.cu` reverb through this same
device-adapter seam and batching per-Sound to amortize transfer. See
`restructure/02_RESULTS.md §2`.
