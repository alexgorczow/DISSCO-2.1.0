/*
  LASS/portable — a viskores-inspired, device-agnostic synthesis layer.

  This is ADDITIVE, opt-in code. It does not modify the original LASS/src.
  See restructure/01_RESTRUCTURE_PLAN.md for the design and parity contract.

  Concepts (mirroring viskores):
    - SynthArray<T>      ~ viskores::cont::ArrayHandle  (device-agnostic buffer)
    - DeviceAdapter*     ~ viskores device adapter      (Schedule / scan primitives)
    - SampleMapWorklet   ~ viskores worklet             (per-sample body, write-once)
    - renderPartial()    ~ viskores filter              (orchestration verb)

  Tier 1 provides the Serial backend and targets BIT-EXACT parity with the
  original Partial::render() for the no-transient sine path.
*/
#ifndef LASS_PORTABLE_SYNTH_H
#define LASS_PORTABLE_SYNTH_H

#include <vector>
#include <cstddef>
#include <cmath>

// Execution-space marker. On CUDA translation units this becomes
// __host__ __device__ so the SAME worklet body compiles for device.
#if defined(__CUDACC__)
  #define LASS_EXEC __host__ __device__
#else
  #define LASS_EXEC
#endif

// Forward declarations from LASS/src (we include the real headers in the .cpp).
class Partial;
class MultiTrack;

namespace portable {

enum class Backend { Serial, Cuda };

/**
 * Device-agnostic 1-D buffer (ArrayHandle analog). Tier 1 keeps data host-side
 * in a std::vector; the Cuda backend (Tier 4) adds an execution-side portal.
 */
template <typename T>
class SynthArray {
public:
  SynthArray() = default;
  explicit SynthArray(std::size_t n, T v = T()) : data_(n, v) {}
  T*       data()        { return data_.data(); }
  const T* data()  const { return data_.data(); }
  std::size_t size() const { return data_.size(); }
  void resize(std::size_t n) { data_.resize(n); }
  T&       operator[](std::size_t i)       { return data_[i]; }
  const T& operator[](std::size_t i) const { return data_[i]; }
private:
  std::vector<T> data_;
};

/**
 * The write-once per-sample worklet: sample[s] = amplitude[s] * sin(2*pi*phase[s]).
 *
 * This reproduces the exact final arithmetic of Partial.cpp:350
 *   sample = amplitude * ( sin(2.0 * M_PI * phase) );
 * (amplitude is float, the sine is evaluated in double, the product is rounded
 * back to float). Keeping types identical is what makes Serial parity exact.
 */
struct SampleMapWorklet {
  // Defined inline (header-only, viskores-style) so the SAME body compiles for
  // host (g++) and device (nvcc marks it __host__ __device__ via LASS_EXEC).
  // Verbatim from Partial.cpp:350  sample = amplitude * ( sin(2.0*M_PI*phase) );
  // amplitude is float, the sine is double, the product rounds back to float.
  LASS_EXEC void operator()(std::size_t s,
                            const float* amplitude,
                            const float* phase,
                            float* wave,
                            float* ampOut) const {
    wave[s]   = amplitude[s] * (sin(2.0 * M_PI * phase[s]));
    ampOut[s] = amplitude[s];
  }
};

// CUDA map dispatch (defined in PartialRendererCuda.cu, compiled by nvcc).
// Declared unconditionally; the Serial build never references it.
//
// Computes only wave[s] = amplitude[s]*sin(2*pi*phase[s]) on the device. The
// amp channel (ampOut[s] = amplitude[s] in the worklet) is a pure identity, so
// the caller fills it with a host memcpy instead of a redundant GPU round-trip.
// Uses persistent thread_local device buffers (grown on demand) so repeated
// per-partial calls do not re-cudaMalloc/free.
void renderMapCuda(const float* amplitude, const float* phase,
                   float* wave, long n);

/**
 * Returns true if the partial can be rendered by the portable no-transient
 * kernel with guaranteed parity. Partials using amplitude/frequency transients
 * or a random wave type return false and MUST fall back to Partial::render().
 */
bool canRenderPortably(Partial& p);

/**
 * Device-agnostic replacement for Partial::render(). For the no-transient path
 * it produces a MultiTrack bit-identical (Serial) to Partial::render(); for
 * any other partial it delegates to Partial::render() unchanged.
 *
 * Ownership: caller deletes the returned MultiTrack (same contract as
 * Partial::render()).
 */
MultiTrack* renderPartial(Partial& p,
                          int numChannels,
                          long sampleCount,
                          float duration,
                          unsigned int samplingRate,
                          Backend backend = Backend::Serial);

/**
 * The opt-in seam used by Sound::render(). Chooses the backend from the
 * LASS_PORTABLE_BACKEND environment variable:
 *   unset / "off" / unknown -> original Partial::render() (default, unchanged)
 *   "serial"                -> portable Serial kernel
 *   "cuda"                  -> portable CUDA kernel (falls back if unavailable)
 * Keeping this logic in one place makes the Sound.cpp change a single call swap.
 */
MultiTrack* renderPartialDispatch(Partial& p,
                                  int numChannels,
                                  long sampleCount,
                                  float duration,
                                  unsigned int samplingRate);

} // namespace portable

#endif // LASS_PORTABLE_SYNTH_H
