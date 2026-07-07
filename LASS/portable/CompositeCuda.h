/*
  LASS/portable/CompositeCuda.h — deterministic GPU score composite (Tier B of
  restructure/06_DETERMINISTIC_COMPOSITE.md).

  The score buffer (wave + amp per channel) lives on the GPU; each rendered
  sound is committed by stream-ordered `score[off+i] += src[i]` kernels launched
  in canonical sequence order by the composite thread. Because the composite is
  ONLY IEEE-754 float additions (no transcendentals, no FMA), the GPU adds are
  bit-identical to the CPU adds: det-gpu output == det (CPU) output, md5-equal.

  Threading contract: every function here is called by the single composite
  thread only (plus compositeCudaFetch at drain end) — no locking inside.
*/
#ifndef LASS_PORTABLE_COMPOSITE_CUDA_H
#define LASS_PORTABLE_COMPOSITE_CUDA_H

class MultiTrack;

namespace portable {

/**
 * Grow the device score to numTracks x numSamples (wave+amp each), preserving
 * existing content exactly (D2D prefix copy) and zero-filling the tail.
 * No-op if already large enough. First call allocates.
 */
void compositeCudaEnsure(int numTracks, long numSamples);

/**
 * Stream-ordered commit of one source array into one device score array:
 *   score[track][kind][offset + i] += src[i]  for i in [0, n)
 * n is clipped against the current device length exactly like
 * SoundSample::composite clips against the host buffer. kindAmp selects the
 * amp array instead of the wave array.
 */
void compositeCudaAdd(int track, bool kindAmp, const float* src, long n,
                      long offset);

/**
 * Synchronize, copy the device score back into a freshly allocated MultiTrack
 * (with amp), and free all device state. Returns NULL if nothing was ever
 * committed (caller keeps its existing buffer).
 */
MultiTrack* compositeCudaFetch(int numChannels, unsigned int samplingRate);

} // namespace portable

#endif // LASS_PORTABLE_COMPOSITE_CUDA_H
