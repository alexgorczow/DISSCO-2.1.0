/*
  LASS/portable/GpuFastSound.h — gpu-fast v1: fused device-resident loudness +
  synthesis for one Sound (restructure/07_GPU_FAST_SPIKE.md, step 3).

  Opt-in via LASS_PIPELINE=gpu-fast. Replaces the two dominant per-Sound stages
  (loudness ~63%, partial synthesis ~26% post-cleanup) with device kernels:

    host:   per-partial dynamic variables iterated once into RLE-compressed
            parameter streams (constants collapse to one run)
    device: expand -> loudness map (24-band model, float transcendentals,
            including the reference's bandGamma[0] max-write quirk) ->
            tremolo/vibrato/frequency phase DOUBLE prefix scans -> sinf synth
            -> deterministic in-order partial sum -> one D2H (mono wave+amp)
    host:   placeholder-spatialized MultiTrack (all channels = mono / nCh),
            handed to the UNCHANGED filter/reverb/Pan/composite pipeline.

  Contract (measured basis in 07 doc): md5-deterministic per (GPU, CUDA)
  config; accuracy vs the double-precision reference at sub-LSB for phase and
  -163 dBFS class for scans; raw diff vs the float CPU render is dominated by
  the CPU chain's own accumulated phase noise and is reported, not asserted.

  Returns NULL when the sound uses features outside the fast path (transients,
  random wave, detune envelopes, per-partial reverb, disabled loudness) —
  caller falls back to the unchanged CPU path.
*/
#ifndef LASS_PORTABLE_GPU_FAST_SOUND_H
#define LASS_PORTABLE_GPU_FAST_SOUND_H

class Sound;
class MultiTrack;

namespace portable {

/** True iff LASS_PIPELINE=gpu-fast (cached). */
bool gpuFastEnabled();

/**
 * Fused device render of the partial-composite stage of Sound::render().
 * Returns the composite MultiTrack (pre filter/reverb/sound-spatialize), or
 * NULL if ineligible / no CUDA device — caller uses the original loop.
 * Thread-safe: device work is serialized internally (VRAM arena).
 */
MultiTrack* renderSoundGpuFast(Sound& snd, int numChannels, long sampleCount,
                               float duration, unsigned int samplingRate);

} // namespace portable

#endif
