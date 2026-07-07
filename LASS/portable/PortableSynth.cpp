/*
  LASS/portable/PortableSynth.cpp — Serial backend (Tier 1).

  renderPartial() reproduces Partial::render() for the no-transient sine path,
  decomposed the viskores way:
    1. materialize each dynamic parameter into a SynthArray (ArrayHandle step)
    2. a faithful sequential pre-pass builds amplitude[] and phase[]
       (the scan-carrying part: tremolo/vibrato/freq phase accumulators)
    3. SampleMapWorklet computes wave[]=amplitude*sin(2*pi*phase) as a pure map,
       dispatched by DeviceAdapterSerial::Schedule
    4. the original Track + Spatializer tail assembles the MultiTrack

  Steps 1-3 are bit-identical to Partial.cpp:216-357 because the loop locals keep
  the original types (m_value_type = float) and the expressions are verbatim, so
  every intermediate rounding matches. Any partial that uses transients, a random
  wave, or per-partial reverb is delegated to the original Partial::render().
*/
#include "PortableSynth.h"

#include "../src/Partial.h"
#include "../src/SoundSample.h"
#include "../src/Track.h"
#include "../src/Spatializer.h"
#include "../src/MultiTrack.h"
#include "../src/DynamicVariable.h"
#include "../src/Iterator.h"

#include <cmath>
#include <functional>
#include <cstdlib>
#include <cstring>

namespace portable {

// -------------------------------------------------------------------------//
// Minimal Serial device adapter: the reference Schedule (parallel-for).
struct DeviceAdapterSerial {
  template <typename Functor>
  static void Schedule(std::size_t n, Functor f) {
    for (std::size_t i = 0; i < n; ++i) f(i);
  }
};

// Identical to Partial::pmod (Partial.cpp:401).
static inline m_value_type pmod_(m_value_type num) {
  while (num > 1.0) num -= 1.0;
  return num;
}

// -------------------------------------------------------------------------//
void SampleMapWorklet::operator()(std::size_t s,
                                  const float* amplitude,
                                  const float* phase,
                                  float* wave,
                                  float* ampOut) const {
  // Verbatim from Partial.cpp:350  sample = amplitude * ( sin(2.0*M_PI*phase) );
  // amplitude is float, the sine is double, the product rounds back to float.
  wave[s]   = amplitude[s] * (sin(2.0 * M_PI * phase[s]));
  ampOut[s] = amplitude[s];
}

// -------------------------------------------------------------------------//
bool canRenderPortably(Partial& p) {
  if ((int)p.getParam(WAVE_TYPE) == 1) return false;                 // random wave
  if (p.getParam(AMPTRANS_AMP_ENV).getMaxValue()  != 0.0) return false;
  if (p.getParam(FREQTRANS_AMP_ENV).getMaxValue() != 0.0) return false;
  // Indirect probe for per-partial reverb (reverbObj is private):
  // getTotalDuration(d) = d + reverbDecay, so > d iff a reverb is attached.
  if (p.getTotalDuration(1.0) > 1.0) return false;
  return true;
}

// -------------------------------------------------------------------------//
MultiTrack* renderPartial(Partial& p,
                          int numChannels,
                          long sampleCount,
                          float duration,
                          unsigned int samplingRate,
                          Backend backend) {
  // Correctness-preserving fallback (Tier 1 supports Serial no-transient path).
  if (backend != Backend::Serial || !canRenderPortably(p)) {
    return p.render(numChannels, sampleCount, duration, samplingRate);
  }

  const long N = (long)(duration * (m_time_type)samplingRate);   // Partial.cpp:68

  SoundSample* waveSample = new SoundSample(sampleCount, samplingRate);
  SoundSample* ampSample  = new SoundSample(sampleCount, samplingRate);

  // ---- iterator setup, mirroring Partial::render (86-177) ----
  p.getParam(FREQUENCY).setDuration(duration);
  p.getParam(WAVE_SHAPE).setDuration(duration);
  p.getParam(TREMOLO_AMP).setDuration(duration);
  p.getParam(TREMOLO_RATE).setDuration(duration);
  p.getParam(VIBRATO_AMP).setDuration(duration);
  p.getParam(VIBRATO_RATE).setDuration(duration);
  p.getParam(PHASE).setDuration(duration);
  p.getParam(LOUDNESS_SCALAR).setDuration(duration);
  p.getParam(FREQ_ENV).setDuration(duration);
  p.getParam(DETUNING_ENV).setDuration(duration);

  p.getParam(FREQUENCY).setSamplingRate(samplingRate);
  p.getParam(WAVE_SHAPE).setSamplingRate(samplingRate);
  p.getParam(TREMOLO_AMP).setSamplingRate(samplingRate);
  p.getParam(TREMOLO_RATE).setSamplingRate(samplingRate);
  p.getParam(VIBRATO_AMP).setSamplingRate(samplingRate);
  p.getParam(VIBRATO_RATE).setSamplingRate(samplingRate);
  p.getParam(PHASE).setSamplingRate(samplingRate);
  p.getParam(LOUDNESS_SCALAR).setSamplingRate(samplingRate);
  p.getParam(FREQ_ENV).setSamplingRate(samplingRate);
  p.getParam(DETUNING_ENV).setSamplingRate(samplingRate);

  DynamicVariable* frequency_env = p.getParam(FREQUENCY).clone();
  frequency_env->setDuration(duration);
  DynamicVariable* freq_env = p.getParam(FREQ_ENV).clone();
  freq_env->setDuration(duration);
  DynamicVariable* detuning_env = p.getParam(DETUNING_ENV).clone();
  detuning_env->setDuration(duration);

  typedef Iterator<m_value_type> ValIter;
  ValIter frequency_it     = frequency_env->valueIterator();
  ValIter wave_shape_it    = p.getParam(WAVE_SHAPE).valueIterator();
  ValIter tremolo_amp_it   = p.getParam(TREMOLO_AMP).valueIterator();
  ValIter tremolo_rate_it  = p.getParam(TREMOLO_RATE).valueIterator();
  ValIter vibrato_amp_it   = p.getParam(VIBRATO_AMP).valueIterator();
  ValIter vibrato_rate_it  = p.getParam(VIBRATO_RATE).valueIterator();
  ValIter phase_it         = p.getParam(PHASE).valueIterator();
  ValIter loudnes_scalar_it= p.getParam(LOUDNESS_SCALAR).valueIterator();
  ValIter freq_it          = freq_env->valueIterator();
  ValIter detuning_it      = detuning_env->valueIterator();

  // ---- faithful sequential pre-pass -> amplitude[], phase[] ----
  SynthArray<float> amp((std::size_t)N), ph((std::size_t)N);

  m_value_type tremolo_phase = 0.0;
  m_value_type vibrato_phase = 0.0;
  m_value_type freq_phase    = 0.0;

  for (long s = 0; s < N; ++s) {
    m_value_type tremolo = tremolo_amp_it.next() * sin(2.0 * M_PI * tremolo_phase);
    tremolo_phase = pmod_(tremolo_phase + (tremolo_rate_it.next() / samplingRate));

    m_value_type amplitude =
        loudnes_scalar_it.next() * wave_shape_it.next() * (1.0 + tremolo);

    m_value_type vibrato = vibrato_amp_it.next() * sin(2.0 * M_PI * vibrato_phase);
    vibrato_phase = pmod_(vibrato_phase + (vibrato_rate_it.next() / samplingRate));

    m_value_type frequency =
        frequency_it.next() * freq_it.next() * detuning_it.next();
    frequency *= 1.0 + vibrato;

    freq_phase = pmod_(freq_phase + (frequency / samplingRate));
    m_value_type phase = freq_phase + phase_it.next();

    amp[(std::size_t)s] = amplitude;
    ph[(std::size_t)s]  = phase;
  }

  // ---- pure map worklet, dispatched by the device adapter ----
  SampleMapWorklet worklet;
  float* waveData = waveSample->getData();
  float* ampData  = ampSample->getData();
  const float* ampSrc = amp.data();
  const float* phSrc  = ph.data();
  DeviceAdapterSerial::Schedule((std::size_t)N, [&](std::size_t s) {
    worklet(s, ampSrc, phSrc, waveData, ampData);
  });

  // ---- original Track + Spatializer tail (Partial.cpp:360-397) ----
  // No per-partial reverb here (canRenderPortably rejected it). In the CMOD
  // pipeline a Partial always carries the placeholder base Spatializer (see
  // Partial.cpp:374), which spatialize_Track reproduces exactly.
  Track* track = new Track(waveSample, ampSample);
  Spatializer spat;
  MultiTrack* returnTrack = spat.spatialize_Track(*track, numChannels);
  delete track;

  delete frequency_env;
  delete freq_env;
  delete detuning_env;

  return returnTrack;
}

// -------------------------------------------------------------------------//
MultiTrack* renderPartialDispatch(Partial& p,
                                  int numChannels,
                                  long sampleCount,
                                  float duration,
                                  unsigned int samplingRate) {
  const char* b = std::getenv("LASS_PORTABLE_BACKEND");
  if (b != nullptr) {
    if (std::strcmp(b, "serial") == 0)
      return renderPartial(p, numChannels, sampleCount, duration, samplingRate,
                           Backend::Serial);
    if (std::strcmp(b, "cuda") == 0)
      return renderPartial(p, numChannels, sampleCount, duration, samplingRate,
                           Backend::Cuda);
  }
  // Default / unknown: original behavior, unchanged.
  return p.render(numChannels, sampleCount, duration, samplingRate);
}

} // namespace portable
