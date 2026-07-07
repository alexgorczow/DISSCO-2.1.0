#!/usr/bin/env python3
"""Generate a render-dominated DISSCO synthesis benchmark family (.dissco XML).

Purpose: stress the LASS *synthesis* pipeline (additive synth + Loudness +
reverb + spatialize), so the LASS render -- not the CMOD event-tree build --
carries the cost and scales ~linearly with musical length.

Why the note-event COUNT is held constant instead of scaled with duration:
the CMOD stochastic event-tree build is super-linear in event count (measured:
360 sounds -> 29 s build; 3600 sounds -> 673 s build, ~23x for 10x). If we scaled
the count, the serial CMOD build would dominate (~95% of wall at 30 min) and bury
the synthesis cost we actually want to benchmark. So instead we keep the count
fixed (~N_TOP*SOUNDS_PER_S1 sounds, build ~constant ~30 s) and scale the
SYNTHESIZED AUDIO by making each Sound (and its s1 window) longer as the piece
lengthens. Total synthesized samples ~ sounds * partials * sound_dur then scale
1:10:30 with duration, and the RENDER stage carries it.

Event tree (mirrors the proven-valid tutorial.dissco template):
    Top (type 0, eventTop)     -- whole piece, N_TOP children "s1"
      s1 (type 4, eventBottom) -- window ~2*sound_dur, SOUNDS_PER_S1 children
        sp1 (type 5, eventSound) -- becomes a LASS Sound with PARTIALS partials

Leaf LASS Sounds = N_TOP * SOUNDS_PER_S1 (constant). Each hits every hot path:
additive synthesis (PARTIALS partials), Loudness (24-band), sound Reverb
(comb+all-pass), Spatialization, tremolo/vibrato Modifiers. s1 events and their
sounds are spread across the timeline (not stacked at t=0 like tutorial.dissco),
giving ~N_TOP*SOUNDS_PER_S1*sound_dur/duration concurrent voices -- dense but
memory-bounded (leak-fixed engine: peak RSS tracks the composite queue + score
length, not sound count).
"""

# --- fixed knobs (identical across all durations) ----------------------------
PARTIALS      = 24     # partials per Sound  -> additive-synthesis stress
N_TOP         = 30     # s1 events (top children) -> drives the CMOD build cost
SOUNDS_PER_S1 = 12     # sp1 children per s1 event; sounds = N_TOP*SOUNDS_PER_S1 = 360
# Each leaf Sound's dry length = duration / DUR_DIVISOR, so synthesized audio
# (and render cost) scales linearly with piece length while the event count -- and
# thus the CMOD build cost -- stays fixed. 1min->2s, 10min->20s, 30min->60s.
# Capped at 60 s (DUR_DIVISOR=30) so the 30-min's peak RSS stays ~10 GB on a
# 15 GB box: render memory ~ concurrency x sound-buffer ~ 0.16 GB per second of
# sound length (measured), and a longer sound_dur (e.g. 120 s) OOMs.
DUR_DIVISOR   = 30
# Per-sound reverb (REV_Advanced) with explicit comb/LP gains so the decay tail
# is bounded and musical. The LASS decay tail is T_r = -3*delay/log(alpha) with
# alpha = max(comb_gain/(1-lp_gain)); these gains give alpha~0.561, so
# delay=0.3 s yields a ~1.5 s tail. (The tutorial's REV_Medium + 0.5 s delay
# leaves alpha~1, a hundreds-of-seconds tail that makes every sound a ~450 MB
# buffer -- it OOMs at scale.) This keeps the comb+allpass hot path exercised
# while per-sound memory stays proportional to the sound's own length.
REVERB_DELAY  = 0.3
COMB_GAINS    = "0.46, 0.48, 0.50, 0.52, 0.53, 0.55"
LP_GAINS      = "0.05, 0.06, 0.07, 0.05, 0.04, 0.02"
SEED          = 8675309
THREADS       = 8      # default; benchmark.py sweeps this in a scratch copy

VERSIONS = [
    ("bench_1min",  60),
    ("bench_10min", 600),
    ("bench_30min", 1800),
]


def spectrum(p):
    """P partials, harmonic amplitude rolloff via EnvLib env 1."""
    lines = []
    for i in range(p):
        scale = round(1.0 - i * (0.96 / (p - 1)), 3)
        lines.append(
            f"        <Partial><Fun><Name>EnvLib</Name><Env>1</Env>"
            f"<Scale>{scale}</Scale></Fun></Partial>"
        )
    return "\n".join(lines)


def build(title, duration):
    n_top = N_TOP
    sound_dur = duration // DUR_DIVISOR         # leaf Sound dry length (scales)
    s1_duration = 2 * sound_dur                 # s1 window holds its sounds
    top_start_hi = max(1, duration - s1_duration)   # keep s1 events in the piece
    s1_start_hi = max(1, s1_duration - sound_dur)   # keep sounds in their s1
    return f"""<?xml version='1.0'?>
<ProjectRoot>
  <ProjectConfiguration>
    <Title>{title}</Title>
    <FileFlag>THMLBsnv</FileFlag>
    <TopEvent>0</TopEvent>
    <PieceStartTime>0</PieceStartTime>
    <Duration>{duration}</Duration>
    <Synthesis>True</Synthesis>
    <Score>False</Score>
    <GrandStaff>False</GrandStaff>
    <NumberOfStaff>1</NumberOfStaff>
    <NumberOfChannels>2</NumberOfChannels>
    <SampleRate>44100</SampleRate>
    <SampleSize>16</SampleSize>
    <NumberOfThreads>{THREADS}</NumberOfThreads>
    <OutputParticel>True</OutputParticel>
    <Seed>{SEED}</Seed>
  </ProjectConfiguration>
  <NoteModifiers>
    <DefaultModifiers>1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1</DefaultModifiers>
    <CustomModifiers>
    </CustomModifiers>
  </NoteModifiers>
  <EnvelopeLibrary>
1
Envelope 1
3
0.000     0.000     EXPONENTIAL         FLEXIBLE    0.500
0.500     1.000     EXPONENTIAL         FLEXIBLE    0.500
1.000     0.000
  </EnvelopeLibrary>
  <MarkovModelLibrary>
0
  </MarkovModelLibrary>
  <Events>
    <Event orderInPalette=' -1'>
      <EventType>0</EventType>
      <Name>0</Name>
      <MaxChildDuration>{s1_duration}</MaxChildDuration>
      <EDUPerBeat>6</EDUPerBeat>
      <TimeSignature>
        <Entry1>4</Entry1>
        <Entry2>4</Entry2>
      </TimeSignature>
      <Tempo>
        <MethodFlag>0</MethodFlag>
        <Prefix>0</Prefix>
        <NoteValue>0</NoteValue>
        <FractionEntry1></FractionEntry1>
        <FractionEntry2>60</FractionEntry2>
        <ValueEntry>60</ValueEntry>
      </Tempo>
      <NumberOfChildren>
        <MethodFlag>0</MethodFlag>
        <Entry1>{n_top}</Entry1>
        <Entry2>0</Entry2>
        <Entry3>0</Entry3>
      </NumberOfChildren>
      <ChildEventDefinition>
        <Entry1><Fun><Name>Random</Name><Low>0</Low><High>{top_start_hi}</High></Fun></Entry1>
        <Entry2>0</Entry2>
        <Entry3>{s1_duration}</Entry3>
        <AttackSieve></AttackSieve>
        <DurationSieve></DurationSieve>
        <DefinitionFlag>0</DefinitionFlag>
        <StartTypeFlag>2</StartTypeFlag>
        <DurationTypeFlag>2</DurationTypeFlag>
      </ChildEventDefinition>
      <Layers>
        <Layer>
          <ByLayer></ByLayer>
          <DiscretePackages>
            <Package>
              <EventName>s1</EventName>
              <EventType>4</EventType>
              <Weight></Weight>
              <AttackEnvelope></AttackEnvelope>
              <AttackEnvelopeScale></AttackEnvelopeScale>
              <DurationEnvelope></DurationEnvelope>
              <DurationEnvelopeScale></DurationEnvelopeScale>
            </Package>
          </DiscretePackages>
        </Layer>
      </Layers>
      <Spatialization></Spatialization>
      <Reverb></Reverb>
      <Filter></Filter>
      <Modifiers>
      </Modifiers>
    </Event>
    <Event orderInPalette=' 0'>
      <EventType>4</EventType>
      <Name>s1</Name>
      <MaxChildDuration>{s1_duration}</MaxChildDuration>
      <EDUPerBeat>6</EDUPerBeat>
      <TimeSignature>
        <Entry1>4</Entry1>
        <Entry2>4</Entry2>
      </TimeSignature>
      <Tempo>
        <MethodFlag>0</MethodFlag>
        <Prefix>0</Prefix>
        <NoteValue>0</NoteValue>
        <FractionEntry1></FractionEntry1>
        <FractionEntry2>60</FractionEntry2>
        <ValueEntry>60</ValueEntry>
      </Tempo>
      <NumberOfChildren>
        <MethodFlag>0</MethodFlag>
        <Entry1>{SOUNDS_PER_S1}</Entry1>
        <Entry2>0</Entry2>
        <Entry3>0</Entry3>
      </NumberOfChildren>
      <ChildEventDefinition>
        <Entry1><Fun><Name>Random</Name><Low>0</Low><High>{s1_start_hi}</High></Fun></Entry1>
        <Entry2>0</Entry2>
        <Entry3>{sound_dur}</Entry3>
        <AttackSieve></AttackSieve>
        <DurationSieve></DurationSieve>
        <DefinitionFlag>0</DefinitionFlag>
        <StartTypeFlag>2</StartTypeFlag>
        <DurationTypeFlag>2</DurationTypeFlag>
      </ChildEventDefinition>
      <Layers>
        <Layer>
          <ByLayer></ByLayer>
          <DiscretePackages>
            <Package>
              <EventName>sp1</EventName>
              <EventType>5</EventType>
              <Weight></Weight>
              <AttackEnvelope></AttackEnvelope>
              <AttackEnvelopeScale></AttackEnvelopeScale>
              <DurationEnvelope></DurationEnvelope>
              <DurationEnvelopeScale></DurationEnvelopeScale>
            </Package>
          </DiscretePackages>
        </Layer>
      </Layers>
      <Spatialization></Spatialization>
      <Reverb></Reverb>
      <Filter></Filter>
      <ExtraInfo>
        <FrequencyInfo>
          <FrequencyFlag>0</FrequencyFlag>
          <FrequencyContinuumFlag>0</FrequencyContinuumFlag>
          <FrequencyEntry1><Fun><Name>RandomInt</Name><LowBound>36</LowBound><HighBound>84</HighBound></Fun></FrequencyEntry1>
          <FrequencyEntry2></FrequencyEntry2>
        </FrequencyInfo>
        <Loudness>100</Loudness>
        <Spatialization><Fun><Name>SPA</Name><Method>STEREO</Method><Apply>SOUND</Apply><Channels><Partials><P><Fun><Name>EnvLib</Name><Env>1</Env><Scale>1.0</Scale></Fun></P></Partials></Channels></Fun></Spatialization>
        <Reverb><Fun><Name>REV_Advanced</Name><Apply>SOUND</Apply><Percents><Percent><Fun><Name>EnvLib</Name><Env>1</Env><Scale>1.0</Scale></Fun></Percent></Percents><CombGainLists><CombGainList>{COMB_GAINS}</CombGainList></CombGainLists><LPGainLists><LPGainList>{LP_GAINS}</LPGainList></LPGainLists><AllPasses><AllPass>0.1</AllPass></AllPasses><Delays><Delay>{REVERB_DELAY}</Delay></Delays></Fun></Reverb>
        <Filter></Filter>
        <ModifierGroup><Fun><Name>Select</Name><List>1,2</List><Index><Fun><Name>RandomInt</Name><LowBound>0</LowBound><HighBound>1</HighBound></Fun></Index></Fun></ModifierGroup>
        <Modifiers>
            <Modifier>
              <Type>1</Type>
              <ApplyHow>0</ApplyHow>
              <Probability><Fun><Name>EnvLib</Name><Env>1</Env><Scale>1.0</Scale></Fun></Probability>
              <Amplitude><Fun><Name>EnvLib</Name><Env>1</Env><Scale>1.0</Scale></Fun></Amplitude>
              <Rate><Fun><Name>EnvLib</Name><Env>1</Env><Scale>1.0</Scale></Fun></Rate>
              <Width></Width>
              <DetuneSpread></DetuneSpread>
              <DetuneDirection></DetuneDirection>
              <DetuneVelocity></DetuneVelocity>
              <GroupName>1,2</GroupName>
              <PartialResultString></PartialResultString>
            </Modifier>
            <Modifier>
              <Type>0</Type>
              <ApplyHow>0</ApplyHow>
              <Probability><Fun><Name>EnvLib</Name><Env>1</Env><Scale>1.0</Scale></Fun></Probability>
              <Amplitude><Fun><Name>EnvLib</Name><Env>1</Env><Scale>1.0</Scale></Fun></Amplitude>
              <Rate><Fun><Name>EnvLib</Name><Env>1</Env><Scale>1.0</Scale></Fun></Rate>
              <Width></Width>
              <DetuneSpread></DetuneSpread>
              <DetuneDirection></DetuneDirection>
              <DetuneVelocity></DetuneVelocity>
              <GroupName>2</GroupName>
              <PartialResultString></PartialResultString>
            </Modifier>
        </Modifiers>
      </ExtraInfo>
    </Event>
    <Event orderInPalette=' 0'>
      <EventType>5</EventType>
      <Name>sp1</Name>
      <NumberOfPartials>{PARTIALS}</NumberOfPartials>
      <Deviation>0</Deviation>
      <GenerateSpectrum></GenerateSpectrum>
      <Spectrum>
{spectrum(PARTIALS)}
      </Spectrum>
    </Event>
  </Events>
</ProjectRoot>
"""


def main():
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "pieces")
    os.makedirs(out_dir, exist_ok=True)
    sounds = N_TOP * SOUNDS_PER_S1
    for title, dur in VERSIONS:
        path = os.path.join(out_dir, f"{title}.dissco")
        with open(path, "w") as f:
            f.write(build(title, dur))
        sound_dur = dur // DUR_DIVISOR
        sound_seconds = sounds * sound_dur       # total synthesized dry audio
        print(f"{path}: {dur}s piece, {sounds} sounds x {sound_dur}s x "
              f"{PARTIALS} partials -> {sound_seconds}s synth audio "
              f"({sound_seconds * PARTIALS} partial-seconds)")


if __name__ == "__main__":
    main()
