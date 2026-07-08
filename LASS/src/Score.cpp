/*
LASS (additive sound synthesis library)
Copyright (C) 2005  Sever Tipei (s-tipei@uiuc.edu)
Modified by Ming-ching Chiu 2013

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
*/

//----------------------------------------------------------------------------//
//
//	Score.cpp
//
//----------------------------------------------------------------------------//

#ifndef __SCORE_CPP
#define __SCORE_CPP

//----------------------------------------------------------------------------//

#include "Score.h"
#include "Types.h"
#include "../../restructure/profiling/StageProfiler.h"  // opt-in stage timing
#ifdef HAVE_CUDA
#include "../portable/CompositeCuda.h"  // deterministic GPU composite backend
#endif

//----------------------------------------------------------------------------//

// This struct passes data between the main thread and the worker threads
struct ThreadEntry{
  //pointer to the score
  Score* score;
  int threadID;
  //pointer to the (sound, insertion-seq) work queue of the score
  std::deque<std::pair<Sound*, long> >* sounds;
  int numChannels;
  int samplingRate;
  pthread_mutex_t* mutexSoundVector;
  int* soundsRendered;
  int* soundObjectsCreated;

  sem_t* semEmptySlotsSounds; 
  sem_t* semFullSlotsSounds;  
  sem_t* semEmptySlotsRendered; 
  sem_t* semFullSlotsRendered;  

  MultiTrack** scoreMultiTrack;

};


//------------------------------------------------------------------------------
void Score::add(Sound* _sound){

  // To prevent unnecessary memory use, this function is temporarily blocked
  // when there are too many sounds objects waiting to be rendered.
  // 200 is just an arbitrary number.
  // This funcion can be upgraded to block the main thread in a fancier way, but
  // for now this is enough.
  //                                           --Ming-ching May 06, 2013

  // Rubin Du July 2024: Integrated the resource management, producer thread / consumer thread synchronization, and optimized concurrency by implementing semaphores

  sem_wait(&semEmptySlotsSounds);  // Wait for an empty slot in sounds vector

  // Lock the sounds vector
  pthread_mutex_lock( &mutexSoundVector );
  // The insertion index is the canonical commit order for the det modes.
  sounds.push_back(std::make_pair(_sound, (long)soundObjectsCreated));
  seqStartTimes.push_back(_sound->getParam(START_TIME));
  soundObjectsCreated ++;

  //update scoreEndTime
  m_time_type soundEndTime = _sound->getParam(START_TIME) +
                                   _sound->getTotalDuration();
  if (soundEndTime > scoreEndTime) {
    scoreEndTime = soundEndTime;
    //cout<<"Score End Time Updated: "<< scoreEndTime << "seconds"<<endl;
  }
    // figure in the reverb die-out period
  if(reverbObj != NULL)
  // atomic<float> has no operator+= before C++20; this runs under
  // mutexSoundVector so the load/store pair cannot lose an update.
  scoreEndTime.store(scoreEndTime.load() + reverbObj->getDecay());

  // Unlock the sounds vector
  pthread_mutex_unlock( &mutexSoundVector );

  sem_post(&semFullSlotsSounds);  // Signal that there is a full slot in sounds vector

}

//------------------------------------------------------------------------------
/**
* This function is the entry of the worker threads.
* It gets the Sound objects from the score, renders them, and put all the
* rendered sounds in renderedThreadScore.
**/
void *render(void *_threadEntry){
  ThreadEntry threadEntry = *((ThreadEntry*) _threadEntry);
  bool deterministic =
      threadEntry.score->getCompositeMode() != Score::COMPOSITE_LEGACY;
  while(true){
    sem_wait(threadEntry.semFullSlotsSounds); // Wait for a full slot in sounds vector

    /* Deterministic modes reserve the OUTPUT slot before taking a sound.
       With FIFO dispatch this keeps every in-flight/buffered sound inside the
       window [nextCommitSeq, nextCommitSeq + MAX_RENDERED_OBJECTS): the head of
       the window always holds a slot, so it always lands and commits, freeing
       slots -- the bounded reorder buffer cannot deadlock. (Legacy acquires the
       slot after rendering, inside addRenderedSound, as before.) */
    if (deterministic)
      sem_wait(threadEntry.semEmptySlotsRendered);

    //Lock the sounds vector
    pthread_mutex_lock( threadEntry.mutexSoundVector );

      if (threadEntry.sounds->size()) {
          *(threadEntry.soundsRendered) = *(threadEntry.soundsRendered) + 1;
          cout<<"Thread #"<<threadEntry.threadID<<": Rendering sound #"<<
                *(threadEntry.soundsRendered)<<" of "<<
                *(threadEntry.soundObjectsCreated)<<endl;

          // legacy: LIFO (back) -- historical order, byte-identical output.
          // det:    FIFO (front) -- contiguous dispatch window.
          std::pair<Sound*, long> entry =
              deterministic ? threadEntry.sounds->front()
                            : threadEntry.sounds->back();
          if (deterministic) threadEntry.sounds->pop_front();
          else               threadEntry.sounds->pop_back();
          Sound* sound = entry.first;
          pthread_mutex_unlock( threadEntry.mutexSoundVector );

          // Render the sound outside the critical section
          MultiTrack* renderedSound = sound->render(threadEntry.numChannels,threadEntry.samplingRate);

          threadEntry.score->addRenderedSound(sound->getParam(START_TIME), renderedSound, entry.second);

          delete sound;

          sem_post(threadEntry.semEmptySlotsSounds);  // Signal that there is an empty slot in sounds vector
      }
      else {
          pthread_mutex_unlock(threadEntry.mutexSoundVector);
          sem_post(threadEntry.semFullSlotsSounds);  // Put back the full slot since no sound was consumed
          if (deterministic)
            sem_post(threadEntry.semEmptySlotsRendered); // return unused reservation
          if (threadEntry.score->isDoneGettingSoundObjects())   // If the main thread is done adding sounds, return
            return _threadEntry;
      }
  }// end of main while loop

}


void *composite(void *_score){
  ((Score*) _score)->compositeRenderedSounds();
  return NULL;
}

//----------------------------------------------------------------------------//
Score::Score(int _numThreads, int _numChannels, int _samplingRate )
    : cmm_(NONE),
    soundsRendered(0),
    soundObjectsCreated(0),
    doneGettingSoundObjects(false),
    workerThreadsAllJoined(false),
    numChannels(_numChannels),
    samplingRate(_samplingRate)
{
  nextCommitSeq = 0;
  compositeMode_ = COMPOSITE_LEGACY;
  // Composite ordering backend (see Score.h). Same opt-in pattern as
  // LASS_REVERB / LASS_PORTABLE_BACKEND: default preserves historical output.
  if (const char* cm = getenv("LASS_COMPOSITE")) {
    if (strcmp(cm, "det") == 0) compositeMode_ = COMPOSITE_DET;
    else if (strcmp(cm, "det-gpu") == 0) {
#ifdef HAVE_CUDA
      compositeMode_ = COMPOSITE_DET_GPU;
#else
      cout << "Score: LASS_COMPOSITE=det-gpu but no CUDA in this build; "
           << "using det (CPU). Output is bit-identical either way." << endl;
      compositeMode_ = COMPOSITE_DET;
#endif
    }
    else if (strcmp(cm, "legacy") != 0)
      cout << "Score: unknown LASS_COMPOSITE '" << cm << "', using legacy." << endl;
  }
  if (compositeMode_ != COMPOSITE_LEGACY)
    cout << "Score: deterministic composite ("
         << (compositeMode_ == COMPOSITE_DET_GPU ? "GPU" : "CPU")
         << ") -- output is bit-identical across thread counts." << endl;

  // Streaming preview (restructure/10): append finalized windows of the score
  // to a raw f32 stream while rendering. Requires the CPU det composite (the
  // committed prefix must be host-readable in canonical order).
  streamFile = NULL;
  streamFlushedSamples = 0;
  streamWindowSamples = (long)(0.5 * samplingRate);
  if (const char* sp = getenv("LASS_STREAM")) {
    if (compositeMode_ != COMPOSITE_DET) {
      cout << "Score: LASS_STREAM requires LASS_COMPOSITE=det (CPU); "
           << "streaming disabled." << endl;
    } else {
      if (const char* w = getenv("LASS_STREAM_WINDOW")) {
        double ws = atof(w);
        if (ws > 0.01 && ws < 30.0) streamWindowSamples = (long)(ws * samplingRate);
      }
      streamFile = fopen(sp, "wb");
      if (!streamFile) {
        cout << "Score: cannot open LASS_STREAM path '" << sp << "'." << endl;
      } else {
        unsigned int hdr[3] = {0x52545344u /*'DSTR' LE*/,
                               (unsigned int)samplingRate,
                               (unsigned int)numChannels};
        fwrite(hdr, sizeof(unsigned int), 3, streamFile);
        fflush(streamFile);
        cout << "Score: streaming preview to " << sp << " (window "
             << (double)streamWindowSamples / samplingRate << "s)." << endl;
      }
    }
  }

  scoreEndTime = 1; //start with a small number
  scoreMultiTrackLength = scoreEndTime;
  m_sample_count_type newNumSamples =
        (m_sample_count_type) (scoreMultiTrackLength * float(samplingRate));
  scoreMultiTrack = new MultiTrack
        (numChannels,newNumSamples,samplingRate);


  reverbObj = NULL;
  numThreads = _numThreads;

  //create threads for sound rendering
  threads = new pthread_t[_numThreads];
  pthread_mutex_init(&mutexSoundVector, NULL);
  pthread_mutex_init(&mutexVectorRenderedSound, NULL);
  sem_init(&semEmptySlotsSounds, 0, MAX_SOUND_OBJECTS);
  sem_init(&semFullSlotsSounds, 0, 0);    
  sem_init(&semEmptySlotsRendered, 0, MAX_RENDERED_OBJECTS);
  sem_init(&semFullSlotsRendered, 0, 0);  

  for (int i = 0; i < _numThreads; i ++){
    ThreadEntry* threadEntry = new ThreadEntry;
    threadEntry->score = this;
    threadEntry->threadID = i;
    threadEntry->sounds = &sounds;
    threadEntry->numChannels = _numChannels;
    threadEntry->samplingRate = _samplingRate;
    threadEntry->mutexSoundVector=&mutexSoundVector;
    threadEntry->soundsRendered = &soundsRendered;
    threadEntry->soundObjectsCreated = &soundObjectsCreated;
    threadEntry->scoreMultiTrack =&scoreMultiTrack;
    threadEntry->semEmptySlotsSounds = &semEmptySlotsSounds;
    threadEntry->semFullSlotsSounds = &semFullSlotsSounds;
    threadEntry->semEmptySlotsRendered = &semEmptySlotsRendered;
    threadEntry->semFullSlotsRendered = &semFullSlotsRendered;
    pthread_create(&threads[i], NULL, render, (void*) threadEntry);
  }

   //start the compositeThread
   pthread_create(&compositeThread, NULL, composite, (void*) this);

}


void Score::addRenderedSound(m_time_type _startTime, MultiTrack* _renderedSound,
                             long _seq){

  RenderedSound* newEntry = new RenderedSound(_startTime, _renderedSound, _seq);
  // active waiting for the vector size to be below 20 (if the main thread can't
  // composite the rendered sound as fast the speed of worker threads produce
  // rendered sounds, there is no need for the worker threads to rush.
    if (compositeMode_ == COMPOSITE_LEGACY)
      sem_wait(&semEmptySlotsRendered);  // Wait for an empty slot
    // (det modes reserved their slot in render() before dispatch)

    pthread_mutex_lock(&mutexVectorRenderedSound);
    renderedSounds.push_back(newEntry);
    pthread_mutex_unlock(&mutexVectorRenderedSound);

    sem_post(&semFullSlotsRendered);  // Signal that there is a full slot in renderedSounds vector

}

//------------------------------------------------------------------------------
// Commit one rendered sound into the score: grow the score buffer if needed,
// add the sound's samples (wave + amp) at its start offset, free the sound.
// In legacy mode commits happen in ARRIVAL order (nondeterministic across
// runs/thread counts); in det modes the caller guarantees insertion order.
void Score::commitRenderedSound(RenderedSound* rs){
  { PROFILE_SCOPE(prof::COMPOSITE);
    checkScoreMultiTrackLength();
#ifdef HAVE_CUDA
    if (compositeMode_ == COMPOSITE_DET_GPU) {
      // Keep the device score exactly as long as the host score would be
      // (covers the initial 1-second buffer when no growth ever triggers).
      portable::compositeCudaEnsure(numChannels,
          (m_sample_count_type)(scoreMultiTrackLength * float(samplingRate)));
      // Same adds, applied on the device in the same order. The offset uses
      // the EXACT expression from SoundSample::composite so index arithmetic
      // matches the CPU path bit-for-bit.
      m_sample_count_type skip =
          m_sample_count_type(rs->startTime * float(samplingRate));
      int tracks = rs->mt->size() < numChannels ? rs->mt->size() : numChannels;
      for (int t = 0; t < tracks; ++t) {
        Track* tr = rs->mt->get(t);
        SoundSample& w = tr->getWave();
        portable::compositeCudaAdd(t, false, w.getData(), w.getSampleCount(), skip);
        if (tr->hasAmp()) {
          SoundSample& a = tr->getAmp();
          portable::compositeCudaAdd(t, true, a.getData(), a.getSampleCount(), skip);
        }
      }
    }
    else
#endif
    scoreMultiTrack->composite(*(rs->mt), rs->startTime);
  }
  delete rs->mt;
  delete rs;
}


void Score::compositeRenderedSounds(){
  while (true){
    sem_wait(&semFullSlotsRendered);

    pthread_mutex_lock( &mutexVectorRenderedSound );

    if(renderedSounds.size()){
      RenderedSound* thisEntry = renderedSounds.back();
      renderedSounds.pop_back();
      pthread_mutex_unlock( &mutexVectorRenderedSound );

      if (compositeMode_ == COMPOSITE_LEGACY) {
        // Historical behavior: free the slot at pop, commit in arrival order.
        sem_post(&semEmptySlotsRendered);
        commitRenderedSound(thisEntry);
      }
      else {
        /* Deterministic drain: park the arrival in the reorder buffer, then
           commit every consecutive sequence number that is now available.
           Slots are released only at commit, matching the reserve-at-dispatch
           in render() so the window stays bounded by MAX_RENDERED_OBJECTS. */
        reorderBuffer[thisEntry->seq] = thisEntry;
        while (!reorderBuffer.empty() &&
               reorderBuffer.begin()->first == nextCommitSeq) {
          RenderedSound* head = reorderBuffer.begin()->second;
          reorderBuffer.erase(reorderBuffer.begin());
          commitRenderedSound(head);
          nextCommitSeq++;
          sem_post(&semEmptySlotsRendered);
        }
        if (streamFile) streamFlush(false);
      }
    }
    else{
      pthread_mutex_unlock( &mutexVectorRenderedSound );
      sem_post(&semFullSlotsRendered);
      if (workerThreadsAllJoined) {
        if (!reorderBuffer.empty()) {
          // Cannot happen if every dispatched sound arrived; defensive only.
          cerr << "WARNING: Score: " << reorderBuffer.size()
               << " rendered sounds never committed (missing seq "
               << nextCommitSeq << ")." << endl;
        }
        if (streamFile) {
          streamFlush(true);
          fclose(streamFile);
          streamFile = NULL;
        }
#ifdef HAVE_CUDA
        if (compositeMode_ == COMPOSITE_DET_GPU) {
          // Bring the device score home once; replaces the untouched stub.
          MultiTrack* fetched =
              portable::compositeCudaFetch(numChannels, samplingRate);
          if (fetched) {
            delete scoreMultiTrack;
            scoreMultiTrack = fetched;
          }
        }
#endif
        return;
      }
    }

  }



}
//------------------------------------------------------------------------------
// Streaming preview flush (composite thread only). Window [a,b) is final once
// every sound with startTime < b has been committed; with in-order commits the
// uncommitted set is seq >= nextCommitSeq, so the frontier is the min start
// over that suffix (O(pending), pending is bounded by the dispatch window +
// producer lead). Flushing starts only after the producer is done adding
// (before that, an earlier-starting sound could still arrive). Samples are
// interleaved float32, hard-clamped to +-1 (the stream is pre-anticlip; see
// restructure/10 for the preview contract). Never mutates render state, so
// the final AIFF stays byte-identical.
void Score::streamFlush(bool final){
  if (!streamFile) return;

  long frontierSamples;
  if (final) {
    frontierSamples = (long)scoreMultiTrack->get(0)->getWave().getSampleCount();
  } else {
    if (!doneGettingSoundObjects) return;
    m_time_type minStart = -1;
    pthread_mutex_lock(&mutexSoundVector);
    for (long q = nextCommitSeq; q < (long)seqStartTimes.size(); ++q)
      if (minStart < 0 || seqStartTimes[q] < minStart) minStart = seqStartTimes[q];
    pthread_mutex_unlock(&mutexSoundVector);
    if (minStart < 0) return;                    // nothing pending: wait for final
    frontierSamples = (long)(minStart * (float)samplingRate);
    long len = (long)scoreMultiTrack->get(0)->getWave().getSampleCount();
    if (frontierSamples > len) frontierSamples = len;
  }

  int nCh = scoreMultiTrack->size();
  std::vector<float> frame(nCh);
  while (streamFlushedSamples +
         (final ? 1 : streamWindowSamples) <= frontierSamples) {
    long end = final ? frontierSamples
                     : streamFlushedSamples + streamWindowSamples;
    if (end > frontierSamples) end = frontierSamples;
    for (long s = streamFlushedSamples; s < end; ++s) {
      for (int c = 0; c < nCh; ++c) {
        float v = scoreMultiTrack->get(c)->getWave()[s];
        if (v > 1.0f) v = 1.0f;
        if (v < -1.0f) v = -1.0f;
        frame[c] = v;
      }
      fwrite(frame.data(), sizeof(float), nCh, streamFile);
    }
    streamFlushedSamples = end;
    fflush(streamFile);
    if (final && streamFlushedSamples >= frontierSamples) break;
  }
}

//------------------------------------------------------------------------------
MultiTrack* Score::joinThreadsAndMix(){
  //Join the threads
  for (int i = 0; i < numThreads; i ++){
    void* threadEntry;
    pthread_join(threads[i], &threadEntry);
    cout<< "thread Joined: Thread #"<<
      ((ThreadEntry*) threadEntry)-> threadID<<endl;

    delete (ThreadEntry*)threadEntry;
  }
  cout<<"=======================Threads all joined.==================="<<endl;
  delete [] threads;
  workerThreadsAllJoined = true;
  sem_post(&semFullSlotsRendered); //unblock the composite thread for proper termination

  //join the composite thread
  pthread_join(compositeThread, NULL);

  pthread_mutex_destroy(&mutexSoundVector);
  pthread_mutex_destroy(&mutexVectorRenderedSound);
  sem_destroy(&semEmptySlotsSounds);
  sem_destroy(&semFullSlotsSounds);
  sem_destroy(&semEmptySlotsRendered);
  sem_destroy(&semFullSlotsRendered);

  // do the reverb
  if(reverbObj != NULL)
  {
    cout << "Applying reverb to the score..." << endl;
    PROFILE_SCOPE(prof::FINAL_REVERB);
    MultiTrack *tmp = & reverbObj->do_reverb_MultiTrack(*scoreMultiTrack);
    delete scoreMultiTrack;
    scoreMultiTrack = tmp;
  }

  // perform Clipping management on the composite:
  cout << "Managing Clipping for the score..." << endl;
  { PROFILE_SCOPE(prof::CLIP);
    manageClipping(scoreMultiTrack, cmm_);
  }
  // return the composite:
  return scoreMultiTrack;
}


//----------------------------------------------------------------------------//

void Score::checkScoreMultiTrackLength(){

  if ( scoreEndTime > scoreMultiTrackLength ){
      scoreMultiTrackLength = scoreEndTime;
      m_sample_count_type newNumSamples =
        (m_sample_count_type) (scoreMultiTrackLength * float(samplingRate));

#ifdef HAVE_CUDA
      if (compositeMode_ == COMPOSITE_DET_GPU) {
        // The score lives on the device; grow it there (zero-fill + exact
        // prefix copy). The host stub buffer is left untouched until fetch.
        portable::compositeCudaEnsure(numChannels, newNumSamples);
        cout<<"Get a longer score with length = " << scoreEndTime.load() << " seconds."<<endl;
        return;
      }
#endif
      MultiTrack* newScoreMultiTrack = new MultiTrack
        (numChannels,newNumSamples,samplingRate);

      newScoreMultiTrack->composite(*scoreMultiTrack, 0);
      delete scoreMultiTrack;
      scoreMultiTrack = newScoreMultiTrack;

      //pthread_mutex_unlock( &mutexVectorRenderedSound );
      cout<<"Get a longer score with length = " << scoreEndTime.load() << " seconds."<<endl;

  }
}


//----------------------------------------------------------------------------//

void Score::setClippingManagementMode(ClippingManagementMode mode)
{
    cout << "Score::setClippingManagementMode - " << mode << endl;
    cmm_ = mode;
}

//----------------------------------------------------------------------------//

Score::ClippingManagementMode Score::getClippingManagementMode()
{
    cout << "Score::ClippingManagementMode - cmm_ " << cmm_ << endl;
    return cmm_;
}


//----------------------------------------------------------------------------//

void Score::manageClipping(MultiTrack* mt, ClippingManagementMode mode)
{
    cout << "Score::manageClipping - mode - " << mode << endl;
    switch (mode)
    {
        case NONE:		break;
        case CLIP: 		clip(mt); break;
        case SCALE : 		scale(mt); break;
        case CHANNEL_SCALE : 	channelScale(mt); break;
        case ANTICLIP :		anticlip(mt); break;
        case CHANNEL_ANTICLIP :	channelAnticlip(mt); break;
        default : 		break;
    }
}

//----------------------------------------------------------------------------//
void Score::use_reverb(Reverb *newReverbObj)
{
	reverbObj = newReverbObj;
}

//----------------------------------------------------------------------------//
// 	PRIVATE FUNCTIONS:
//----------------------------------------------------------------------------//


//----------------------------------------------------------------------------//
void Score::clip(MultiTrack* mt)
{
    cout << "Performing CLIP" << endl;

    // for each track
    for (int t=0; t<mt->size(); t++)
    {
        SoundSample& wave = mt->get(t)->getWave();
        SoundSample& amp = mt->get(t)->getAmp();

        // for each sample
        m_sample_count_type numSamples = wave.getSampleCount();
        for (m_sample_count_type s=0; s<numSamples; s++)
        {
            if (wave[s] >  1.0) wave[s] =  1.0;
            if (wave[s] < -1.0) wave[s] = -1.0;
            if (amp[s] >  1.0) amp[s] =  1.0;
            if (amp[s] < -1.0) amp[s] = -1.0;
        }

    }
}


//----------------------------------------------------------------------------//

void Score::scale(MultiTrack* mt)
{
    cout << "Performing SCALE" << endl;

    // -----
    // first, find the maximum amplitude:

    m_sample_type maxAmp = 0;
    // for each track
    for (int t=0; t<mt->size(); t++)
    {
        SoundSample& amp = mt->get(t)->getAmp();
        // for each sample
        m_sample_count_type numSamples = amp.getSampleCount();
        for (m_sample_count_type s=0; s<numSamples; s++)
        {
            if (amp[s] > maxAmp) maxAmp = amp[s];
        }
    }

    // -----
    // create a scaling factor:
    m_sample_type scalingFactor = 1.0 / maxAmp;

    // -----
    // scale every value by this factor

    // for each track
    for (int t=0; t<mt->size(); t++)
    {
        SoundSample& wave = mt->get(t)->getWave();
        SoundSample& amp = mt->get(t)->getAmp();

        // for each sample
        m_sample_count_type numSamples = wave.getSampleCount();
        for (m_sample_count_type s=0; s<numSamples; s++)
        {
            wave[s] *= scalingFactor;
            amp[s] *= scalingFactor;
        }
    }
}


//----------------------------------------------------------------------------//

void Score::channelScale(MultiTrack* mt)
{
    cout << "Performing CHANNEL_SCALE" << endl;

    // -----
    // for each track
    for (int t=0; t<mt->size(); t++)
    {
        // -----
        // find the maximum amplitude:
        SoundSample& wave = mt->get(t)->getWave();
        SoundSample& amp = mt->get(t)->getAmp();

        m_sample_type maxAmp = 0;

        // for each sample
        m_sample_count_type numSamples = wave.getSampleCount();
        for (m_sample_count_type s=0; s<numSamples; s++)
        {
            if (amp[s] > maxAmp) maxAmp = amp[s];
        }

        // -----
        // create a scaling factor:
        m_sample_type scalingFactor = 1.0 / maxAmp;

        // -----
        // scale every value by this factor
        for (m_sample_count_type s=0; s<numSamples; s++)
        {
            wave[s] *= scalingFactor;
            amp[s] *= scalingFactor;
        }

    }
}


//----------------------------------------------------------------------------//

void Score::anticlip(MultiTrack* mt)
{
    cout << "Performing ANTICLIP" << endl;

    // this is a dificult one, because we need fast cros-track access.
    // make our own data-structure for this one.
    vector<SoundSample*> wave;
    vector<SoundSample*> amp;
    int numTracks = mt->size();
    for (int i=0; i<numTracks; i++)
    {
        wave.push_back(& mt->get(i)->getWave());
        amp.push_back(& mt->get(i)->getAmp());
    }

    // now, for each sample:
    m_sample_count_type numSamples = wave[0]->getSampleCount();
    for (m_sample_count_type s=0; s<numSamples; s++)
    {
        // find the total amplitude across all tracks.
        m_sample_type totalAmp = 0.0;
        for (int t=0; t<numTracks; t++)
        {
            totalAmp += (*amp[t])[s];
        }

        // scale if necessary
        if (totalAmp > 1.0)
        {
            m_sample_type scalingFactor = 1.0 / totalAmp;
            for (int t=0; t<numTracks; t++)
            {
                (*amp[t] )[s] *= scalingFactor;
                (*wave[t])[s] *= scalingFactor;
            }
        }
    }
}


//----------------------------------------------------------------------------//

m_sample_type todB(m_sample_type x)
{
  return (m_sample_type)(log10((double)x) * 20.0);
}


//----------------------------------------------------------------------------//

m_sample_type fromdB(m_sample_type x)
{
  return (m_sample_type)pow(10.0, ((double)x * 0.05));
}


//----------------------------------------------------------------------------//

m_sample_type compressSound(m_sample_type x, m_sample_type peak,
  m_sample_type dBCompressionPoint)
{
/*				sever commented out 6/11/16
  m_sample_type xdB = todB(x);
  m_sample_type cdB = dBCompressionPoint;
  m_sample_type pdB = todB(peak);
*/

  if(x < fromdB(dBCompressionPoint))
    return x;

      m_sample_type c = fromdB(dBCompressionPoint);
      x = 1 - (peak - x) * (1 - c) / (peak - c);	//sever 6/11/16
//    x = fromdB((pdB - xdB) * (pdB * xdB - cdB * cdB) / ((cdB - pdB) * (cdB - pdB)));		//Andrew Burnson
      return x;

//return fromdB((pdB - xdB) * (pdB * xdB - cdB * cdB) /
//  ((cdB - pdB) * (cdB - pdB)));
}


//----------------------------------------------------------------------------//

void Score::channelAnticlip(MultiTrack* mt)
{
    cout << "Performing CHANNEL_ANTICLIP" << endl;

    // for each track
    m_sample_type maxAmplitude = 0.0;
    int peakPlace = 0;
    for (int t=0; t<mt->size(); t++)
    {
	// cout << "Score::channelAnticlip - first time track=" << t << endl;
        SoundSample& wave = mt->get(t)->getWave();
        SoundSample& amp = mt->get(t)->getAmp();

        // for each sample
        m_sample_count_type numSamples = wave.getSampleCount();
        for (m_sample_count_type s=0; s<numSamples; s++)
        {

            if (amp[s] > 1.0) {
/* 				deprecated
                scale this sample:
                wave[s] *= 1.0 / amp[s];
                amp[s] = 1.0;
*/
            }

            m_sample_type cur = wave[s];
            if(cur < 0) cur = -cur;
            if(cur > maxAmplitude)
            {
              maxAmplitude = cur;
              peakPlace = s;
            }
        }
    }

    if(maxAmplitude <= 0.99)
    {
      cout << "Peak at " << maxAmplitude << endl;
      //cout << "Normalizing to achieve better signal-to-noise ratio.";
    }
    else if(maxAmplitude >= 0.99)
    {
      cout << "Warning: peak is " << todB(maxAmplitude) << " dB at " <<
        ((double)peakPlace / (double)mt->get(0)->getWave().getSamplingRate())
           << " seconds. Compressing [-6, " << todB(maxAmplitude)
           << ") to [-6, 0) dB" << endl;
     maxAmplitude /= 0.99; //Never actually allow it to hit 0dB.

      //m_sample_type normalizeValue = maxAmplitude;
      for (int t=0; t<mt->size(); t++)
      {
          SoundSample& wave = mt->get(t)->getWave();
          SoundSample& amp = mt->get(t)->getAmp();

          // for each sample
          m_sample_count_type numSamples = wave.getSampleCount();
          for (m_sample_count_type s=0; s<numSamples; s++)
          {
              amp[s] = 1.0;

 	    if (wave[s] < 0) {
	      wave[s] *= -1;
              wave[s] = compressSound(wave[s], maxAmplitude, -6.0);
	      wave[s] *= -1;
            } else {
              wave[s] = compressSound(wave[s], maxAmplitude, -6.0);
	    }

/*
             cout << "Score::channelAnticlip - wave[" << s << "]= "
	      << wave[s] << endl;
`
              if(s == 0)
          cout << "Score::channelAnticlip Test - showing compression curve used:"
               << endl;
              wave[s] = compressSound((float)s / (float)numSamples * 2.f, 2.f, -6.0);
*/
          }
      }
    }
}


//----------------------------------------------------------------------------//

MultiTrack* Score::doneAddingSounds(){

  pthread_mutex_lock( &mutexSoundVector );
  doneGettingSoundObjects= true;
  sem_post(&semFullSlotsSounds); // unblock the worker threads for proper termination
  pthread_mutex_unlock( &mutexSoundVector );

  return joinThreadsAndMix();

}

//----------------------------------------------------------------------------//
#endif //__SCORE_CPP
