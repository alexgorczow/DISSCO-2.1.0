/* LASS (additive sound synthesis library) Copyright (C) 2005  Sever Tipei (s-tipei@uiuc.edu)
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
//	Score.h	
//
//----------------------------------------------------------------------------//

#ifndef __SCORE_H
#define __SCORE_H

//----------------------------------------------------------------------------//
#include "StandardHeaders.h"

#include <atomic>
#include <deque>
#include <map>
#include <utility>
#include "XmlReader.h"
#include "Types.h"
#include "Collection.h"
#include "MultiTrack.h"
#include "Sound.h"
#include "Reverb.h"
#include "XmlReader.h"

class Utilities;

//----------------------------------------------------------------------------//

#define MAX_SOUND_OBJECTS 200
#define MAX_RENDERED_OBJECTS 20

/**
*	A Score simply consists of a collection of Sounds.
*	In addition to this, it provides functionality for 
*	managing clipping in a piece.
*	\author Braden Kowitz
*
* Update May 2013 by Ming-ching:
* Score is now responsible for lauching multi-threaded rendering and the final
* mix down.
**/

class Score{
public:

    /**
    *	This is the default constructor.  It sets the ClippingManagementMode 
    *	to NONE.
    **/
    Score(int _numThreads,  int _numChannels, int _samplingRate );
    
    /**
    * Sound objects created by CMOD is added to the score via this function.
    **/
    void add(Sound* _sound);
    
    /**
    * Worker threads give the renderedSound back to the main thread to compositie.
    * _seq is the sound's Score::add insertion index (canonical commit order for
    * the deterministic composite modes; ignored by the legacy mode).
    **/
    void addRenderedSound(m_time_type _startTime, MultiTrack* _renderedSound,
                          long _seq);

    /**
    * This function is called by the working threads to test if CMOD
    * has finished adding sounds to the score. If vector<Sound*> sounds has
    * size 0 and doneGettingSoundObjects is true, the working thread returns.
    **/
    bool isDoneGettingSoundObjects(){return doneGettingSoundObjects;}

    /**
    * Composite ordering mode, selected once via the LASS_COMPOSITE env var:
    *   unset/"legacy" -> arrival-order commits (historical behavior, default)
    *   "det"          -> canonical insertion-order commits on the CPU
    *   "det-gpu"      -> canonical order, adds performed on the GPU
    * Both det modes produce ONE bit-identical output for any thread count.
    **/
    enum CompositeMode { COMPOSITE_LEGACY, COMPOSITE_DET, COMPOSITE_DET_GPU };
    CompositeMode getCompositeMode(){return compositeMode_;}
    
    /**
    * This function is called by the compositeThread.
    **/
    void compositeRenderedSounds();
    
    /**
    * The final mix down and clean up after all the sound objects are rendered.
    **/
    MultiTrack* joinThreadsAndMix();
    
    /**
    * Called by the working threads to determine if they need to update the
    * duration of the MultiTrack objects currently in use.
    **/
    m_time_type getScoreEndTime(){return scoreEndTime;}
    
    /**
    *	\enum ClippingManagementMode
    *	This sets the clipping management mode for this score.
    *	This post-process is run after the score is rendered.
    **/

    /**
    *	\var ClippingManagementMode NONE
    *		- No clipping management is taken at all.
    *		- This lets the composer render once,
    *		  then try different post-processes, saving some time.
    **/

    /**	
    *	\var ClippingManagementMode CLIP
    *		- Any value over 1 or under -1 is clipped to these limits.
    **/

    /**
    *	\var ClippingManagementMode SCALE
    *		- The max amplitude value is found in the entire score
    *             (all tracks).
    *		- Each track is then scaled by 1/maxAmplitude.
    **/

    /**
    *	\var ClippingManagementMode CHANNEL_SCALE
    *		- For each track, a max amplitude value if found.
    *		- This track is then scaled by 1/maxAmplitude
    **/

    /**
    *	\var ClippingManagementMode ANTICLIP
    *		- If the total amplitude accross tracks at any given
    *		  time is greater than 1, then that sample is scaled
    *		  on all tracks by 1/totalAmplitude.
    **/

    /**
    *	\var ClippingManagementMode CHANNEL_ANTICLIP
    *		- For each channel, and each sample
    *		- if a sample has an amplitude greater than 1,
    *		  then that sample is scaled by 1/amplitude.
    **/	
    enum ClippingManagementMode
    {
        NONE,
        CLIP,
        SCALE,
        CHANNEL_SCALE,
        ANTICLIP,
        CHANNEL_ANTICLIP
    };

    /**
    *	This function sets the ClippingManagementMode for this score.
    *	\param mode The ClippingManagementMode to set
    **/
    void setClippingManagementMode(ClippingManagementMode mode);

    /**
    *	This function gets the current ClippingManagementMode for this score.
    *	\return The ClippingManagementMode
    **/
    ClippingManagementMode getClippingManagementMode();

    /**
    *	This function manages clipping on a MultiTrack object with the 
    *	specified mode. This is performed automatically when a score is 
    *	rendered, but is available for the user to render once, then 
    *	post-process many times.
    *	\param mt The MultiTrack to clip
    *	\param mode The ClippingManagementMode
    **/
    static void manageClipping(MultiTrack* mt, ClippingManagementMode mode);

    /**
    *   This function performs reverb in the render() method.
    *	\param newReverbObj The Reverb object
    **/
	  void use_reverb(Reverb *newReverbObj);
	
	
	  /**
	  * returns the final rendered score.
	  **/
	  MultiTrack* doneAddingSounds();
	  
	  /**
	  * Increase the length of MultiTrack* score to fit sounds
	  **/
	  void checkScoreMultiTrackLength(); 
  
    
//    /** 
//    *	\deprecated
//    *	This outputs an XML representation of the object to STDOUT
//    *
//    **/
//    void xml_print( );
//    /**
//    *	\deprecated
//    **/
//    void xml_print( ofstream& xmlOutput );
//    /**
//    *	\deprecated
//    **/
//    void xml_print( const char * xmlOutputPath );
//    /**
//    *	\deprecated
//    **/
//    void xml_read( XmlReader::xmltag *scoretag);
    
    DISSCO_HASHMAP<long, Reverb *>* reverbHash;
    DISSCO_HASHMAP<long, DynamicVariable *>* dvHash;

private:
    ClippingManagementMode cmm_;
	
    Reverb *reverbObj;
   
    /**
    * This private function clips the MultiTrack.
    * \param mt The MultiTrack to clip
    **/
    static void clip(MultiTrack* mt);
   
    /**
    * This private function scales the MultiTrack.
    * \param mt The MultiTrack to scale
    **/
    static void scale(MultiTrack* mt);

    /**
    * This private function scales the channels in the MultiTrack.
    * \param mt The MultiTrack to scale
    **/
    static void channelScale(MultiTrack* mt);

    /**
    * This private function unclips the MultiTrack.
    * \param mt The MultiTrack to unclip
    **/
    static void anticlip(MultiTrack* mt);

    /**
    * This private function unclips the channels in a MultiTrack.
    * \param mt The MultiTrack to unclip
    **/
    static void channelAnticlip(MultiTrack* mt);
    
  
   
  
    /**
    * stores the Sound objects to be rendered, tagged with their insertion
    * sequence number (the canonical order for the deterministic modes).
    * Legacy mode pops from the back (preserving the historical LIFO order
    * exactly); det modes pop from the front (FIFO) so the in-flight window
    * stays contiguous and the reorder buffer cannot deadlock.
    **/
    std::deque<std::pair<Sound*, long> > sounds;

    /**
    * A rendered sound waiting to be committed into the score.
    **/
    struct RenderedSound {
        m_time_type startTime;
        MultiTrack* mt;
        long seq;
        RenderedSound(m_time_type t, MultiTrack* m, long s)
            : startTime(t), mt(m), seq(s) {}
    };

    /**
    * stores the rendered sounds for the composite thread
    **/
    std::vector<RenderedSound*> renderedSounds;

    /**
    * Deterministic modes only (touched by the composite thread exclusively):
    * out-of-order arrivals wait here until their sequence number is next.
    **/
    std::map<long, RenderedSound*> reorderBuffer;
    long nextCommitSeq;
    CompositeMode compositeMode_;

    /**
    * Streaming preview (LASS_STREAM, det CPU composite only; see
    * restructure/10_REALTIME_LISTENING.md). startTime of every added sound,
    * indexed by seq (written under mutexSoundVector); the composite thread
    * flushes finalized time windows: [a,b) is final once every sound with
    * startTime < b is committed, i.e. up to min(start of seq >= nextCommitSeq).
    **/
    std::vector<m_time_type> seqStartTimes;
    FILE* streamFile;
    long streamFlushedSamples;
    long streamWindowSamples;
    void streamFlush(bool final);

    /**
    * Commit one rendered sound into the score (grow + composite + free).
    * Used by both the legacy drain and the deterministic in-order drain.
    **/
    void commitRenderedSound(RenderedSound* rs);
    
    /**
    * The MultiTrack object which holds the actual score
    **/
    MultiTrack* scoreMultiTrack;
    
    /**
    * The max end time seen among the added sound objects
    **/
    // Shared across the main (add), worker (render), and composite threads.
    // Written under mutexSoundVector but read lock-free by the composite thread
    // (checkScoreMultiTrackLength), so make it atomic to avoid a data race.
    std::atomic<m_time_type> scoreEndTime;
    m_time_type scoreMultiTrackLength;   // touched only by the composite thread
    
    /**
    * Number of threads
    **/
    int numThreads;
    
    /**
    * An array to hold thread objects
    **/
    pthread_t* threads;
    pthread_t compositeThread;
    
    /**
    * counter: # of sounds rendered.
    **/
    int soundsRendered;
    
    /**
    * counter: # of sound Objects passing in so far.
    **/
    int soundObjectsCreated;
    
    /**
    * A flag to indicate that the Score object has done receiving all the 
    * sound objects.
    **/
    // Cross-thread flags: written by main/composite, read lock-free by workers
    // and the composite thread. Atomic to avoid data races.
    std::atomic<bool> doneGettingSoundObjects;
    std::atomic<bool> workerThreadsAllJoined;
  
    /**
    * mutex to protect vector<Sound*> sounds
    **/
    pthread_mutex_t mutexSoundVector;
    
    /**
    * mutex to protect MultiTrack* scoreMultiTrack
    **/
    pthread_mutex_t mutexVectorRenderedSound;
    
    // Rubin Du July 2024: restructured and replaced conditions with semaphores for better resources management and synchronization
    ///**
    //* cond to brocast to worker threads the status of vector<Sound*> sounds
    //**/
    //pthread_cond_t conditionSoundVector;

    /**
    * semaphores for work queue management
    **/
    sem_t semEmptySlotsSounds;
    sem_t semFullSlotsSounds;
    sem_t semEmptySlotsRendered;
    sem_t semFullSlotsRendered;
    
    /**
    * Number of channels
    **/
    int numChannels;
    
    /**
    * Sampling rate
    **/
    int samplingRate;
    
};


//----------------------------------------------------------------------------//
#endif //__SCORE_H


