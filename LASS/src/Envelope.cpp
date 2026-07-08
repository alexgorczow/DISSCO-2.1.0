/*
LASS (additive sound synthesis library)
Copyright (C) 2005  Sever Tipei (s-tipei@uiuc.edu)

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
// Envelope.cpp
//
//----------------------------------------------------------------------------//


#include "Envelope.h"
#include "EnvelopeIterator.h"
#include "InterpolatorTypes.h"
#include "EnvelopeLibrary.h"

EnvelopeLibrary envlib;

using std::setw;

//----------------------------------------------------------------------------//
// This is the default constructor
Envelope::Envelope()
{
    segments_ = new Collection<envelope_segment>();
    interpolators_ = new Collection<Interpolator*>();
    generatedSegmentLengths_ = new Collection<m_value_type>();
    totalLength_ = -1;
    currentInterpolatorRate_ = 0;
}

//----------------------------------------------------------------------------//
// This is the copy constructor
Envelope::Envelope(const Envelope &env)
{
	//cout << "Copy in Envelope from " << &env << " to " << this << endl;

	segments_ = new Collection<envelope_segment>( *env.segments_ );

	// there's not an easy way to transfer these, so the copy does not contain
	// the generated interpolators, but these are only stored to help
	// optimize the code
	interpolators_ = new Collection<Interpolator*>();

	generatedSegmentLengths_ = new Collection<m_value_type>( *env.generatedSegmentLengths_ );

	totalLength_ = env.totalLength_;
	currentInterpolatorRate_ = env.currentInterpolatorRate_;
}

//----------------------------------------------------------------------------//
// This constructor uses points and segments to build an envelope
Envelope::Envelope(Collection<xy_point> xy_points, Collection<envelope_segment> segs)
{
	segments_ = new Collection<envelope_segment>();
	interpolators_ = new Collection<Interpolator*>();
	generatedSegmentLengths_ = new Collection<m_value_type>();
	totalLength_ = -1;
	currentInterpolatorRate_ = 0;

    envelope_segment seg;
    xy_point tpt = xy_points.get(0);
    seg.x = tpt.x;
    seg.y = tpt.y;
    segments_->add(seg);
    for (int i = 1; i < xy_points.size(); i++)
	{
	    seg = segs.get(i - 1);
	    seg.x = xy_points.get(i).x;
	    seg.y = xy_points.get(i).y;
	    segments_->add(seg);
	}

    // define the shape, given the Collection
    defineShape();
}

//----------------------------------------------------------------------------//
// This constructor just uses a collection of segments
Envelope::Envelope(Collection<envelope_segment> segs)
{
	// Copy the segments passed in
	segments_ = new Collection<envelope_segment>(segs);

	interpolators_ = new Collection<Interpolator*>();
	generatedSegmentLengths_ = new Collection<m_value_type>();
	totalLength_ = -1;
	currentInterpolatorRate_ = 0;

  // define the shape, given the Collections
  defineShape();
}

//----------------------------------------------------------------------------//
Envelope::~Envelope()
{
	delete(segments_);

	if (interpolators_) {
		for (int i = 0; i < interpolators_->size(); i++) {
			delete(interpolators_->get(i));
		}
		delete(interpolators_);
	}

	delete(generatedSegmentLengths_);
}

//----------------------------------------------------------------------------//
void Envelope::print()
{
    if (segments_->size() == 0)
    {
        cout << "Envelope is empty.\n";
        return;
    }

    int iNumSegments = segments_->size();
    int iLoop = 0;

    cout << endl;
    cout << "Envelope has " << iNumSegments - 1 << " segments." << endl;

    for (iLoop = 0; iLoop < iNumSegments - 1; iLoop++)
        {
            // print a line with form (x, y):
            cout << " x, y: " << "(";
            cout << setw(5) << segments_->get(iLoop).x;
            cout << ", ";
            cout << setw(5) << segments_->get(iLoop).y;
            cout << ")" << endl;

            // print the entry number
            cout << "     Entry " << iLoop << ":" << endl;

            // print the length type
            cout << "       Length type: ";
            if (this->getSegmentLengthType(iLoop) == FLEXIBLE)
                {
                    cout << "FLEXIBLE" << endl;
                }
            else
                {
                    cout << "FIXED" << endl;
                }

            // print the length value
            cout << "       Length value: ";
            cout << this->getSegmentLength(iLoop) << endl;

            // print the interpolation type
            cout << "       Interpolation Type: ";
            switch (this->getSegmentInterpolationType(iLoop))
                {
                case LINEAR:
                    cout << "LINEAR" << endl;
                    break;
                case EXPONENTIAL:
                    cout << "EXPONENTIAL" << endl;
                    break;
                case CUBIC_SPLINE:
                    cout << "CUBIC_SPLINE" << endl;
                    break;
                }
        }

    // output the last point
    cout << " x, y: " << "(";
    cout << setw(5) << segments_->get(iLoop).x;
    cout << ", ";
    cout << setw(5) << segments_->get(iLoop).y;
    cout << ")" << endl;

    cout << "End of Envelope." << endl;
    cout << endl;
}


//----------------------------------------------------------------------------//
Envelope* Envelope::clone()
{
	// give them a new envelope that is made using the current envelope's data
	return new Envelope(*segments_);
}


//----------------------------------------------------------------------------//
m_value_type Envelope::getValue(m_value_type x, m_value_type totalLength)
{
	if (x > totalLength || x < 0) {
		// invalid values, so let them know
		cout << "Envelope::getValue(...): invalid x or totalLength parameter" << endl;
	}

	//Handle special case where absolute end of envelope is asked for
	if (x == totalLength) {
		return segments_->get(segments_->size() - 1).y;
	}

    // Bit-exact memoization: build the per-segment 101-value tables once, and
    // only (re)generate segment lengths when totalLength changes. The original
    // rebuilt an interpolator and stepped it up to 100 times on EVERY call.
    if (!valueTablesBuilt_) {
        buildValueTables_();
    }
    if (getValueCachedLength_ != totalLength) {
        generateLengths(totalLength);
        getValueCachedLength_ = totalLength;
        lastSearchX_ = -1; lastSearchCurrent_ = 0; lastSearchIndex_ = 0;
    }

    /* Find the segment this x refers to: a float prefix sum over the segment
       lengths. Monotonic-sweep resume (bit-exact): we cache the PRE-correction
       loop state (current = first prefix sum >= previous x, x_Index = number of
       segments added so far). For x >= previous x, continuing the loop from
       that state performs exactly the adds a from-scratch search would perform
       (the prefix sums are one fixed float sequence), so the reached state --
       and therefore the returned value -- is identical. The overshoot
       correction below runs on locals and never touches the cached
       pre-correction state (note (S-L)+L != S in floats, which is why the
       POST-correction state must not be cached). Callers that sweep x forward
       (the per-sample reverb mix at x = i/N) drop to O(1) amortized. */
    m_value_type current;
    int x_Index;
    if (x < lastSearchX_) {           // backwards: full restart
        current = 0;
        x_Index = 0;
    } else {                          // forwards (or equal): resume
        current = lastSearchCurrent_;
        x_Index = lastSearchIndex_;
    }
    while (current < x) {
	    current += generatedSegmentLengths_->get(x_Index);
	    x_Index++;
	}
    lastSearchX_ = x;
    lastSearchCurrent_ = current;
    lastSearchIndex_ = x_Index;

    //If we have overshot, lets go back
    if (current != x) {
	    x_Index--;
	    current -= generatedSegmentLengths_->get(x_Index);
	}

    // Which of the 101 quantized samples of this segment do we want?
    int sample = (int) (round(((x - current) / generatedSegmentLengths_->get(x_Index)) * 100.0));

    // Return the exact value the freshly-built interpolator would have produced.
    return valueTables_[x_Index][sample];
}

//----------------------------------------------------------------------------//
// Builds, once, the 101 exact interpolator outputs for every segment. Mirrors
// the interpolator construction that getValue() used to perform per call, so the
// cached values are bit-identical to the original iterate-to-sample result.
void Envelope::buildValueTables_()
{
    int numSegments = segments_->size() - 1;
    valueTables_.assign(numSegments, std::vector<m_value_type>());

    for (int seg = 0; seg < numSegments; seg++) {
        Interpolator *interp;
        switch (getSegmentInterpolationType(seg)) {
          case EXPONENTIAL:  interp = new ExponentialInterpolator(); break;
          case CUBIC_SPLINE: interp = new CubicSplineInterpolator(); break;
          default:           interp = new LinearInterpolator();      break;
        }
        interp->setSamplingRate(100);
        interp->setDuration(1.0);
        xy_point left  = getPoint(seg);
        xy_point right = getPoint(seg + 1);
        interp->addEntry(0, left.y);
        interp->addEntry(1, right.y);

        Iterator<m_value_type> it = interp->valueIterator();
        valueTables_[seg].reserve(101);
        for (int s = 0; s <= 100; s++) {
            valueTables_[seg].push_back(it.next());
        }
        delete interp;
    }
    valueTablesBuilt_ = true;
}


/* //The old getValue
m_value_type Envelope::getValue (m_time_type time,
                                                m_time_type totalLength)
{
    m_rate_type newSamplingRate = 50;
    m_rate_type oldSamplingRate = this->getSamplingRate();
    m_time_type oldDuration = this->getDuration();
    m_value_type returnValue = 0;

    if (time > totalLength || time < 0)
    {
        // they are dumb for passing in invalid values, so let them know
        cout << "Envelope::getValue(...): invalid time or totalLength parameter" << endl;
    }
    else
    {
        // set the dynamic variable parameters to temporary values
        this->setSamplingRate(newSamplingRate);
        this->setDuration(totalLength);

        // get an iterator for this shape
        Iterator<m_value_type> *tempIterator =
            new Iterator<m_value_type> (valueIterator());

        // we can calculate how many samples are needed for the floor (time)
        m_time_type timeSum = (m_time_type) floor ((double) time);
        int iNumGetValCalls = newSamplingRate * ((int) floor (time));

        // now, calculate how many samples are needed for (time - floor (time))
        while (timeSum < time)
        {
            timeSum = timeSum + (1.0 / newSamplingRate);    
            iNumGetValCalls++;
        }

        // it seems to be a few samples off...this is a hack to get good values
        iNumGetValCalls += 2;

        // iterate, based on the values calculated
        for (int iLoop = 0; iLoop < iNumGetValCalls; iLoop++)
        {
            returnValue = tempIterator->next();
        }

        // cout << "  iNumGetValCalls = " << iNumGetValCalls << endl;
        // cout << "  timeSum = " << timeSum << endl;

        // set the dynamic variable paramaters back to what they were
        this->setSamplingRate(oldSamplingRate);
        this->setDuration(oldDuration);
    }

    return (returnValue);

}
*/

//----------------------------------------------------------------------------//
void Envelope::defineShape()
{
  invalidateGetValueCache_();
	// set our local variables to copies of the variables passed in
	//maybe make this its own function???
	//===
	envelope_segment temp_seg;
	m_value_type temp = 0;
	for (int i = 0; i < segments_->size(); i++)	{
		temp_seg = segments_->get(i);
		temp_seg.length = temp_seg.x - temp;
		segments_->set(i, temp_seg);
		temp = temp_seg.x;
	}
}

//----------------------------------------------------------------------------//
void Envelope::addToShape(Collection<envelope_segment> segs)
{
  invalidateGetValueCache_();
	int startIndex;  // want to start at 0 if segments_ is empty; otherwise 1
	m_value_type x_Offset;
	if (segments_->size() == 0) {
		x_Offset = 0;
		startIndex = 0;
	} else {
		x_Offset = segments_->get(segments_->size() - 1).x;
		startIndex = 1;
		
		// verify that the last old point and the first new point have the
	    // same y value (usually zero), that they can be appended
	    if ((segments_->get(segments_->size() - 1).y) != (segs.get(0).y))
		{
		    cout << "WARNING: Y values of last point of first envelope\n";
		    cout << "and first point of second envelope do not match.\n";
		    cout << "This implementation corrupts the first point of the\n";
		    cout << "second envelope\n";
		}
	}
    m_value_type x_Last = x_Offset;
    envelope_segment segTemp;


    // for every element in the new collections
    for (int i = startIndex; i < segs.size(); i++)
	{
		// add the current segment and point + 1 to our existing Collections
	    segTemp = segs.get(i);
	    segTemp.x = segTemp.x + x_Offset;
	    segTemp.length = segTemp.x - x_Last;
	    x_Last = segTemp.x;
	    segments_->add(segTemp);
	}
}

//----------------------------------------------------------------------------//
void Envelope::addToShape(Envelope * shape)
{
  invalidateGetValueCache_();
    if (shape->segments_ == NULL)
	{
	    cout << "ERROR: Envelope to be added is undefined.\n";
	    return;
	}

    addToShape(*shape->segments_);
}

//----------------------------------------------------------------------------//

Envelope* Envelope::multiply(Envelope & env1, Envelope & env2)
{
	// new envelope that will be returned based on this:
  Collection<envelope_segment> result;

	// X and Y segments
	Collection<envelope_segment>* env1Segs = env1.getSegments();
	Collection<envelope_segment>* env2Segs = env2.getSegments();

	envelope_segment seg1, seg2, newseg;
	m_value_type length=0, max1=0, max2=0;
	// find the biggest x values
	for (int i = 0; i < env1Segs->size(); i++) {
		if (env1Segs->get(i).x > max1) {
			max1 = env1Segs->get(i).x;
		}
	}
	for (int i = 0; i < env2Segs->size(); i++) {
		if (env2Segs->get(i).x > max2) {
			max2 = env2Segs->get(i).x;
		}
	}
	if (max1 >= max2) {
		length = max1;
	} else {
		length = max2;
	}

	// multiply - find y values for every x value in either envelope
    Iterator<envelope_segment> segments1 = env1Segs->iterator();
    Iterator<envelope_segment> segments2 = env2Segs->iterator();

	envelope_segment *add_seg, *first_seg, *second_seg;

	first_seg = &seg1;
	second_seg = &seg2;
	add_seg = NULL;

	while (first_seg != NULL || second_seg != NULL) {
		// advance pointers depending on which seg was added last
		if (add_seg == &seg1) {
			printf("was 1\n");
			if (segments1.hasNext())
				seg1 = segments1.next();
			else
				first_seg = NULL;
		} else if (add_seg == &seg2) {
			printf("was 2\n");
			if (segments2.hasNext())
				seg2 = segments2.next();
			else
				second_seg = NULL;
		} else {	// this is the first time
			seg1 = segments1.next();
			seg2 = segments2.next();
			printf("first time\n");
		}

		m_value_type seg_len;		
		// determine which point to add now
		if ((((seg1.x / max1) * length) < ((seg2.x / max2) * length) 
				&& first_seg != NULL) || second_seg == NULL) {
			add_seg = &seg1;
			seg_len = max1;
		} else {
			add_seg = &seg2;
			seg_len = max2;
		}

		// MULTIPLICATION
		// right now you get all FLEXABLE segs, this is good
		// right now all EXPONENTIAL segs, this is BOGUS :-)

		newseg.x = (add_seg->x / seg_len) * length;
		newseg.y = env1.getValue(newseg.x, length) *
			env2.getValue(newseg.x, length);
		printf("x=%f, length=%f\n", newseg.x, length);
		newseg.lengthType = FLEXIBLE;
		newseg.interType = EXPONENTIAL;

		// check for duplicate of x value
		int duplicate = 0;
		Iterator <envelope_segment> res_it = result.iterator();

		while (res_it.hasNext()) {
			envelope_segment oldseg = res_it.next();
			if (newseg.x < (oldseg.x + ENV_MULT_EPSILON) &&	newseg.x > (oldseg.x - ENV_MULT_EPSILON)) {
				duplicate = 1;
				break;
			}
		}

		printf("doing x=%f", newseg.x);
		// add only if not a duplicate
		if (duplicate == 0) {
			result.add(newseg);
			printf(" added.");
		}
		printf("\n");
		if (first_seg == NULL)
			printf("first_seg=null\n");
		if (second_seg == NULL)
			printf("second_seg=null\n");
	} // while

    return (new Envelope(result));
}

//----------------------------------------------------------------------------//
Collection<xy_point>* Envelope::getPoints()
{
    //extract point information out of the segments
    Collection < xy_point > *points = new Collection<xy_point>();
    xy_point
	xy;
    for (int i = 0; i < segments_->size(); i++) {
	    xy.x = segments_->get(i).x;
	    xy.y = segments_->get(i).y;
	    points->add(xy);
	}
    return points;
}

//----------------------------------------------------------------------------//
Collection<envelope_segment>* Envelope::getSegments()
{
    // the point data are stored in a member variable, so return a copy of it
    return (new Collection<envelope_segment>(*segments_));
}

//----------------------------------------------------------------------------//
void Envelope::setSegment(int index, envelope_segment segment)
{
  invalidateGetValueCache_();
    index++;
    if (checkValidSegmentIndex(index)) {
	    if (checkValidSegmentIndex(index - 1)) {
		    segment.length = segment.x - (segments_->get(index - 1)).x;
		} else {
			segment.length = segment.x;
		}
	    segments_->set(index, segment);
	} else {
		cout << "Invalid setSegment index!!!\n";
	}
}

//----------------------------------------------------------------------------//
envelope_segment Envelope::getSegment(int index)
{
    index++;
    // lets the Collection class hand an invalid index
    return segments_->get(index);
}

//----------------------------------------------------------------------------//
void Envelope::setPoint(int index, xy_point point)
{
  invalidateGetValueCache_();
    envelope_segment seg;
    seg = segments_->get(index);
    seg.x = point.x;
    seg.y = point.y;
    segments_->set(index, seg);
}

//----------------------------------------------------------------------------//
xy_point Envelope::getPoint(int index)
{
    //lets the Collection class hand an invalid index
    envelope_segment tempseg;
    xy_point pt;
    tempseg = segments_->get(index);
    pt.x = tempseg.x;
    pt.y = tempseg.y;
    return pt;
}

//----------------------------------------------------------------------------//
void Envelope::addSegment(envelope_segment segment)
{
    //make sure it is an addable segment
    segment.length = segment.x - segments_->get(segments_->size() - 1).x;
    segments_->add(segment);
}

//----------------------------------------------------------------------------//
void Envelope::addPoint(xy_point point)
{
    envelope_segment seg;
    seg.x = point.x;
    seg.y = point.y;
    seg.lengthType = FIXED;
    seg.interType = LINEAR;
    addSegment(seg);
}

//----------------------------------------------------------------------------//
m_value_type Envelope::getSegmentLength(int index)
{
    index++;
    // lets the Collection class hand an invalid index
    return (segments_->get(index).length);
}

//----------------------------------------------------------------------------//
void Envelope::setSegmentLength(int index, m_value_type length)
{
  invalidateGetValueCache_();
    index++;
    // make sure the time is positive, otherwise bad things will happen
    if (length < 0) {
	    length = -length;
	}

    // assumes a valid index is supplied
    envelope_segment segTemp = segments_->get(index);
    segTemp.length = length;
    segments_->set(index, segTemp);
}

//----------------------------------------------------------------------------//
stretch_type Envelope::getSegmentLengthType(int index)
{
    index++;
    // assumes a valid index is supplied
    return (segments_->get(index).lengthType);
}

//----------------------------------------------------------------------------//
void Envelope::setSegmentLengthType(int index, stretch_type lengthType)
{
  invalidateGetValueCache_();
    index++;
    // assumes a valid index is supplied
    envelope_segment segTemp = segments_->get(index);
    segTemp.lengthType = lengthType;
    segments_->set(index, segTemp);
}

//----------------------------------------------------------------------------//
interpolation_type Envelope::getSegmentInterpolationType(int index)
{
    index++;
    // assumes a valid index is supplied
    return (segments_->get(index).interType);
}

//----------------------------------------------------------------------------//
void Envelope::setSegmentInterpolationType(int index, interpolation_type interType)
{
  invalidateGetValueCache_();
    index++;
    // assumes a valid index is supplied
    envelope_segment
	segTemp = segments_->get(index);
    segTemp.interType = interType;
    segments_->set(index, segTemp);
}

//----------------------------------------------------------------------------//
inline bool Envelope::checkValidSegmentIndex(int index)
{
    // check if index is valid
    return ((index >= 0) && (segments_->size() > index));
}

//----------------------------------------------------------------------------//
Iterator<m_value_type> Envelope::valueIterator()
{
    // optimize so that if the envelope is used repeatedly without changing
    // the time value that is assigned, interpolators are not re-created
    // every time
    if (totalLength_ != getDuration() || currentInterpolatorRate_ != getSamplingRate()) {

	    totalLength_ = getDuration();
	    currentInterpolatorRate_ = getSamplingRate();
	    // populate the lengths member variable
	    generateLengths(totalLength_);

	    // populate the interpolator member variable with interpolators, based
	    // on the lengths just set
	    addInterpolators(currentInterpolatorRate_);
	}

    // give them the big iterator
    return (Iterator<m_value_type> (new EnvelopeIterator(*interpolators_)));
}

//----------------------------------------------------------------------------//
// gpu-fast support: export the exact per-segment step structure valueIterator()
// walks (see Envelope.h). Reuses valueIterator()'s own regeneration logic so
// generatedSegmentLengths_ matches the current duration/samplingRate, then
// mirrors addInterpolators' (prev -> segments_[i+1].y) pairing and the
// interpolator's steps = (count)(segmentLength * rate) computation (the
// interpolator's internal rate rescale (duration/maxTime)*rate is exactly
// rate here because maxTime == duration for the 2-entry interpolators).
bool Envelope::exportDeviceSegments(std::vector<DeviceSegment>& out)
{
    out.clear();
    if (segments_->size() < 2) return false;

    if (totalLength_ != getDuration() || currentInterpolatorRate_ != getSamplingRate()) {
        totalLength_ = getDuration();
        currentInterpolatorRate_ = getSamplingRate();
        generateLengths(totalLength_);
        addInterpolators(currentInterpolatorRate_);
    }

    int numSegments = segments_->size() - 1;
    m_value_type prev = segments_->get(0).y;
    for (int i = 0; i < numSegments; i++) {
        int t = getSegmentInterpolationType(i);
        if (t == CUBIC_SPLINE) { out.clear(); return false; }
        DeviceSegment ds;
        ds.type  = (t == EXPONENTIAL) ? 1 : 0;
        ds.vFrom = prev;
        ds.vTo   = segments_->get(i + 1).y;
        m_value_type segLen = generatedSegmentLengths_->get(i);
        ds.steps = (long)((segLen - (m_value_type)0.0) *
                          (m_value_type)currentInterpolatorRate_);
        out.push_back(ds);
        prev = segments_->get(i + 1).y;
    }
    return true;
}

//----------------------------------------------------------------------------//
void Envelope::generateLengths(m_time_type totalLength)
{
	int iNumSegments = segments_->size() - 1;
	double dTotalFixedLength = 0;
	double dTotalFlexPercent = 0;
	double dScaleRatio = 1;

	totalLength_ = totalLength;

	// get rid of previously generated lengths
	generatedSegmentLengths_->clear();

	// figure out how they want to allocate length for each segment and populate
	// the generated lengths with their current values
	for (int iIndex = 0; iIndex < iNumSegments; iIndex++) {
		if (this->getSegmentLengthType(iIndex) == FIXED) {
			// if this segment has fixed-length, increment fixed length running total
			dTotalFixedLength += getSegmentLength(iIndex);
		} else {
			// else increment flex-length running total
			dTotalFlexPercent += getSegmentLength(iIndex);
		}
		// append their current value to our values for generated lengths
		generatedSegmentLengths_->add(getSegmentLength(iIndex));
	}

    // now, correct the generated values as necessary (i.e. the rest of this)

    // if they specified more FIXED length than actual length, or if there is no
    // FIEXIBLE length and they did not specify enough FIXED length, scale the
    // FIXED lengths to the actual length (and set total amount of FLEXIBLE length
    // to zero)
    if (dTotalFixedLength > totalLength_ 
			|| (dTotalFixedLength < totalLength_ && dTotalFlexPercent == 0)) {
		if (dTotalFixedLength > totalLength_) {
			cout << "WARNING: Duration specified was less than added FIXED lengths\n"
				 << "FIXED lengths will also be scaled down\n";
		} else {
			cout << "WARNING: Duration specified was more than added FIXED lengths\n"
				 << "Since no lengths are FLEXIBLE, FIXED lengths will be scaled up\n";
		}
	    // no flexible length is available
	    dTotalFlexPercent = 0;

	    // get a ratio to scale all fixed length entries by
	    dScaleRatio = totalLength_ / dTotalFixedLength;

	    for (int iIndex = 0; iIndex < iNumSegments; iIndex++) {
		    // scale every FIXED length value
		    if (this->getSegmentLengthType(iIndex) == FIXED) {
				generatedSegmentLengths_->set(iIndex, getSegmentLength(iIndex) * dScaleRatio);
			    setSegmentLength(iIndex, getSegmentLength(iIndex) * dScaleRatio);
			} else {
			    // all flex length entries are set to 0
			    generatedSegmentLengths_->set(iIndex, 0);
			    setSegmentLength(iIndex, 0);
			}
		}
	} else if (dTotalFlexPercent != 1.0 && dTotalFlexPercent != 0) {
		// else if their percentages do not add up to 100% (i.e. 1.0) and they are
    	// trying to use flex length, scale the flex length values
	    dScaleRatio = 1.0 / dTotalFlexPercent;

	    // scale percentages
	    for (int iIndex = 0; iIndex < iNumSegments; iIndex++) {
		    if (getSegmentLengthType(iIndex) == FLEXIBLE) {
				generatedSegmentLengths_->set(iIndex, getSegmentLength(iIndex) * dScaleRatio);
			}
		}
	}

    // if we have FLEXIBLE length to set, the values stored right now are only
    // ratios of length available, so we need to set them to actual lengths
    if (dTotalFlexPercent > 0) {
	    // calculate the length available to use for flex durations
	    double flexLengthAvailable = totalLength_ - dTotalFixedLength;

	    // set actual durations for FLEXIBLE length segments
	    for (int iIndex = 0; iIndex < iNumSegments; iIndex++) {
		    // if this is a flex-length entry, set length accordingly
		    if (getSegmentLengthType(iIndex) == FLEXIBLE) {
			    generatedSegmentLengths_->set(iIndex, generatedSegmentLengths_->get(iIndex) * flexLengthAvailable);
			}
		}
	}
}

//----------------------------------------------------------------------------//
void Envelope::addInterpolators(m_rate_type rate)
{
	int iNumSegments = segments_->size();
	// clear out interpolators that are stored, if there are any stored
	for (int i = 0; i < interpolators_->size(); i++) {
		delete(interpolators_->get(i));
	}
	interpolators_->clear();

    // initialize a new Collection for interpolators
    m_value_type prev = segments_->get(0).y;


    // for each segment that we have, store a new interpolator
    for (int iIndex = 0; iIndex < iNumSegments - 1; iIndex++) {
	    // add an interpolator for the current segment to the Collection
	    switch (getSegmentInterpolationType(iIndex)) {
		  case EXPONENTIAL:
		      interpolators_->add(new ExponentialInterpolator());
		      break;

		  case CUBIC_SPLINE:
		      interpolators_->add(new CubicSplineInterpolator());
		      break;

		  default:
		      interpolators_->add(new LinearInterpolator());
		      break;
		}

	    // set the Interpolator's sampling rate to the rate specified
	    interpolators_->get(iIndex)->setSamplingRate(rate);
	    m_value_type tempLengthValue = generatedSegmentLengths_->get(iIndex);

	    // add a start and end point to this interpolator:
	    interpolators_->get(iIndex)->addEntry(0, prev);
	    interpolators_->get(iIndex)->addEntry(tempLengthValue, segments_->get(iIndex + 1).y);

	    // set interpolator duration
	    interpolators_->get(iIndex)->setDuration(tempLengthValue);
	    prev = segments_->get(iIndex + 1).y;
	}
}

//----------------------------------------------------------------------------//
void Envelope::scale(m_value_type factor)
{
  invalidateGetValueCache_();
	int iNumSegments = segments_->size();
	envelope_segment segTemp;

	// for every point that we've got stored
	for (int iIndex = 0; iIndex < iNumSegments; iIndex++) {
		// scale this value by the factor given
		segTemp = segments_->get(iIndex);
		segTemp.y *= factor;
		segments_->set(iIndex, segTemp);
	}
}


//----------------------------------------------------------------------------//
m_value_type Envelope::getMaxValue()
{
    int iNumSegments = segments_->size();
    m_value_type maxVal = 0.0;

    // for every point that we've got stored
    for (int iIndex = 0; iIndex < iNumSegments; iIndex++) {
	    // if this point's value is greater than the max, set max = this value
	    if (segments_->get(iIndex).y > maxVal) {
		    maxVal = segments_->get(iIndex).y;
		}
	}

    return maxVal;
}

//----------------------------------------------------------------------------//
float Envelope::getScaledValueNew(double checkPoint, float coef) 
{
	// the value we will return
	float value;

	// scale it by the given coefficient
	scale(coef);

	// Find the envelope value
	value = getValue((float)checkPoint, 1.);

	return value;
}

//----------------------------------------------------------------------------//
/*Envelope * FitEnvelopeNew(double checkPoint, int offset) 
{
    int envNum;
    float coeff;

    Envelope * env;

    envNum = ReadComputeInt(checkPoint, offset);
    coeff = ReadComputeFloat(checkPoint, offset);
    env = envlib.getEnvelope(envNum);
    env->scale(coeff);

    cout << "Utility::FitEnvelope  - envNum=" << envNum << " coeff=" << coeff
     << " scaled" << endl;

    return env;
}*/

//----------------------------------------------------------------------------//
void Envelope::xml_print(ofstream & xmlOutput)
{
    DynamicVariable *pnt2dyn = this;

    xmlOutput << "<dv id=\"" << (long) pnt2dyn << "\">" << endl;
    xmlOutput << "\t<dv_type value=\"envelope\" />" << endl;
    xmlOutput << "\t<duration value=\"" << getDuration() << "\" />" << endl;
    xmlOutput << "\t<rate value=\"" << getSamplingRate() << "\" />" << endl;

    //Print out private vars and collections for Envelope here
    xmlOutput << "\t<totalLength value=\"" << totalLength_ << "\" />" << endl;
    xmlOutput << "\t<currentInterpolatorRate value=\"" <<
	currentInterpolatorRate_ << "\" />" << endl;
    envelope_segment myseg;

    Iterator < envelope_segment > it_segments = segments_->iterator();
    while (it_segments.hasNext()) {
	    myseg = it_segments.next();
	    xmlOutput << "\t<segment>" << endl;
	    xmlOutput << "\t\t<type value=\"";
	    switch (myseg.interType) {
		  case LINEAR:
		      xmlOutput << "LINEAR";
		      break;
		  case EXPONENTIAL:
		      xmlOutput << "EXPONENTIAL";
		      break;
		  case CUBIC_SPLINE:
		      xmlOutput << "CUBIC_SPLINE";
		      break;
		}
	    xmlOutput << "\" />" << endl;

	    xmlOutput << "\t\t<length type=\"";
	    switch (myseg.lengthType) {
		  case FIXED:
		      xmlOutput << "FIXED";
		      break;
		  case FLEXIBLE:
		      xmlOutput << "FLEXIBLE";
		      break;
		}
	    xmlOutput << "\" value=\"" << myseg.length << "\" />" << endl;

	    xmlOutput << "\t\t<xyPoint x=\"" << myseg.x << "\" ";
	    xmlOutput << "y=\"" << myseg.y << "\" />" << endl;
	    xmlOutput << "\t</segment>" << endl;
	}
    //Note: the interpolators_ and generatedSegmentLengths_ collections
    //      doesn't need to be backed up; it is used for internal
    //      optimization only, not storing data

    xmlOutput << "</dv>" << endl;
}

//----------------------------------------------------------------------------//
void Envelope::xml_print(ofstream & xmlOutput, list < DynamicVariable * >&dynObjs)
{
    DynamicVariable *pnt2dyn = this;

    //Print the pointer value as an ID, then the "meat" gets printed later
    xmlOutput << "\t\t\t\t<dv_type value=\"envelope\" />" << endl;
    xmlOutput << "\t\t\t\t<dv_ptr id=\"" << (long) pnt2dyn << "\" />" << endl;

    // Update dynamic variable list if necessary
    list < DynamicVariable * >::const_iterator dynit;
    for (dynit = dynObjs.begin(); dynit != dynObjs.end(); dynit++) {
	    if ((*dynit) == this)
		break;
	}
    if (dynit == dynObjs.end()) {
	    dynObjs.push_back(this);
	}

}

//----------------------------------------------------------------------------//
void Envelope::xml_read(XmlReader::xmltag * envtag) 
{
	char *value;
    envelope_segment seg;

  if((value = envtag->findChildParamValue("duration", "value")) != 0)
    setDuration(atof(value));

  if((value = envtag->findChildParamValue("rate", "value")) != 0) {
    setSamplingRate(atoi(value));
  }
    XmlReader::xmltag * segtag;
    while ((segtag = envtag->children->find("segment")) != 0) {
	    XmlReader::xmltag * xy = segtag->children->find("xyPoint");
	    if ((value = xy->getParamValue("x")) != 0) {
		    seg.x = atof(value);
		}
	    if ((value = xy->getParamValue("y")) != 0) {
		    seg.y = atof(value);
		}
	    if ((value = segtag->findChildParamValue("type", "value")) != 0) {
		    if (strcmp(value, "LINEAR") == 0)
			seg.interType = LINEAR;
		    if (strcmp(value, "EXPONENTIAL") == 0)
			seg.interType = EXPONENTIAL;
		    if (strcmp(value, "CUBIC_SPLINE") == 0)
			seg.interType = CUBIC_SPLINE;
		}
	    if ((value = segtag->findChildParamValue("length", "type")) != 0) {
		    if (strcmp(value, "FIXED") == 0)
				seg.lengthType = FIXED;
		    if (strcmp(value, "FLEXIBLE") == 0)
				seg.lengthType = FLEXIBLE;
		}
	    segments_->add(seg);
	}

    defineShape();
}
