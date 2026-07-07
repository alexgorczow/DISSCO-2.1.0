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
//	SampledVariable.h
//
//----------------------------------------------------------------------------//

#ifndef __SAMPLED_VARIABLE_H
#define __SAMPLED_VARIABLE_H

//----------------------------------------------------------------------------//
#include "StandardHeaders.h"
#include "Types.h"
#include "DynamicVariable.h"
#include "AbstractIterator.h"

#include <vector>
#include <utility>

/**
*	A DynamicVariable that stores a fixed per-sample sequence in run-length
*	form. It exists to replace the pathological use of a LinearInterpolator as
*	the LOUDNESS_SCALAR (Loudness added ONE interpolator entry per audio sample,
*	and the iterator then rebuilt a list<Entry> of the same length -- hundreds
*	of MB per concurrent sound). This variable stores the same per-sample values
*	as run-length pairs and its iterator emits them directly, with no per-sample
*	list, so memory collapses (and constant stretches cost O(1)).
*
*	valueIterator() emits exactly the stored per-sample sequence; reading past
*	the end holds the final value (matching the old iterator's behavior).
**/
class SampledVariable : public DynamicVariable
{
public:

	SampledVariable();

	/**
	*	Build from a per-sample array. The array must already be the EXACT
	*	sequence the consumer will read (the caller is responsible for any
	*	end-of-sequence adjustments). Internally run-length compressed.
	**/
	void setFromArray(const std::vector<m_value_type>& values);

	DynamicVariable* clone();
	Iterator<m_value_type> valueIterator();
	void scale(m_value_type factor);
	m_value_type getMaxValue();

	void xml_print(ofstream& xmlOutput, list<DynamicVariable*>& dynObjs);
	void xml_print(ofstream& xmlOutput);

	//------------------------------------------------------------------------//
	class SampledVariableIterator : public AbstractIterator<m_value_type>
	{
	public:
		// Holds a pointer to the parent's runs (valid for the parent's
		// lifetime, which outlives the iterator on the render path).
		SampledVariableIterator(
			const std::vector<std::pair<m_value_type, m_sample_count_type> >* runs);
		SampledVariableIterator* clone();
		bool hasNext();
		m_value_type& next();
	private:
		const std::vector<std::pair<m_value_type, m_sample_count_type> >* runs_;
		std::size_t runIdx_;
		m_sample_count_type emittedInRun_;
		m_value_type current_;
	};

private:
	// (value, run-length) pairs; total samples = sum of the lengths.
	std::vector<std::pair<m_value_type, m_sample_count_type> > runs_;
	m_value_type maxValue_;
};

//----------------------------------------------------------------------------//
#endif //__SAMPLED_VARIABLE_H
