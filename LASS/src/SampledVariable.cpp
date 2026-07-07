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
//	SampledVariable.cpp
//
//----------------------------------------------------------------------------//

#ifndef __SAMPLED_VARIABLE_CPP
#define __SAMPLED_VARIABLE_CPP

#include "SampledVariable.h"

//----------------------------------------------------------------------------//
SampledVariable::SampledVariable()
	: maxValue_(0)
{
}

//----------------------------------------------------------------------------//
// Run-length compress the per-sample sequence. Consecutive samples with the
// exact same float value collapse into one run, so a constant loudness stretch
// becomes a single (value,count) pair. Bit-exact: the iterator re-emits the same
// float for each sample of the run.
void SampledVariable::setFromArray(const std::vector<m_value_type>& values)
{
	runs_.clear();
	maxValue_ = 0;
	if (values.empty()) return;

	maxValue_ = values[0];
	m_value_type runVal = values[0];
	m_sample_count_type runLen = 1;
	for (std::size_t i = 1; i < values.size(); ++i)
	{
		if (values[i] > maxValue_) maxValue_ = values[i];
		if (values[i] == runVal)
		{
			runLen++;
		}
		else
		{
			runs_.push_back(std::make_pair(runVal, runLen));
			runVal = values[i];
			runLen = 1;
		}
	}
	runs_.push_back(std::make_pair(runVal, runLen));
}

//----------------------------------------------------------------------------//
DynamicVariable* SampledVariable::clone()
{
	return new SampledVariable(*this);
}

//----------------------------------------------------------------------------//
Iterator<m_value_type> SampledVariable::valueIterator()
{
	return Iterator<m_value_type>(new SampledVariableIterator(&runs_));
}

//----------------------------------------------------------------------------//
void SampledVariable::scale(m_value_type factor)
{
	maxValue_ *= factor;
	for (std::size_t i = 0; i < runs_.size(); ++i)
		runs_[i].first *= factor;
}

//----------------------------------------------------------------------------//
m_value_type SampledVariable::getMaxValue()
{
	return maxValue_;
}

//----------------------------------------------------------------------------//
// Deprecated/legacy LASS XML path (not used by the cmod render pipeline; the
// .particel output does not include LOUDNESS_SCALAR). Emit a constant with the
// first value so anything that does parse it gets a harmless, valid variable.
void SampledVariable::xml_print(ofstream& xmlOutput)
{
	m_value_type v = runs_.empty() ? 0 : runs_[0].first;
	xmlOutput << "\t\t\t\t<dv_type value=\"constant\" />" << endl;
	xmlOutput << "\t\t\t\t<duration value=\"" << getDuration() << "\" />" << endl;
	xmlOutput << "\t\t\t\t<rate value=\"" << getSamplingRate() << "\" />" << endl;
	xmlOutput << "\t\t\t\t<value value=\"" << v << "\" />" << endl;
}

//----------------------------------------------------------------------------//
void SampledVariable::xml_print(ofstream& xmlOutput, list<DynamicVariable*>& dynObjs)
{
	(void) dynObjs.size();
	xml_print(xmlOutput);
}

//----------------------------------------------------------------------------//
// SampledVariable::SampledVariableIterator
//----------------------------------------------------------------------------//
SampledVariable::SampledVariableIterator::SampledVariableIterator(
	const std::vector<std::pair<m_value_type, m_sample_count_type> >* runs)
	: runs_(runs), runIdx_(0), emittedInRun_(0),
	  current_(runs && !runs->empty() ? (*runs)[0].first : 0)
{
}

//----------------------------------------------------------------------------//
SampledVariable::SampledVariableIterator*
SampledVariable::SampledVariableIterator::clone()
{
	SampledVariableIterator* c = new SampledVariableIterator(runs_);
	c->runIdx_ = runIdx_;
	c->emittedInRun_ = emittedInRun_;
	c->current_ = current_;
	return c;
}

//----------------------------------------------------------------------------//
bool SampledVariable::SampledVariableIterator::hasNext()
{
	return runs_ != 0 && runIdx_ < runs_->size();
}

//----------------------------------------------------------------------------//
m_value_type& SampledVariable::SampledVariableIterator::next()
{
	if (runs_ != 0 && runIdx_ < runs_->size())
	{
		current_ = (*runs_)[runIdx_].first;
		emittedInRun_++;
		if (emittedInRun_ >= (*runs_)[runIdx_].second)
		{
			runIdx_++;
			emittedInRun_ = 0;
		}
	}
	// Past the end: hold the last value (matches the old interpolator behavior).
	return current_;
}

//----------------------------------------------------------------------------//
#endif //__SAMPLED_VARIABLE_CPP
