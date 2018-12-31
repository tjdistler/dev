/*
 * Copyright (c) 2011, Tom Distler (tdistler.com)
 * All rights reserved.
 *
 * The BSD License
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice, 
 *   this list of conditions and the following disclaimer.
 *
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * - Neither the name of the XSDK nor the names of its contributors may
 *   be used to endorse or promote products derived from this software without
 *   specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef _XSDK_FAST_CLOCK_H_
#define _XSDK_FAST_CLOCK_H_

#include "xsdk/common.h"

namespace xsdk {
namespace time
{
    /** A class for high-performance timing. */
    class FastClock
    {
    public:
        /** 
         * Returns the frequency of the clock. The frequency doesn't vary over
         * time, so it is safe to cache to this value.
         * @return The number of clock 'ticks' per second.
         */
        XSDK_API static uint64_t Frequency();

        /**
         * Returns the current clock value.
         * @return The current clock value in 'ticks'.
         */
        XSDK_API static uint64_t Now();

        /**
         * Calculates the amount of time that has elapsed between start and end.
         * @param start The clock value at the start of the interval.
         * @param end The clock value at the end of the interval.
         * @return The elapsed time in seconds.
         */
        XSDK_API static struct timeval ElapsedTime(const uint64_t& start, const uint64_t& end);

        /**
         * Calculates the amount of time that has elapsed between start and end.
         * @param start The clock value at the start of the interval.
         * @param end The clock value at the end of the interval.
         * @param freq The clock frequency.
         * @return The elapsed time.
         */
        XSDK_API static struct timeval ElapsedTime(const uint64_t& start, const uint64_t& end, const uint64_t& freq);

    }; //class

}; }; //namespace

#endif //_XSDK_FAST_CLOCK_H_
