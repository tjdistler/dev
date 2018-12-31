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

#include "xsdk/time/fast_clock.h"
#include "xsdk/timeutils.h"
#include "TestFastClock.h"

#define INTERVAL    2        // sec
#define TOLERANCE   250000   // usec
#define LOOP_TIME   10000000 // usec

#define SEC_TO_USEC(t)   ((t) * 1000000)

using namespace xsdk;
using namespace xsdk::time;

void TestFastClock::TestBasic()
{
    timeval duration;
    uint64_t start, end, freq;

    TEST_ASSERT_NO_THROW( start = FastClock::Now() );
    sleep(INTERVAL);
    TEST_ASSERT_NO_THROW( end = FastClock::Now() );
    TEST_ASSERT_NO_THROW( duration = FastClock::ElapsedTime(start, end) );
    TEST_ASSERT( TimevalToUSec(duration) > ( SEC_TO_USEC(INTERVAL) - TOLERANCE ) );
    TEST_ASSERT( TimevalToUSec(duration) < ( SEC_TO_USEC(INTERVAL) + TOLERANCE ) );

    TEST_ASSERT_NO_THROW( freq  = FastClock::Frequency() );
    TEST_ASSERT_NO_THROW( duration = FastClock::ElapsedTime(start, end, freq) );
    TEST_ASSERT( TimevalToUSec(duration) > ( SEC_TO_USEC(INTERVAL) - TOLERANCE ) );
    TEST_ASSERT( TimevalToUSec(duration) < ( SEC_TO_USEC(INTERVAL) + TOLERANCE ) );
}

void TestFastClock::TestMonotonic()
{
    uint64_t cur, prev;
    timeval begin, now;
    long runtime = 0;
    uint64_t max = (uint64_t)-1;

    gettimeofday(&begin, 0);
    TEST_ASSERT_NO_THROW( prev = FastClock::Now() );
    while (runtime < LOOP_TIME)
    {
        TEST_ASSERT_NO_THROW( cur = FastClock::Now() );
        TEST_ASSERT( cur - prev < max / 2 );
        prev = cur;

        gettimeofday(&now, 0);
        runtime = TimeDuration(begin, now);
    }
}
