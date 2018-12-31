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

#if defined(IS_POSIX)
    #include <time.h>
    #define XSDK_CLOCK_TYPE   CLOCK_MONOTONIC
    #define XSDK_NANO_SEC     1000000000
#endif

#define XSDK_MICRO_SEC  1000000

using namespace xsdk;
using namespace xsdk::time;


uint64_t FastClock::Frequency()
{
    uint64_t freq;

#if defined(IS_WINDOWS)
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
#elif defined(IS_POSIX)
    struct timespec cfreq;
    clock_getres(XSDK_CLOCK_TYPE, &cfreq);
    /* 1/tick-duration converts time-between-ticks to ticks-per-second. */
    freq = (uint64_t)( 1.0 / ((double)cfreq.tv_sec + ((double)cfreq.tv_nsec / (double)XSDK_NANO_SEC)) );
#endif

    return freq;
}

uint64_t FastClock::Now()
{
    uint64_t now;

#if defined(IS_WINDOWS)
    QueryPerformanceCounter((LARGE_INTEGER*)&now);
#elif defined(IS_POSIX)
    struct timespec cnow;
    clock_gettime(XSDK_CLOCK_TYPE, &cnow);
    /* Convert to nanosecond ticks. */
    now = (uint64_t)cnow.tv_sec;
    now *= XSDK_NANO_SEC;
    now += cnow.tv_nsec;
#endif

    return now;
}

struct timeval FastClock::ElapsedTime(const uint64_t& start, const uint64_t& end)
{
    return ElapsedTime(start, end, Frequency());
}

struct timeval FastClock::ElapsedTime(const uint64_t& start, const uint64_t& end, const uint64_t& freq)
{
    uint64_t frac = (uint64_t)( (end - start) % freq );
    uint64_t sec  = (uint64_t)( (end - start) - frac );
    struct timeval duration;
    duration.tv_sec  = (long)( sec / freq );
    duration.tv_usec = (long)( frac / (freq / XSDK_MICRO_SEC) );
    return duration;
}
