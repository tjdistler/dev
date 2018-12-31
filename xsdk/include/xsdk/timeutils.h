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

#ifndef _XSDK_TIME_UTILS_H_
#define _XSDK_TIME_UTILS_H_

#include "xsdk/common.h"

#if defined(IS_WINDOWS)
    #if !defined(XSDK_EXCLUDE_TIME_FUNCTIONS)
        int gettimeofday(struct timeval* tv, void* obsolete);
        void sleep(unsigned int seconds);
        void usleep(unsigned int usec);
        void timeradd(struct timeval *a, struct timeval *b, struct timeval *res);
        void timersub(struct timeval *a, struct timeval *b, struct timeval *res);
    #endif
#elif defined(IS_POSIX)
    #include <sys/time.h>
    #include <unistd.h>
#endif

/** 
 * Calculates the difference between 2 timevals in microseconds.
 * @param start The time at the start of the interval.
 * @param end The time at the end of the interval.
 * @return The duration between start and end (in microseconds).
 */
inline long TimeDuration(const struct timeval &start, const struct timeval &end)
{
    long result;
    result =  (end.tv_sec   * 1000000) + end.tv_usec;
    result -= (start.tv_sec * 1000000) + start.tv_usec;
    return result;
}

/** Converts a timeval to it's microseconds representation.
 * @note The method returns a 'long' which may overflow. This method is best
 * used when the timeval represents a duration, not wall time.
 * @param t The timeval to convert.
 * @return The timeval represented as microseconds.
 */
inline long TimevalToUSec(const struct timeval &t)
{
    return (long)( (t.tv_sec * 1000000) + t.tv_usec );
}

#endif //_XSDK_TIME_UTILS_H_
