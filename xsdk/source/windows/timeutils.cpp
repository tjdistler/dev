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

#include "xsdk/timeutils.h"

#if defined(_MSC_VER) || defined(_MSC_EXTENSIONS)
    #define DELTA_EPOCH_IN_MICROSECS  11644473600000000Ui64
#else
    #define DELTA_EPOCH_IN_MICROSECS  11644473600000000ULL
#endif

int gettimeofday(struct timeval* tv, void* obsolete)
{
    FILETIME ft;
    uint64_t tmpres = 0;

    if( tv )
    {
        GetSystemTimeAsFileTime( &ft );

        tmpres |= ft.dwHighDateTime;
        tmpres <<= 32;
        tmpres |= ft.dwLowDateTime;

        tmpres /= 10;

        tmpres -= DELTA_EPOCH_IN_MICROSECS;

        tv->tv_sec = (long)(tmpres / 1000000UL);
        tv->tv_usec = (int)(tmpres % 1000000UL);

        return 0;
    }

    return -1;
}

void sleep(unsigned int seconds)
{
    Sleep((DWORD)(seconds * 1000));
}

void usleep(unsigned int usec)
{
    Sleep((DWORD)(usec / 1000));
}

void timeradd(struct timeval *a, struct timeval *b, struct timeval *res)
{
    res->tv_sec  = a->tv_sec  + b->tv_sec; 
    res->tv_usec = a->tv_usec + b->tv_usec;
    if( res->tv_usec >= 1000000 )
    {
        res->tv_sec++;
        res->tv_usec -= 1000000;
    }
}

void timersub(struct timeval *a, struct timeval *b, struct timeval *res)
{
    res->tv_sec  = a->tv_sec  - b->tv_sec;
    res->tv_usec = a->tv_usec - b->tv_usec;
    if( res->tv_usec < 0 )
    {
        res->tv_sec--;
        res->tv_usec += 1000000;
    }
}
