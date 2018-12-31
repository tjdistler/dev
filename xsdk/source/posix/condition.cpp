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

#include "xsdk/condition.h"
#include "xsdk/exception.h"
#include "xsdk/timeutils.h"

#include <errno.h>

using namespace xsdk;

Condition::Condition(Mutex &lock) :
    _cond(),
    _lock(lock)
{
    int err = pthread_cond_init(&_cond, 0);
    if(err < 0)
        throw Exception(F_SYSTEM_ERROR, __FILE__, __LINE__, "Unable to allocate condition variable. error = %d.", err);
}

Condition::~Condition() throw()
{
    pthread_cond_destroy(&_cond);
}

void Condition::Wait()
{
    int err = pthread_cond_wait(&_cond, &_lock._lock);
    if(err < 0)
        throw Exception(F_SYSTEM_ERROR, __FILE__, __LINE__, "Unable to wait on the condition variable. error = %d.", err);
}

bool Condition::Wait(const timeval& timeout)
{
    struct timespec ts;
    ts.tv_sec  = timeout.tv_sec;
    ts.tv_nsec = timeout.tv_usec * 1000;

    int err = pthread_cond_timedwait(&_cond, &_lock._lock, &ts);
    if (err == ETIMEDOUT)
        return false;
    if (err < 0)
        throw Exception(F_SYSTEM_ERROR, __FILE__, __LINE__, "Unable to wait on the condition variable. error = %d.", err);
    return true;
}

void Condition::Signal()
{
    int err = pthread_cond_signal(&_cond);
    if(err < 0)
        throw Exception(F_SYSTEM_ERROR, __FILE__, __LINE__, "Unable to signal the condition variable. error = %d.", err);
}

void Condition::Broadcast()
{
    int err = pthread_cond_broadcast(&_cond);
    if(err < 0)
        throw Exception(F_SYSTEM_ERROR, __FILE__, __LINE__, "Unable to broadcast the condition variable. error = %d.", err);
}
