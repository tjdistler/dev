/*
 * Copyright (c) 2011, Tony DiCroce, Tom Distler (tdistler.com)
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

#include "xsdk/mutex.h"
#include "xsdk/exception.h"

#include <errno.h>

using namespace xsdk;

Mutex::Mutex() :
    _lock()
{
    int err = pthread_mutexattr_init(&_attr);
    if( err < 0 )
        throw Exception(F_SYSTEM_ERROR, __FILE__, __LINE__, "Unable to allocate mutex attributes. error = %d.", err);

    err = pthread_mutexattr_settype(&_attr, PTHREAD_MUTEX_RECURSIVE);
    if( err < 0 )
        throw Exception(F_SYSTEM_ERROR, __FILE__, __LINE__, "Unable to specify recursive mutex. error = %d.", err);

    err = pthread_mutex_init(&_lock, &_attr);
    if( err < 0 )
        throw Exception(F_SYSTEM_ERROR, __FILE__, __LINE__, "Unable to allocate a mutex. error = %d.", err);
}

Mutex::~Mutex() throw()
{
    pthread_mutex_destroy(&_lock);
    pthread_mutexattr_destroy(&_attr);
}

void Mutex::Acquire()
{
    int err = pthread_mutex_lock(&_lock);
    if( err != 0 )
        throw Exception(F_SYSTEM_ERROR, __FILE__, __LINE__, "pthread_mutex_lock() failed. error = %d.", err);
}

void Mutex::Release()
{
    int err = pthread_mutex_unlock(&_lock);
    if( err != 0 )
        throw Exception(F_SYSTEM_ERROR, __FILE__, __LINE__, "pthread_mutex_unlock() failed. error = %d.", err);
}

bool Mutex::TryAcquire()
{
    int err = pthread_mutex_trylock(&_lock);
    if( err == 0 )
        return true;
    else if( err == EBUSY )
        return false;
    throw Exception(F_SYSTEM_ERROR, __FILE__, __LINE__, "pthread_mutex_trylock() failed. error = %d.", err);
}
