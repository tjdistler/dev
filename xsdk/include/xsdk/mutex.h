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

#ifndef _XSDK_MUTEX_H_
#define _XSDK_MUTEX_H_

#include "xsdk/common.h"

namespace xsdk
{
    class Condition;

    /** A recursive mutex */
    class Mutex
    {
        friend class Condition;

    public:
        XSDK_API Mutex();
        XSDK_API virtual ~Mutex() throw();

        /** Blocks until the mutex is aquired. */
        XSDK_API void Acquire();

        /** Releases the aquired mutex. */
        XSDK_API void Release();

        /**
         * Attempts to aquire the mutex. Doesn't block.
         * @return 'true' if the lock was available and is now aquired. 'false' if another thread held the lock.
         */
        XSDK_API bool TryAcquire();

    private:
        Mutex(const Mutex&);
        Mutex& operator= (const Mutex&);

        #if defined(IS_WINDOWS)
            CRITICAL_SECTION _lock;
        #elif defined(IS_POSIX)
            pthread_mutex_t _lock;
            pthread_mutexattr_t _attr;
        #endif

    }; //class

}; //namespace

#endif //_XSDK_MUTEX_H_

