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

#ifndef _XSDK_CONDITION_H_
#define _XSDK_CONDITION_H_

#include "xsdk/common.h"
#include "xsdk/mutex.h"
#include "xsdk/timeutils.h"

namespace xsdk
{
    /**
     * This class behaves similar to a posix condition.
     * @note On Windows, it is NOT safe to swap the "waiter" and the
     * "signaller" once using the condition. Only signal in one direction.
     *
     * @code
     * Mutex lock;
     * Condition cond(lock);
     * ...
     * lock.Acquire();
     * while (!ready)
     *     cond.Wait();
     * ...
     * lock.Release();
     *
     * lock.Acquire();
     * timeval timeout = {30, 0}; // 30 second timeout
     * bool timedout = false;
     * while (!ready & !timedout)
     *     timedout = !cond.Wait(Condition::MakeAbsolute(timeout));
     * if (timedout)
     *     ...
     * ...
     * lock.Release();
     * @endcode
     */
    class Condition
    {
    public:
        XSDK_API Condition(xsdk::Mutex &lock);
        XSDK_API virtual ~Condition() throw();

        /** 
         * Releases the mutex and blocks the caller until signalled. The mutex
         * is re-acquired before this method returns.
         */
        XSDK_API void Wait();

        /**
         * Releases the mutex and blocks the caller until the condition is
         * signalled, or the call timesout. The mutex is re-acquired before
         * this method returns (even if it times out).
         * @note The static MakeAbsolute() method can help create the timeout
         * value.
         * @param timeout The maximum amount of time to wait. This is an
         * absolute time (e.g. 1:30pm); not relative (e.g. 30 seconds).
         * @return 'true' if the condition was signalled, 'false' if it
         * timed out.
         */
        XSDK_API bool Wait(const timeval& timeout);

        /** Wakes up one thread waiting on the condition. */
        XSDK_API void Signal();

        /** Wakes up all threads waiting on the condition. */
        XSDK_API void Broadcast();

        /**
         * Takes a relative 'time' (e.g. 30 seconds) and converts it to 'now' +
         * 'time'.
         * @param time The relative time to convert.
         * @return A converted time.
         */
        XSDK_API static timeval MakeAbsolute(const timeval& time)
        {
            timeval now, abs;
            gettimeofday(&now, 0);
            timeradd(&now, const_cast<timeval*>(&time), &abs);
            return abs;
        }

    private:
        Condition(const Condition&);
        Condition& operator = (const Condition&);

    #if defined(IS_WINDOWS)
        enum { SIGNAL=0, BROADCAST=1, MAX_EVENTS=2 } ;
        HANDLE _events[Condition::MAX_EVENTS];
        volatile unsigned int _numWaitingThreads;
        CRITICAL_SECTION _threadCountLock;
    #elif defined(IS_POSIX)
        pthread_cond_t _cond;
    #endif

        xsdk::Mutex& _lock;

    }; //class

}; //namespace

#endif //_XSDK_CONDITION_H_
