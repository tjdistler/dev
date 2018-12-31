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

#ifndef _XSDK_THREAD_H_
#define _XSDK_THREAD_H_

#include "xsdk/common.h"
#include "xsdk/mutex.h"

namespace xsdk
{

    class Thread
    {
    public:
        XSDK_API Thread();
        XSDK_API virtual ~Thread() throw();

        /**
         * Starts the thread.
         * @param ptr User-defined data that is passed to the thread entry point.
         */
        XSDK_API void Start(void *ptr=0);

        /**
         * Waits for the thread to exit.
         * @return The thread return code.
         */
        XSDK_API int Join();

        /** @return 'true' if the thread is running. */
        XSDK_API bool Running() const;

        /**Aborts the thread. DON'T USE UNDER NORMAL CIRCUMSTANCES! */
        XSDK_API void Cancel();

    protected:
        /**
         * Thread execution starts here. Derived classes implement this.
         * @param ptr The user-defined data passed to the Start() method.
         * @return Thread return code.
         */
        XSDK_API virtual int ThreadMain(void *ptr) = 0;

    private:
        Thread(const Thread&);
        Thread& operator= (const Thread&) { return *this; }

        /** Thread execution REALLY starts here. */
        static void* _Start(void *obj);

        mutable xsdk::Mutex _lock;
        volatile bool       _running;
        volatile int        _returnCode;
        void*               _userData;

        #if defined(IS_WINDOWS)
            HANDLE _thread;
        #elif defined(IS_POSIX)
            pthread_t _thread;
        #endif

    }; //class

}; //namespace

#endif //_XSDK_THREAD_H_
