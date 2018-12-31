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

#ifndef _XSDK_TEST_MUTEX_H_
#define _XSDK_TEST_MUTEX_H_

#include "xsdk/mutex.h"
#include "xsdk/thread.h"
#include "BaseTest.h"

class TestMutex : public BaseTest, public xsdk::Thread
{
public:
    TestMutex() {}
    virtual ~TestMutex() throw() {}

    TEST_BEGIN("xsdk::Mutex")
        TEST_FUNC(TestLock);
        TEST_FUNC(TestRecursiveLock);
        TEST_FUNC(TestManyLocks);
    TEST_END

private:
    virtual int ThreadMain(void *ptr);

    virtual void Init() throw();
    virtual void Fini() throw();

    void TestLock();
    void TestRecursiveLock();
    void TestManyLocks();

    xsdk::Mutex _lock;
    volatile bool _mainFlag;
    volatile bool _threadFlag;
    volatile bool _threadStarted;

}; //class

#endif
