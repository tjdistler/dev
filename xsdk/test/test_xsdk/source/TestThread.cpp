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

#include "TestThread.h"
#include "xsdk/timeutils.h"

// We can't use conditions yet... they aren't tested.
#define BUSYWAIT(flag) \
    XSDK_MACRO_BEGIN \
    for (int ii=0; ii<20 && !(flag); ++ii) \
        usleep(500000); \
    XSDK_MACRO_END

using namespace xsdk;

void TestThread::Init() throw()
{
    _mainFlag = false;
    _threadFlag = false;
    _threadStarted = false;
}

void TestThread::Fini() throw()
{
    this->Join();
}

void TestThread::TestStartStop()
{
    TEST_ASSERT_NO_THROW(this->Start());
    BUSYWAIT(_threadStarted);
    TEST_ASSERT(_threadStarted);
    _mainFlag = true;
    int result = 0;
    TEST_ASSERT_NO_THROW(result = this->Join());
    TEST_ASSERT(!this->Running());
    TEST_ASSERT(result == 42);
    TEST_ASSERT(_threadFlag);
}

void TestThread::TestCancel()
{
    TEST_ASSERT_NO_THROW(this->Start());
    BUSYWAIT(_threadStarted);
    TEST_ASSERT(_threadStarted);
    TEST_ASSERT_NO_THROW(this->Cancel());
}

int TestThread::ThreadMain(void *ptr)
{
    _threadStarted = true;
    BUSYWAIT(_mainFlag);
    if (!_mainFlag) {
        printf(">> TestThread::ThreadMain timed out waiting for main thread.\n");
        return -1;
    }
    _threadFlag = true;
    return 42;
}
