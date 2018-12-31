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

#include "TestCondition.h"
#include "xsdk/timeutils.h"
#include "xsdk/guard.h"

using namespace xsdk;

#define NUM_THREADS     10

// We can't use conditions yet... they aren't tested.
#define BUSYWAIT(flag) \
    XSDK_MACRO_BEGIN \
    for (int ii=0; ii<20 && !(flag); ++ii) \
        usleep(500000); \
    XSDK_MACRO_END

// Thread for testing broadcasts.
class TestBCastThread : public Thread
{
public:
    TestBCastThread() : _wait(true) {}
    virtual ~TestBCastThread() throw() {}
    void Setup(Mutex* lock, Condition* cond) { _lock = lock; _cond = cond; }
    void Finish() { _wait = false; }
protected:
    Mutex* _lock;
    Condition* _cond;
    bool _wait;
    virtual int ThreadMain(void *ptr)
    {
        Guard g(*_lock);
        while (_wait)
            _cond->Wait();
        return 42;
    }
};


TestCondition::TestCondition() :
_lock(),
_cond(_lock)
{
}

void TestCondition::Init() throw()
{
    _wait = true;
    _threadStarted = false;
    _threadDone = false;
}

void TestCondition::Fini() throw()
{
    this->Join();
}

void TestCondition::TestSignal()
{
    // Have the thread wait on the condition, then signal it.
    _testType = TTYPE_WAIT;
    TEST_ASSERT_NO_THROW( this->Start() );
    BUSYWAIT( _threadStarted );
    TEST_ASSERT( _threadStarted );

    sleep(1);
    {
        Guard g(_lock);
        TEST_ASSERT( _threadDone == false ); // Assert thread is blocked
        _wait = false;
        TEST_ASSERT_NO_THROW( _cond.Signal() );
    }
    sleep(1);
    TEST_ASSERT( _threadDone == true ); // Assert thread unblocked
}

void TestCondition::TestWait()
{
    // Wait on the condition and have the thread signal it.
    _testType = TTYPE_SIGNAL;
    TEST_ASSERT_NO_THROW( this->Start() );
    BUSYWAIT( _threadStarted );
    TEST_ASSERT( _threadStarted );

    {
        Guard g(_lock);
        _wait = true;
        while (_wait) {
            TEST_ASSERT_NO_THROW( _cond.Wait() );
        }
    }
    sleep(1);
    TEST_ASSERT( _threadDone == true );
}

void TestCondition::TestTimeout()
{
    // Wait on the condition but timeout.
    _testType = TTYPE_SIGNAL;
    TEST_ASSERT_NO_THROW( this->Start() );
    BUSYWAIT( _threadStarted );
    TEST_ASSERT( _threadStarted );

    {
        Guard g(_lock);
        _wait = true;
        timeval time;
        time.tv_sec = 1;
        time.tv_usec = 0;
        bool timedout = false;
        while (_wait & !timedout) {
            TEST_ASSERT_NO_THROW( timedout = !_cond.Wait(Condition::MakeAbsolute(time)) );
        }
        TEST_ASSERT( timedout == true );
        TEST_ASSERT( _wait == true );

        time.tv_sec = 5;
        timedout = false;
        while (_wait & !timedout) {
            TEST_ASSERT_NO_THROW( timedout = !_cond.Wait(Condition::MakeAbsolute(time)) );
        }
        TEST_ASSERT( timedout == false );
        TEST_ASSERT( _wait == false );
    }
    sleep(1);
    TEST_ASSERT( _threadDone == true );
}

void TestCondition::TestBroadcast()
{
    TestBCastThread threads[NUM_THREADS];
    for (int ii=0; ii<NUM_THREADS; ++ii) {
        TEST_ASSERT_NO_THROW( threads[ii].Setup(&_lock, &_cond) );
        TEST_ASSERT_NO_THROW( threads[ii].Start() );
    }

    sleep(1);

    for (int ii=0; ii<NUM_THREADS; ++ii)
        TEST_ASSERT( threads[ii].Running() == true );
    
    for (int ii=0; ii<NUM_THREADS; ++ii)
        threads[ii].Finish();

    TEST_ASSERT_NO_THROW( _cond.Broadcast() );
    sleep(1);

    for (int ii=0; ii<NUM_THREADS; ++ii)
        TEST_ASSERT( threads[ii].Running() == false );

    for (int ii=0; ii<NUM_THREADS; ++ii)
        TEST_ASSERT( threads[ii].Join() == 42 );
}

int TestCondition::ThreadMain(void *ptr)
{
    _threadStarted = true;

    if (_testType == TTYPE_WAIT)
    {
        Guard g(_lock);
        while (_wait)
            _cond.Wait();
    }
    else
    {
        sleep(2);
        Guard g(_lock);
        _wait = false;
        _cond.Signal();
    }

    _threadDone = true;
    return 0;
}
