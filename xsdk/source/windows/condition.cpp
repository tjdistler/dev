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

using namespace xsdk;

Condition::Condition(Mutex &lock) :
    _lock(lock),
    _numWaitingThreads(0)
{
    InitializeCriticalSection(&_threadCountLock);

    // We have to use both types of Win32 events to get the exact behavior we
    // expect from a POSIX condition variable. We use an auto-reset event for
    // the signal() call so only a single thread is woken up. We rely on the
    // "stickyness" of a manual-reset event to properly implement broadcast().
    _events[SIGNAL]    = CreateEvent(0, false, false, 0); // auto-reset event
    _events[BROADCAST] = CreateEvent(0, true, false, 0);  // manual-reset event
    if(_events[SIGNAL] == 0 || _events[BROADCAST] == 0)
        throw std::exception("Unable to allocate Win32 event objects.");
}

Condition::~Condition() throw()
{
    if (_events[SIGNAL])
        CloseHandle(_events[SIGNAL]);
    if (_events[BROADCAST] )
        CloseHandle(_events[BROADCAST]);

    DeleteCriticalSection(&_threadCountLock);
}

void Condition::Wait()
{
    // We keep track of the number of waiting threads so we can guarantee that
    // all waiting threads are woken up by a broadcast() call.
    EnterCriticalSection(&_threadCountLock);
    _numWaitingThreads++;
    LeaveCriticalSection(&_threadCountLock);

    // Release the lock before waiting... this is standard POSIX condition
    // semantics.
    _lock.Release();

    // NOTE: We avoid the "lost wakeup" bug common in many Win32 condition
    // implementations because we are using a manual-reset event. If we were
    // only using an auto-reset event, it would be possible to lose a signal
    // right here due to a race condition.

    // Wait for either event to be signaled.
    DWORD result = WaitForMultipleObjects(MAX_EVENTS, _events, false, INFINITE);
    if( (result < WAIT_OBJECT_0) || (result > WAIT_OBJECT_0 + MAX_EVENTS) )
        throw Exception(F_SYSTEM_ERROR, __FILE__, __LINE__, "Error 0x%08X occurred what waiting for the Win32 event objects.", GetLastError());

    // The following code is only required for the broadcast(). Basically, if
    // broadcast was called AND we are the last waiting thread, then reset the
    // manual-reset event.
    EnterCriticalSection(&_threadCountLock);
    _numWaitingThreads--;
    bool lastThreadWaiting = (result == WAIT_OBJECT_0 + BROADCAST) && (_numWaitingThreads == 0);
    LeaveCriticalSection(&_threadCountLock);

    if(lastThreadWaiting)
        ResetEvent(_events[BROADCAST]);

    // Reacquire the lock before returning... again, this is standard POSIX
    // condition semantics.
    _lock.Acquire();
}

bool Condition::Wait(const timeval& timeout)
{
    //
    // NOTE: See the Wait() method for more details about how this works
    // on Windows.
    //

    EnterCriticalSection(&_threadCountLock);
    _numWaitingThreads++;
    LeaveCriticalSection(&_threadCountLock);

    timeval diff, now;
    gettimeofday(&now, 0);
    timersub(const_cast<timeval*>(&timeout), &now, &diff);
    DWORD msec = (diff.tv_sec * 1000) + (diff.tv_usec / 1000);

    _lock.Release();

    DWORD result = WaitForMultipleObjects(MAX_EVENTS, _events, false, (DWORD)msec);
    bool timedOut = false;
    if (result == WAIT_TIMEOUT)
        timedOut = true;
    if (!timedOut && ( (result < WAIT_OBJECT_0) || (result > WAIT_OBJECT_0 + MAX_EVENTS) ) )
        throw Exception(F_SYSTEM_ERROR, __FILE__, __LINE__, "Error 0x%08X occurred what waiting for the Win32 event objects.", GetLastError());

    EnterCriticalSection(&_threadCountLock);
    _numWaitingThreads--;
    bool lastThreadWaiting = (result == WAIT_OBJECT_0 + BROADCAST) && (_numWaitingThreads == 0);
    LeaveCriticalSection(&_threadCountLock);

    if (lastThreadWaiting)
        ResetEvent(_events[BROADCAST]);

    _lock.Acquire();

    return !timedOut;
}

void Condition::Signal()
{
    EnterCriticalSection(&_threadCountLock);
    bool threadsWaiting = _numWaitingThreads > 0;
    LeaveCriticalSection(&_threadCountLock);

    if(threadsWaiting)
        SetEvent(_events[SIGNAL]);
}

void Condition::Broadcast()
{
    EnterCriticalSection(&_threadCountLock);
    bool threadsWaiting = _numWaitingThreads > 0;
    LeaveCriticalSection(&_threadCountLock);

    if(threadsWaiting)
        SetEvent(_events[BROADCAST]);
}

