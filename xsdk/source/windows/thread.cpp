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

#include "xsdk/thread.h"
#include "xsdk/guard.h"
#include "xsdk/exception.h"

using namespace xsdk;

Thread::Thread() :
    _running(false),
    _returnCode(0),
    _userData(0),
    _thread(0)
{
}

Thread::~Thread() throw()
{
    try {
        Join();
    }
    catch(...) {}
}

void Thread::Start(void *ptr)
{
    Guard lock(_lock);
    if (_running)
        return;

    _returnCode = 0;
    _userData = ptr;
    _thread = CreateThread(0, 0, (LPTHREAD_START_ROUTINE)_Start, this, 0, 0);

    if (_thread == 0)
        throw Exception(F_SYSTEM_ERROR, __FILE__, __LINE__, "Unable to create thread.");
    _running = true;
}

int Thread::Join()
{
    if (_thread != 0)
    {
        WaitForSingleObject(_thread, INFINITE);
        CloseHandle(_thread);
        _thread = 0;
    }
    return _returnCode;
}

bool Thread::Running() const
{
    Guard lock(_lock);
    return _running;
}

void Thread::Cancel()
{
    Guard lock(_lock);
    TerminateThread(_thread, 0);
    _running = false;
}

void* Thread::_Start(void *obj)
{
    Thread *td = (Thread*)obj;
    if (td) {
        int returnCode = td->ThreadMain(td->_userData);
        {
            Guard lock(td->_lock);
            td->_running = false;
            td->_returnCode = returnCode;
        }
    }
    return 0;
}
