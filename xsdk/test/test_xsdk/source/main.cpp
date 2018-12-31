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

#include "TestException.h"
#include "TestThread.h"
#include "TestMutex.h"
#include "TestGuard.h"
#include "TestCondition.h"
#include "TestBuffer.h"

#include <stdio.h>
#include <list>
#include <string>

using namespace std;

int main(int argc, char *argv[])
{
    list<string> failures;

    printf("\n");
    { TestException test; test.Run(failures); }
    { TestThread    test; test.Run(failures); }
    { TestMutex     test; test.Run(failures); }
    { TestGuard     test; test.Run(failures); }
    { TestCondition test; test.Run(failures); }
    { TestBuffer    test; test.Run(failures); }

    /* ADD MORE TESTS HERE */
    
    printf("\n");

    if (failures.empty())
        printf("RESULT: Success\n\n");
    else
    {
        list<string>::iterator it = failures.begin();
        for (; it != failures.end(); ++it)
            printf("%s\n\n", it->c_str());
        printf("\nRESULT: *** FAIL (%u) ***\n\n", (unsigned)failures.size());
    }

    return failures.size();
}
