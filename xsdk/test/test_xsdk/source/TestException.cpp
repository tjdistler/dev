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
#include <string.h>

using namespace std;
using namespace xsdk;

void TestException::TestConstruction()
{
    Exception *e = 0;

    TEST_ASSERT_NO_THROW(e = new Exception());
    delete e;

    TEST_ASSERT_NO_THROW(e = new Exception("Error %d occurred", 42));
    TEST_ASSERT( strcmp(e->Message(), "Error 42 occurred") == 0 );
    delete e;

    TEST_ASSERT_NO_THROW(e = new Exception(7, "Error %d occurred", 42));
    TEST_ASSERT( e->Type() == 7 );
    TEST_ASSERT( strcmp(e->Message(), "Error 42 occurred") == 0 );
    delete e;

    TEST_ASSERT_NO_THROW(e = new Exception(7, __FILE__, 77, "Error %d occurred", 42));
    TEST_ASSERT( e->Type() == 7 );
    TEST_ASSERT( strcmp(e->File(), __FILE__) == 0 );
    TEST_ASSERT( e->Line() == 77 );
    TEST_ASSERT( strcmp(e->Message(), "Error 42 occurred") == 0 );
    delete e;
}

void TestException::TestGetSet()
{
    Exception e;

    // Test initial state.
    TEST_ASSERT( e.Type() == F_UNDEFINED );
    TEST_ASSERT( strcmp(e.File(), "") == 0 );
    TEST_ASSERT( e.Line() == -1 );
    TEST_ASSERT( strcmp(e.Message(), "") == 0 );

    TEST_ASSERT_NO_THROW( e.SetType(F_OUT_OF_RANGE) );
    TEST_ASSERT( e.Type() == F_OUT_OF_RANGE );

    TEST_ASSERT_NO_THROW( e.SetFile("Foo") );
    TEST_ASSERT( strcmp(e.File(), "Foo") == 0 );

    TEST_ASSERT_NO_THROW( e.SetLine(42) );
    TEST_ASSERT( e.Line() == 42 );

    TEST_ASSERT_NO_THROW( e.SetMessage("Foo %d", 42) );
    TEST_ASSERT( strcmp(e.Message(), "Foo 42") == 0 );
}

void TestException::TestWhat()
{
    int line = __LINE__;
    Exception e(F_OUT_OF_RANGE, __FILE__, line, "Error %d occurred", 42);
    string w = e.what();

    TEST_ASSERT( w.find("F_OUT_OF_RANGE") != string::npos );
    TEST_ASSERT( w.find(__FILE__) != string::npos );
    TEST_ASSERT( w.find("Error 42 occurred") != string::npos );

    char lineStr[8];
    snprintf(lineStr, sizeof(lineStr), "%d", line);
    TEST_ASSERT( w.find(lineStr) != string::npos );
}

void TestException::TestAssignment()
{
    Exception e(F_OUT_OF_RANGE, __FILE__, __LINE__, "Error %d occurred", 42);
    Exception e2 = e;

    TEST_ASSERT( strcmp(e.what(), e2.what()) == 0 );
    TEST_ASSERT( e2.Type() == e.Type() );
    TEST_ASSERT( strcmp(e2.File(), e.File()) == 0 );
    TEST_ASSERT( e2.Line() == e.Line() );
    TEST_ASSERT( strcmp(e2.Message(), e.Message()) == 0 );
}
