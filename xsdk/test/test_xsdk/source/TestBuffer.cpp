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

#include "TestBuffer.h"
#include "xsdk/exception.h"
#include <string.h>

using namespace xsdk;

const unsigned char data[] = { 0xde, 0xad, 0xbe, 0xef };

void TestBuffer::TestConstructor()
{
    TEST_ASSERT_NO_THROW( Buffer b );
    TEST_ASSERT_NO_THROW( Buffer b(4242) );
    {
        Buffer b;
        TEST_ASSERT( b.Length() == 0 );
    }
    {
        Buffer b(7777);
        TEST_ASSERT( b.Length() == 7777 );
    }
}

void TestBuffer::TestCopy()
{
    Buffer b(sizeof(data));
    TEST_ASSERT_NO_THROW( b.Copy(data, sizeof(data)) );
    TEST_ASSERT( b.Length() == sizeof(data) );
    void *p = 0;
    TEST_ASSERT_NO_THROW( p = b.Ptr() );
    TEST_ASSERT( p );
    TEST_ASSERT( memcmp(p, data, sizeof(data)) == 0 );

    // Test copy constructor for deep copy.
    Buffer *pB = 0;
    TEST_ASSERT_NO_THROW( pB = new Buffer(b) );
    TEST_ASSERT( memcmp(pB->Ptr(), b.Ptr(), b.Length()) == 0 );
    TEST_ASSERT( pB->Ptr() != b.Ptr() ); // make sure the buffers are different.
    TEST_ASSERT( pB->Length() == b.Length() );
    delete pB;

    // Test assignment operator.
    Buffer b2;
    {
        Buffer b3(42);
        TEST_ASSERT_NO_THROW( b3 = b2 = b );
        TEST_ASSERT( b2.Length() == b.Length() );
        TEST_ASSERT( b3.Length() == b2.Length() );
        TEST_ASSERT( memcmp(b2.Ptr(), b.Ptr(), b.Length()) == 0 );
        TEST_ASSERT( memcmp(b3.Ptr(), b.Ptr(), b.Length()) == 0 );
    }
}

void TestBuffer::TestFill()
{
    Buffer b1(5000);
    Buffer b2(5000);
    TEST_ASSERT_NO_THROW( b1.Fill(0) );
    TEST_ASSERT_NO_THROW( b2.Fill(0xA5) );

    uint8_t* p1 = (uint8_t*)b1.Ptr();
    uint8_t* p2 = (uint8_t*)b2.Ptr();
    for (int ii=0; ii<5000; ++ii)
    {
        TEST_ASSERT( p1[ii] == 0 );
        TEST_ASSERT( p2[ii] == 0xA5 );
    }
}

void TestBuffer::TestResize()
{
    Buffer b(1024);
    TEST_ASSERT( b.Length() == 1024 );
    TEST_ASSERT_NO_THROW( b.Fill(0xF0) );
    TEST_ASSERT_NO_THROW( b.SetLength(2048) );
    TEST_ASSERT( b.Length() == 2048 );

    // Verify the data was copied.
    uint8_t* p = (uint8_t*)b.Ptr();
    for (int ii=0; ii<1024; ++ii)
        TEST_ASSERT( p[ii] == 0xF0 );
}

void TestBuffer::TestArrayAccess()
{
    {
        Buffer b(sizeof(data));
        TEST_ASSERT_NO_THROW( b.Copy(data, sizeof(data)) );
        TEST_ASSERT( b.Length() == sizeof(data) );

        for (size_t ii=0; ii<sizeof(data); ++ii)
        {
            TEST_ASSERT_NO_THROW( b[ii] );
            TEST_ASSERT( b[ii] == data[ii] );
        }

        // Try updating the buffer using array access.
        TEST_ASSERT_NO_THROW( b[0] = 0xA5 );
        TEST_ASSERT_NO_THROW( b[1] = 0x5A );
        TEST_ASSERT( b[0] == 0xA5 );
        TEST_ASSERT( b[1] == 0x5A );

        // Test out of range index.
        TEST_ASSERT_THROW( b[sizeof(data)], Exception );
    }

    {
        Buffer b;
        TEST_ASSERT_THROW( b[0], Exception );
    }
}
