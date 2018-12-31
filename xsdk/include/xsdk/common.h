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

#ifndef _XSDK_COMMON_H_
#define _XSDK_COMMON_H_

#include <cstdarg>

#if defined(WIN32) || defined(WIN64) || defined(_MSC_VER)
    #undef  IS_WINDOWS
    #define IS_WINDOWS
#elif defined(linux)
    #undef  IS_LINUX
    #define IS_LINUX
#elif defined(__NetBSD__) || defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__bsdi__)
    #undef  IS_BSD
    #define IS_BSD
#else
    #error ">> Unknown Operating System. You must hand-modify this file to make it work."
#endif

#if defined(IS_LINUX) || defined(IS_BSD)
    #undef  IS_POSIX
    #define IS_POSIX
#endif

#if defined(NDEBUG) || defined(_DEBUG) || defined(DEBUG)
    #undef  IS_DEBUG
    #define IS_DEBUG
#endif

#define XSDK_MACRO_BEGIN do {
#define XSDK_MACRO_END   }while(0)

#define XSDK_STR_EXPAND(tok) #tok
#define XSDK_STR(tok) XSDK_STR_EXPAND(tok)

// Windows
#if defined(IS_WINDOWS)

    #include <winsock2.h>
    #include <windows.h>

    #if !defined(XSDK_EXCLUDE_INTTYPES_H)
        #include "xsdk/windows/inttypes.h"
    #endif

    #define XSDK_API  __declspec(dllexport)

    #if !defined(snprintf)
        #define snprintf _snprintf
    #endif

    #if !defined(va_copy)
        #define va_copy(dst, src) ((void)((dst) = (src)))
    #endif


#else

    #include <pthread.h>
    #include <inttypes.h>
    #include "sys/types.h"

    #define XSDK_API

#endif //OS


// Compile-time assert verifies the size of explicit-length datatypes.
struct ___XsdkTypesAssert
{
    unsigned int SizeofInt8  : sizeof(int8_t)  == 1 && sizeof(uint8_t)  == 1;
    unsigned int SizeofInt16 : sizeof(int16_t) == 2 && sizeof(uint16_t) == 2;
    unsigned int SizeofInt32 : sizeof(int32_t) == 4 && sizeof(uint32_t) == 4;
    unsigned int SizeofInt64 : sizeof(int64_t) == 8 && sizeof(uint64_t) == 8;
};

#endif
