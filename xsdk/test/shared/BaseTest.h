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

#ifndef _XSDK_BASE_TEST_H_
#define _XSDK_BASE_TEST_H_

#include "xsdk/common.h"

#include <stdio.h>
#include <string>
#include <exception>
#include <list>

class __TestException : public std::exception
{
public:
    __TestException(char const *msg, char const *exp, char const *file, int line);
    virtual ~__TestException() throw();
    const char* what() const throw();
private:
    std::string _msg;
}; //class


#define TEST_ASSERT(exp) \
    XSDK_MACRO_BEGIN if (!(exp)) throw __TestException("Assert failed", XSDK_STR(exp), __FILE__, __LINE__); \
    XSDK_MACRO_END

#define TEST_ASSERT_THROW(exp, extype) \
    XSDK_MACRO_BEGIN const char *file=__FILE__; int line=__LINE__; \
    try { exp; } \
    catch(extype) { /*success*/ return; } \
    catch(...) { throw __TestException("Wrong exception type thrown", XSDK_STR(exp), file, line); } \
    throw __TestException("The expected exception was not thrown", XSDK_STR(exp), file, line); \
    XSDK_MACRO_END

#define TEST_ASSERT_NO_THROW(exp) \
    XSDK_MACRO_BEGIN const char *file=__FILE__; int line=__LINE__; \
    try { exp; } \
    catch(...) { throw __TestException("Exception thrown", XSDK_STR(exp), file, line); } \
    XSDK_MACRO_END


#define TEST_BEGIN(title) \
    virtual bool Run(std::list<std::string> &failures) { \
        printf("%s\n", (title)); \
        bool result = true;

#define TEST_FUNC(func) \
    XSDK_MACRO_BEGIN \
        Init(); \
        try { \
            printf("  %s", XSDK_STR(func)); \
            fflush(stdout); \
            (func)(); \
            printf("\n"); \
        } catch(__TestException &ex) { \
            printf(" \t**FAILED**\n"); \
            result = false; \
            failures.push_back(std::string(ex.what())); \
        } catch(std::exception &ex) { \
            printf(" \t**FAILED**\n"); \
            result = false; \
            failures.push_back(std::string(ex.what())); \
        } \
        Fini(); \
    XSDK_MACRO_END

#define TEST_END \
        return result; \
    }


class BaseTest
{
public:
    BaseTest() {}
    virtual ~BaseTest() throw() {}

    virtual bool Run(std::list<std::string>&) = 0;

protected:
    // Called before/after each test.
    virtual void Init() throw() {}
    virtual void Fini() throw() {}

}; //class

#endif //_XSDK_BASE_TEST_H_
