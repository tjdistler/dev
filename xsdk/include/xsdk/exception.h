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

#ifndef _XSDK_EXCEPTION_H_
#define _XSDK_EXCEPTION_H_

#include "xsdk/common.h"
#include <exception>
#include <string>

namespace xsdk
{

    /** Defines exception error types. */
    const int F_UNDEFINED       = 0; // Don't use when setting the exception type.
    const int F_OUT_OF_RANGE    = 1;
    const int F_VALUE_NOT_SET   = 2;
    const int F_SYSTEM_ERROR    = 3;    ///< Error while calling a system function (e.g pthread_create() or Win32 API).


    /** An extension to the std::exception class */
    class Exception : public std::exception
    {
    public:
        XSDK_API Exception();
        /**
         * Creates an exception with an error message.
         * @param msg Text describing the error. May contain printf-style tags.
         */
        XSDK_API Exception(const char* msg, ...);
        /**
         * Create an exception with an error code and message.
         * @param type The type of exception. See error.h
         * @param msg Text describing the error. May contain printf-style tags.
         */
        XSDK_API Exception(int type, const char* msg, ...);
        /**
         * Create an exception with an error code and message.
         * @param type The type of exception. See error.h
         * @param file The file name where the exception was thrown.
         * @param line The line number where the exception was thrown.
         * @param msg Text describing the error. May contain printf-style tags.
         */
        XSDK_API Exception(int type, const char* file, int line, const char* msg, ...);

        XSDK_API virtual ~Exception() throw();

        // Using default copy constructor and assignment operator.

        /**
         * Overrides std::exception::what() to provide a more detailed error
         * description.
         */
        XSDK_API virtual const char* what() const throw();

        /** Returns the exception type. */
        XSDK_API int Type() { return _type; }
        /** Returns the line number. */
        XSDK_API int Line()  { return _line; }
        /** Returns the file name. */
        XSDK_API const char* File()    { return _file.c_str(); }
        /** Returns the expanded message string. */
        XSDK_API const char* Message() { return _message.c_str(); }
        
        /** Sets the exception type. */
        XSDK_API void SetType(int type);
        /** Sets the line number. */
        XSDK_API void SetLine(int line);
        /** Sets the file name. */
        XSDK_API void SetFile(const char* file);
        /** Sets the text describing the error. */
        XSDK_API void SetMessage(const char* msg, ...);

    protected:

        /** Creates the message returned by what(). */
        XSDK_API void _BuildWhatMessage();
        XSDK_API void _FormatMessage(std::string& dst, const char* format, ...);
        XSDK_API void _FormatMessage(std::string& dst, const char* format, va_list& args);
        
        int _type;
        int _line;
        std::string _file;
        std::string _message;
        std::string _whatMessage;

    }; //class

}; //namespace

#endif //_XSDK_EXCEPTION_H_
