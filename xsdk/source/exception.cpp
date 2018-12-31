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

#include "xsdk/exception.h"
#include "xsdk/common.h"
#include <cstdio>

using namespace std;
using namespace xsdk;

Exception::Exception() :
_type(F_UNDEFINED),
_line(-1)
{
}

Exception::Exception(const char* msg, ...) :
_type(F_UNDEFINED),
_line(-1)
{
    va_list args;
    va_start(args, msg);
    _FormatMessage(_message, msg, args);
    va_end(args);
    _BuildWhatMessage();
}

Exception::Exception(int type, const char* msg, ...) :
_type(type),
_line(-1)
{
    va_list args;
    va_start(args, msg);
    _FormatMessage(_message, msg, args);
    va_end(args);
    _BuildWhatMessage();
}

Exception::Exception(int type, const char* file, int line, const char* msg, ...) :
_type(type),
_line(line),
_file(file)
{
    va_list args;
    va_start(args, msg);
    _FormatMessage(_message, msg, args);
    va_end(args);
    _BuildWhatMessage();
}

Exception::~Exception() throw()
{
}

const char* Exception::what() const throw()
{
    return _whatMessage.c_str();
}

void Exception::SetType(int type)
{
    _type = type;
    _BuildWhatMessage();
}

void Exception::SetLine(int line)
{
    _line = line;
    _BuildWhatMessage();
}

void Exception::SetFile(const char* file)
{
    _file = file;
    _BuildWhatMessage();
}

void Exception::SetMessage(const char* msg, ...)
{
    va_list args;
    va_start(args, msg);
    _FormatMessage(_message, msg, args);
    va_end(args);
    _BuildWhatMessage();
}

void Exception::_BuildWhatMessage()
{
    string type;
    switch(_type) {
        case F_UNDEFINED:     type = "<undefined>";             break;
        case F_OUT_OF_RANGE:  type = XSDK_STR(F_OUT_OF_RANGE);  break;
        case F_VALUE_NOT_SET: type = XSDK_STR(F_VALUE_NOT_SET); break;
        case F_SYSTEM_ERROR:  type = XSDK_STR(F_SYSTEM_ERROR);  break;
        default: type = "<unrecognized>";
    };

    _FormatMessage(_whatMessage,
        "xsdk::Exception [%s] thrown from %s:%d - \"%s\"",
        type.c_str(),
        _file.c_str(),
        _line,
        _message.c_str());
}

void Exception::_FormatMessage(std::string& dst, const char* format, ...)
{
    va_list args;
    va_start(args, format);
    _FormatMessage(dst, format, args);
    va_end(args);
}

void Exception::_FormatMessage(std::string& dst, const char* format, va_list& args)
{
#if defined(IS_WINDOWS)
    int len = vsnprintf(0, 0, format, args);
#else
    // On some platforms (i.e. the powerpc), vsnprintf modifies the variable
    // arguments list so we can't use it again. Since we call this method twice
    // (once to get the required buffer length and once to actually format the
    // string), we need to pass a COPY of the list to the first call so the
    // list is still usable for the second call.
    va_list args_copy;
    va_copy(args_copy, args);
    int len = vsnprintf(0, 0, format, args_copy);
    va_end(args_copy);
#endif

    if (len <= 0)
        return;
    len++;
    char* msg = new char[len];
    len = vsnprintf(msg, len, format, args);
    if (len <= 0)
        return;
    dst = msg;
    delete [] msg;
}
