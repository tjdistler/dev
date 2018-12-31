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

#include "xsdk/buffer.h"
#include "xsdk/exception.h"
#include <string.h>

using namespace xsdk;

Buffer::Buffer() :
_targetLen(0),
_bufferLen(0),
_buffer(0)
{
}

Buffer::Buffer(size_t length) :
_targetLen(length),
_bufferLen(0),
_buffer(0)
{
}

Buffer::~Buffer() throw()
{
    delete [] _buffer;
    _buffer = 0;
    _bufferLen = 0;
}

Buffer::Buffer(const Buffer& other) :
_targetLen(0),
_bufferLen(0),
_buffer(0)
{
    *this = other;
}

Buffer& Buffer::operator= (const Buffer& other)
{
    _targetLen = other._targetLen;
    if (other._buffer)
        memcpy(_Buffer(), other._buffer, other._bufferLen);
    return *this;
}

uint8_t& Buffer::operator[](size_t index)
{
    if (index >= _targetLen)
        throw Exception(xsdk::F_OUT_OF_RANGE, __FILE__, __LINE__, "Index is out of range.");

    return _Buffer()[index];
}

void* Buffer::Ptr()
{
    return _Buffer();
}

void* Buffer::Copy(const void* source, size_t length)
{
    _targetLen = length;
    _Buffer();
    return memcpy(_buffer, source, length);
}

void* Buffer::Fill(uint8_t value)
{
    _Buffer();
    return memset(_buffer, value, _bufferLen);
}

uint8_t* Buffer::_Buffer()
{
    if (_targetLen == 0)
        throw Exception(xsdk::F_VALUE_NOT_SET, __FILE__, __LINE__, "The size of the buffer has not been set.");

    if (_bufferLen < _targetLen)
    {
        // Allocate a larger buffer and copy the previous data.
        uint8_t* dst = new uint8_t[_targetLen];
        if (_buffer)
            memcpy(dst, _buffer, _bufferLen);
        delete [] _buffer;
        _bufferLen = _targetLen;
        _buffer = dst;
    }

    return _buffer;
}
