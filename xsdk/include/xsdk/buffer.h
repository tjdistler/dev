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

#ifndef _XSDK_BUFFER_H_
#define _XSDK_BUFFER_H_

#include "xsdk/common.h"

namespace xsdk
{

    /**
     * A class to make dealing with dynamically allocated memory simple.
     * This class has many of the same semantics as using a raw array pointer.
     * However, all memory management details are controlled by the lifetime
     * of this object.
     *
     * @code
     * buffer p(10);
     * uint8_t x = p[10] // throws
     * uint8_t y = p[0] * 2;
     * buffer p2 = p;  // deep copy
     * char* raw = (char*)p2.Ptr();
     * @endcode
     */
    class Buffer
    {
    public:
        XSDK_API Buffer();

        /**
         * Constructor that sets the size of the buffer.
         * @param length The size of the buffer in bytes.
         */
        XSDK_API Buffer(size_t length);

        /** Frees any allocated memory. */
        XSDK_API virtual ~Buffer() throw();

        /** Copies the contents of the other buffer. */
        XSDK_API Buffer(const Buffer& other);

        /** Copies the contents of the other buffer. */
        XSDK_API Buffer& operator= (const Buffer& other);

        /**
         * Provides access to the buffer contents using the array operator.
         * @return A reference to the byte at the specified index.
         */
        XSDK_API uint8_t& operator[](size_t index);

        /** Sets the length of the buffer. */
        XSDK_API void SetLength(size_t length) { _targetLen = length; }

        /** Returns the length of the buffer. */
        XSDK_API size_t Length() const { return _targetLen; }

        /** 
         * Returns a pointer to the buffer memory.
         * @note An exception is thrown if no length has been set on the
         * buffer.
         */
        XSDK_API void* Ptr();

        /**
         * Copies the source buffer into the object.
         * @param source The buffer data to copy.
         * @param length The number of bytes to copy from the source buffer.
         * @return A pointer to the destination buffer memory.
         */
        XSDK_API void* Copy(const void* source, size_t length);

        /**
         * Fills the buffer with the specified value.
         * @note An exception is thrown if no length has been set on the
         * buffer.
         * @return A pointer to the buffer memory.
         */
        XSDK_API void* Fill(uint8_t value);

    private:

        /**
         * Returns a pointer to the raw buffer memory. This method ensures
         * that the buffer is at least as large as _targetLen. If it isn't,
         * then this method allocates a new buffer and copies the previous
         * data over.
         */
        uint8_t* _Buffer();

        size_t   _targetLen;///< The requested length of the buffer.
        size_t   _bufferLen;///< The actual length of the allocated memory.
        uint8_t* _buffer;   ///< The buffer memory (of length _bufferLen).

    }; //class

}; //namespace

#endif //_XSDK_BUFFER_H_
