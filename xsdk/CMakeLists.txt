cmake_minimum_required(VERSION 2.6)
project(xsdk)

include_directories(SHARED include)

#-----------------------------------------------------------------------------
# Collect all platform INDEPENDENT files.
#-----------------------------------------------------------------------------
file(GLOB SRCS source/*.cpp)
file(GLOB INCS include/xsdk/*.h)
file(GLOB INCS_TIME include/xsdk/time/*.h)

#-----------------------------------------------------------------------------
# Collect all platform SPECIFIC source files.
#-----------------------------------------------------------------------------
if (WIN32)
    file(GLOB SRCS_PLATFORM source/windows/*.cpp)
    file(GLOB INCS_PLATFORM include/xsdk/windows/*.h)
    source_group("Source Files (Windows)" FILES ${SRCS_PLATFORM})
    source_group("Header Files (Windows)" FILES ${INCS_PLATFORM})
    source_group("Header Files (Time)" FILES ${INCS_TIME})
elseif (UNIX)
    file(GLOB SRCS_PLATFORM source/posix/*.cpp)
endif()

# Set compiler flags
if (WIN32)
    add_definitions( "/wd4996" ) # disable compile warnings
elseif (UNIX)
    add_definitions( "-Wall -g" )
endif()

#-----------------------------------------------------------------------------
# Add source files to build.
#-----------------------------------------------------------------------------
add_library(xsdk ${SRCS} ${SRCS_PLATFORM} ${INCS} ${INCS_PLATFORM} ${INCS_TIME})

#-----------------------------------------------------------------------------
# Build unit tests
#-----------------------------------------------------------------------------
add_subdirectory(test/test_xsdk)
add_subdirectory(test/test_xsdk_time)
