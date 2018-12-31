# Set the test application name HERE
set(TEST_NAME "test_xsdk_time")

set(TEST_BASE_DIR ${xsdk_SOURCE_DIR}/test)
set(TEST_SHARED ${TEST_BASE_DIR}/shared)

# TEST_DIR is the root of this suite of tests.
set(TEST_DIR ${TEST_BASE_DIR}/${TEST_NAME})

# Print paths for debug
message(STATUS "  TEST_BASE_DIR: " ${TEST_BASE_DIR})
message(STATUS "  TEST_DIR: " ${TEST_DIR})

include_directories(SHARED include)
include_directories(${xsdk_INCLUDE_DIR})
include_directories(${TEST_SHARED})

#-----------------------------------------------------------------------------
# Collect all necessary files.
#-----------------------------------------------------------------------------
file(GLOB SRCS ${TEST_DIR}/source/*.cpp ${TEST_SHARED}/*.cpp)
file(GLOB INCS ${TEST_DIR}/include/*.h ${TEST_SHARED}/*.h)

# Make sure the linker can find the xsdk library.
link_directories(${xsdk_BINARY_DIR})

# Defines libraries to link to
if (UNIX)
    set(LIBS "-lpthread -lrt")
endif()

# On Windows, disable compile warnings
if (WIN32)
    add_definitions( "/wd4996" )
endif()

add_executable(${TEST_NAME} ${SRCS} ${INCS}) 

# Link the executable. 
target_link_libraries(${TEST_NAME} xsdk ${LIBS})
