cmake_minimum_required(VERSION 3.20)
project(zipnn_c_api)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -O3 -march=native")

set(FSE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/FiniteStateEntropy)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(SOURCES
    data_manipulation_dtype16.c
    zipnn.c
    zipnn_test.c
    ${FSE_DIR}/lib/huf_compress.c
    ${FSE_DIR}/lib/huf_decompress.c
    ${FSE_DIR}/lib/fse_compress.c
    ${FSE_DIR}/lib/fse_decompress.c
    ${FSE_DIR}/lib/entropy_common.c
    ${FSE_DIR}/lib/hist.c
)

set(HEADERS
    data_manipulation_dtype16.h
    zipnn.h
)

find_package(Threads REQUIRED)

add_executable(zipnn_test ${SOURCES} ${HEADERS})
target_include_directories(zipnn_test PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${FSE_DIR}/lib
)
target_link_libraries(zipnn_test PRIVATE ${CMAKE_THREAD_LIBS_INIT} m)

set(LIB_SOURCES
    zipnn.c
    data_manipulation_dtype16.c
    ${FSE_DIR}/lib/huf_compress.c
    ${FSE_DIR}/lib/huf_decompress.c
    ${FSE_DIR}/lib/fse_compress.c
    ${FSE_DIR}/lib/fse_decompress.c
    ${FSE_DIR}/lib/entropy_common.c
    ${FSE_DIR}/lib/hist.c
)
add_library(zipnn_c_api ${LIB_SOURCES} ${HEADERS})
target_include_directories(zipnn_c_api PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${FSE_DIR}/lib
)