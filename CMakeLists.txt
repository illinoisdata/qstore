cmake_minimum_required(VERSION 3.20)

# set(CMAKE_C_COMPILER "/usr/bin/gcc-10")
# set(CMAKE_CXX_COMPILER "/usr/bin/gcc")

# message(STATUS "Toolchain file path: ${CMAKE_CURRENT_SOURCE_DIR}/../vcpkg/scripts/buildsystems/vcpkg.cmake")
set(VCPKG_TARGET_TRIPLET "x64-linux")
set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(QStore LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -flto=auto -mavx2 -mfma -mbmi2 -ffast-math -pthread")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")

enable_testing()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/baselines/zipnn_c_api/)

add_executable(baseline0_no_compression
  baselines/src/baseline0_no_compression.cpp
  src/safetensors_utils.cpp
  src/load_tensors.cpp
  src/utils.cpp
)
target_include_directories(baseline0_no_compression PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(baseline0_no_compression PRIVATE nlohmann_json::nlohmann_json OpenMP::OpenMP_CXX)


add_executable(baseline1_lz4
  baselines/src/baseline1_lz4.cpp
  src/safetensors_utils.cpp
  src/load_tensors.cpp
  src/utils.cpp
)
target_include_directories(baseline1_lz4 PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(baseline1_lz4 PRIVATE OpenMP::OpenMP_CXX lz4::lz4 nlohmann_json::nlohmann_json)


add_executable(baseline2_zstd
  baselines/src/baseline2_zstd.cpp
  src/safetensors_utils.cpp
  src/load_tensors.cpp
  src/utils.cpp
)
target_include_directories(baseline2_zstd PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(baseline2_zstd PRIVATE OpenMP::OpenMP_CXX zstd::libzstd nlohmann_json::nlohmann_json)


add_executable(baseline4_zipnn 
  baselines/src/baseline4_zipnn.cpp
  src/safetensors_utils.cpp
  src/load_tensors.cpp
  src/utils.cpp
)
target_include_directories(baseline4_zipnn PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(baseline4_zipnn PRIVATE zipnn_c_api nlohmann_json::nlohmann_json OpenMP::OpenMP_CXX)


add_executable(final_encode
  src/final_encode.cpp
  src/safetensors_utils.cpp
  src/load_tensors.cpp
  src/utils.cpp
  src/encode_helper.cpp
  baselines/zipnn_c_api/FiniteStateEntropy/lib/huf_compress.c
  baselines/zipnn_c_api/FiniteStateEntropy/lib/huf_decompress.c
  baselines/zipnn_c_api/FiniteStateEntropy/lib/fse_compress.c
  baselines/zipnn_c_api/FiniteStateEntropy/lib/fse_decompress.c
  baselines/zipnn_c_api/FiniteStateEntropy/lib/entropy_common.c
  baselines/zipnn_c_api/FiniteStateEntropy/lib/hist.c 
)
target_include_directories(final_encode PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  baselines/zipnn_c_api/FiniteStateEntropy/lib
)
target_link_libraries(final_encode PRIVATE nlohmann_json::nlohmann_json OpenMP::OpenMP_CXX)

add_executable(final_decode
  src/final_decode.cpp
  src/safetensors_utils.cpp
  src/load_tensors.cpp
  src/utils.cpp
  src/decode_helper.cpp
  baselines/zipnn_c_api/FiniteStateEntropy/lib/huf_compress.c
  baselines/zipnn_c_api/FiniteStateEntropy/lib/huf_decompress.c
  baselines/zipnn_c_api/FiniteStateEntropy/lib/fse_compress.c
  baselines/zipnn_c_api/FiniteStateEntropy/lib/fse_decompress.c
  baselines/zipnn_c_api/FiniteStateEntropy/lib/entropy_common.c
  baselines/zipnn_c_api/FiniteStateEntropy/lib/hist.c 
)
target_include_directories(final_decode PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  baselines/zipnn_c_api/FiniteStateEntropy/lib
)
target_link_libraries(final_decode PRIVATE nlohmann_json::nlohmann_json OpenMP::OpenMP_CXX)

add_executable(final_decode_quantized
  src/final_decode_quantized_only.cpp
  src/safetensors_utils.cpp
  src/load_tensors.cpp
  src/utils.cpp
  src/decode_helper.cpp
  baselines/zipnn_c_api/FiniteStateEntropy/lib/huf_compress.c
  baselines/zipnn_c_api/FiniteStateEntropy/lib/huf_decompress.c
  baselines/zipnn_c_api/FiniteStateEntropy/lib/fse_compress.c
  baselines/zipnn_c_api/FiniteStateEntropy/lib/fse_decompress.c
  baselines/zipnn_c_api/FiniteStateEntropy/lib/entropy_common.c
  baselines/zipnn_c_api/FiniteStateEntropy/lib/hist.c 
)
target_include_directories(final_decode_quantized PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  baselines/zipnn_c_api/FiniteStateEntropy/lib
)
target_link_libraries(final_decode_quantized PRIVATE nlohmann_json::nlohmann_json OpenMP::OpenMP_CXX)


find_package(nlohmann_json CONFIG REQUIRED)
find_package(OpenMP REQUIRED)
find_package(GTest CONFIG REQUIRED)
find_package(lz4 CONFIG REQUIRED)
find_package(zstd CONFIG REQUIRED)
find_package(span-lite CONFIG REQUIRED)