cmake_minimum_required(VERSION 3.27)
project(lopnor_llm_predict LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)

set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to an ONNX Runtime installation containing include/ and lib/")

find_path(ONNXRUNTIME_INCLUDE_DIR
  NAMES onnxruntime_cxx_api.h
  PATHS ${ONNXRUNTIME_ROOT}/include
)
find_library(ONNXRUNTIME_LIBRARY
  NAMES onnxruntime
  PATHS ${ONNXRUNTIME_ROOT}/lib ${ONNXRUNTIME_ROOT}/lib64
)
if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
  message(FATAL_ERROR
    "Could not find ONNX Runtime. Configure with "
    "-DONNXRUNTIME_ROOT=/path/to/onnxruntime")
endif()

find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
  )
  FetchContent_MakeAvailable(nlohmann_json)
endif()

add_executable(lopnor_llm_predict
  src/feature_builder.cpp
  src/input_parser.cpp
  src/main.cpp
  src/metadata.cpp
  src/onnx_runner.cpp
  src/output_writer.cpp
  src/postprocess.cpp
)

target_include_directories(lopnor_llm_predict
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${ONNXRUNTIME_INCLUDE_DIR}
)

target_link_libraries(lopnor_llm_predict
  PRIVATE
    ${ONNXRUNTIME_LIBRARY}
    nlohmann_json::nlohmann_json
)
