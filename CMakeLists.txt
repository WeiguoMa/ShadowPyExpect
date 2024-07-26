cmake_minimum_required(VERSION 3.28)
project(ShadowPyExpect)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# Find Python3
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

if(NOT Python3_FOUND)
    message(FATAL_ERROR "Python 3 not found.")
endif()

message(STATUS "Using Python 3: ${Python3_EXECUTABLE}")

# Ensure pybind11 uses the found Python environment
set(PYBIND11_PYTHON_VERSION ${Python3_VERSION})
set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})
set(PYTHON_INCLUDE_DIR ${Python3_INCLUDE_DIRS})
set(PYTHON_LIBRARY ${Python3_LIBRARIES})

add_subdirectory("./extern/pybind11")
add_subdirectory("./extern/eigen")
include_directories(${CMAKE_SOURCE_DIR}/extern/eigen)

pybind11_add_module(expectationCS src/cpp/expectation_shadow_binding.cpp)
pybind11_add_module(generateMeasureScheme src/cpp/measure_scheme_generator_binding.cpp)
pybind11_add_module(fakeSampler_backend src/cpp/fake_sampler_binding.cpp)
