cmake_minimum_required(VERSION 3.28)
project(predicting_quantum_properties)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

set(PYTHON_EXECUTABLE "/opt/anaconda3/envs/weiguoQ/bin/python")
set(PYTHON_INCLUDE_DIRECTORY "/opt/anaconda3/envs/weiguoQ/include/python3.10")

add_subdirectory("./extern/pybind11")

include_directories("/Users/weiguo_ma/CppPackages/eigen")

pybind11_add_module(expectationCS src/cpp/expectation_shadow_binding.cpp)
pybind11_add_module(generateMeasureScheme src/cpp/measure_scheme_generator_binding.cpp)
pybind11_add_module(fakeSampler_backend src/cpp/fake_sampler_binding.cpp)