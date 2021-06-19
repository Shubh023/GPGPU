cmake_minimum_required(VERSION 3.18)

project(irgpu LANGUAGES CXX CUDA
              VERSION 1.0
              DESCRIPTION "IRGPU Final Project.")

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    #set(CUDA_NVCC_FLAGS -g;-G)                 # Try setting debug symbols (does not work)
endif()

find_package(OpenCV REQUIRED core imgcodecs highgui)
if(OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
endif(OpenCV_FOUND)

set(Boost_USE_STATIC_LIBS OFF) 
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF) 
find_package(Boost COMPONENTS timer) 
if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS}) 
endif()

include(FetchContent)

FetchContent_Declare(
  GoogleBenchmark
  URL https://github.com/google/benchmark/archive/v1.5.5.tar.gz
)

if (NOT GoogleBenchmark_POPULATED)
  FetchContent_Populate(GoogleBenchmark)
  set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "From Gtest")
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "From Gtest")
  add_subdirectory(${googlebenchmark_SOURCE_DIR} ${googlebenchmark_BINARY_DIR})
endif()

#set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -pg")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O0 -g")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")
   
# The compiled library and executable is here
add_subdirectory(src)