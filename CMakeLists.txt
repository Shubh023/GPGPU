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

#set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -pg")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")
   
# The compiled library and executable is here
add_subdirectory(src)