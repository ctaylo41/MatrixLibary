cmake_minimum_required(VERSION 3.0.0)
set(This Matrix)

project(Matrix VERSION 0.1.0 LANGUAGES C CXX)

add_library(Matrix src/matrix.cpp)
set(CMAKE_CXX_STANDARD 14) # Set the C++ standard to C++14
set(CMAKE_CXX_STANDARD_REQUIRED ON) # Make C++14 a requirement
set(CMAKE_POSITION_INDEPENDENT_CODE ON) 

enable_testing()
add_subdirectory(googletest)

set(Headers
    src/matrix.h
    src/vector.h
)
set(Sources
    src/matrix.cpp
    src/vector.cpp
)
add_subdirectory(test)
