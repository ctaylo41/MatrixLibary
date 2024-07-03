cmake_minimum_required(VERSION 3.0)

set(This Test)

set(MatrixSources
    ../src/matrix.cpp
    ../src/vector.cpp
)


set(TestSources
    matrix_tests.cpp
    vector_tests.cpp
)

add_library(MatrixTests STATIC ${MatrixSources})

add_executable(${This} ${TestSources})
target_link_libraries(${This} PUBLIC
    gtest_main
    MatrixTests
)


add_test(
    NAME ${This}
    COMMAND ${This}
)