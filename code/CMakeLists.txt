cmake_minimum_required(VERSION 3.14)
project(Parallel-CBS)
include(CTest)
include_directories(include/)
set (CMAKE_CXX_STANDARD 17) # Use C++17

PROJECT(myproject)

# fetch latest argparse
include(FetchContent)
FetchContent_Declare(
    argparse
    GIT_REPOSITORY https://github.com/p-ranav/argparse.git
)
FetchContent_MakeAvailable(argparse)

# Optimization flags
# SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -O3")

add_executable(CBS src/main.cpp src/MAPFLoader.cpp src/CBSSolver.cpp src/AStar.cpp)

add_executable(TestAStar test/testAStar.cpp src/AStar.cpp src/MAPFLoader.cpp)
add_executable(TestMAPFLoader test/testMAPFLoader.cpp src/MAPFLoader.cpp)
add_executable(TestCBS test/testCBS.cpp src/MAPFLoader.cpp src/CBSSolver.cpp src/AStar.cpp)

add_executable(GenerateInstances src/GenerateInstance.cpp)
target_link_libraries(GenerateInstances argparse)

add_test(NAME TestMAPFLoader
        COMMAND TestMAPFLoader)
add_test(NAME TestAStar
        COMMAND TestAStar)
add_test(NAME TestCBS
        COMMAND TestCBS)
