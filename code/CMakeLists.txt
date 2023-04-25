cmake_minimum_required(VERSION 3.14)
project(Parallel-CBS)
include(CTest)
include_directories(include/)
set (CMAKE_CXX_STANDARD 14) # Use C++14

PROJECT(myproject)

# fetch latest argparse
include(FetchContent)
FetchContent_Declare(
    argparse
    GIT_REPOSITORY https://github.com/p-ranav/argparse.git
)
FetchContent_MakeAvailable(argparse)

#require torch
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Optimization flags
# SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -O3")

# add_executable(TestAStar test/testAStar.cpp src/AStar.cpp src/MAPFLoader.cpp)
# add_executable(TestMAPFLoader test/testMAPFLoader.cpp src/MAPFLoader.cpp)
# add_executable(TestCBS test/testCBS.cpp src/MAPFLoader.cpp src/CBSSolver.cpp src/AStar.cpp)

add_library(cbsSolver src/CBSSolver.cpp src/AStar.cpp src/ConflictModel.cpp)
target_link_libraries(cbsSolver "${TORCH_LIBRARIES}")

# add_executable(CBS src/main.cpp src/MAPFLoader.cpp)
# target_link_libraries(CBS cbsSolver)

add_executable(GenerateData src/GenerateData.cpp src/Instance.cpp src/MAPFLoader.cpp)
target_link_libraries(GenerateData argparse)
target_link_libraries(GenerateData cbsSolver)

add_executable(TrainModel src/TrainModel.cpp src/Instance.cpp src/DataLoader.cpp src/MAPFLoader.cpp src/ConflictModel.cpp)
target_link_libraries(TrainModel argparse)
target_link_libraries(TrainModel cbsSolver)
target_link_libraries(TrainModel "${TORCH_LIBRARIES}")

add_test(NAME TestMAPFLoader
        COMMAND TestMAPFLoader)
add_test(NAME TestAStar
        COMMAND TestAStar)
add_test(NAME TestCBS
        COMMAND TestCBS)
