#include <cstdio>
#include "MAPFLoader.hpp"

int main()
{
    // Load MAPF problem
    MAPFLoader loader;

    std::string fileName = "../instances/random_map.txt";
    loader.loadInstanceFromFile(fileName);

    // Run specific version of CBS

}