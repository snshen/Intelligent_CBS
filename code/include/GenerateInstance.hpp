#ifndef INSTANCE_H
#define INSTANCE_H

#include <string>
#include <vector>
#include <random>

class Instance
{
    public:
        std::vector<char> randMap;
        Instance(int height, int width, float density, int numAgents);
        void generateMap();
        void writeMapToFile(std::string filePath);
        void generateSingleInstance(std::string filePath, std::string name);

    private:
        std::random_device rd;
        int mWidth;
        int mHeight;
        float mDensity;
        int mAgents;
};
#endif