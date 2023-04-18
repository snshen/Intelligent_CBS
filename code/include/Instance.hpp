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
        void generateSingleInstance(std::string filePath);

        int getWidth() const { return mWidth; };
        int getHeight() const { return mHeight; };
        int getDensity() const { return mDensity; };
        int getAgents() const { return mAgents; };

        void setWidth(int newWidth) { mWidth = newWidth; };
        void setHeight(int newHeight) { mHeight = newHeight; };
        void setDensity(int newDensity) { mDensity = newDensity; };
        void setAgents(int newAgents) { mAgents = newAgents; };

    private:
        std::random_device rd;
        int mWidth;
        int mHeight;
        float mDensity;
        int mAgents;
};
#endif