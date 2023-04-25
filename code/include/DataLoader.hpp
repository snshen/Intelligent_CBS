#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <string>
#include <torch/torch.h>
#include "MAPFInstance.hpp"

class DataLoader
{
    public:

        struct Metrics
        {
            int sumOfCosts;
            float elapsedTime; 
            int counter;
            int numConstraint;
        };

        Metrics metrics;
        
        std::string id;
        int height;
        int width;

        torch::Tensor constraintTensor;
        void loadDataFromFile(const std::string& filePath);

    private:
};

#endif