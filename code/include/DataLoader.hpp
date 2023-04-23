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
        int numAgents;

        std::vector<torch::Tensor> pathTensors;

        void loadDataFromFile(const std::string& filePath);
        // Make tensor of tensors, each tensor has a map corresponding to the agents
        //

    private:
        void parseText(std::string text, MAPFInstance &result);
        void parseRowsAndCols(std::string line, MAPFInstance &result);
        void parseMap(std::string map_as_txt, MAPFInstance &result);
        void parseAgentDetails(std::string agent_info, MAPFInstance &result);
};

#endif