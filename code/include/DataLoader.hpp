#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <string>
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
        
        int id;
        int height;
        int width;
        int numAgents;

        DataLoader loadDataFromFile(const std::string& filePath);
        // Make tensor of tensors, each tensor has a map corresponding to the agents
        //

    private:
        void parseText(std::string text, MAPFInstance &result);
        void parseRowsAndCols(std::string line, MAPFInstance &result);
        void parseMap(std::string map_as_txt, MAPFInstance &result);
        void parseAgentDetails(std::string agent_info, MAPFInstance &result);
};

#endif