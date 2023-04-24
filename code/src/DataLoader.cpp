#include "DataLoader.hpp"
#include <iostream>
#include <fstream>

void DataLoader::loadDataFromFile(const std::string& filePath)
{   

    std::ifstream fs;
    fs.open(filePath, std::ios::in);

    std::string filename = filePath.substr(filePath.find_last_of("/") + 1);
    std::string::size_type const p(filename.find_last_of('.'));
    id = filename.substr(0, p);

    if (fs.is_open())
    {
        // get height and width11
        fs >> height >> width;
        // get num of agents
        fs >> numAgents;

        //get paths 
        std::string line;
        for (int i=0; i<numAgents; i++) 
        {   
            fs>>line;
            torch::Tensor t = torch::zeros({height, width});
            std::getline(fs, line);
            
            for (int i=0; i<height; i++){
                for (int j=0; j<width; j++){
                    t[i][j] = line[i * width + j] - '0';
                }
            }
            pathTensors.push_back(t);
        }
        
        fs >> metrics.sumOfCosts >> metrics.elapsedTime >> metrics.counter >> metrics.numConstraint;
        fs.close();
        
    } else std::cout << "Problem with opening file";

}