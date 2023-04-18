#include "Instance.hpp"
#include <iostream>
#include <fstream>

#include <argparse/argparse.hpp>
#include <cassert> 
#include <filesystem>
using namespace std;

Instance::Instance(int height, int width, float density, int numAgents)
    : mWidth(height), mHeight(width), mDensity(density), mAgents(numAgents)
{   
    randMap = vector<char>(mWidth * mHeight, '.');
    generateMap();
}

void Instance::generateMap()
{
    fill(randMap.begin(), randMap.end(), '.');
    int num_obs = (((float)mWidth*(float)mHeight)*mDensity);
    for(int i=0; i<num_obs+2*mAgents; i++)
    {   
        if (i<num_obs) randMap[i] = '@';
        else if (i<num_obs+mAgents) randMap[i] = 's';
        else randMap[i] = 'g';
    }

    mt19937 g(rd());
    shuffle(randMap.begin(), randMap.end(), g);
}

void Instance::writeMapToFile(string filePath)
{
    ofstream fw(filePath);

    if (fw.is_open())
    {   
        fw << mHeight << " " << mWidth << "\n";

        vector<pair<int,int>> agentStarts;
        vector<pair<int,int>> agentGoals;

        for (int i = 0; i < mHeight; i++) {
            for (int j = 0; j < mWidth; j++)
            {   
                char currChar = randMap[i*mWidth+j];
                if (currChar=='s'){
                    agentStarts.push_back(make_pair(i,j));
                    currChar = '.';
                }
                else if (currChar=='g'){
                    agentGoals.push_back(make_pair(i,j));
                    currChar = '.';
                }
                fw << currChar;
                if(j!=mWidth-1){
                    fw << " ";
                } 
            }
            fw << "\n";
        }

        mt19937 g(rd());
        shuffle(agentGoals.begin(), agentGoals.end(), g);

        fw << mAgents << "\n";
        for(int i = 0; i < mAgents; i++){
            pair<int,int> agentStart = agentStarts[i];
            pair<int,int> agentGoal = agentGoals[i];
            fw << agentStart.first << " " << agentStart.second << " " << agentGoal.first << " " << agentGoal.second <<"\n";
        }
        fw.close();
    } else cout << "Problem with opening file";
}

void Instance::generateSingleInstance(string filePath)
{
    generateMap();
    writeMapToFile(filePath);
}     


// int main(int argc, char *argv[])
// {
//     argparse::ArgumentParser program("instance generator");

//     program.add_argument("--map_height").help("height of map enviornment")
//         .default_value(16).scan<'i', int>();

//     program.add_argument("--map_width").help("width of map enviornment")
//         .default_value(16).scan<'i', int>();

//     program.add_argument("--num_agents").help("number of agents in the map enviornment")
//         .default_value(4).scan<'i', int>();
    
//     program.add_argument("--obs_density").help("density of obstacles in the map enviornment")
//         .default_value(0.25f).scan<'f', float>();
    
//     program.add_argument("--num_train").help("number of train instances to generate")
//         .default_value(50).scan<'i', int>();
    
//     program.add_argument("--train_path").help("path to save training data")
//         .default_value(string("../../data/train_instances/"));

//     program.add_argument("--num_test").help("number of test instances to generate")
//         .default_value(10).scan<'i', int>();

//     program.add_argument("--test_path").help("path to save test data")
//         .default_value(string{"../../data/test_instances/"});

//     try
//     {
//         program.parse_args(argc, argv);
//     }
//     catch (const runtime_error &err)
//     {
//         cerr << err.what() << endl;
//         cerr << program;
//         return 1;
//     }

//     Instance instance(program.get<int>("map_height"), program.get<int>("map_width"), program.get<float>("obs_density"), program.get<int>("num_agents"));

//     string trainPath = program.get<string>("train_path");
//     string testPath = program.get<string>("test_path");

//     for(int i=0; i<program.get<int>("num_train"); i++)
//     {   
//         string name = to_string(i);
//         instance.generateSingleInstance(trainPath, name);
//     }

//     for(int i=0; i<program.get<int>("num_test"); i++)
//     {
//          string name = to_string(i);
//         instance.generateSingleInstance(testPath, name);
//     }

//     return 0;
// }