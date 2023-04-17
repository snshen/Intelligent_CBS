#include <iostream>
#include <fstream>

#include <string>
#include <vector>
#include <random>

#include <argparse/argparse.hpp>
#include <cassert> 
#include <filesystem>
using namespace std;

vector<char> randomMapGenerator( int height, int width, float density, int num_agents)
{
    vector<char> flattenedMap(width * height, '.');
    
    int num_obs = (((float)width*(float)height)*density);
    for(int i=0; i<num_obs+2*num_agents; i++)
    {   
        if (i<num_obs)
        {
            flattenedMap[i] = '@';
        }
        else if (i<num_obs+num_agents)
        {
            flattenedMap[i] = 's';
        }
        else
        {
            flattenedMap[i] = 'g';
        }
    }
    random_device rd;
    mt19937 g(rd());
    shuffle(flattenedMap.begin(), flattenedMap.end(), g);

    return flattenedMap;
};

void writeMapToFile(vector<char> randMap, int height, int width, int numAgents, string filePath)
{
    ofstream fw(filePath);

    if (fw.is_open())
    {   
        fw << height << " " << width << "\n";

        vector<pair<int,int>> agentStarts;
        vector<pair<int,int>> agentGoals;

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++)
            {   
                char currChar = randMap[i*width+j];
                if (currChar=='s'){
                    agentStarts.push_back(make_pair(i,j));
                    currChar = '.';
                }
                else if (currChar=='g'){
                    agentGoals.push_back(make_pair(i,j));
                    currChar = '.';
                }
                fw << currChar;
                if(j!=width-1){
                    fw << " ";
                } 
            }
            fw << "\n";
        }
        fw << numAgents << "\n";
        for(int i = 0; i < numAgents; i++){
            pair<int,int> agentStart = agentStarts[i];
            pair<int,int> agentGoal = agentGoals[i];
            fw << agentStart.first << " " << agentStart.second << " " << agentGoal.first << " " << agentGoal.second <<"\n";
        }
        fw.close();
    } else cout << "Problem with opening file";
}

int main(int argc, char *argv[])
{
    argparse::ArgumentParser program("instance generator");

    program.add_argument("--map_height").help("height of map enviornment")
        .default_value(8).scan<'i', int>();

    program.add_argument("--map_width").help("width of map enviornment")
        .default_value(8).scan<'i', int>();

    program.add_argument("--num_agents").help("number of agents in the map enviornment")
        .default_value(4).scan<'i', int>();
    
    program.add_argument("--obs_density").help("density of obstacles in the map enviornment")
        .default_value(0.1f).scan<'f', float>();
    
    program.add_argument("--num_train").help("number of train instances to generate")
        .default_value(50).scan<'i', int>();
    
    program.add_argument("--train_path").help("path to save training data")
        .default_value(string("../data/train_instances/"));

    program.add_argument("--num_test").help("number of test instances to generate")
        .default_value(10).scan<'i', int>();

    program.add_argument("--test_path").help("path to save test data")
        .default_value(string{"../data/test_instances/"});

    try
    {
        program.parse_args(argc, argv);
    }
    catch (const runtime_error &err)
    {
        cerr << err.what() << endl;
        cerr << program;
        return 1;
    }

    int height = program.get<int>("map_height");
    int width = program.get<int>("map_width");
    float density = program.get<float>("obs_density");
    int numAgents = program.get<int>("num_agents");

    for(int i=0; i<program.get<int>("num_train"); i++)
    {
        cout << "Generating train instance: " << i << endl;
        vector<char> randMap = randomMapGenerator(height, width, density, numAgents);
        string filePath = program.get<string>("train_path")+to_string(i)+".txt";
        writeMapToFile(randMap, height, width, numAgents, filePath);
    }

    for(int i=0; i<program.get<int>("num_test"); i++)
    {
        cout << "Generating test instance: " << i << endl;
        vector<char> randMap = randomMapGenerator(height, width, density, numAgents);
        string filePath = program.get<string>("test_path")+to_string(i)+".txt";
        writeMapToFile(randMap, height, width, numAgents, filePath);
    }
    

    return 0;
}