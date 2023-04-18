#include "Instance.hpp"
#include "MAPFLoader.hpp"
#include "CBSSolver.hpp"
#include "AStar.hpp"
#include "TesterUtils.hpp"

#include <argparse/argparse.hpp>
using namespace std;

void writePathsToFile(const Instance &mapInstance, vector<vector<Point2>> paths, string filePath){
    
    int width = mapInstance.getWidth();
    int height = mapInstance.getHeight();
    int numAgents = mapInstance.getAgents();

    vector<char> pathMap(height * width);
    ofstream file;
    file.open ("filePath.txt");

    for(int i=0; i<numAgents; i++){
        std::vector<Point2> path = paths[i];
        fill(pathMap.begin(), pathMap.end(), 0);
        
        for(Point2 loc:path){
            pathMap[loc.x*width + loc.y] = 1;
        }

        // file << path;
    }
    file.close();
    
}

int main(int argc, char *argv[])
{
    argparse::ArgumentParser program("instance generator");

    program.add_argument("--map_height").help("height of map enviornment")
        .default_value(64).scan<'i', int>();

    program.add_argument("--map_width").help("width of map enviornment")
        .default_value(64).scan<'i', int>();

    program.add_argument("--num_agents").help("number of agents in the map enviornment")
        .default_value(8).scan<'i', int>();
    
    program.add_argument("--obs_density").help("density of obstacles in the map enviornment")
        .default_value(0.25f).scan<'f', float>();
    
    program.add_argument("--num_train").help("number of train instances to generate")
        .default_value(10).scan<'i', int>();
    
    program.add_argument("--train_path").help("path to save training data")
        .default_value(string("../../data/instances/train_instances/"));

    program.add_argument("--num_test").help("number of test instances to generate")
        .default_value(2).scan<'i', int>();

    program.add_argument("--test_path").help("path to save test data")
        .default_value(string{"../../data/instances/test_instances/"});

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

    TestTimer ttimer;
    Instance instance(program.get<int>("map_height"), program.get<int>("map_width"), program.get<float>("obs_density"), program.get<int>("num_agents"));

    string trainPath = program.get<string>("train_path");
    string testPath = program.get<string>("test_path");

    for(int i=0; i<program.get<int>("num_train"); i++)
    {   
        string filePath = trainPath + to_string(i) + ".txt";
        instance.generateSingleInstance(filePath);

        // Load MAPF problem
        MAPFLoader loader;
        MAPFInstance mapfProblem = loader.loadInstanceFromFile(filePath);

        // Create CBS solver
        CBSSolver cbsSolver;
        int counter;
        ttimer.start();

        auto optNode = cbsSolver.safeSolve(mapfProblem, counter);
        if (!optNode.has_value()){
            i--;
            printf("generated unsolvable map, trying again \n");
            continue;
        }
        std::vector<std::vector<Point2>> paths = optNode.value()->paths;
        writePathsToFile(instance, paths, filePath);

        std::vector<Constraint> constraintList = optNode.value()->constraintList;
        double elapsedTime = ttimer.elapsed();
        int sumOfCosts=0;
        for(const auto& path : paths)
            sumOfCosts += path.size()-1;

        printf("SOC = %d\n", sumOfCosts);
        printf("Elapsed time = %f ms\n", elapsedTime);
        printf("Nodes generated = %d \n", counter);

        // // Log results for specific test file
        // if(testFile == animateFile){
        //     saveToFile(resultFile, testFile, paths);
        // }
    }

    for(int i=0; i<program.get<int>("num_test"); i++)
    {
        string filePath = trainPath + to_string(i) + ".txt";
        instance.generateSingleInstance(filePath);
    }
    printf("WERE GENIUSESSSSS \n");
    return 0;
};