#include "Instance.hpp"
#include "MAPFLoader.hpp"
#include "CBSSolver.hpp"
#include "TesterUtils.hpp"

#include <argparse/argparse.hpp>
using namespace std;

void writeDataToFile(const Instance &mapInstance, std::vector<Constraint> constraintList, string filePath,
                     int sumOfCosts, float elapsedTime, int counter){
    
    int width = mapInstance.getWidth();
    int height = mapInstance.getHeight();
    int numAgents = mapInstance.getAgents();
    int numConstraint = constraintList.size();

    vector<char> constraintMap(height * width);
    ofstream file(filePath);

    if (file.is_open())
    {   
        file << height << " " << width << "\n";
        
        fill(constraintMap.begin(), constraintMap.end(), '0');
        
        for(Constraint constraint:constraintList){
            constraintMap[constraint.location.first.x*width + constraint.location.first.y] = '1';
        }
        for(char cell:constraintMap){
            file << cell;
        }
        file << "\n";

        file << sumOfCosts << " " << elapsedTime << " " << counter << " " << numConstraint << "\n";
        file.close();
    } else cout << "Problem with opening file";
}

int main(int argc, char *argv[])
{
    argparse::ArgumentParser program("Data Generator");

    program.add_argument("--map_height").help("height of map enviornment")
        .default_value(64).scan<'i', int>();

    program.add_argument("--map_width").help("width of map enviornment")
        .default_value(64).scan<'i', int>();

    program.add_argument("--num_agents").help("number of agents in the map enviornment")
        .default_value(20).scan<'i', int>();
    
    program.add_argument("--obs_density").help("density of obstacles in the map enviornment")
        .default_value(0.25f).scan<'f', float>();
    
    program.add_argument("--num_train").help("number of train instances to generate")
        .default_value(5000).scan<'i', int>();
    
    program.add_argument("--train_path").help("path to save training data")
        .default_value(string("../../data/instances/train_instances/"));
    
    program.add_argument("--start_train").help("id of train instances to start at")
        .default_value(0).scan<'i', int>();

    program.add_argument("--num_test").help("number of test instances to generate")
        .default_value(500).scan<'i', int>();

    program.add_argument("--test_path").help("path to save test data")
        .default_value(string{"../../data/instances/test_instances/"});
    
    program.add_argument("--start_test").help("id of test instances to start at")
        .default_value(0).scan<'i', int>();
    
    program.add_argument("--train_label_path").help("path to save train data")
        .default_value(string{"../../data/labels/train_labels/"});

    program.add_argument("--test_label_path").help("path to save test data")
        .default_value(string{"../../data/labels/test_labels/"});

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

    string instancePath = program.get<string>("train_path");
    string labelPath = program.get<string>("train_label_path");
    string testPath = program.get<string>("test_path");
    int start = program.get<int>("start_train");

    int counter;
    bool unsolvable;
    int id;

    for(int i=0; i<program.get<int>("num_train")+program.get<int>("num_test"); i++)
    {   
        printf("Processing sample %d\n", i);
        
        if (i==program.get<int>("num_train")){
            instancePath = program.get<string>("test_path");
            labelPath = program.get<string>("test_label_path");
            start = program.get<int>("start_test");
        }

        id = i+start;
        string filePath = instancePath + to_string(id) + ".txt";
        instance.generateSingleInstance(filePath);

        // Load MAPF problem
        MAPFLoader loader;
        MAPFInstance mapfProblem = loader.loadInstanceFromFile(filePath);

        // Create CBS solver
        CBSSolver cbsSolver;
        counter =  0;
        ttimer.start();

        unsolvable = false;
        auto optNode = cbsSolver.safeSolve(mapfProblem, counter, unsolvable);
        if (unsolvable){
            i--;
            printf("generated unsolvable map, trying again \n");
            continue;
        }
        
        std::vector<std::vector<Point2>> paths = optNode->paths;
        
        double elapsedTime = ttimer.elapsed();
        int sumOfCosts=0;
        for(const auto& path : paths)
            sumOfCosts += path.size()-1;

        std::vector<Constraint> constraintList = optNode->constraintList;

        filePath = labelPath + to_string(id) + ".txt";
        writeDataToFile(instance, constraintList, filePath, sumOfCosts, elapsedTime, counter);
    }
    return 0;
};