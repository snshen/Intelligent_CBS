#include "Instance.hpp"
#include "MAPFLoader.hpp"
#include "CBSSolver.hpp"
#include "TesterUtils.hpp"

#include <argparse/argparse.hpp>
using namespace std;

int main(int argc, char *argv[])
{
    argparse::ArgumentParser program("Model Training");

    program.add_argument("--num_train").help("number of train instances to use")
        .default_value(10).scan<'i', int>();
    
    program.add_argument("--train_path").help("path to load train instances")
        .default_value(string("../../data/instances/train_instances/"));

    program.add_argument("--num_test").help("number of test instances to use")
        .default_value(2).scan<'i', int>();

    program.add_argument("--test_path").help("path to load test instances")
        .default_value(string{"../../data/instances/test_instances/"});
    
    program.add_argument("--train_label_path").help("path to load train data")
        .default_value(string{"../../data/labels/train_labels/"});

    program.add_argument("--test_label_path").help("path to load test data")
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

    for(int i=0; i<program.get<int>("num_train")+program.get<int>("num_test"); i++)
    {   
        if (i==program.get<int>("num_train")){
            instancePath = program.get<string>("test_path");
            labelPath = program.get<string>("test_label_path");
        }

        string filePath = instancePath + to_string(i) + ".txt";
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

        double elapsedTime = ttimer.elapsed();
        int sumOfCosts=0;
        for(const auto& path : paths)
            sumOfCosts += path.size()-1;
        std::vector<Constraint> constraintList = optNode.value()->constraintList;
        int numConstraint = constraintList.size();

        filePath = labelPath + to_string(i) + ".txt";
        writeDataToFile(instance, paths, filePath, sumOfCosts, elapsedTime, counter, numConstraint);
    }
    return 0;
};