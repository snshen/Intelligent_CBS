#include "Instance.hpp"
#include "MAPFLoader.hpp"
#include "CBSSolver.hpp"
#include "TesterUtils.hpp"
#include "DataLoader.hpp"

#include <argparse/argparse.hpp>
using namespace std;

int main(int argc, char *argv[])
{
    argparse::ArgumentParser program("Model Training");

    program.add_argument("--num_train").help("number of train instances to use")
        .default_value(5000).scan<'i', int>();
    
    program.add_argument("--train_path").help("path to load train instances")
        .default_value(string("../../data/instances/train_instances/"));
    
    program.add_argument("--train_label_path").help("path to load train data")
        .default_value(string{"../../data/labels/train_labels/"});

    program.add_argument("--num_test").help("number of test instances to use")
        .default_value(500).scan<'i', int>();

    program.add_argument("--test_path").help("path to load test instances")
        .default_value(string{"../../data/instances/test_instances/"});

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
    // Device
    auto cuda_available = torch::cuda::is_available();
    torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
    std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

    const double learning_rate = 1e-3;
    const double weight_decay = 1e-3;
    const size_t num_epochs = 10;

    string instancePath = program.get<string>("train_path");
    string labelPath = program.get<string>("train_label_path");

    int counter;
    bool timeout;
    int id;
    bool train = true;

    TestTimer ttimer;
    DataLoader testLoader;
    MAPFLoader loader;

    // Model
    string filePath = instancePath + to_string(0) + ".txt";
    MAPFInstance mapfProblem = loader.loadInstanceFromFile(filePath);

    ConfNet model(mapfProblem.cols, mapfProblem.rows, 64, 1), *modelPtr;
    model->to(device);

    // Optimizer
    torch::optim::Adam optimizer(
        model->parameters(), torch::optim::AdamOptions(learning_rate).weight_decay(weight_decay));

    // Set floating point output precision
    std::cout << std::fixed << std::setprecision(4);
    for (size_t epoch = 0; epoch != num_epochs; ++epoch) {
        for(int i=0; i<program.get<int>("num_train")+program.get<int>("num_test"); i++)
        {   
            printf("Processing sample %d\n", i);
            
            if (i==program.get<int>("num_train")){
                instancePath = program.get<string>("test_path");
                labelPath = program.get<string>("test_label_path");
            }

            filePath = instancePath + to_string(i) + ".txt";
            mapfProblem = loader.loadInstanceFromFile(filePath);

            //Input maps
            torch::Tensor collisionMap = torch::zeros({mapfProblem.rows, mapfProblem.cols});
            torch::Tensor instanceMap = torch::zeros({mapfProblem.rows, mapfProblem.cols});
            torch::Tensor startMap = torch::zeros({mapfProblem.rows, mapfProblem.cols});
            torch::Tensor goalMap = torch::zeros({mapfProblem.rows, mapfProblem.cols});

            for(int i=0; i<mapfProblem.rows; i++){
                for(int j=0; j<mapfProblem.cols; j++){
                    instanceMap[i][j] = int(mapfProblem.map[i][j]);
                }  
            }
            for(int i=0; i<mapfProblem.numAgents; i++){
                Point2 startLoc = mapfProblem.startLocs[i];
                Point2 goalLoc = mapfProblem.goalLocs[i];
                startMap[mapfProblem.startLocs[i].y][mapfProblem.startLocs[i].x] = 1;
                goalMap[mapfProblem.goalLocs[i].y][mapfProblem.goalLocs[i].x] = 1;
            }
            torch::Tensor inputMaps = torch::stack({collisionMap, instanceMap, startMap, goalMap}, 0);

            filePath = labelPath + to_string(i) + ".txt";
            testLoader.loadDataFromFile(filePath);



            // Create CBS solver 
            CBSSolver cbsSolver;
            counter =  0;
            ttimer.start();

            timeout = false;
            auto optNode = cbsSolver.trainSolve(mapfProblem, counter, timeout, testLoader.pathTensors, modelPtr, inputMaps);
            std::vector<std::vector<Point2>> paths = optNode->paths;

            double elapsedTime = ttimer.elapsed();
            int sumOfCosts=0;
            for(const auto& path : paths)
                sumOfCosts += path.size()-1;
            std::vector<Constraint> constraintList = optNode->constraintList;
            int numConstraint = constraintList.size();

            // filePath = labelPath + to_string(id) + ".txt";
            // writeDataToFile(instance, paths, filePath, sumOfCosts, elapsedTime, counter, numConstraint);

        }
    }

    // testLoader.loadDataFromFile("/home/ubuntu/Intelligent_CBS/data/labels/test_labels/5000.txt");

    // TestTimer ttimer;
    // Instance instance(program.get<int>("map_height"), program.get<int>("map_width"), program.get<float>("obs_density"), program.get<int>("num_agents"));

    // string instancePath = program.get<string>("train_path");
    // string labelPath = program.get<string>("train_label_path");
    // string testPath = program.get<string>("test_path");

    // for(int i=0; i<program.get<int>("num_train")+program.get<int>("num_test"); i++)
    // {   
    //     if (i==program.get<int>("num_train")){
    //         instancePath = program.get<string>("test_path");
    //         labelPath = program.get<string>("test_label_path");
    //     }

    //     string filePath = instancePath + to_string(i) + ".txt";
    //     instance.generateSingleInstance(filePath);

    //     // Load MAPF problem
    //     MAPFLoader loader;
    //     MAPFInstance mapfProblem = loader.loadInstanceFromFile(filePath);

    //     // Create CBS solver
    //     CBSSolver cbsSolver;
    //     int counter;
    //     ttimer.start();

    //     bool unsolvable = false;
    //     auto optNode = cbsSolver.safeSolve(mapfProblem, counter, unsolvable);
    //     if (unsolvable){
    //         i--;
    //         printf("generated unsolvable map, trying again \n");
    //         continue;
    //     }
    //     std::vector<std::vector<Point2>> paths = optNode.value()->paths;

    //     double elapsedTime = ttimer.elapsed();
    //     int sumOfCosts=0;
    //     for(const auto& path : paths)
    //         sumOfCosts += path.size()-1;
    //     std::vector<Constraint> constraintList = optNode.value()->constraintList;
    //     int numConstraint = constraintList.size();

    //     filePath = labelPath + to_string(i) + ".txt";
    //     writeDataToFile(instance, paths, filePath, sumOfCosts, elapsedTime, counter, numConstraint);
    // }
    return 0;
};