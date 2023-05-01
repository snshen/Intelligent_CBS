#include "Instance.hpp"
#include "MAPFLoader.hpp"
#include "CBSSolver.hpp"
#include "TesterUtils.hpp"
#include "DataLoader.hpp"

#include <argparse/argparse.hpp>
using namespace std;

trainMetrics runOneInstance(MAPFInstance& mapfProblem, DataLoader& trainLoader, int& counter, TestTimer& ttimer, torch::optim::Adam& optimizer, ConfNet* modelPtr, torch::Device device, bool& timeout){

    //Input maps
    torch::Tensor collisionMap = torch::zeros({mapfProblem.rows, mapfProblem.cols}, device);
    torch::Tensor instanceMap = torch::zeros({mapfProblem.rows, mapfProblem.cols}, device);
    torch::Tensor startMap = torch::zeros({mapfProblem.rows, mapfProblem.cols}, device);
    torch::Tensor goalMap = torch::zeros({mapfProblem.rows, mapfProblem.cols}, device);
    
    for(int i=0; i<mapfProblem.rows; i++){
        for(int j=0; j<mapfProblem.cols; j++){
            instanceMap[i][j] = int(mapfProblem.map[i][j]);
        }  
    }
    
    for(int i=0; i<mapfProblem.numAgents; i++){
        Point2 startLoc = mapfProblem.startLocs[i];
        Point2 goalLoc = mapfProblem.goalLocs[i];
        startMap[mapfProblem.startLocs[i].x][mapfProblem.startLocs[i].y] = 1;
        goalMap[mapfProblem.goalLocs[i].x][mapfProblem.goalLocs[i].y] = 1;
    }
    
    torch::Tensor inputMaps = torch::stack({collisionMap, instanceMap, startMap, goalMap}, 0);
    inputMaps = torch::unsqueeze(inputMaps, 0);
    inputMaps.to(device);
    
    // Create CBS solver 
    CBSSolver cbsSolver;
    counter =  0;
    ttimer.start();

    //track metrics
    trainMetrics metrics;
    metrics.counter =  0;
    metrics.runningLoss = 0;
    metrics.numLoss = 0;

    auto optNode = cbsSolver.testSolve(mapfProblem, timeout, modelPtr, inputMaps, metrics, device);
    
    std::vector<std::vector<Point2>> paths = optNode->paths;

    metrics.elapsedTime = ttimer.elapsed();
    metrics.sumOfCosts=0;
    for(const auto& path : paths)
        metrics.sumOfCosts += path.size()-1;
    std::vector<Constraint> constraintList = optNode->constraintList;
    metrics.numConstraint = constraintList.size();
    metrics.avgLoss = metrics.runningLoss/static_cast<float>(metrics.numLoss);
    
    return metrics;
}

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
    
    program.add_argument("--num").help("number of instances to evaluate over")
        .default_value(100).scan<'i', int>();
    
    program.add_argument("--instance_path").help("path to save training data")
        .default_value(string("../../data/online/instance.txt"));

    program.add_argument("--label_path").help("path to save train data")
        .default_value(string{"../../data/online/label.txt"});
    
    program.add_argument("--model_path").help("path to models")
        .default_value(string{"../../data/models/epoch_2_sample_0.pt"});

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
    int height = program.get<int>("map_height");
    int width = program.get<int>("map_width");
    Instance instance(height, width, program.get<float>("obs_density"), program.get<int>("num_agents"));

    string instancePath = program.get<string>("instance_path");
    string labelPath = program.get<string>("label_path");
    int num = program.get<int>("num");

    int counter;
    bool unsolvable;
    int CBSUnsolvable = 0;
    int ItellUnsolvable = 0;

    //load model
    auto cuda_available = torch::cuda::is_available();
    torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
    std::cout << (cuda_available ? "CUDA available. Using GPU." : "Training on CPU.") << '\n';

    ConfNet model(width, height, 64, 1);
    torch::load(model, program.get<string>("model_path"));

    model->to(device);

    // Optimizer
    torch::optim::Adam optimizer(
        model->parameters(), torch::optim::AdamOptions(0).weight_decay(1e-3));

    model->eval();

    // Set floating point output precision
    std::cout << std::fixed << std::setprecision(4);

    //metric tracking

    std::vector<float> inte_losses;
    std::vector<float> inte_times;
    std::vector<float> inte_counts;
    std::vector<float> inte_constraints;
    
    std::vector<float> orig_times;
    std::vector<float> orig_counts;
    std::vector<float> orig_constraints;

    DataLoader testLoader;

    for(int i=0; i<num; i++)
    {   
        printf("Processing sample %d\n", i);
        
        instance.generateSingleInstance(instancePath);

        // Load MAPF problem
        MAPFLoader loader;
        MAPFInstance mapfProblem = loader.loadInstanceFromFile(instancePath);


        // Check A* solvability
        AStar lowLevelSolver(mapfProblem);
        CBSSolver::CTNodeSharedPtr root = std::make_shared<CBSSolver::CTNode>();
        root->paths.resize(mapfProblem.numAgents);
        bool found = true;
        for (int j = 0; j < mapfProblem.startLocs.size(); j++)
        {
            found = lowLevelSolver.solve(j, root->constraintList, root->paths[j]);
            
            if (!found){
                printf("generated unsolvable map, trying again \n");
                break;
            }
        }   
        if (!found){
            i--;
            continue;
        }

        // CBS solver
        CBSSolver cbsSolver;
        counter =  0;
        ttimer.start();

        unsolvable = false;
        auto optNode = cbsSolver.safeSolve(mapfProblem, counter, unsolvable);
        if (unsolvable){
            CBSUnsolvable++;
        }
        
        std::vector<std::vector<Point2>> paths = optNode->paths;
        
        double elapsedTime = ttimer.elapsed();
        int sumOfCosts=0;
        for(const auto& path : paths)
            sumOfCosts += path.size()-1;

        std::vector<Constraint> constraintList = optNode->constraintList;

        writeDataToFile(instance, constraintList, labelPath, sumOfCosts, elapsedTime, counter);

        testLoader.loadDataFromFile(labelPath);

        //Intelligient Solver
        bool timeout = false;
        trainMetrics metrics = runOneInstance(mapfProblem, testLoader, counter, ttimer, optimizer, &model, device, timeout);
        if(!timeout && !unsolvable){
            cout<<"TRAIN RESULTS | elapsedTime: " << metrics.elapsedTime << ", counter: " << metrics.counter << ", numConstraint: " << metrics.numConstraint << ", unsolved: " << ItellUnsolvable << "\n";
            cout<<"ORIGI RESULTS | elapsedTime: " << testLoader.metrics.elapsedTime << ", counter: " << testLoader.metrics.counter << ", numConstraint: " << testLoader.metrics.numConstraint << ", unsolved: " << CBSUnsolvable << "\n";
            
            inte_losses.push_back(metrics.avgLoss);
            inte_times.push_back(metrics.elapsedTime);
            inte_counts.push_back(metrics.counter);
            inte_constraints.push_back(metrics.numConstraint);

            orig_times.push_back(testLoader.metrics.elapsedTime);
            orig_counts.push_back(testLoader.metrics.counter);
            orig_constraints.push_back(testLoader.metrics.numConstraint);
        }
        else if(timeout){
            ItellUnsolvable++;
        }
        
        if (timeout || unsolvable){
            i--;
        }

    }

    float inte_time = std::accumulate(inte_times.begin(), inte_times.end(), 0.0) / inte_times.size();
    float inte_count = std::accumulate(inte_counts.begin(), inte_counts.end(), 0.0) / inte_counts.size();
    float inte_constraint = std::accumulate(inte_constraints.begin(), inte_constraints.end(), 0.0) / inte_constraints.size();

    float orig_time = std::accumulate(orig_times.begin(), orig_times.end(), 0.0) / orig_times.size();
    float orig_count = std::accumulate(orig_counts.begin(), orig_counts.end(), 0.0) / orig_counts.size();
    float orig_constraint = std::accumulate(orig_constraints.begin(), orig_constraints.end(), 0.0) / orig_constraints.size();

    cout<<"---------------------FINAL RESULTS---------------------\n";
    cout<<"INTELLIGENT | elapsedTime: " << inte_time << ", counter: " << inte_count << ", numConstraint: " << inte_constraint << ", unsolved: " << ItellUnsolvable << "\n";
    cout<<"ORIGINAL | elapsedTime: " << orig_time << ", counter: " << orig_count << ", numConstraint: " << orig_constraint << ", unsolved: " << CBSUnsolvable << "\n";
    
    return 0;
};