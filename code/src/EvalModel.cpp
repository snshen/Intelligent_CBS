#include "Instance.hpp"
#include "MAPFLoader.hpp"
#include "CBSSolver.hpp"
#include "TesterUtils.hpp"
#include "DataLoader.hpp"
// #include <gtest/gtest.hpp>

#include <argparse/argparse.hpp>
using namespace std;

void writeMetricsToFile(trainMetrics metrics, string filePath){
    ofstream file(filePath, std::ios_base::app);
    if (file.is_open())
    {   
        file << metrics.sumOfCosts << " " << metrics.elapsedTime << " " << metrics.counter << " " << metrics.numConstraint << "\n";
        file.close();
    } else cout << "Problem with opening file";
}

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



int main(int argc, char *argv[])
{
    argparse::ArgumentParser program("Model Evaluation");

    program.add_argument("--num_train").help("number of train instances to use")
        .default_value(3500).scan<'i', int>();
    
    program.add_argument("--train_path").help("path to load train instances")
        .default_value(string("../../data/instances/train_instances/"));
    
    program.add_argument("--train_label_path").help("path to load train data")
        .default_value(string{"../../data/labels/train_labels/"});

    program.add_argument("--num_test").help("number of test instances to use")
        .default_value(100).scan<'i', int>();

    program.add_argument("--test_path").help("path to load test instances")
        .default_value(string{"../../data/instances/test_instances/"});

    program.add_argument("--test_label_path").help("path to load test data")
        .default_value(string{"../../data/labels/test_labels/"});

    program.add_argument("--output_path").help("path to outputs")
        .default_value(string{"../../data/outputs/train_outputs.txt"});
    
    program.add_argument("--model_path").help("path to models")
        .default_value(string{"../../data/models/latest.pt"});
    
    program.add_argument("--lr").help("learning rate for model training")
        .default_value(0.003f).scan<'f', float>();
    
    program.add_argument("--eval_freq").help("frequecy to save model with")
        .default_value(1000).scan<'i', int>();

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

    const double learning_rate = program.get<float>("lr");
    const double weight_decay = 1e-3;
    const size_t num_epochs = 10;

    string instancePath = program.get<string>("train_path");
    string labelPath = program.get<string>("train_label_path");
    string outPath = program.get<string>("output_path");
    string modelPath = program.get<string>("model_path");

    int numTest = program.get<int>("num_test");
    int numTrain = program.get<int>("num_train");

    int counter;
    bool timeout;
    int id;
    bool train = true;

    TestTimer ttimer;
    DataLoader testLoader;
    MAPFLoader loader;
    float bestLoss = 100;

    // Model
    string filePath = instancePath + to_string(0) + ".txt";
    MAPFInstance mapfProblem = loader.loadInstanceFromFile(filePath);

    ConfNet model(mapfProblem.cols, mapfProblem.rows, 64, 1); //, *modelPtr;
    torch::load(model, modelPath);

    model->to(device);

    // Optimizer
    torch::optim::Adam optimizer(
        model->parameters(), torch::optim::AdamOptions(0).weight_decay(weight_decay));

    model->eval();

    // Set floating point output precision
    std::cout << std::fixed << std::setprecision(4);
    
    std::vector<float> inte_losses;
    std::vector<float> inte_times;
    std::vector<float> inte_counts;
    std::vector<float> inte_constraints;
    
    std::vector<float> orig_times;
    std::vector<float> orig_counts;
    std::vector<float> orig_constraints;

    for(int i=0; i<numTest; i++)
    {   
        printf("Processing sample %d\n", i);
        
        int id = i+numTrain;
        filePath = instancePath + to_string(id) + ".txt";
        mapfProblem = loader.loadInstanceFromFile(filePath);
        
        filePath = labelPath + to_string(id) + ".txt";
        testLoader.loadDataFromFile(filePath);

        bool timeout = false;
        trainMetrics metrics = runOneInstance(mapfProblem, testLoader, counter, ttimer, optimizer, &model, device, timeout);
        if(!timeout){
            cout<<"TRAIN RESULTS | elapsedTime: " << metrics.elapsedTime << ", counter: " << metrics.counter << ", numConstraint: " << metrics.numConstraint << "\n";
            cout<<"ORIGI RESULTS | elapsedTime: " << testLoader.metrics.elapsedTime << ", counter: " << testLoader.metrics.counter << ", numConstraint: " << testLoader.metrics.numConstraint << "\n";
            
            inte_losses.push_back(metrics.avgLoss);
            inte_times.push_back(metrics.elapsedTime);
            inte_counts.push_back(metrics.counter);
            inte_constraints.push_back(metrics.numConstraint);

            orig_times.push_back(testLoader.metrics.elapsedTime);
            orig_counts.push_back(testLoader.metrics.counter);
            orig_constraints.push_back(testLoader.metrics.numConstraint);
        }else{
            numTest++;
        }

        
    }
    float inte_loss = std::accumulate(inte_losses.begin(), inte_losses.end(), 0.0) / inte_losses.size();
    float inte_time = std::accumulate(inte_times.begin(), inte_times.end(), 0.0) / inte_times.size();
    float inte_count = std::accumulate(inte_counts.begin(), inte_counts.end(), 0.0) / inte_counts.size();
    float inte_constraint = std::accumulate(inte_constraints.begin(), inte_constraints.end(), 0.0) / inte_constraints.size();

    float orig_time = std::accumulate(orig_times.begin(), orig_times.end(), 0.0) / orig_times.size();
    float orig_count = std::accumulate(orig_counts.begin(), orig_counts.end(), 0.0) / orig_counts.size();
    float orig_constraint = std::accumulate(orig_constraints.begin(), orig_constraints.end(), 0.0) / orig_constraints.size();

    cout<<"---------------------FINAL RESULTS---------------------\n";
    cout<<"INTELLIGENT | elapsedTime: " << inte_time << ", counter: " << inte_count << ", numConstraint: " << inte_constraint << ", loss: " << inte_loss << "\n";
    cout<<"ORIGINAL | elapsedTime: " << orig_time << ", counter: " << orig_count << ", numConstraint: " << orig_constraint << "\n";
    
    return 0;
};