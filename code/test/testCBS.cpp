#include <algorithm>
#include "TesterUtils.hpp"
#include "MAPFLoader.hpp"
#include "CBSSolver.hpp"

// Answers
std::unordered_map<std::string, int> answers = {
    {"../instances/test_1.txt",41},
    {"../instances/test_10.txt",19},
    {"../instances/test_11.txt",35},
    {"../instances/test_12.txt",36},
    {"../instances/test_13.txt",36},
    {"../instances/test_14.txt",24},
    {"../instances/test_15.txt",50},
    {"../instances/test_16.txt",51},
    {"../instances/test_17.txt",39},
    {"../instances/test_18.txt",32},
    {"../instances/test_19.txt",47},
    {"../instances/test_2.txt",18},
    {"../instances/test_20.txt",28},
    {"../instances/test_21.txt",46},
    {"../instances/test_22.txt",51},
    {"../instances/test_23.txt",32},
    {"../instances/test_24.txt",47},
    {"../instances/test_25.txt",40},
    {"../instances/test_26.txt",42},
    {"../instances/test_27.txt",40},
    {"../instances/test_28.txt",41},
    {"../instances/test_29.txt",48},
    {"../instances/test_3.txt",28},
    {"../instances/test_30.txt",43},
    {"../instances/test_31.txt",39},
    {"../instances/test_32.txt",30},
    {"../instances/test_33.txt",28},
    {"../instances/test_34.txt",33},
    {"../instances/test_35.txt",30},
    {"../instances/test_36.txt",23},
    {"../instances/test_37.txt",38},
    {"../instances/test_38.txt",28},
    {"../instances/test_39.txt",35},
    {"../instances/test_4.txt",32},
    {"../instances/test_40.txt",24},
    {"../instances/test_41.txt",45},
    {"../instances/test_42.txt",57},
    {"../instances/test_43.txt",43},
    {"../instances/test_44.txt",33},
    {"../instances/test_45.txt",24},
    {"../instances/test_46.txt",57},
    {"../instances/test_47.txt",65},
    {"../instances/test_48.txt",36},
    {"../instances/test_49.txt",42},
    {"../instances/test_5.txt",26},
    {"../instances/test_50.txt",48},
    {"../instances/test_6.txt",24},
    {"../instances/test_7.txt",34},
    {"../instances/test_8.txt",38},
    {"../instances/test_9.txt",24},
    {"../instances/test_51.txt",102},
    {"../instances/test_52.txt",299},
    {"../instances/test_53.txt",143},
    {"../instances/test_54.txt",51},
    {"../instances/test_55.txt",202},
    {"../instances/test_56.txt",100},
    {"../instances/test_57.txt",70},
    {"../instances/test_60.txt",16},
    {"../instances/test_61.txt",6},
    {"../instances/test_62.txt",15},
    {"../instances/test_63.txt",7},
    {"../instances/test_64.txt",8}
};

int main()
{
    std::string path = "../../data/train_instances";
    std::string resultFile = "../outputs/result.txt";
    std::string animateFile = "../instances/exp0.txt";
    std::vector<std::string> ignoreFiles = {
            "../instances/evaluation",
            "../instances/min-sum-of-cost.csv",
            "../instances/maze_map.txt",
            "../instances/random_map.txt",
            "../instances/test_56.txt", // For now skip
    };
    const int testAmount = 5;
    int testResult = 0;
    TestTimer ttimer;
    for (const auto & entry : std::filesystem::directory_iterator(path)){
        // Skip if necessary
        std::string testFile = entry.path();
        if(std::find(ignoreFiles.begin(), ignoreFiles.end(), testFile) != ignoreFiles.end())
            continue;

        // Load MAPF problem
        MAPFLoader loader;
        printf("Test file: %s\n", testFile.c_str());
        MAPFInstance mapfProblem = loader.loadInstanceFromFile(testFile);

        // Create CBS solver
        CBSSolver cbsSolver;

        // Solve Problem
        double elapsedTime = 0;
        std::vector<std::vector<Point2>> paths;
        for(int i=0; i<testAmount; i++)
        {
            ttimer.start();
            paths = cbsSolver.solve(mapfProblem);
            elapsedTime += ttimer.elapsed();
        }
        elapsedTime /= testAmount;

        // Evaluate
        int sumOfCosts=0;
        for(const auto& path : paths)
            sumOfCosts += path.size()-1;
        if (answers.find(testFile) != answers.end()) {
            int expectedSOC = answers[testFile];
            if(expectedSOC != sumOfCosts)
            {
                printf("* Incorrect SOC; expected %d, got %d\n", expectedSOC, sumOfCosts);
                testResult = 1;
            }
            else
                printf("SOC = %d\n", sumOfCosts);
        }
        else
        {
            printf("SOC = %d (not tested)\n", sumOfCosts);
        }
        printf("Elapsed time = %f ms\n", elapsedTime);

        // Log results for specific test file
        if(testFile == animateFile){
            saveToFile(resultFile, testFile, paths);
        }
    }
    if(testResult)
        printf("* Test failed\n");
    else
        printf("Passed all tests!\n");
    return testResult;

}