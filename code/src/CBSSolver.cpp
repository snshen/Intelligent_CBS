#include "CBSSolver.hpp"
#include <queue>
#include <chrono>
#include <bits/stdc++.h>

CBSSolver::CBSSolver()
: numNodesGenerated(0) 
{
}

std::vector<std::vector<Point2>> CBSSolver::solve(MAPFInstance instance)
{
    // Initialize low level solver
    AStar lowLevelSolver(instance);

    // Create priority queue
    std::priority_queue <CTNodeSharedPtr, 
                         std::vector<CTNodeSharedPtr>, 
                         CTNodeComparator 
                        > pq;

    CTNodeSharedPtr root = std::make_shared<CTNode>();
    root->paths.resize(instance.numAgents);
    root->id = 0;
    numNodesGenerated++;

    // Create paths for all agents
    for (int i = 0; i < instance.startLocs.size(); i++)
    {
        bool found = lowLevelSolver.solve(i, root->constraintList, root->paths[i]);
        
        if (!found)
            throw NoSolutionException();
    }

    root->cost = 0;
    detectCollisions(root->paths, root->collisionList);

    pq.push(root);

    while (!pq.empty())
    {
        CTNodeSharedPtr cur = pq.top();
        pq.pop();

        // If no collisions in the node then return solution
        if (cur->collisionList.size() == 0)
        {
            return cur->paths;
        }

        // Get first collision and create two nodes (each containing a new plan for the two agents in the collision)
        for (Constraint &c : resolveCollision(cur->collisionList[0]))
        {
            // Add new constraint
            CTNodeSharedPtr child = std::make_shared<CTNode>();
            child->constraintList = cur->constraintList;
            child->constraintList.push_back(c);
            child->paths = cur->paths;

            // Replan only for the agent that has the new constraint
            child->paths[c.agentNum].clear();
            bool success = lowLevelSolver.solve(c.agentNum, child->constraintList, child->paths[c.agentNum]);

            if (success)
            {
                // Update cost and find collisions
                child->cost = computeCost(child->paths);
                detectCollisions(child->paths, child->collisionList);
                pq.push(child);
            }
        }
    }

    throw NoSolutionException();
}

CBSSolver::CTNodeSharedPtr CBSSolver::safeSolve(MAPFInstance instance, int& counter, bool& unsolvable)
{
    // Initialize low level solver  
    AStar lowLevelSolver(instance);

    // Create priority queue
    std::priority_queue <CTNodeSharedPtr, 
                         std::vector<CTNodeSharedPtr>, 
                         CTNodeComparator 
                        > pq;

    CTNodeSharedPtr root = std::make_shared<CTNode>();
    root->paths.resize(instance.numAgents);
    root->id = 0;
    numNodesGenerated++;

    // Create paths for all agents
    for (int i = 0; i < instance.startLocs.size(); i++)
    {
        bool found = lowLevelSolver.solve(i, root->constraintList, root->paths[i]);
        
        if (!found)
            return {};
    }

    root->cost = 0;
    detectCollisions(root->paths, root->collisionList);

    pq.push(root);
    counter++;

    std::chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();;
    std::chrono::duration<double, std::milli> elapsedTime;
    while (!pq.empty())
    {   
        CTNodeSharedPtr cur = pq.top();
        pq.pop();

        elapsedTime = std::chrono::high_resolution_clock::now() - start;
        if (elapsedTime.count()>300000){
            printf("Instance timeout, took more than 5mins to solve \n");
            unsolvable = true;
            return root;
        }
        
        // If no collisions in the node then return solution
        if (cur->collisionList.size() == 0)
        {
            return cur;
        }
        
        // Get first collision and create two nodes (each containing a new plan for the two agents in the collision)
        for (Constraint &c : resolveCollision(cur->collisionList[0]))
        {
            // Add new constraint
            CTNodeSharedPtr child = std::make_shared<CTNode>();
            child->constraintList = cur->constraintList;
            child->constraintList.push_back(c);
            child->paths = cur->paths;

            // Replan only for the agent that has the new constraint
            child->paths[c.agentNum].clear();
            bool success = lowLevelSolver.solve(c.agentNum, child->constraintList, child->paths[c.agentNum]);
            // loss
            if (success)
            {
                // Update cost and find collisions
                child->cost = computeCost(child->paths);
                detectCollisions(child->paths, child->collisionList);

                // Add to search queue
                pq.push(child);
                counter++;

            }
        }
    }
    unsolvable = true;
    return root;
}


CBSSolver::CTNodeSharedPtr CBSSolver::trainSolve(MAPFInstance instance,
                                                bool& timeout, 
                                                std::vector<torch::Tensor> gtPaths, 
                                                ConfNet* modelPtr,
                                                torch::optim::Adam& optimizer,
                                                torch::Tensor inputMaps,
                                                trainMetrics& metrics,
                                                torch::Device device)
{ 
    // Initialize low level solver  
    AStar lowLevelSolver(instance);

    // Create priority queue
    std::priority_queue <CTNodeSharedPtr, 
                         std::vector<CTNodeSharedPtr>, 
                         CTNodeComparator 
                        > pq;

    CTNodeSharedPtr root = std::make_shared<CTNode>();
    root->paths.resize(instance.numAgents);
    root->id = 0;
    numNodesGenerated++;

    // Create paths for all agents
    for (int i = 0; i < instance.startLocs.size(); i++)
    {
        bool found = lowLevelSolver.solve(i, root->constraintList, root->paths[i]);
        
        if (!found)
            return {};
    }

    root->cost = 0;
    detectCollisions(root->paths, root->collisionList);

    pq.push(root);
    metrics.counter++;

    std::chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();;
    std::chrono::duration<double, std::milli> elapsedTime;
    while (!pq.empty())
    {   
        CTNodeSharedPtr cur = pq.top();
        pq.pop();

        elapsedTime = std::chrono::high_resolution_clock::now() - start;
        if (elapsedTime.count()>300000){
            printf("Instance timeout, took more than 5mins to solve \n");
            timeout = true;
            return root;
        }
        
        // If no collisions in the node then return solution
        if (cur->collisionList.size() == 0)
        {
            return cur;
        }
        
        // Get first collision and create two nodes (each containing a new plan for the two agents in the collision)
        std::vector<Collision> collisionList = cur->collisionList;
        //maps int of a flattened array of the Instance map to the index of the collision list
        std::unordered_map<int, int> collisionInds;
        
        for(int i=0; i<collisionList.size(); i++){
            Collision currCollision = collisionList[0];
            collisionInds[currCollision.location.first.x*instance.cols+currCollision.location.first.y] = i;
            if (currCollision.location.first == currCollision.location.second){
                inputMaps[0][0][currCollision.location.first.x][currCollision.location.first.y] = 1;
            }
            else{
                inputMaps[0][0][currCollision.location.first.x][currCollision.location.first.y] = 2;
            }
        }
        
        torch::Tensor output = (*modelPtr)->forward(inputMaps);
        // std::cout << "dims " << output.sizes() << std::endl;
        output = output[0][0];
        
        double running_loss = 0.0;
        int bestInd;
        float bestVal = 0;
        
        for(auto ind=collisionInds.begin(); ind!=collisionInds.end(); ind++){
            int x=ind->first%instance.cols;
            int y=ind->first/instance.cols;
            float currVal = output[y][x].item<float>();
            if(currVal>=bestVal){
                bestVal = currVal;
                bestInd = ind->second;
            }
        }

        Collision bestCollision = collisionList[bestInd];
        
        bool backprop = true;
        bool second = false;
        torch::Tensor outputPred = torch::zeros({2, output.sizes()[0], output.sizes()[1]});
        torch::Tensor outputLabel= torch::zeros({2, output.sizes()[0], output.sizes()[1]});
        for (Constraint &c : resolveCollision(bestCollision))
        {
            // Add new constraint
            CTNodeSharedPtr child = std::make_shared<CTNode>();
            child->constraintList = cur->constraintList;
            child->constraintList.push_back(c);
            child->paths = cur->paths;

            // Replan only for the agent that has the new constraint
            child->paths[c.agentNum].clear();
            bool success = lowLevelSolver.solve(c.agentNum, child->constraintList, child->paths[c.agentNum]);
            // loss
            if (success)
            {   
                torch::Tensor label = gtPaths[c.agentNum].to(device);
                std::vector<Point2> path = child->paths[c.agentNum];

                if(!second){
                    outputLabel[0] = label;
                    for(Point2 loc:path){
                        outputPred[0][loc.x][loc.y] = 1;
                    }
                } 
                else{
                    outputLabel[1] = label;
                    for(Point2 loc:path){
                        outputPred[1][loc.x][loc.y] = 1;
                    }
                }
                second = true;

                // Update cost and find collisions
                child->cost = computeCost(child->paths);
                detectCollisions(child->paths, child->collisionList);

                // Add to search queue
                pq.push(child);
                metrics.counter++;

            }else backprop = false; 
        }
        if(backprop){
            // outputPred.to(device);
            // outputLabel.to(device);
            torch::Tensor loss = torch::nn::functional::mse_loss(outputPred, outputLabel).requires_grad_(true);
            metrics.runningLoss += loss.item<double>();
            metrics.numLoss++;
            loss.backward();
        }
    }
    timeout = true;
    return root;
}

int inline CBSSolver::computeCost(const std::vector<std::vector<Point2>> &paths)
{
    int result = 0;

    for (int i = 0; i < paths.size(); i++)
    {
        result += paths[i].size();
    }

    return result;
}


void CBSSolver::detectCollisions(const std::vector<std::vector<Point2>> &paths, std::vector<Collision> &collisionList)
{
    Collision col;
    collisionList.clear();

    // N^2 alg - seems like there should be a better way to detect collisions
    for (int i = 0; i < paths.size() - 1; i++)
    {
        for (int j = i + 1; j < paths.size(); j++)
        {
            if (detectCollision(i, j, paths[i], paths[j], col))
                collisionList.push_back(col);
        }
    }

    // TODO: Call model here to sort and replace collision list
}

inline bool CBSSolver::detectCollision(int agent1, int agent2, const std::vector<Point2> &pathA, const std::vector<Point2> &pathB, Collision &col)
{
    int maxTime = std::max(pathA.size(), pathB.size());

    for (int t = 0; t < maxTime; t++)
    {
        if (getLocation(pathA, t) == getLocation(pathB, t))
        {
            col = createVertexCollision(agent1, agent2, t, getLocation(pathA, t));
            return true;
        }

        if (getLocation(pathA, t) == getLocation(pathB, t + 1) && getLocation(pathA, t + 1) == getLocation(pathB, t))
        {
            col = createEdgeCollision(agent1, agent2, t + 1, getLocation(pathA, t), getLocation(pathA, t + 1));
            return true;
        }
    }

    return false;
}

inline Point2 CBSSolver::getLocation(const std::vector<Point2> &path, int t)
{
    if (t >= path.size())
        return path[path.size() - 1];
    else
        return path[t];
}

inline std::vector<Constraint> CBSSolver::resolveCollision(const Collision& col)
{
    if (col.isVertexCollision)
    {
        return std::vector<Constraint> {Constraint{col.agent1, col.t, true, col.location}, Constraint{col.agent2, col.t, true, col.location}};
    }
    else
    {
        return std::vector<Constraint> {Constraint{col.agent1, col.t, false, col.location}, Constraint{col.agent2, col.t, false, col.location}};
    }
}