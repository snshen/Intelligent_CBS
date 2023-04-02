#include "CBSSolver.hpp"
#include <queue>

CBSSolver::CBSSolver()
: numNodesGenerated(0) 
{
}

std::vector<std::vector<Point2>> CBSSolver::solve(MAPFInstance instance)
{
    // Create priority queue
    std::priority_queue <CTNodeSharedPtr, 
                         std::vector<CTNodeSharedPtr>, 
                         CTNodeComparator 
                        > pq;

    printf("test\n");

    CTNodeSharedPtr root = std::make_shared<CTNode>();
    printf("A %d\n", instance.numAgents);
    root->paths.reserve(instance.numAgents);
    root->id = 0;
    numNodesGenerated++;
    

    printf("B\n");
    // Create paths for all agents
    for (int i = 0; i < instance.startLocs.size(); i++)
    {
        // root->paths.push_back(lowLevelSolver.solve(instance.startLocs[i], instance.goalLocs[i], ));
    }

    printf("A\n");
    root->cost = computeCost(root->paths);
    detectCollisions(root->paths, root->collisionList);

    pq.push(root);

    while (!pq.empty())
    {
        CTNodeSharedPtr cur = pq.top();
        pq.pop();

        // If no collisions in the node then return solution
        if (cur->collisionList.size() == 0)
            return cur->paths;

        // Get first collision and create two nodes (each containing a new plan for the two agents in the collision)
        for (Constraint &c : resolveCollision(cur->collisionList[0]))
        {
            CTNodeSharedPtr child = std::make_shared<CTNode>();
            child->constraintList = cur->constraintList;
            child->constraintList.push_back(c);

            child->paths = cur->paths;

            // std::vector<Point2> newPath = lowLevelSolver.solve(instance.startLocs[c.agentNum], instance.goalLocs[c.agentNum], );
            std::vector<Point2> newPath;

            if (newPath.size() > 0)
            {
                child->paths[c.agentNum] = std::move(newPath);
                child->cost = computeCost(child->paths);
                detectCollisions(child->paths, child->collisionList);

                pq.push(child);
            }
        }
    }

    throw NoSolutionException();
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
}

inline bool CBSSolver::detectCollision(int agent1, int agent2, const std::vector<Point2> &pathA, const std::vector<Point2> &pathB, Collision &col)
{
    int maxTime = std::max(pathA.size(), pathB.size());

    for (int t = 0; t < maxTime; t++)
    {
        if (getLocation(pathA, t) == getLocation(pathB, t))
            col = createVertexCollision(agent1, agent2, t, pathA[t]);
            return true;

        if (getLocation(pathA, t) == getLocation(pathB, t + 1) && getLocation(pathA, t + 1) == getLocation(pathB, t))
            col = createEdgeCollision(agent1, agent2, t + 1, getLocation(pathA, t), getLocation(pathA, t + 1));
            return true;
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
        return std::vector<Constraint> {createVertexConstraint(col.agent1, col.location.first, col.t), createVertexConstraint(col.agent2, col.location.first, col.t)};
    }
    else
    {
        return std::vector<Constraint> {createEdgeConstraint(col.agent1, col.location, col.t), createEdgeConstraint(col.agent2, col.location, col.t)};
    }
}