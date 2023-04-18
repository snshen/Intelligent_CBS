#ifndef ASTAR_H
#define ASTAR_H

#include <vector>
#include <memory>
#include <unordered_map>
#include "MAPFInstance.hpp"
#include "SolverUtils.hpp"

#define DIAGONAL (1.41421356)

struct Node
{
    Point2 pos;
    float g; // Cost to come
    float h; // Cost to go (heuristic)
    float f; // Total cost (estimate)
    int t; // Timestep

    bool isClosed;
    std::shared_ptr<Node> parent;
};

typedef std::shared_ptr<Node> NodeSharedPtr;
typedef std::unordered_map<int, std::vector<Constraint>> ConstraintsTable;

class AStar
{
public:
    AStar(const MAPFInstance& mapfInstance) : _problem(mapfInstance), nbrConnectedness(5){computeHeuristicMap();};
    bool solve(const int agent_id, const std::vector<Constraint>& constraints, std::vector<Point2> &outputPath);
private:
    void computePath(NodeSharedPtr goal, std::vector<Point2> &outputPath);
    void computeHeuristicMap();
    int computeHash(const Point2& pos, const int t);
    ConstraintsTable buildConstraintsTable(const std::vector<Constraint>& constraints, const int agent_id, int& maxTimestep);
    bool isConstrained(const Point2& currLoc, const Point2& nextLoc, const int nextTime, const ConstraintsTable& constraintsTable);
    
    const int nbrConnectedness;
    const MAPFInstance& _problem; // MAPF problem
    std::vector<std::vector<std::vector<float>>> _heuristicMap; // Look up table for heuristic values for each agent
    const int _dx[9] = {1, -1, 0, 0, 0, 1, 1, -1, -1};
    const int _dy[9] = {0, 0, 1, -1, 0, 1, -1, 1, -1};
    const float _travel_cost[9] = {1, 1, 1, 1, 1, DIAGONAL, DIAGONAL, DIAGONAL, DIAGONAL};
};

class NodeComparator
{
public:
    bool operator()(const NodeSharedPtr &a, const NodeSharedPtr &b) const
    {
        return a->f > b->f;
    }
};

#endif