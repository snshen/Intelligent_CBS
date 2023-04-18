#ifndef CBS_SOLVER_H
#define CBS_SOLVER_H

#include <memory>
#include <tuple>
#include "MAPFInstance.hpp"
#include "SolverUtils.hpp"
#include "AStar.hpp"
#include <optional>

class CBSSolver
{
public:
    CBSSolver();

    std::vector<std::vector<Point2>> solve(MAPFInstance instance);
    std::optional<std::vector<std::vector<Point2>>> safeSolve(MAPFInstance instance, bool saveData = false, bool useModel = false);

private:
    int inline computeCost(const std::vector<std::vector<Point2>> &paths);
    void detectCollisions(const std::vector<std::vector<Point2>> &paths, std::vector<Collision> &collisionList);
    inline bool detectCollision(int agent1, int agent2, const std::vector<Point2> &pathA, const std::vector<Point2> &pathB, Collision &col);
    inline Point2 getLocation(const std::vector<Point2> &path, int t);
    inline std::vector<Constraint> resolveCollision(const Collision &col);

    struct CTNode
    {
        float cost;
        std::vector<std::vector<Point2>> paths;
        std::vector<Collision> collisionList;
        std::vector<Constraint> constraintList;
        int id;
    };

    typedef std::shared_ptr<CTNode> CTNodeSharedPtr;

    class CTNodeComparator
    {
    public:
        bool operator()(const CTNodeSharedPtr &a, const CTNodeSharedPtr &b) const
        {
            if (a->cost == b->cost)
            {
                if (a->collisionList.size() == b->collisionList.size())
                {
                    return a->id > b->id;
                }

                return a->collisionList.size() > b->collisionList.size();
            }

            return a->cost > b->cost;
        }
    };

    class NoSolutionException : public std::exception
    {
        char *what()
        {
            return (char*)"No Solution exists for the given MAPF instance";
        }
    };

    int numNodesGenerated;
};

#endif