
#ifndef MAP_H
#define MAP_H

#include <ostream>
#include <vector>

struct Point2 {
    int x;
    int y;
    
    bool operator== (const Point2& rhs) const
    {
        return x == rhs.x && y == rhs.y;
    }
};

struct MAPFInstance
{
    int rows;
    int cols;
    int numAgents;

    std::vector<std::vector<bool>> map;

    std::vector<Point2> startLocs;
    std::vector<Point2> goalLocs;
};

#endif