#include "MAPFLoader.hpp"
#include <iostream>
#include <fstream>
#include <bits/stdc++.h>
#include <algorithm>

MAPFInstance MAPFLoader::loadInstanceFromFile(const std::string &fileName)
{
    std::fstream txtFile;
    txtFile.open(fileName, std::ios::in);

    MAPFInstance result;

    if (txtFile.is_open())
    {
        std::stringstream ss;
        std::string contents;

        ss << txtFile.rdbuf();
        contents = ss.str();

        parseText(contents, result);
    }
    else
    {
        printf("ERROR: File was not opened\n");
    }

    txtFile.close();

    return result;
}

void MAPFLoader::parseText(std::string text, MAPFInstance &result)
{
    int end_first_line = text.find("\n");
    parseRowsAndCols(text.substr(0, end_first_line), result);
    text.erase(text.begin(), text.begin() + end_first_line + 1);

    std::string digits = "0123456789";
    int map_len = text.find_first_of(digits);
    parseMap(text.substr(0, map_len), result);
    text.erase(text.begin(), text.begin() + map_len);

    parseAgentDetails(text, result);
}

void MAPFLoader::parseRowsAndCols(std::string line, MAPFInstance &result)
{
    std::string value;
    std::stringstream ss(line);
    bool first = true;

    while (ss >> value)
    {
        if (first)
        {
            result.rows = std::stoi(value);
            first = false;
        }
        else
        {
            result.cols = std::stoi(value);
            break;
        }
    }
}

void MAPFLoader::parseMap(std::string mapAsTxt, MAPFInstance &result)
{
    // Remove any whitespace from the map
    mapAsTxt.erase(std::remove(mapAsTxt.begin(), mapAsTxt.end(), ' '), mapAsTxt.end());
    mapAsTxt.erase(std::remove(mapAsTxt.begin(), mapAsTxt.end(), '\r'), mapAsTxt.end());

    result.map.resize(result.rows);

    int curPos = 0;
    for (int r = 0; r < result.rows; r++)
    {
        result.map[r].resize(result.cols);

        for (int c = 0; c < result.cols; c++)
        {
            if (mapAsTxt[curPos] == '@')
                result.map[r][c] = true;
            else if (mapAsTxt[curPos] == '.')
                result.map[r][c] = false;

            curPos++;
        }
        curPos++;
    }
}

void MAPFLoader::parseAgentDetails(std::string agentInfo, MAPFInstance &result)
{
    int endFirstLine = agentInfo.find("\n");
    result.numAgents = std::stoi(agentInfo.substr(0, endFirstLine));

    agentInfo.erase(agentInfo.begin(), agentInfo.begin() + endFirstLine + 1);

    result.startLocs.reserve(result.numAgents);
    result.goalLocs.reserve(result.numAgents);

    int values[4];
    for (int i = 0; i < result.numAgents; i++)
    {
        // std::cout << "Agent Info = " << agentInfo << std::endl;
        // std::cout << "-------" << std::endl;
        int endLine = agentInfo.find("\n");
        std::string line = agentInfo.substr(0, endLine);
        for (int j = 0; j < 3; j++)
        {
            int endNextInt = line.find(" ");
            values[j] = std::stoi(line.substr(0, endNextInt));
            line.erase(line.begin(), line.begin() + endNextInt + 1);
        }
        values[3] = std::stoi(line.substr(0, line.length()));

        result.startLocs.push_back(Point2{values[0], values[1]});
        result.goalLocs.push_back(Point2{values[2], values[3]});

        agentInfo.erase(agentInfo.begin(), agentInfo.begin() + endLine + 1);
    }
}
