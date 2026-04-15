#pragma once
#include <vector>

void traceAndShade(std::vector<std::vector<double>>& vertexBuffer,
                   std::vector<int>& indexBuffer,
                   std::vector<std::vector<double>>& normalBuffer,
                    double lx, double ly, double lz,
    double rx, double ry, double rz, int i);
