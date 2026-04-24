#pragma once
#include <vector>
#include <unordered_map>
#include <cmath>
#include "Vec3.h"

// grid-based spatial hash for O(1) amortized neighbor lookup
class SpatialHash {
    float cellSize;
    std::unordered_map<int64_t, std::vector<int>> table;

    int64_t cellKey(int ix, int iy, int iz) const {
        constexpr int64_t p1 = 73856093LL;
        constexpr int64_t p2 = 19349663LL;
        constexpr int64_t p3 = 83492791LL;
        return (int64_t)ix * p1 ^ (int64_t)iy * p2 ^ (int64_t)iz * p3;
    }

public:
    explicit SpatialHash(float cellSize) : cellSize(cellSize) {
        table.reserve(4096);
    }

    void clear() { table.clear(); }

    void insert(int particleIdx, const Vec3& pos) {
        int ix = std::floor(pos.x / cellSize);
        int iy = std::floor(pos.y / cellSize);
        int iz = std::floor(pos.z / cellSize);
        table[cellKey(ix, iy, iz)].push_back(particleIdx);
    }

    void query(const Vec3& pos, float radius, std::vector<int>& out) const {
        out.clear();
        int ir = std::ceil(radius / cellSize);
        int ix = std::floor(pos.x / cellSize);
        int iy = std::floor(pos.y / cellSize);
        int iz = std::floor(pos.z / cellSize);

        for (int dx = -ir; dx <= ir; ++dx) {
            for (int dy = -ir; dy <= ir; ++dy) {
                for (int dz = -ir; dz <= ir; ++dz) {
                    auto it = table.find(cellKey(ix+dx, iy+dy, iz+dz));
                    if (it != table.end())
                        for (int j : it->second) out.push_back(j);
                }
            }
        }
    }
};
