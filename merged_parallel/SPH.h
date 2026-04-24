#pragma once
#include <vector>
#include <cuda_runtime.h>
#include "Vec3.h"

// GPU spatial hash - defined here so March.cu can reconstruct it from SPH's device pointers
struct GpuHash {
    static constexpr int HASH_SIZE = 65536;
    int* d_head;
    int* d_next;
    double cellSize;

    __host__ __device__ int cellKey(int ix, int iy, int iz) const {
        constexpr int64_t p1 = 73856093LL;
        constexpr int64_t p2 = 19349663LL;
        constexpr int64_t p3 = 83492791LL;
        int64_t h = (int64_t)ix * p1 ^ (int64_t)iy * p2 ^ (int64_t)iz * p3;
        return (int)(h & (int64_t)(HASH_SIZE - 1));
    }

    __device__ void insert(int idx, const Vec3& pos) {
        int ix = (int)floor(pos.x / cellSize);
        int iy = (int)floor(pos.y / cellSize);
        int iz = (int)floor(pos.z / cellSize);
        int key = cellKey(ix, iy, iz);
        d_next[idx] = atomicExch(&d_head[key], idx);
    }
};

struct Particle {
    Vec3 pos;
    Vec3 vel;
    Vec3 acc;
    double density;
    double pressure;
};

// 'Weakly Compressible' SPH simulation.
//
// The implementation and kernel functions used are based on:
// https://cg.informatik.uni-freiburg.de/publications/2014_EG_SPH_STAR.pdf
//
// step() launches 4 CUDA kernels (build grid → density/pressure → forces → integrate).
// particles (host vector) is kept in sync after each step() for use by March.cpp.
class SPH {
public:
    static constexpr double H = 0.13;
    static constexpr double H2 = H * H;
    static constexpr double RHO0 = 1000.0;
    static constexpr double SPACING = H * 0.75;
    static constexpr double MASS = RHO0 * SPACING * SPACING * SPACING;
    static constexpr double GAMMA = 7.0;
    static constexpr double C_SOUND = 30.0;
    static constexpr double B_PRESS = RHO0 * C_SOUND * C_SOUND / GAMMA;
    static constexpr double MU = 0.13;
    static constexpr double GRAVITY = 9.81;
    static constexpr double DT = 0.00025;
    static constexpr double BMIN = -3.0;
    static constexpr double BMAX = 3.0;
    static constexpr double RESTITUTION = 0.4;

    // kernel constants. can't use std library lol
    static constexpr double K_PI = 3.14159265358979323846;
    static constexpr double K_POLY6 = 315.0 / (64.0 * K_PI * H*H*H*H*H*H*H*H*H);
    static constexpr double K_SPIKY = -45.0 / (K_PI * H*H*H*H*H*H);
    static constexpr double K_VISC = 45.0 / (K_PI * H*H*H*H*H*H);

    std::vector<Particle> particles; // host copy of particles

    SPH();
    ~SPH();
    void reset();
    void step();
    void syncToHost();

    // for March.cu
    Particle* getParticles() const { return d_particles; }
    int* getHashHead()       const { return d_hashHead; }
    int* getHashNext()       const { return d_hashNext; }

private:
    Particle* d_particles;
    int* d_hashHead;
    int* d_hashNext;
    int maxParticles;
};
