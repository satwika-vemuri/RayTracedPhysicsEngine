#pragma once
#include <vector>
#include "Vec3.h"

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

private:
    Particle* d_particles;
    int* d_hashHead;
    int* d_hashNext;
    int maxParticles;
};
