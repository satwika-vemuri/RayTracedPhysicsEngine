#pragma once
#include <vector>
#include "Vec3.h"
#include "SpatialHash.h"

struct Particle {
    Vec3 pos;
    Vec3 vel;
    Vec3 acc;
    float density;
    float pressure;
};

// 'Weakly Compressible' SPH simulation.
// 
// The implimentation and kernel functions used are based
// on the following paper:
// https://cg.informatik.uni-freiburg.de/publications/2014_EG_SPH_STAR.pdf
//
// I'll refer to this paper as 'the STAR paper' throughout these files
class SPH {
public:
    static constexpr float H = 0.13f; // smoothing length used in smoothing kernel
    static constexpr float H2 = H * H;

    static constexpr float RHO0 = 1000.0f; // rest density
    // particle spacing. note that increasing/decreasing
    // this value increases and decreases the number of particles
    // spawned.
    static constexpr float SPACING = H * 0.75f;
    static constexpr float MASS = RHO0 * SPACING * SPACING * SPACING; // explicit mass

    // some parameters for the so called 'equation of state (EOS)' calculations
    static constexpr float GAMMA = 7.0f;
    static constexpr float C_SOUND = 30.0f;
    static constexpr float B_PRESS = RHO0 * C_SOUND * C_SOUND / GAMMA;

    static constexpr float MU = 0.13f; // dynamic viscosity (may be increased or decreased as needed)
    static constexpr float GRAVITY = 9.81f;

    // for time integration
    static constexpr float DT = 0.00025f;

    // some paramters for the axi-alligned bounding box
    static constexpr float BMIN = -3.0f;
    static constexpr float BMAX =  3.0f;
    static constexpr float RESTITUTION = 0.4f; // wall collision damping

    // precomputed kernel constants that are defined in SPH.cpp
    static const float K_POLY6;
    static const float K_SPIKY;
    static const float K_VISC;

    std::vector<Particle> particles;

    SPH();
    void reset();
    // advance by DT (build grid --> density --> forces --> integrate)
    void step();

private:
    SpatialHash grid;
    std::vector<int> nbrs; // buffer for neighbor queries in grid

    // the kernel functions
    // read the STAR paper for more details on these
    float W_poly6(float r2) const;
    Vec3  gradW_spiky(const Vec3& rij, float r) const;
    float lapW_visc(float r) const;

    // simulation passes called in step()
    void buildGrid();
    void computeDensityPressure();
    void computeForces();
    void integrate();
    void enforceBoundary(Particle& p);
};
