#pragma once
#include <vector>
#include "vec3.h"
#include "SpatialHash.h"

struct Particle {
    Vec3 pos;
    Vec3 vel;
    Vec3 acc;
    double density;
    double pressure;
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
    static constexpr double H = 0.15; // smoothing length used in smoothing kernel
    static constexpr double H2 = H * H;

    static constexpr double RHO0 = 1000.0; // rest density
    // particle spacing. note that increasing/decreasing
    // this value increases and decreases the number of particles
    // spawned.
    static constexpr double SPACING = H * 0.85;
    static constexpr double MASS = RHO0 * SPACING * SPACING * SPACING; // explicit mass

    // some parameters for the so called 'equation of state (EOS)' calculations
    static constexpr double GAMMA = 7.;
    static constexpr double C_SOUND = 30.0;
    static constexpr double B_PRESS = RHO0 * C_SOUND * C_SOUND / GAMMA;

    static constexpr double MU = 0.08; // dynamic viscosity (may be increased or decreased as needed)
    static constexpr double GRAVITY = 9.81;

    // for time integration
    static constexpr double DT = 0.001;

    // some paramters for the axi-alligned bounding box
    static constexpr double BMIN = -1.5;
    static constexpr double BMAX =  1.5;
    static constexpr double RESTITUTION = 0.4; // wall collision damping

    // precomputed kernel constants that are defined in SPH.cpp
    static const double K_POLY6;
    static const double K_SPIKY;
    static const double K_VISC;

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
    double W_poly6(double r2) const;
    Vec3  gradW_spiky(const Vec3& rij, double r) const;
    double lapW_visc(double r) const;

    // simulation passes called in step()
    void buildGrid();
    void computeDensityPressure();
    void computeForces();
    void integrate();
    void enforceBoundary(Particle& p);
};
