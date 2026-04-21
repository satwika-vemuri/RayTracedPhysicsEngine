#include "SPH.h"
#include <cmath>
#include <algorithm>

// defining the kernel constants
const double SPH::K_POLY6 = 315.0 / (64.0 * (double)M_PI * std::pow(H, 9.0));
const double SPH::K_SPIKY = -45.0 / ((double)M_PI * std::pow(H, 6.0));
const double SPH::K_VISC  =  45.0 / ((double)M_PI * std::pow(H, 6.0));

// *** Kernel Functions ***
// from the STAR paper

// good for density accumulation
double SPH::W_poly6(double r2) const {
    if (r2 >= H2) return 0.0;
    double d = H2 - r2;
    return K_POLY6 * d * d * d;
}

// good for a repulsive contribution in the pressure force sum
Vec3 SPH::gradW_spiky(const Vec3& rij, double r) const {
    if (r >= H || r < 1e-6f) return {};
    double d = H - r;
    double coef = K_SPIKY * d * d / r;
    return rij * coef;
}

double SPH::lapW_visc(double r) const {
    if (r >= H) return 0.0;
    return K_VISC * (H - r);
}

// *** Public Interface ***

SPH::SPH() : grid(H) {
    reset();
}

void SPH::reset() {
    particles.clear();
    for (double x = BMIN + SPACING; x < 0.0; x += SPACING) { // left half of the box
        for (double y = BMIN + SPACING; y < BMAX - SPACING; y += SPACING) { // full height
            for (double z = BMIN + SPACING; z < BMAX - SPACING; z += SPACING) { // full depth
                Particle p;
                p.pos = {x, y, z};
                p.vel = {0.0, 0.0, 0.0};
                p.acc = {0.0, 0.0, 0.0};
                p.density  = RHO0;
                p.pressure = 0.0;
                particles.push_back(p);
            }
        }
    }
}

void SPH::step() {
    buildGrid();
    computeDensityPressure();
    computeForces();
    integrate();
}

// *** Simulation Passes ***

// initalized the spacial hash grid with particles (and their positions)
void SPH::buildGrid() {
    grid.clear();
    for (int i = 0; i < particles.size(); ++i) grid.insert(i, particles[i].pos);
}

// computes both pressure and density for all particles using the equations
// mentioned in the STAR paper
void SPH::computeDensityPressure() {
    for (auto& pi : particles) {
        grid.query(pi.pos, H, nbrs);

        // density first
        pi.density = 0.0;
        for (int j : nbrs) {
            Vec3  rij = pi.pos - particles[j].pos;
            double r2  = rij.length2();
            pi.density += MASS * W_poly6(r2);
        }
        // to avoid division by zero or negative pressure
        pi.density = std::max(pi.density, RHO0 * 0.1);

        // pressure
        // tait equation from STAR paper
        double ratio = pi.density / RHO0;
        double r2t = ratio * ratio;
        double r4t = r2t * r2t;
        double r7 = r4t * r2t * ratio;
        pi.pressure = std::max(0.0, B_PRESS * (r7 - 1.0));
    }
}

// use previously calculated pressure and density along with other items
// here to calculate forces on particles
void SPH::computeForces() {
    const Vec3 gravity{0.0, -GRAVITY, 0.0};

    for (int i = 0; i < particles.size(); ++i) {
        auto& pi = particles[i];
        Vec3 a_press{}, a_visc{};

        grid.query(pi.pos, H, nbrs);
        for (int j : nbrs) {
            if (j == i) continue;
            auto& pj = particles[j];

            Vec3  rij = pi.pos - pj.pos;
            double r   = rij.length();
            if (r >= H || r < 1e-6f) continue;

            // acceleration from pressure
            double pterm = pi.pressure / (pi.density * pi.density)
                + pj.pressure / (pj.density * pj.density);
            a_press += gradW_spiky(rij, r) * (-MASS * pterm);

            // acceleration from viscosity (damps velocity)
            double wlap = lapW_visc(r);
            a_visc += (pj.vel - pi.vel) * (MU * MASS / pj.density * wlap);
        }

        // add acceleration onto acceleration of current particle
        pi.acc = a_press + a_visc + gravity;
    }
}

// apply acceleration onto velocity and velocity onto position
void SPH::integrate() {
    for (auto& p : particles) {
        p.vel += p.acc * DT;
        p.pos += p.vel * DT;
        enforceBoundary(p);
    }
}

void SPH::enforceBoundary(Particle& p) {
    auto reflect = [&](double& pos, double& vel, double mn, double mx) {
        if (pos < mn) {
            pos = mn + 2e-4f; if (vel < 0.0) vel *= -RESTITUTION;
        }
        if (pos > mx) {
            pos = mx - 2e-4f; if (vel > 0.0) vel *= -RESTITUTION;
        }
    };
    reflect(p.pos.x, p.vel.x, BMIN, BMAX);
    reflect(p.pos.y, p.vel.y, BMIN, BMAX);
    reflect(p.pos.z, p.vel.z, BMIN, BMAX);
}
