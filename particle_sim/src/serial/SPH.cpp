#include "SPH.h"
#include <cmath>
#include <algorithm>

// defining the kernel constants
const float SPH::K_POLY6 = 315.0f / (64.0f * (float)M_PI * std::pow(H, 9.0f));
const float SPH::K_SPIKY = -45.0f / ((float)M_PI * std::pow(H, 6.0f));
const float SPH::K_VISC  =  45.0f / ((float)M_PI * std::pow(H, 6.0f));

// *** Kernel Functions ***
// from the STAR paper

// good for density accumulation
float SPH::W_poly6(float r2) const {
    if (r2 >= H2) return 0.0f;
    float d = H2 - r2;
    return K_POLY6 * d * d * d;
}

// good for a repulsive contribution in the pressure force sum
Vec3 SPH::gradW_spiky(const Vec3& rij, float r) const {
    if (r >= H || r < 1e-6f) return {};
    float d = H - r;
    float coef = K_SPIKY * d * d / r;
    return rij * coef;
}

float SPH::lapW_visc(float r) const {
    if (r >= H) return 0.0f;
    return K_VISC * (H - r);
}

// *** Public Interface ***

SPH::SPH() : grid(H) {
    reset();
}

void SPH::reset() {
    particles.clear();
    for (float x = (0.5f * BMIN); x < (0.5f * BMAX); x += SPACING) { // left half of the box
        for (float y = BMIN + SPACING; y < (0.5f * BMAX); y += SPACING) { // full height
            for (float z = BMIN + SPACING; z < BMAX - SPACING; z += SPACING) { // full depth
                Particle p;
                p.pos = {x, y, z};
                p.vel = {0.0f, 0.0f, 0.0f};
                p.acc = {0.0f, 0.0f, 0.0f};
                p.density  = RHO0;
                p.pressure = 0.0f;
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
    for (int i = 0; i < (int)particles.size(); ++i) grid.insert(i, particles[i].pos);
}

// computes both pressure and density for all particles using the equations
// mentioned in the STAR paper
void SPH::computeDensityPressure() {
    for (auto& pi : particles) {
        grid.query(pi.pos, H, nbrs);

        // density first
        pi.density = 0.0f;
        for (int j : nbrs) {
            Vec3  rij = pi.pos - particles[j].pos;
            float r2  = rij.length2();
            pi.density += MASS * W_poly6(r2);
        }
        // to avoid division by zero or negative pressure
        pi.density = std::max(pi.density, RHO0 * 0.1f);

        // pressure
        // tait equation from STAR paper
        float ratio = pi.density / RHO0;
        float r2t = ratio * ratio;
        float r4t = r2t * r2t;
        float r7 = r4t * r2t * ratio;
        pi.pressure = std::max(0.0f, B_PRESS * (r7 - 1.0f));
    }
}

// use previously calculated pressure and density along with other items
// here to calculate forces on particles
void SPH::computeForces() {
    const Vec3 gravity{0.0f, -GRAVITY, 0.0f};

    for (int i = 0; i < (int)particles.size(); ++i) {
        auto& pi = particles[i];
        Vec3 a_press{}, a_visc{};

        grid.query(pi.pos, H, nbrs);
        for (int j : nbrs) {
            if (j == i) continue;
            auto& pj = particles[j];

            Vec3  rij = pi.pos - pj.pos;
            float r   = rij.length();
            if (r >= H || r < 1e-6f) continue;

            // acceleration from pressure
            float pterm = pi.pressure / (pi.density * pi.density)
                + pj.pressure / (pj.density * pj.density);
            a_press += gradW_spiky(rij, r) * (-MASS * pterm);

            // acceleration from viscosity (damps velocity)
            float wlap = lapW_visc(r);
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
    auto reflect = [&](float& pos, float& vel, float mn, float mx) {
        if (pos < mn) {
            pos = mn + 2e-4f; if (vel < 0.0f) vel *= -RESTITUTION;
        }
        if (pos > mx) {
            pos = mx - 2e-4f; if (vel > 0.0f) vel *= -RESTITUTION;
        }
    };
    reflect(p.pos.x, p.vel.x, BMIN, BMAX);
    reflect(p.pos.y, p.vel.y, BMIN, BMAX);
    reflect(p.pos.z, p.vel.z, BMIN, BMAX);
}
