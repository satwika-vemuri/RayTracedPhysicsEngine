#include "SPH.h"
#include <cuda_runtime.h>
#include <cstdio>

/****************************
*****************************
** GPU Spatial Hash *********
*****************************
****************************/

// atomic linked-list hash that's safe for concurrent insertion via atomicExch
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

/****************************
*****************************
** Kernel Helper Functions **
*****************************
****************************/

__device__ static double d_w_poly6(double r2) {
    if (r2 >= SPH::H2) return 0.0;
    double d = SPH::H2 - r2;
    return SPH::K_POLY6 * d * d * d;
}

__device__ static Vec3 d_grad_w_spiky(const Vec3& rij, double r) {
    if (r >= SPH::H || r < 1e-6) return {};
    double d = SPH::H - r;
    return rij * (SPH::K_SPIKY * d * d / r);
}

__device__ static double d_lap_w_visc(double r) {
    if (r >= SPH::H) return 0.0;
    return SPH::K_VISC * (SPH::H - r);
}

/****************************
*****************************
** my sexy cuda kernels!! ***
*****************************
****************************/

// one thread per particle: insert into the linked-list hash
__global__ void buildGrid_kernel(Particle* particles, int n, GpuHash hash) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    hash.insert(i, particles[i].pos);
}

// one thread per particle: compute density and pressure
__global__ void computeDensityPressure_kernel(Particle* particles, int n, GpuHash hash) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const Vec3 pos_i = particles[i].pos;
    double density = 0.0;

    int ix = (int)floor(pos_i.x / hash.cellSize);
    int iy = (int)floor(pos_i.y / hash.cellSize);
    int iz = (int)floor(pos_i.z / hash.cellSize);

    // check 3x3x3 neighborhood
    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dz = -1; dz <= 1; ++dz) {
                int key = hash.cellKey(ix+dx, iy+dy, iz+dz);
                int j = hash.d_head[key];
                while (j != -1) {
                    Vec3 rij = pos_i - particles[j].pos;
                    density += SPH::MASS * d_w_poly6(rij.length2());
                    j = hash.d_next[j];
                }
            }
        }
    }

    density = fmax(density, SPH::RHO0 * 0.1);
    particles[i].density = density;

    // tait equation of state from STAR paper
    double ratio = density / SPH::RHO0;
    double r7 = ratio * ratio * ratio * ratio * ratio * ratio * ratio;
    particles[i].pressure = fmax(0.0, SPH::B_PRESS * (r7 - 1.0));
}

// one thread per particle: compute pressure and viscosity accelerations
__global__ void computeForces_kernel(Particle* particles, int n, GpuHash hash) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const Vec3 gravity{0.0, -SPH::GRAVITY, 0.0};
    const Vec3 pos_i = particles[i].pos;
    const Vec3 vel_i = particles[i].vel;
    const double rho_i = particles[i].density;
    const double prs_i = particles[i].pressure;

    Vec3 a_press{}, a_visc{};

    int ix = (int)floor(pos_i.x / hash.cellSize);
    int iy = (int)floor(pos_i.y / hash.cellSize);
    int iz = (int)floor(pos_i.z / hash.cellSize);

    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dz = -1; dz <= 1; ++dz) {
                int key = hash.cellKey(ix+dx, iy+dy, iz+dz);
                int j = hash.d_head[key];
                while (j != -1) {
                    if (j != i) {
                        Vec3 rij = pos_i - particles[j].pos;
                        double r2 = rij.length2();
                        if (r2 < SPH::H2 && r2 > 1e-12) {
                            double r = sqrt(r2);
                            double pterm = prs_i / (rho_i * rho_i)
                            + particles[j].pressure / (particles[j].density * particles[j].density);
                            a_press += d_grad_w_spiky(rij, r) * (-SPH::MASS * pterm);
                            a_visc  += (particles[j].vel - vel_i)
                            * (SPH::MU * SPH::MASS / particles[j].density * d_lap_w_visc(r));
                        }
                    }
                    j = hash.d_next[j];
                }
            }
        }
    }

    particles[i].acc = a_press + a_visc + gravity;
}

// one thread per particle: symplectic Euler integration + boundary enforcement
__global__ void integrate_kernel(Particle* particles, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    particles[i].vel += particles[i].acc * SPH::DT;
    particles[i].pos += particles[i].vel * SPH::DT;

    if (particles[i].pos.x < SPH::BMIN) {
        particles[i].pos.x = SPH::BMIN + 2e-4;
        if (particles[i].vel.x < 0.0) particles[i].vel.x *= -SPH::RESTITUTION;
    } else if (particles[i].pos.x > SPH::BMAX) {
        particles[i].pos.x = SPH::BMAX - 2e-4;
        if (particles[i].vel.x > 0.0) particles[i].vel.x *= -SPH::RESTITUTION;
    }
    if (particles[i].pos.y < SPH::BMIN) {
        particles[i].pos.y = SPH::BMIN + 2e-4;
        if (particles[i].vel.y < 0.0) particles[i].vel.y *= -SPH::RESTITUTION;
    } else if (particles[i].pos.y > SPH::BMAX) {
        particles[i].pos.y = SPH::BMAX - 2e-4;
        if (particles[i].vel.y > 0.0) particles[i].vel.y *= -SPH::RESTITUTION;
    }
    if (particles[i].pos.z < SPH::BMIN) {
        particles[i].pos.z = SPH::BMIN + 2e-4;
        if (particles[i].vel.z < 0.0) particles[i].vel.z *= -SPH::RESTITUTION;
    } else if (particles[i].pos.z > SPH::BMAX) {
        particles[i].pos.z = SPH::BMAX - 2e-4;
        if (particles[i].vel.z > 0.0) particles[i].vel.z *= -SPH::RESTITUTION;
    }
}

/****************************
*****************************
** SPH Member Functions *****
*****************************
****************************/

SPH::SPH()
    : d_particles(nullptr), d_hashHead(nullptr), d_hashNext(nullptr), maxParticles(0)
{
    reset();
}

SPH::~SPH() {
    cudaFree(d_particles);
    cudaFree(d_hashHead);
    cudaFree(d_hashNext);
}

void SPH::reset() {
    particles.clear();
    for (double x = (0.5 * BMIN); x < (0.5 * BMAX); x += SPACING)
        for (double y = BMIN + SPACING; y < (0.5 * BMAX); y += SPACING) 
            for (double z = BMIN + SPACING; z < BMAX - SPACING; z += SPACING) {
                Particle p;
                p.pos = {x, y, z};
                p.vel = {};
                p.acc = {};
                p.density = RHO0;
                p.pressure = 0.0;
                particles.push_back(p);
            }

    int n = (int)particles.size();
    if (n > maxParticles) {
        cudaFree(d_particles);
        cudaFree(d_hashHead);
        cudaFree(d_hashNext);
        cudaMalloc(&d_particles, n * sizeof(Particle));
        cudaMalloc(&d_hashHead, GpuHash::HASH_SIZE * sizeof(int));
        cudaMalloc(&d_hashNext, n * sizeof(int));
        maxParticles = n;
    }
}

void SPH::step() {
    int n = (int)particles.size();
    if (n == 0) return;

    // copy particles from host to device
    cudaMemcpy(d_particles, particles.data(), n * sizeof(Particle), cudaMemcpyHostToDevice);

    // clear cash to all -1
    cudaMemset(d_hashHead, 0xFF, GpuHash::HASH_SIZE * sizeof(int));

    GpuHash gpuHash{ d_hashHead, d_hashNext, H };

    constexpr int BLOCK = 256;
    int numBlocks = (n + BLOCK - 1) / BLOCK;

    buildGrid_kernel <<<numBlocks, BLOCK>>>(d_particles, n, gpuHash);
    computeDensityPressure_kernel<<<numBlocks, BLOCK>>>(d_particles, n, gpuHash);
    computeForces_kernel <<<numBlocks, BLOCK>>>(d_particles, n, gpuHash);
    integrate_kernel <<<numBlocks, BLOCK>>>(d_particles, n);

    // device to host
    cudaMemcpy(particles.data(), d_particles, n * sizeof(Particle), cudaMemcpyDeviceToHost);
}
