#include "SPH.h"
#include <cuda_runtime.h>
#include <cstdio>


/****************************
*****************************
** Kernel Helper Functions **
*****************************
****************************/

__device__ static float d_w_poly6(float r2) {
    if (r2 >= SPH::H2) return 0.0;
    float d = SPH::H2 - r2;
    return SPH::K_POLY6 * d * d * d;
}

__device__ static Vec3 d_grad_w_spiky(const Vec3& rij, float r) {
    if (r >= SPH::H || r < 1e-6f) return {};
    float d = SPH::H - r;
    return rij * (SPH::K_SPIKY * d * d / r);
}

__device__ static float d_lap_w_visc(float r) {
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
    float density = 0.0;

    int ix = (int)floor(pos_i.x / hash.cellSize);
    int iy = (int)floor(pos_i.y / hash.cellSize);
    int iz = (int)floor(pos_i.z / hash.cellSize);

    // check 3x3x3 neighborhood
    #pragma unroll
    for (int dx = -1; dx <= 1; ++dx) {
        #pragma unroll
        for (int dy = -1; dy <= 1; ++dy) {
            #pragma unroll
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

    density = fmaxf(density, SPH::RHO0 * 0.1f);
    particles[i].density = density;

    // tait equation of state from STAR paper
    float ratio = density / SPH::RHO0;
    float r7 = ratio * ratio * ratio * ratio * ratio * ratio * ratio;
    particles[i].pressure = fmaxf(0.0f, SPH::B_PRESS * (r7 - 1.0f));
}

// one thread per particle: compute pressure and viscosity accelerations
__global__ void computeForces_kernel(Particle* particles, int n, GpuHash hash) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const Vec3 gravity{0.0, -SPH::GRAVITY, 0.0};
    const Vec3 pos_i = particles[i].pos;
    const Vec3 vel_i = particles[i].vel;
    const float rho_i = particles[i].density;
    const float prs_i = particles[i].pressure;

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
                        float r2 = rij.length2();
                        if (r2 < SPH::H2 && r2 > 1e-12f) {
                            float r = sqrtf(r2);
                            float pterm = prs_i / (rho_i * rho_i)
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
    for (float x = (0.5 * BMIN); x < (0.5 * BMAX); x += SPACING)
        for (float y = BMIN + SPACING; y < (0.5 * BMAX); y += SPACING) 
            for (float z = BMIN + SPACING; z < BMAX - SPACING; z += SPACING) {
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
    cudaMemcpy(d_particles, particles.data(), n * sizeof(Particle), cudaMemcpyHostToDevice);
}

void SPH::step() {
    int n = (int)particles.size();
    if (n == 0) return;

    // clear cash to all -1
    cudaMemset(d_hashHead, 0xFF, GpuHash::HASH_SIZE * sizeof(int));

    GpuHash gpuHash{ d_hashHead, d_hashNext, H };

    constexpr int BLOCK = 256;
    int numBlocks = (n + BLOCK - 1) / BLOCK;

    buildGrid_kernel <<<numBlocks, BLOCK>>>(d_particles, n, gpuHash);
    computeDensityPressure_kernel<<<numBlocks, BLOCK>>>(d_particles, n, gpuHash);
    computeForces_kernel <<<numBlocks, BLOCK>>>(d_particles, n, gpuHash);
    integrate_kernel <<<numBlocks, BLOCK>>>(d_particles, n);
}

void SPH::syncToHost() {
    int n = (int)particles.size();
    if (n == 0) return;
    cudaMemcpy(particles.data(), d_particles, n * sizeof(Particle), cudaMemcpyDeviceToHost);
}