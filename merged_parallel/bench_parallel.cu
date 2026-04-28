#include <vector>
#include <chrono>
#include <cstdio>
#include <cuda_runtime.h>

#include "SPH.h"
#include "March.h"

constexpr int FRAMES   = 20;
constexpr int SUBSTEPS = 20;

int main() {
    SPH sim;
    initMarchTables();
    printf("particles: %ld\n", sim.particles.size());
    printf("GRID_N: %d\n", GRID_N);

    std::vector<Point3>    vertexBuffer;
    std::vector<uint32_t>  indexBuffer;
    std::vector<Vec3>      normalBuffer;

    int64_t total_phys  = 0;
    int64_t total_march = 0;

    for (int frame = 0; frame < FRAMES; ++frame) {
        auto ps = std::chrono::steady_clock::now();
        for (int s = 0; s < SUBSTEPS; ++s) sim.step();
        auto pe = std::chrono::steady_clock::now();
        int64_t phys_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(pe - ps).count();

        vertexBuffer.clear(); indexBuffer.clear(); normalBuffer.clear();

        auto ms = std::chrono::steady_clock::now();
        buildScalarField(sim.getParticles(), (int)sim.particles.size(),
                         sim.getHashHead(), sim.getHashNext());
        marchCubes(vertexBuffer, indexBuffer, normalBuffer);
        cudaDeviceSynchronize();
        auto me = std::chrono::steady_clock::now();
        int64_t march_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(me - ms).count();

        total_phys  += phys_ns;
        total_march += march_ns;

        printf("frame %d  phys=%ldms  march=%ldms  tris=%ld\n",
               frame, phys_ns/1000000, march_ns/1000000, vertexBuffer.size()/3);
    }

    printf("\n--- averages over %d frames ---\n", FRAMES);
    printf("physics (parallel): %ld ms/frame\n", total_phys  / FRAMES / 1000000);
    printf("march   (parallel): %ld ms/frame\n", total_march / FRAMES / 1000000);

    return 0;
}
