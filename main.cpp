#include <vector>
#include <algorithm>
#include <string>
#include <sstream>
#include <cstdint>
#include <iomanip>
#include <cmath>
#include <chrono>

// physics sim includes
#include "particle_sim/src/serial/March.h"
#include "particle_sim/src/serial/SPH.h"

// serial ray tracer (CUDA-free)
#include "rendering/rayTrace_serial.h"

/*
    ******************************************************
    ******************************************************
    ******************************************************
    **** Temporary. Delete after TODO resolved ***********
    ******************************************************
    ******************************************************
    ******************************************************
*/

void convertMeshBuffers(
    const std::vector<std::vector<double>>& vertexBuffer_meshout,
    const std::vector<int>& indexBuffer_meshout,
    const std::vector<std::vector<double>>& normalBuffer_meshout,
    std::vector<Point3>& vertexBuffer_rayin,
    std::vector<uint32_t>& indexBuffer_rayin,
    std::vector<Vec3>& normalBuffer_rayin
) {
    vertexBuffer_rayin.reserve(vertexBuffer_meshout.size());
    for (const auto& v : vertexBuffer_meshout)
        vertexBuffer_rayin.push_back(Point3{v[0], v[1], v[2]});

    indexBuffer_rayin.reserve(indexBuffer_meshout.size());
    for (int idx : indexBuffer_meshout)
        indexBuffer_rayin.push_back(static_cast<uint32_t>(idx));

    normalBuffer_rayin.reserve(normalBuffer_meshout.size());
    for (const auto& n : normalBuffer_meshout)
        normalBuffer_rayin.push_back(Vec3{n[0], n[1], n[2]});
}

/*
    ******************************************************
    ******************************************************
    ******************************************************
    ******************** Main ****************************
    ******************************************************
    ******************************************************
    ******************************************************
*/

int main() {
    // buffers. used throughout the program
    // TODO: pick one of these buffer styles and stick with it
    std::vector<std::vector<double>> vertexBuffer_meshout;
    std::vector<int> indexBuffer_meshout;
    std::vector<std::vector<double>> normalBuffer_meshout;
    std::vector<Point3> vertexBuffer_rayin;
    std::vector<uint32_t> indexBuffer_rayin;
    std::vector<Vec3> normalBuffer_rayin;

    /*
        *********************
        fluid simulator setup
        *********************
    */
    SPH sim;
    printf("Number of particles: %ld\n", sim.particles.size());
    // corners of the box containing the fluid
    static const Vec3 boxCorners[8] = {
    {sim.BMIN,sim.BMIN,sim.BMIN},
    {sim.BMAX,sim.BMIN,sim.BMIN},
    {sim.BMAX, sim.BMAX,sim.BMIN},
    {sim.BMIN, sim.BMAX,sim.BMIN},
    {sim.BMIN,sim.BMIN, sim.BMAX},
    {sim.BMAX,sim.BMIN, sim.BMAX},
    {sim.BMAX, sim.BMAX, sim.BMAX},
    {sim.BMIN, sim.BMAX, sim.BMAX}
    };
    // edges of the box containing the fluid (indexes into boxCorners)
    static constexpr int boxEdges[12][2] = {
    {0,1},{1,2},{2,3},{3,0}, {4,5},{5,6},{6,7},{7,4}, {0,4},{1,5},{2,6},{3,7}
    };
    // num sub-steps per rendered frame
    // 5 × Dt(0.001 s) = 0.005 s of simulated time per visual frame
    constexpr int SUBSTEPS = 5;

    /*
        ray tracer setup
    */
    Point3 leftCorner  = boxCorners[0];   // {BMIN, BMIN, BMIN}
    Point3 rightCorner = boxCorners[6];   // {BMAX, BMAX, BMAX}

    /*
        *********
        main loop
        *********
    */
   int frame = 0;
    while (true) {
        printf("starting on frame #%d\n", frame);

        // aways clear
        vertexBuffer_meshout.clear();
	    indexBuffer_meshout.clear();
        normalBuffer_meshout.clear();
        vertexBuffer_rayin.clear();
        indexBuffer_rayin.clear();
        normalBuffer_rayin.clear();

        // physics
        auto physics_start = std::chrono::steady_clock::now();
        for (int s = 0; s < SUBSTEPS; ++s) sim.step();
        auto physics_end = std::chrono::steady_clock::now();
        int64_t physics_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(physics_end - physics_start).count();
        printf("done with the physics\n");
        printf("physics benchmarking:\n\t%ld ns elapsed\n\t%ld particles simulated\n\t%d substeps\n\t~%ld ns/sim step\n",
            physics_elapsed, sim.particles.size(), SUBSTEPS, physics_elapsed/SUBSTEPS);

        // particles to mesh
        auto mesh_start = std::chrono::steady_clock::now();
	    buildScalarField(sim.particles);
	    marchCubes(vertexBuffer_meshout, indexBuffer_meshout, normalBuffer_meshout);
        auto mesh_end = std::chrono::steady_clock::now();
        int64_t mesh_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(mesh_end - mesh_start).count();
        printf("done with the mesh construction\n");
        printf("mesh benchmarking:\n\t%ld ns elapsed\n", mesh_elapsed);
        

        // TODO: fix this!!! we want allignment on buffers
        convertMeshBuffers(
            vertexBuffer_meshout,
	        indexBuffer_meshout,
            normalBuffer_meshout,
            vertexBuffer_rayin,
            indexBuffer_rayin,
            normalBuffer_rayin
        );

        auto ray_start = std::chrono::steady_clock::now();
        // ray trace
        rayTrace(
            vertexBuffer_rayin,
            indexBuffer_rayin,
            normalBuffer_rayin,
            frame,
            leftCorner,
            rightCorner);
         printf("done with the ray tracing\n");
        auto ray_end = std::chrono::steady_clock::now();
        int64_t ray_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(ray_end - ray_start).count();
        printf("ray benchmarking:\n\t%ld ns elapsed\n\t%d rays traced\n", ray_elapsed, IMAGE_HEIGHT * IMAGE_WIDTH);
        printf("done with ray tracing");
        printf("finishing frame #%d\n", frame);
        ++frame;
    }

    return 0;
}
