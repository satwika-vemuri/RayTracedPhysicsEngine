#include <vector>
#include <algorithm>
#include <string>
#include <sstream>
#include <cstdint>
#include <iomanip>
#include <cmath>

// physics sim includes
#include "particle_sim/src/March.h"
#include "particle_sim/src/SPH.h"

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
        for (int s = 0; s < SUBSTEPS; ++s) sim.step();
        printf("done with the physics\n");

        // particles to mesh
	    buildScalarField(sim.particles);
	    marchCubes(vertexBuffer_meshout, indexBuffer_meshout, normalBuffer_meshout);
        printf("done with the mesh construction\n");

        // TODO: fix this!!! we want allignment on buffers
        convertMeshBuffers(
            vertexBuffer_meshout,
	        indexBuffer_meshout,
            normalBuffer_meshout,
            vertexBuffer_rayin,
            indexBuffer_rayin,
            normalBuffer_rayin
        );

        // ray trace
        rayTrace(
            vertexBuffer_rayin,
            indexBuffer_rayin,
            normalBuffer_rayin,
            frame,
            leftCorner,
            rightCorner);
         printf("done with the ray tracing\n");

        printf("finishing frame #%d\n", frame);
        ++frame;
    }

    return 0;
}
