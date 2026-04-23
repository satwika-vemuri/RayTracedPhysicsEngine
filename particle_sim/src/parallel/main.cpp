#include <vector>
#include <algorithm>
#include <string>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include "March.h"
#include "SPH.h"

// main
int main() {

    SPH sim;
    printf("Number of particles: %ld\n", sim.particles.size());
    // the box outline
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
    static constexpr int boxEdges[12][2] = {
    {0,1},{1,2},{2,3},{3,0}, {4,5},{5,6},{6,7},{7,4}, {0,4},{1,5},{2,6},{3,7}
    };

    // num sub-steps per rendered frame
    constexpr int SUBSTEPS = 20;

    std::vector<std::vector<double>> vertexBuffer;
    std::vector<int> indexBuffer;
    std::vector<std::vector<double>> normalBuffer;

    // main loop
    for (int frame = 0; frame < 100; ++frame) {

        auto physics_start = std::chrono::steady_clock::now();
        for (int s = 0; s < SUBSTEPS; ++s) sim.step();

        sim.syncToHost(); // important!

        auto physics_end = std::chrono::steady_clock::now();
        int64_t physics_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(physics_end - physics_start).count();
        printf("done with the physics\n");
        printf("physics benchmarking:\n\t%ld ns elapsed\n\t%ld particles simulated\n\t%d substeps\n\t~%ld ns/sim step\n",
            physics_elapsed, sim.particles.size(), SUBSTEPS, physics_elapsed/SUBSTEPS);

	    vertexBuffer.clear();
	    indexBuffer.clear();

	    buildScalarField(sim.particles);
	    marchCubes(vertexBuffer, indexBuffer, vertexBuffer);
        printf("Done w/frame #%d\n", frame++);
    }

    return 0;
}
