#include <vector>
#include <string>
#include <sstream>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <fstream>
#include <cstdio>
// physics sim includes
#include "March.h"
#include "SPH.h"

// serial ray tracer (CUDA-free)
#include "color.h"
#include "consts.h"
#include "rayTrace.h"
#include "phong.h"


using std::vector;



inline Color getAntialiasedColor(int r, int c, Color* rayColors) {
    int valid = 0;
    Color sum;

    if (r - 1 >= 0) { sum += rayColors[(r-1)*IMAGE_WIDTH + c]; ++valid; }
    if (r + 1 < IMAGE_HEIGHT) { sum += rayColors[(r+1)*IMAGE_WIDTH + c]; ++valid; }
    if (c - 1 >= 0) { sum += rayColors[r*IMAGE_WIDTH + (c-1)]; ++valid; }
    if (c + 1 < IMAGE_WIDTH) { sum += rayColors[r*IMAGE_WIDTH + (c+1)]; ++valid; }

    return (sum / valid) * 0.5 + rayColors[r * IMAGE_WIDTH + c] * 0.5;
}

Color colorPixel(int r, int c, const Point3& cameraPos,const vector<Triangle>& sceneTriangles, SceneConstants& scene){
    // define image plane metrics
    double imagePlaneDistance = 1;
    double theta = PI/4; // 45 degree FOV
    double planeHeight = 2*imagePlaneDistance*tan(theta/2);
    double planeWidth =  ((double)IMAGE_WIDTH / IMAGE_HEIGHT) * planeHeight;

    double u = (c + 0.5) / IMAGE_WIDTH;
    double v = (r + 0.5) / IMAGE_HEIGHT;

    // map to [-0.5, 0.5]
    double m_x = (u - 0.5) * planeWidth;
    double m_y = (0.5 - v) * planeHeight;  // flip here
    double m_z = imagePlaneDistance;

    //slight bug fix here
    Point3 m(cameraPos.x + m_x, cameraPos.y + m_y, cameraPos.z + m_z); // world-space point on image plane
    Vec3 dir(m.x - cameraPos.x,  m.y - cameraPos.y, m.z - cameraPos.z);
    Ray ray(cameraPos, dir); // ray direction = m - camera

    HitRecord h = findIntersectingTriangle(ray, sceneTriangles);
    return phong(h, cameraPos, scene);
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


inline void rayTrace(
    const std::vector<Point3>& vertexBuffer,
    const std::vector<uint32_t>& indexBuffer,
    const std::vector<Vec3>& normalBuffer,
    int frame,
    const Point3& leftCorner,
    const Point3& rightCorner)
{
    std::ofstream outFile("frames/image" + std::to_string(frame) + ".ppm");
    if (!outFile.is_open()) {
        fprintf(stderr, "rayTrace: could not open output file for frame %d\n", frame);
        return;
    }
    outFile << "P3\n" << IMAGE_WIDTH << ' ' << IMAGE_HEIGHT << "\n255\n";

    SceneConstants scene = {
        Light(Point3(200.0, 100.0, 100.0), Color{1, 1, 1}, 1.0f),
        Color{0.05, 0.15, 0.4},
        Color{0.0, 0.0, 0.0}
    };

    Point3 cameraPos = computeCameraPosition(leftCorner, rightCorner);
    std::vector<Triangle> tris = constructSceneTriangles(vertexBuffer, indexBuffer, normalBuffer);

    Color* rayColors = new Color[IMAGE_WIDTH * IMAGE_HEIGHT];

    for (int r = 0; r < IMAGE_HEIGHT; ++r) {
        for (int c = 0; c < IMAGE_WIDTH; ++c) {
            rayColors[r * IMAGE_WIDTH + c] = colorPixel(r, c, cameraPos, tris, scene);
        }
        // printf("not stuck. just taking time... (getting pixel colors for row %d)\n", r);
    }

    for (int r = 0; r < IMAGE_HEIGHT; ++r) {
        for (int c = 0; c < IMAGE_WIDTH; ++c) {
            write_Color(outFile, getAntialiasedColor(r, c, rayColors));
        }
        // printf("not stuck. just taking time... (writing pixel colors to outfile for row %d)\n", r);
    }

    delete[] rayColors;
    outFile.close();
}

int main() {
    // buffers. shared between marching cubes and ray tracer
    std::vector<Point3> vertexBuffer;
    std::vector<uint32_t> indexBuffer;
    std::vector<Vec3> normalBuffer;

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
    constexpr int SUBSTEPS = 20;

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

        // always clear
        vertexBuffer.clear();
        indexBuffer.clear();
        normalBuffer.clear();

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
        marchCubes(vertexBuffer, indexBuffer, normalBuffer);
        auto mesh_end = std::chrono::steady_clock::now();
        int64_t mesh_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(mesh_end - mesh_start).count();
        printf("done with the mesh construction\n");
        printf("mesh benchmarking:\n\t%ld ns elapsed\n", mesh_elapsed);

        auto ray_start = std::chrono::steady_clock::now();
        // ray trace
        rayTrace(
            vertexBuffer,
            indexBuffer,
            normalBuffer,
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
