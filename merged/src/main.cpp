#include <vector>
#include <string>
#include <sstream>
#include <cstdint>
#include <cinttypes>
#include <cmath>
#include <chrono>
#include <fstream>
#include <cstdio>
#include <limits>
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

    if (r - 1 >= 0)          { sum += rayColors[(r-1)*IMAGE_WIDTH + c]; ++valid; }
    if (r + 1 < IMAGE_HEIGHT) { sum += rayColors[(r+1)*IMAGE_WIDTH + c]; ++valid; }
    if (c - 1 >= 0)           { sum += rayColors[r*IMAGE_WIDTH + (c-1)]; ++valid; }
    if (c + 1 < IMAGE_WIDTH)  { sum += rayColors[r*IMAGE_WIDTH + (c+1)]; ++valid; }

    return (sum / valid) * 0.5 + rayColors[r * IMAGE_WIDTH + c] * 0.5;
}



struct BFTiming { int64_t construct_ns, raytrace_ns, filewrite_ns; };

BFTiming rayTraceBruteForce(
    const std::vector<Point3>& vertexBuffer,
    const std::vector<uint32_t>& indexBuffer,
    const std::vector<Vec3>& normalBuffer,
    int frame,
    const Point3& leftCorner,
    const Point3& rightCorner)
{
    BFTiming t{};

    auto t0 = std::chrono::steady_clock::now();
    std::vector<Triangle> tris = constructSceneTriangles(vertexBuffer, indexBuffer, normalBuffer);
    auto t1 = std::chrono::steady_clock::now();
    t.construct_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

    SceneConstants scene = {
        Light(Point3(200.0, 100.0, 100.0), Color{1, 1, 1}, 1.0f),
        Color{0.05, 0.15, 0.4},
        Color{0.0, 0.0, 0.0}
    };

    Point3 cameraPos = computeCameraPosition(leftCorner, rightCorner);

    double imagePlaneDistance = 1.0;
    double theta = PI / 4;
    double planeHeight = 2 * imagePlaneDistance * std::tan(theta / 2);
    double planeWidth  = ((double)IMAGE_WIDTH / IMAGE_HEIGHT) * planeHeight;

    Color* rayColors = new Color[IMAGE_WIDTH * IMAGE_HEIGHT];

    auto t2 = std::chrono::steady_clock::now();
    for (int r = 0; r < IMAGE_HEIGHT; ++r) {
        for (int c = 0; c < IMAGE_WIDTH; ++c) {
            double u = (c + 0.5) / IMAGE_WIDTH;
            double v = (r + 0.5) / IMAGE_HEIGHT;

            double m_x = (u - 0.5) * planeWidth;
            double m_y = (0.5 - v) * planeHeight;
            double m_z = imagePlaneDistance;

            Point3 m(cameraPos.x + m_x, cameraPos.y + m_y, cameraPos.z + m_z);
            Vec3 dir(m.x - cameraPos.x, m.y - cameraPos.y, m.z - cameraPos.z);
            Ray ray(cameraPos, dir);

            HitRecord h = findIntersectingTriangle(ray, tris);
            rayColors[r * IMAGE_WIDTH + c] = phong(h, cameraPos, scene);
        }
    }
    auto t3 = std::chrono::steady_clock::now();
    t.raytrace_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();

    // ── 3. Anti-aliasing + file write (ASCII P3) ──────────────────────────────
    auto t4 = std::chrono::steady_clock::now();
    std::ofstream outFile("frames/image_bf_" + std::to_string(frame) + ".ppm");
    if (outFile.is_open()) {
        outFile << "P3\n" << IMAGE_WIDTH << ' ' << IMAGE_HEIGHT << "\n255\n";
        for (int r = 0; r < IMAGE_HEIGHT; ++r)
            for (int c = 0; c < IMAGE_WIDTH; ++c)
                write_Color(outFile, getAntialiasedColor(r, c, rayColors));
    }
    auto t5 = std::chrono::steady_clock::now();
    t.filewrite_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t5 - t4).count();

    delete[] rayColors;
    return t;
}




struct DDATiming { int64_t construct_ns, assign_ns, flatten_ns, raytrace_ns, filewrite_ns; };

DDATiming rayTraceDDA(
    const std::vector<Point3>& vertexBuffer,
    const std::vector<uint32_t>& indexBuffer,
    const std::vector<Vec3>& normalBuffer,
    int frame,
    const Point3& leftCorner,
    const Point3& rightCorner)
{
    DDATiming t{};

 
    auto t0 = std::chrono::steady_clock::now();
    std::vector<Triangle> sceneTriangles = constructSceneTriangles(vertexBuffer, indexBuffer, normalBuffer);
    auto t1 = std::chrono::steady_clock::now();
    t.construct_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();


    auto t2 = std::chrono::steady_clock::now();
    vector<vector<int>> trianglesPerBox = assignTriangles(sceneTriangles, leftCorner, rightCorner, BOXDIMENSION);
    auto t3 = std::chrono::steady_clock::now();
    t.assign_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();

    auto t4 = std::chrono::steady_clock::now();
    int numCells = BOXDIMENSION * BOXDIMENSION * BOXDIMENSION;
    vector<int> cellStart(numCells + 1, 0);
    for (int i = 0; i < numCells; i++)
        cellStart[i + 1] = cellStart[i] + (int)trianglesPerBox[i].size();

    vector<int> cellTriangles(cellStart[numCells]);
    vector<int> offset = cellStart;
    for (int i = 0; i < numCells; i++)
        for (int triIdx : trianglesPerBox[i])
            cellTriangles[offset[i]++] = triIdx;
    auto t5 = std::chrono::steady_clock::now();
    t.flatten_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t5 - t4).count();

    SceneConstants scene = {
        Light(Point3(200.0, 100.0, 100.0), Color{1, 1, 1}, 1.0f),
        Color{0.05, 0.15, 0.4},
        Color{0.0, 0.0, 0.0}
    };

    Point3 cameraPos = computeCameraPosition(leftCorner, rightCorner);
    Point3 sceneCenter((leftCorner.x + rightCorner.x) / 2.0,
                       (leftCorner.y + rightCorner.y) / 2.0,
                       (leftCorner.z + rightCorner.z) / 2.0);

    double imagePlaneDistance = 1.0;
    Vec3 cameraForward = (sceneCenter - cameraPos).normalized();
    Vec3 sceneUp(0, 1, 0);
    Vec3 cameraRight = cross(cameraForward, sceneUp).normalized();
    Vec3 cameraUp    = cross(cameraRight, cameraForward);

    double theta       = PI / 4;
    double planeHeight = 2.0 * imagePlaneDistance * std::tan(theta / 2.0);
    double planeWidth  = ((double)IMAGE_WIDTH / IMAGE_HEIGHT) * planeHeight;

    Color* rayColors = new Color[IMAGE_WIDTH * IMAGE_HEIGHT];

    auto t6 = std::chrono::steady_clock::now();
    for (int r = 0; r < IMAGE_HEIGHT; ++r) {
        for (int c = 0; c < IMAGE_WIDTH; ++c) {
            double u   = (c + 0.5) / IMAGE_WIDTH;
            double v   = (r + 0.5) / IMAGE_HEIGHT;

            double m_x = (u - 0.5) * planeWidth;
            double m_y = (0.5 - v) * planeHeight;
            double m_z = imagePlaneDistance;

            Point3 m = cameraPos + cameraRight * m_x + cameraUp * m_y + cameraForward * m_z;
            Vec3 dir  = m - cameraPos;
            Ray ray(cameraPos, dir);

            int cells[BOXDIMENSION * 3];
            int numHitCells = findIntersectingCubes(ray, leftCorner, rightCorner,
                                                    BOXDIMENSION, cells, 3 * BOXDIMENSION);

            HitRecord best;
            best.hit = false;

            for (int ci = 0; ci < numHitCells; ci++) {
                HitRecord h = findIntersectingTriangleInVoxel(
                    ray, cellTriangles, sceneTriangles, cellStart, cells[ci]);
                if (h.hit && (!best.hit || h.distance < best.distance))
                    best = h;
            }

            rayColors[r * IMAGE_WIDTH + c] = phong(best, cameraPos, scene);
        }
    }
    auto t7 = std::chrono::steady_clock::now();
    t.raytrace_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t7 - t6).count();

    auto t8 = std::chrono::steady_clock::now();
    std::ofstream outFile("frames/image" + std::to_string(frame) + ".ppm",
                          std::ios::binary);
    if (outFile.is_open()) {
        outFile << "P6\n" << IMAGE_WIDTH << ' ' << IMAGE_HEIGHT << "\n255\n";
        std::vector<unsigned char> pixels(IMAGE_WIDTH * IMAGE_HEIGHT * 3);
        for (int r = 0; r < IMAGE_HEIGHT; ++r) {
            for (int c = 0; c < IMAGE_WIDTH; ++c) {
                int idx = r * IMAGE_WIDTH + c;
                pack_Color(&pixels[idx * 3], getAntialiasedColor(r, c, rayColors), true);
            }
        }
        outFile.write(reinterpret_cast<const char*>(pixels.data()), (std::streamsize)pixels.size());
    }
    auto t9 = std::chrono::steady_clock::now();
    t.filewrite_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t9 - t8).count();

    delete[] rayColors;
    return t;
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
    std::vector<Point3>   vertexBuffer;
    std::vector<uint32_t> indexBuffer;
    std::vector<Vec3>     normalBuffer;

    SPH sim;
    printf("Number of particles: %ld\n", sim.particles.size());

    static const Vec3 boxCorners[8] = {
        {sim.BMIN, sim.BMIN, sim.BMIN},
        {sim.BMAX, sim.BMIN, sim.BMIN},
        {sim.BMAX, sim.BMAX, sim.BMIN},
        {sim.BMIN, sim.BMAX, sim.BMIN},
        {sim.BMIN, sim.BMIN, sim.BMAX},
        {sim.BMAX, sim.BMIN, sim.BMAX},
        {sim.BMAX, sim.BMAX, sim.BMAX},
        {sim.BMIN, sim.BMAX, sim.BMAX}
    };
    constexpr int SUBSTEPS = 20;

    Point3 leftCorner  = boxCorners[0];
    Point3 rightCorner = boxCorners[6];


    int64_t total_assign_ns         = 0;  // construct + assign + flatten
    int64_t total_dda_raytrace_ns   = 0;
    int64_t total_alias_pack_write_ns = 0;
    int64_t total_bf_raytrace_ns    = 0;


    std::ofstream csvFile("benchmark_serial.csv");
    if (csvFile.is_open()) {
        csvFile << "frame,construct_ns,assign_ns,flatten_ns,"
                   "dda_raytrace_ns,bf_raytrace_ns,filewrite_ns\n";
    }

    int frame = 0;
    while (true) {
        printf("FRAME: #%d\n", frame);

        vertexBuffer.clear();
        indexBuffer.clear();
        normalBuffer.clear();


        auto physics_start = std::chrono::steady_clock::now();
        for (int s = 0; s < SUBSTEPS; ++s) sim.step();
        auto physics_end   = std::chrono::steady_clock::now();
        int64_t physics_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                 physics_end - physics_start).count();
        printf("\tphysics benchmarking:\n"
               "\t\t%" PRId64 " ns elapsed\n"
               "\t\t%ld particles simulated\n"
               "\t\t%d substeps\n"
               "\t\t~%" PRId64 " ns/sim step\n",
               physics_ns, (long)sim.particles.size(), SUBSTEPS, physics_ns / SUBSTEPS);

        auto mesh_start = std::chrono::steady_clock::now();
        buildScalarField(sim.particles);
        marchCubes(vertexBuffer, indexBuffer, normalBuffer);
        auto mesh_end   = std::chrono::steady_clock::now();
        int64_t mesh_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                              mesh_end - mesh_start).count();
        printf("\tmesh benchmarking:\n\t\t%" PRId64 " ns elapsed\n", mesh_ns);


        DDATiming dda = rayTraceDDA(vertexBuffer, indexBuffer, normalBuffer,
                                    frame, leftCorner, rightCorner);

        int64_t assign_ns = dda.construct_ns + dda.assign_ns + dda.flatten_ns;
        total_assign_ns         += assign_ns;
        total_dda_raytrace_ns   += dda.raytrace_ns;
        total_alias_pack_write_ns += dda.filewrite_ns;

        printf("\tDDA render benchmarking:\n"
               "\t\tconstruct triangles:  %" PRId64 " ns\n"
               "\t\tAABB voxel binning:   %" PRId64 " ns\n"
               "\t\tCSR flatten:          %" PRId64 " ns\n"
               "\t\tray trace + shading:  %" PRId64 " ns\n"
               "\t\taa + pack + write:    %" PRId64 " ns\n"
               "\t\t%d rays traced\n",
               dda.construct_ns, dda.assign_ns, dda.flatten_ns,
               dda.raytrace_ns, dda.filewrite_ns,
               IMAGE_WIDTH * IMAGE_HEIGHT);


        BFTiming bf = rayTraceBruteForce(vertexBuffer, indexBuffer, normalBuffer,
                                         frame, leftCorner, rightCorner);
        total_bf_raytrace_ns += bf.raytrace_ns;

        printf("\tBrute-force render benchmarking:\n"
               "\t\tconstruct triangles:  %" PRId64 " ns\n"
               "\t\tray trace + shading:  %" PRId64 " ns\n"
               "\t\taa + write:           %" PRId64 " ns\n"
               "\t\t%d rays traced\n",
               bf.construct_ns, bf.raytrace_ns, bf.filewrite_ns,
               IMAGE_WIDTH * IMAGE_HEIGHT);

        if (csvFile.is_open()) {
            csvFile << frame << ','
                    << dda.construct_ns << ','
                    << dda.assign_ns    << ','
                    << dda.flatten_ns   << ','
                    << dda.raytrace_ns  << ','
                    << bf.raytrace_ns   << ','
                    << dda.filewrite_ns << '\n';
            csvFile.flush();
        }

        printf("\tfinishing frame #%d\n\n", frame);
        ++frame;
    }

    printf("Total time constructing scene and assigning triangles = %f sec\n",
           total_assign_ns / 1e9);
    printf("Total render time (DDA, serial CPU) = %f sec\n",
           total_dda_raytrace_ns / 1e9);
    printf("Total render time (brute-force, serial CPU) = %f sec\n",
           total_bf_raytrace_ns / 1e9);
    printf("Total time packing, antialiasing, and writing pixels = %f sec\n",
           total_alias_pack_write_ns / 1e9);

    return 0;
}
