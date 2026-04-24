
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <string>
#include <cuda_runtime.h>
#include <cstdio>
#include <time.h>
#include <vector>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <filesystem>
#include <cstdlib>

#include "Vec3.h"
#include "rayTrace_gpu.h"
#include "color.h"
#include "March.h"
#include "SPH.h"

#define FRAMES 720
#define BOXDIMENSION 32
#define MAX_BOXDIMENSION 128

using std::vector;

struct BenchmarkConfig {
    int frames = FRAMES;
    int substeps = 20;
    int boxDimension = BOXDIMENSION;
    bool writeFrames = true;
    bool bruteForce = false;
    bool includeSphere = false;
    std::string csvPath = "benchmark.csv";
};

static constexpr int PROGRESS_PRINT_INTERVAL = 60;

struct FrameStats {
    int frame = 0;
    double physics_ms = 0.0;
    double march_ms = 0.0;
    double tri_construct_ms = 0.0;
    double binning_ms = 0.0;
    double flatten_ms = 0.0;
    double h2d_ms = 0.0;
    double ray_ms = 0.0;
    double d2h_ms = 0.0;
    double file_ms = 0.0;
    double total_ms = 0.0;
    size_t num_particles = 0;
    size_t num_triangles = 0;
    size_t flat_triangle_indices = 0;
    size_t nonempty_cells = 0;
    size_t max_tris_per_cell = 0;
    double avg_tris_per_nonempty_cell = 0.0;
    double duplication_factor = 0.0;
    size_t h2d_bytes = 0;
    size_t d2h_bytes = 0;
};

static double nsToMs(int64_t ns) {
    return static_cast<double>(ns) / 1.0e6;
}

static void writeCsvHeader(std::ofstream& csv) {
    csv << "frame,box_dimension,substeps,write_frames,bruteforce,include_sphere,"
           "physics_ms,march_ms,tri_construct_ms,binning_ms,flatten_ms,"
           "h2d_ms,ray_ms,d2h_ms,file_ms,total_ms,"
           "num_particles,num_triangles,flat_triangle_indices,nonempty_cells,"
           "max_tris_per_cell,avg_tris_per_nonempty_cell,duplication_factor,"
           "h2d_bytes,d2h_bytes,h2d_bandwidth_gbps,d2h_bandwidth_gbps\n";
}

static void writeCsvRow(std::ofstream& csv, const BenchmarkConfig& cfg, const FrameStats& s) {
    double h2dBw = (s.h2d_ms > 0.0) ? (static_cast<double>(s.h2d_bytes) / (s.h2d_ms / 1000.0)) / 1.0e9 : 0.0;
    double d2hBw = (s.d2h_ms > 0.0) ? (static_cast<double>(s.d2h_bytes) / (s.d2h_ms / 1000.0)) / 1.0e9 : 0.0;

    csv << s.frame << ','
        << cfg.boxDimension << ','
        << cfg.substeps << ','
        << (cfg.writeFrames ? 1 : 0) << ','
        << (cfg.bruteForce ? 1 : 0) << ','
        << (cfg.includeSphere ? 1 : 0) << ','
        << s.physics_ms << ','
        << s.march_ms << ','
        << s.tri_construct_ms << ','
        << s.binning_ms << ','
        << s.flatten_ms << ','
        << s.h2d_ms << ','
        << s.ray_ms << ','
        << s.d2h_ms << ','
        << s.file_ms << ','
        << s.total_ms << ','
        << s.num_particles << ','
        << s.num_triangles << ','
        << s.flat_triangle_indices << ','
        << s.nonempty_cells << ','
        << s.max_tris_per_cell << ','
        << s.avg_tris_per_nonempty_cell << ','
        << s.duplication_factor << ','
        << s.h2d_bytes << ','
        << s.d2h_bytes << ','
        << h2dBw << ','
        << d2hBw << '\n';
}

static std::string summaryPathFromCsv(const std::string& csvPath) {
    std::filesystem::path p(csvPath);
    p.replace_extension(".txt");
    return p.string();
}

static void writeSummaryFile(const std::string& path,
                             const FrameStats& avg,
                             size_t frameCount,
                             const BenchmarkConfig& cfg,
                             float executionTimeSeconds) {
    std::ofstream out(path, std::ios::out);
    if (!out.is_open()) {
        std::cerr << "Failed to open benchmark summary file: " << path << "\n";
        return;
    }

    out << "Benchmark Summary\n";
    out << "  Frames: " << frameCount << "\n";
    out << "  Mode: " << (cfg.bruteForce ? "bruteforce" : "dda") << "\n";
    out << "  Box dimension: " << cfg.boxDimension << "\n";
    out << "  Output frames: " << (cfg.writeFrames ? "enabled" : "disabled") << "\n";
    out << "  Sphere: " << (cfg.includeSphere ? "enabled" : "disabled") << "\n";
    out << "  Average timings (ms/frame):\n";
    out << "    physics: " << avg.physics_ms << "\n";
    out << "    march:   " << avg.march_ms << "\n";
    out << "    tri (constructSceneTriangles): " << avg.tri_construct_ms << "\n";
    out << "    bin (assignTriangles):         " << avg.binning_ms << "\n";
    out << "    flat:    " << avg.flatten_ms << "\n";
    out << "    h2d:     " << avg.h2d_ms << "\n";
    out << "    ray:     " << avg.ray_ms << "\n";
    out << "    d2h:     " << avg.d2h_ms << "\n";
    out << "    file:    " << avg.file_ms << "\n";
    out << "    total:   " << avg.total_ms << "\n";
    out << "  Average grid quality:\n";
    out << "    avg triangles/nonempty cell: " << avg.avg_tris_per_nonempty_cell << "\n";
    out << "    duplication factor:          " << avg.duplication_factor << "\n";
    out << "Execution time = " << executionTimeSeconds << " s\n";
}

static void printAverageSummary(const FrameStats& avg, size_t frameCount, const BenchmarkConfig& cfg) {
    printf("\nBenchmark Summary\n");
    printf("  Frames: %zu\n", frameCount);
    printf("  Mode: %s\n", cfg.bruteForce ? "bruteforce" : "dda");
    printf("  Box dimension: %d\n", cfg.boxDimension);
    printf("  Output frames: %s\n", cfg.writeFrames ? "enabled" : "disabled");
    printf("  Sphere: %s\n", cfg.includeSphere ? "enabled" : "disabled");
    printf("  Average timings (ms/frame):\n");
    printf("    physics: %.3f\n", avg.physics_ms);
    printf("    march:   %.3f\n", avg.march_ms);
    printf("    tri (constructSceneTriangles):     %.3f\n", avg.tri_construct_ms);
    printf("    bin (assignTriangles):     %.3f\n", avg.binning_ms);
    printf("    flat:    %.3f\n", avg.flatten_ms);
    printf("    h2d:     %.3f\n", avg.h2d_ms);
    printf("    ray:     %.3f\n", avg.ray_ms);
    printf("    d2h:     %.3f\n", avg.d2h_ms);
    printf("    file:    %.3f\n", avg.file_ms);
    printf("    total:   %.3f\n", avg.total_ms);
    printf("  Average grid quality:\n");
    printf("    avg triangles/nonempty cell: %.3f\n", avg.avg_tris_per_nonempty_cell);
    printf("    duplication factor:          %.3f\n", avg.duplication_factor);
}

static BenchmarkConfig parseArgs(int argc, char** argv) {
    BenchmarkConfig cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.rfind("--frames=", 0) == 0) {
            cfg.frames = std::stoi(arg.substr(9));
        } else if (arg.rfind("--substeps=", 0) == 0) {
            cfg.substeps = std::stoi(arg.substr(11));
        } else if (arg.rfind("--boxdim=", 0) == 0) {
            cfg.boxDimension = std::stoi(arg.substr(9));
        } else if (arg.rfind("--csv=", 0) == 0) {
            cfg.csvPath = arg.substr(6);
        } else if (arg == "--no-output") {
            cfg.writeFrames = false;
        } else if (arg == "--bruteforce") {
            cfg.bruteForce = true;
        } else if (arg == "--with-sphere") {
            cfg.includeSphere = true;
        } else if (arg == "--help") {
            std::cout << "Usage: ./physics_sim [--frames=N] [--substeps=N] [--boxdim=N] "
                         "[--csv=PATH] [--no-output] [--bruteforce] [--with-sphere]\n";
            std::exit(0);
        }
    }
    return cfg;
}

__device__
Color phong(const HitRecord& pos,
            const Point3& cameraPos,
            SceneConstants scene) {

    if (!pos.hit) return scene.dark;

    Vec3 N = pos.interpolatedNormal();
    Vec3 L = (scene.light.coords() - pos.point).normalized();
    Vec3 V = (cameraPos - pos.point).normalized();
    Vec3 R = reflect(L, N);

    float diffuse = fmaxf(0.0f, N.dot(L));
    float spec    = pow(fmaxf(0.0f, R.dot(V)), SHININESS);

    Color ambientTerm =
        scene.surfaceColor * AMBIENT * scene.light.brightness();

    Color diffuseTerm =
        scene.surfaceColor * scene.light.color() *
        scene.light.brightness() * diffuse;

    Color specularTerm =
        scene.light.color() *
        scene.light.brightness() *
        REFLECTIVENESS * spec;

    return ambientTerm + diffuseTerm + specularTerm;
}

void generateSphere(
    vector<Point3>& vertexBuffer,
    vector<uint32_t>& indexBuffer,
    vector<Vec3>& normalBuffer
) {
    int latSteps = 20;
    int lonSteps = 20;

    float cx = 500.0f, cy = 500.0f, cz = 500.0f;
    float radius = 200.0f;

    for (int i = 0; i <= latSteps; i++) {
        float theta = PI * i / latSteps;

        for (int j = 0; j <= lonSteps; j++) {
            float phi = 2.0f * PI * j / lonSteps;

            float x = cx + radius * sin(theta) * cos(phi);
            float y = cy + radius * sin(theta) * sin(phi);
            float z = cz + radius * cos(theta);

            vertexBuffer.push_back(Point3{x,y,z});

            float nx = x - cx;
            float ny = y - cy;
            float nz = z - cz;

            float len = sqrt(nx * nx + ny * ny + nz * nz);
            normalBuffer.push_back(Vec3{nx, ny, nz} / len);
        }
    }

    int stride = lonSteps + 1;

    for (int i = 0; i < latSteps; i++) {
        for (int j = 0; j < lonSteps; j++) {
            int v1 = i * stride + j;
            int v2 = v1 + 1;
            int v3 = (i + 1) * stride + j;
            int v4 = v3 + 1;

            indexBuffer.push_back(v1);
            indexBuffer.push_back(v3);
            indexBuffer.push_back(v2);

            indexBuffer.push_back(v2);
            indexBuffer.push_back(v3);
            indexBuffer.push_back(v4);
        }
    }
}

Color getAntialiasedColor(int r, int c, Color* rayColors) {
    int valid_count = 0;
    Color neighbourSum;
    
    if(r-1 >= 0) {
        valid_count++;
        neighbourSum += rayColors[(r-1)*IMAGE_WIDTH + c];
    }
    if(r+1 < IMAGE_HEIGHT) {
        valid_count++;
        neighbourSum += rayColors[(r+1)*IMAGE_WIDTH + c];
    }
    if(c-1 >= 0) {
        valid_count++;
        neighbourSum += rayColors[r *IMAGE_WIDTH + (c-1)];
    }
    if(c+1 < IMAGE_WIDTH) {
        valid_count++;
        neighbourSum += rayColors[r *IMAGE_WIDTH + (c+1)];
    }
    
    Color final_col = neighbourSum/valid_count;
    final_col = final_col * 0.5 + rayColors[r * IMAGE_WIDTH + c] * 0.5;
    return final_col;
}

__global__
void computeRayColors(Color* rayColors,
                      Point3 leftCorner,
                      Point3 rightCorner,
                      Point3 sceneCenter,
                      int* cellStart,
                      int* cellTriangles,
                      Triangle* sceneTriangles,
                      int numTriangles,
                      int width,
                      int height,
                      SceneConstants scene,
                      Vec3 cameraForward,
                      Vec3 cameraRight,
                      Vec3 cameraUp,
                      double planeWidth,
                      double planeHeight,
                      Point3 cameraPos,
                      int boxDimension,
                      int useBruteForce) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r >= height || c >= width) return;
    int idx = r * width + c;

    float imagePlaneDistance = 1.0f;
    float u = (c + 0.5f) / width;
    float v = (r + 0.5f) / height;

    float m_x = (u - 0.5f) * planeWidth;
    float m_y = (0.5f - v) * planeHeight;
    float m_z = imagePlaneDistance;

    Point3 m = cameraPos + cameraRight * m_x + cameraUp * m_y + cameraForward * m_z;
    Vec3 dir = m - cameraPos;
    Ray ray(cameraPos, dir);
    
    HitRecord best;
    best.hit = false;

    if (useBruteForce) {
        for (int i = 0; i < numTriangles; i++) {
            HitRecord h = mollerTrumbore(ray, sceneTriangles[i]);
            if (h.hit && (!best.hit || h.distance < best.distance)) {
                best = h;
            }
        }
    } else {
        int cells[MAX_BOXDIMENSION * 3];
        int numCells = findIntersectingCubes(ray, leftCorner, rightCorner, boxDimension, cells, 3 * boxDimension);
    
        for (int ci = 0; ci < numCells; ci++) {
            HitRecord h = findIntersectingTriangle(ray, cellTriangles, sceneTriangles, cellStart, cells[ci]);
            if (h.hit && (!best.hit || h.distance < best.distance)) {
                best = h;
            }
        }
    }

    rayColors[idx] = phong(best, cameraPos, scene);
}

int main(int argc, char** argv) {
    struct timespec start, stop; 
    float time;
    if (clock_gettime(CLOCK_REALTIME, &start) == -1) {
        perror("clock gettime");
    }

    BenchmarkConfig cfg = parseArgs(argc, argv);
    if (cfg.boxDimension <= 0 || cfg.boxDimension > MAX_BOXDIMENSION) {
        std::cerr << "boxDimension must be in [1, " << MAX_BOXDIMENSION << "]\n";
        return 1;
    }

    SPH sim;
    initMarchTables();
    printf("Number of particles: %ld\n", sim.particles.size());

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

    Point3 leftCorner  = boxCorners[0];
    Point3 rightCorner = boxCorners[6];

    std::vector<unsigned char> pixels(IMAGE_WIDTH * IMAGE_HEIGHT * 3);

    vector<Point3> vertexBuffer;
    vector<uint32_t> indexBuffer;
    vector<Vec3> normalBuffer;
    Color* rayColors = new Color[IMAGE_HEIGHT * IMAGE_WIDTH];

    Point3 sceneCenter(0.0, -1.0, 0.0);
    Point3 cameraPos = sceneCenter + Vec3(8.0, 6.0, 10.0);
    double imagePlaneDistance = 1.0;
    Vec3 cameraForward = (sceneCenter - cameraPos).normalized();
    Vec3 sceneUp(0, 1, 0);
    Vec3 cameraRight = (cross(cameraForward, sceneUp)).normalized();
    Vec3 cameraUp = cross(cameraRight, cameraForward);
    double theta = PI/4;
    double planeHeight = 2 * imagePlaneDistance * tan(theta / 2);
    double planeWidth = ((double)IMAGE_WIDTH / IMAGE_HEIGHT) * planeHeight;

    SceneConstants scene = {
        Light(Point3(200.0, 100.0, 100.0), Color{1,1,1}, 1.0f),
        Color{0.05, 0.15, 0.4},
        Color{0.0, 0.0, 0.0}
    };

    Color*    d_rayColors = nullptr;
    int*      d_cellStart = nullptr;
    Triangle* d_sceneTriangles = nullptr;
    int*      d_cellTriangles  = nullptr;
    size_t    triCapacity  = 0;
    size_t    flatCapacity = 0;

    int numCells = cfg.boxDimension * cfg.boxDimension * cfg.boxDimension;

    cudaMalloc(&d_rayColors, IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(Color));
    if (!cfg.bruteForce) {
        cudaMalloc(&d_cellStart, (numCells + 1) * sizeof(int));
    }

    std::ofstream csv(cfg.csvPath, std::ios::out);
    if (!csv.is_open()) {
        std::cerr << "Failed to open benchmark CSV: " << cfg.csvPath << "\n";
        return 1;
    }
    writeCsvHeader(csv);

    if (cfg.writeFrames) {
        std::filesystem::create_directories("frames");
    }

    std::vector<FrameStats> allStats;
    allStats.reserve(cfg.frames);

    printf("Benchmark config: frames=%d substeps=%d boxDim=%d mode=%s output=%s sphere=%s csv=%s\n",
           cfg.frames,
           cfg.substeps,
           cfg.boxDimension,
           cfg.bruteForce ? "bruteforce" : "dda",
           cfg.writeFrames ? "frames" : "disabled",
           cfg.includeSphere ? "on" : "off",
           cfg.csvPath.c_str());

    for (int frame = 0; frame < cfg.frames; frame++) {
        auto frame_start = std::chrono::steady_clock::now();
        FrameStats stats;
        stats.frame = frame;

        auto physics_start = std::chrono::steady_clock::now();
        for (int s = 0; s < cfg.substeps; ++s) {
            sim.step();
        }
        sim.syncToHost();
        auto physics_end = std::chrono::steady_clock::now();
        int64_t physics_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(physics_end - physics_start).count();
        stats.physics_ms = nsToMs(physics_elapsed);
        stats.num_particles = sim.particles.size();

        vertexBuffer.clear();
        indexBuffer.clear();
        normalBuffer.clear();

        buildScalarField(sim.getParticles(),
                         (int)sim.particles.size(),
                         sim.getHashHead(),
                         sim.getHashNext());

        auto march_start = std::chrono::steady_clock::now();
        marchCubes(vertexBuffer, indexBuffer, normalBuffer);
        auto march_end = std::chrono::steady_clock::now();
        stats.march_ms = nsToMs(std::chrono::duration_cast<std::chrono::nanoseconds>(march_end - march_start).count());

        if (cfg.includeSphere) {
            generateSphere(vertexBuffer, indexBuffer, normalBuffer);
        }

        std::ofstream outFile;
        if (cfg.writeFrames) {
            outFile.open("frames/image" + std::to_string(frame) + ".ppm", std::ios::binary);
            if (outFile.is_open()) {
                outFile << "P6\n" << IMAGE_WIDTH << ' ' << IMAGE_HEIGHT << "\n255\n";
            } else {
                std::cerr << "Unable to open file\n";
            }
        }

        auto tri_construct_start = std::chrono::steady_clock::now();
        vector<Triangle> sceneTriangles = constructSceneTriangles(vertexBuffer, indexBuffer, normalBuffer);
        auto tri_construct_end = std::chrono::steady_clock::now();
        stats.tri_construct_ms = nsToMs(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    tri_construct_end - tri_construct_start).count());
        stats.num_triangles = sceneTriangles.size();

        vector<int> cellStart(numCells + 1, 0);
        vector<int> flatTriangleIdx;
        if (!cfg.bruteForce) {
            auto binning_start = std::chrono::steady_clock::now();
            vector<vector<int>> trianglesPerBox = assignTriangles(sceneTriangles, leftCorner, rightCorner, cfg.boxDimension);
            auto binning_end = std::chrono::steady_clock::now();
            stats.binning_ms = nsToMs(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    binning_end - binning_start).count());

            size_t totalAssigned = 0;
            size_t nonemptyCells = 0;
            size_t maxPerCell = 0;
            for (const auto& cell : trianglesPerBox) {
                totalAssigned += cell.size();
                if (!cell.empty()) {
                    nonemptyCells++;
                    maxPerCell = std::max(maxPerCell, cell.size());
                }
            }
            stats.flat_triangle_indices = totalAssigned;
            stats.nonempty_cells = nonemptyCells;
            stats.max_tris_per_cell = maxPerCell;
            stats.avg_tris_per_nonempty_cell = (nonemptyCells > 0)
                ? static_cast<double>(totalAssigned) / static_cast<double>(nonemptyCells)
                : 0.0;
            stats.duplication_factor = sceneTriangles.empty()
                ? 0.0
                : static_cast<double>(totalAssigned) / static_cast<double>(sceneTriangles.size());

            auto flatten_start = std::chrono::steady_clock::now();
            for (int i = 0; i < numCells; i++) {
                cellStart[i + 1] = cellStart[i] + trianglesPerBox[i].size();
            }

            flatTriangleIdx.resize(cellStart[numCells]);
            vector<int> offset = cellStart;

            for (int i = 0; i < numCells; i++) {
                for (int triIdx : trianglesPerBox[i]) {
                    if (triIdx < 0 || triIdx >= (int)sceneTriangles.size()) {
                        printf("BAD TRI IDX: %d\n", triIdx);
                        continue;
                    }
                    flatTriangleIdx[offset[i]++] = triIdx;
                }
            }
            auto flatten_end = std::chrono::steady_clock::now();
            stats.flatten_ms = nsToMs(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    flatten_end - flatten_start).count());
        } else {
            stats.flat_triangle_indices = sceneTriangles.size();
            stats.nonempty_cells = sceneTriangles.empty() ? 0 : 1;
            stats.max_tris_per_cell = sceneTriangles.size();
            stats.avg_tris_per_nonempty_cell = sceneTriangles.size();
            stats.duplication_factor = sceneTriangles.empty() ? 0.0 : 1.0;
        }

        int numTriangles = sceneTriangles.size();
        if ((size_t)numTriangles > triCapacity) {
            if (d_sceneTriangles) {
                cudaFree(d_sceneTriangles);
            }
            triCapacity = numTriangles;
            cudaMalloc(&d_sceneTriangles, triCapacity * sizeof(Triangle));
        }
        if (!cfg.bruteForce && flatTriangleIdx.size() > flatCapacity) {
            if (d_cellTriangles) {
                cudaFree(d_cellTriangles);
            }
            flatCapacity = flatTriangleIdx.size();
            cudaMalloc(&d_cellTriangles, flatCapacity * sizeof(int));
        }

        auto h2d_start = std::chrono::steady_clock::now();
        stats.h2d_bytes = numTriangles * sizeof(Triangle);
        if (!cfg.bruteForce) {
            stats.h2d_bytes +=
                (numCells + 1) * sizeof(int) +
                flatTriangleIdx.size() * sizeof(int);
        }

        cudaMemcpy(d_sceneTriangles, sceneTriangles.data(),
                   numTriangles * sizeof(Triangle),
                   cudaMemcpyHostToDevice);
        if (!cfg.bruteForce) {
            cudaMemcpy(d_cellStart, cellStart.data(),
                       (numCells + 1) * sizeof(int), cudaMemcpyHostToDevice);
            if (!flatTriangleIdx.empty()) {
                cudaMemcpy(d_cellTriangles, flatTriangleIdx.data(),
                           flatTriangleIdx.size() * sizeof(int), cudaMemcpyHostToDevice);
            }
        }
        auto h2d_end = std::chrono::steady_clock::now();
        stats.h2d_ms = nsToMs(std::chrono::duration_cast<std::chrono::nanoseconds>(h2d_end - h2d_start).count());

        dim3 blockSize(16, 16);
        dim3 gridSize((IMAGE_WIDTH + 15) / 16,
                      (IMAGE_HEIGHT + 15) / 16);

        auto ray_start = std::chrono::steady_clock::now();
        computeRayColors<<<gridSize, blockSize>>>(
            d_rayColors,
            leftCorner,
            rightCorner,
            sceneCenter,
            d_cellStart,
            d_cellTriangles,
            d_sceneTriangles,
            numTriangles,
            IMAGE_WIDTH,
            IMAGE_HEIGHT,
            scene,
            cameraForward,
            cameraRight,
            cameraUp,
            planeWidth,
            planeHeight,
            cameraPos,
            cfg.boxDimension,
            cfg.bruteForce ? 1 : 0
        );

        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        }
        auto ray_end = std::chrono::steady_clock::now();
        int64_t ray_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(ray_end - ray_start).count();
        stats.ray_ms = nsToMs(ray_elapsed);

        auto d2h_start = std::chrono::steady_clock::now();
        cudaMemcpy(rayColors, d_rayColors,
                   IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(Color),
                   cudaMemcpyDeviceToHost);
        auto d2h_end = std::chrono::steady_clock::now();
        stats.d2h_ms = nsToMs(std::chrono::duration_cast<std::chrono::nanoseconds>(d2h_end - d2h_start).count());
        stats.d2h_bytes = IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(Color);

        auto file_start = std::chrono::steady_clock::now();
        if (cfg.writeFrames) {
            for (int r = 0; r < IMAGE_HEIGHT; r++) {
                for (int c = 0; c < IMAGE_WIDTH; c++) {
                    int idx = r * IMAGE_WIDTH + c;
                    Color aa = getAntialiasedColor(r, c, rayColors);
                    pack_Color(&pixels[idx * 3], aa, true);
                }
            }
            if (outFile.is_open()) {
                outFile.write(reinterpret_cast<const char*>(pixels.data()), pixels.size());
                outFile.close();
            }
        }
        auto file_end = std::chrono::steady_clock::now();
        stats.file_ms = nsToMs(std::chrono::duration_cast<std::chrono::nanoseconds>(file_end - file_start).count());

        auto frame_end = std::chrono::steady_clock::now();
        stats.total_ms = nsToMs(std::chrono::duration_cast<std::chrono::nanoseconds>(frame_end - frame_start).count());

        writeCsvRow(csv, cfg, stats);
        allStats.push_back(stats);

        bool shouldPrintProgress =
            (frame == 0) ||
            ((frame + 1) % PROGRESS_PRINT_INTERVAL == 0) ||
            (frame + 1 == cfg.frames);
        if (shouldPrintProgress) {
            printf("Progress: frame %d/%d | total=%.3f ms/frame | physics=%.3f ms | ray=%.3f ms | tris=%zu | dup=%.3f\n",
                   frame + 1,
                   cfg.frames,
                   stats.total_ms,
                   stats.physics_ms,
                   stats.ray_ms,
                   stats.num_triangles,
                   stats.duplication_factor);
        }
    }

    cudaFree(d_rayColors);
    cudaFree(d_sceneTriangles);
    if (d_cellStart) {
        cudaFree(d_cellStart);
    }
    if (d_cellTriangles) {
        cudaFree(d_cellTriangles);
    }
    delete[] rayColors;

    if (clock_gettime(CLOCK_REALTIME, &stop) == -1) {
        perror("clock gettime");
    }
    time = (stop.tv_sec - start.tv_sec) + (float)(stop.tv_nsec - start.tv_nsec) / 1e9f;

    if (!allStats.empty()) {
        FrameStats avg;
        for (const FrameStats& s : allStats) {
            avg.physics_ms += s.physics_ms;
            avg.march_ms += s.march_ms;
            avg.tri_construct_ms += s.tri_construct_ms;
            avg.binning_ms += s.binning_ms;
            avg.flatten_ms += s.flatten_ms;
            avg.h2d_ms += s.h2d_ms;
            avg.ray_ms += s.ray_ms;
            avg.d2h_ms += s.d2h_ms;
            avg.file_ms += s.file_ms;
            avg.total_ms += s.total_ms;
            avg.avg_tris_per_nonempty_cell += s.avg_tris_per_nonempty_cell;
            avg.duplication_factor += s.duplication_factor;
        }

        double denom = static_cast<double>(allStats.size());
        avg.physics_ms /= denom;
        avg.march_ms /= denom;
        avg.tri_construct_ms /= denom;
        avg.binning_ms /= denom;
        avg.flatten_ms /= denom;
        avg.h2d_ms /= denom;
        avg.ray_ms /= denom;
        avg.d2h_ms /= denom;
        avg.file_ms /= denom;
        avg.total_ms /= denom;
        avg.avg_tris_per_nonempty_cell /= denom;
        avg.duplication_factor /= denom;

        printAverageSummary(avg, allStats.size(), cfg);
        printf("benchmark csv written to %s\n", cfg.csvPath.c_str());

        std::string summaryPath = summaryPathFromCsv(cfg.csvPath);
        writeSummaryFile(summaryPath, avg, allStats.size(), cfg, time);
        printf("benchmark summary written to %s\n", summaryPath.c_str());
    }
    
    printf("Execution time = %.6f s\n", time);
    std::cout << sizeof(Triangle) << "\n";

    return 0;
}
