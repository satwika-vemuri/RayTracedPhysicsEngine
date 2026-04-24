
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


#include "Vec3.h"
#include "rayTrace_gpu.h"
#include "color.h"
#include "March.h"
#include "SPH.h"

void initMarchTables();
void buildScalarField(Particle* d_particles,
                      int numParticles,
                      int* hashHead,
                      int* hashNext);

#define FRAMES 20
#define BOXDIMENSION 64

using std::vector;


//Phong shading


__device__
Color phong(const HitRecord& pos,
            const Point3& cameraPos,
            SceneConstants scene) {

    if (!pos.hit) return scene.dark;

    Vec3 N = pos.interpolatedNormal();
    Vec3 L = (scene.light.coords() - pos.point).normalized();
    Vec3 V = (cameraPos - pos.point).normalized();
    Vec3 R = reflect(L, N);

    double diffuse = fmax(0.0, N.dot(L));
    double spec    = pow(fmax(0.0, R.dot(V)), SHININESS);

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


Color getAntialiasedColor(int r, int c, Color* rayColors) {
    int valid_count = 0;
    Color neighbourSum;
    
    // all adjacent rays are considered
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
    
    // average these with weight of main ray as .5 and weight of the sum
    // of all other rays as .5
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
                      int height, SceneConstants scene,
                      Vec3 cameraForward,
                      Vec3 cameraRight,
                      Vec3 cameraUp,
                      double planeWidth,
                      double planeHeight,
                      Point3 cameraPos) {
    



    // compute r & c as a function of the thread and block idx
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r >= height || c >= width) return;
    int idx = r * width + c;

    // variables for ray tracing
    // FOR CAMERA AT UPPER RIGHT

    double imagePlaneDistance = 1;


    double u = (c + 0.5) / width;
    double v = (r + 0.5) / height;

    double m_x = (u - 0.5) * planeWidth;
    double m_y = (0.5 - v) * planeHeight;
    double m_z = imagePlaneDistance;



    // camera at top right
    Point3 m = cameraPos + cameraRight * m_x + cameraUp * m_y + cameraForward * m_z;
    Vec3 dir = m - cameraPos;
    Ray ray(cameraPos, dir);
    
    // ex: rightCorner = (1000, 1000, 1000), sceneCenter = (500, 500, 500), calculate leftCorner to be (0, 0, 0)

    // new version with 3D DDA
    int cells[BOXDIMENSION * 3];
    int numCells = findIntersectingCubes(ray, leftCorner, rightCorner, BOXDIMENSION, cells, 3*BOXDIMENSION);
    
    HitRecord best;
    best.hit = false;
    
    for (int ci = 0; ci < numCells; ci++) {
        HitRecord h = findIntersectingTriangle(ray, cellTriangles, sceneTriangles, cellStart, cells[ci]);
        if (h.hit && (!best.hit || h.distance < best.distance)) {
            best = h;
        }
    }
    rayColors[idx] = phong(best, cameraPos, scene);
    return;
}



int main() {
    // time constants
    struct timespec start, stop; 
    double time;
    if( clock_gettime(CLOCK_REALTIME, &start) == -1) { perror("clock gettime");}
    
    SPH sim;
    initMarchTables();
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

    Point3 leftCorner  = boxCorners[0];   // {BMIN, BMIN, BMIN}
    Point3 rightCorner = boxCorners[6];   // {BMAX, BMAX, BMAX}

    // num sub-steps per rendered frame
    constexpr int SUBSTEPS = 20;


    vector<Point3> vertexBuffer;
    vector<uint32_t> indexBuffer;
    vector<Vec3> normalBuffer;
    Color* rayColors = new Color[IMAGE_HEIGHT * IMAGE_WIDTH];

    Point3 sceneCenter(0.0, -1.0, 0.0);
    Point3 cameraPos = sceneCenter + Vec3(8.0, 6.0, 10.0);
    
    double imagePlaneDistance = 1.0;
    
    Vec3 cameraForward = (sceneCenter - cameraPos).normalized(); // points the camera to the center of the scene
    Vec3   sceneUp(0, 1, 0);
    Vec3 cameraRight = (cross(cameraForward, sceneUp)).normalized(); 
   
    Vec3 cameraUp = cross(cameraRight, cameraForward);
    double theta = PI/4;
    double planeHeight = 2 * imagePlaneDistance * tan(theta/2);
    double planeWidth = ((double)IMAGE_WIDTH / IMAGE_HEIGHT) * planeHeight;


    SceneConstants scene = {
        Light(Point3(200.0, 100.0, 100.0), Color{1,1,1}, 1.0f),
        Color{0.05, 0.15, 0.4},
        Color{0.0, 0.0, 0.0}
    };

    
    Color*    d_rayColors;
    int*      d_cellStart;
    Triangle* d_sceneTriangles = nullptr;
    int*      d_cellTriangles  = nullptr;
    size_t    triCapacity  = 0;
    size_t    flatCapacity = 0;


    int numCells = BOXDIMENSION * BOXDIMENSION * BOXDIMENSION;

    cudaMalloc(&d_rayColors, IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(Color));
    cudaMalloc(&d_cellStart, (numCells + 1) * sizeof(int));

    for(int frame = 0; frame < FRAMES; frame++){
        printf("FRAME: #%d\n", frame);

        auto physics_start = std::chrono::steady_clock::now();
        for (int s = 0; s < SUBSTEPS; ++s) sim.step();

        sim.syncToHost(); // important!

        auto physics_end = std::chrono::steady_clock::now();
        int64_t physics_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(physics_end - physics_start).count();
        printf("\tdone with the physics\n");
        printf("\tphysics benchmarking:\n\t\t%ld ns elapsed\n\t\t%ld particles simulated\n\t\t%d substeps\n\t\t~%ld ns/sim step\n",
            physics_elapsed, sim.particles.size(), SUBSTEPS, physics_elapsed/SUBSTEPS);

        // clear buffers for every new frame
	    vertexBuffer.clear();
	    indexBuffer.clear();
        normalBuffer.clear();

	    buildScalarField(sim.getParticles(),
                         (int)sim.particles.size(),
                         sim.getHashHead(),
                         sim.getHashNext());
	    marchCubes(vertexBuffer, indexBuffer, normalBuffer);
        printf("\tbuffers created\n");

        // file output information
        std::ofstream outFile("frames/image" + std::to_string(frame) + ".ppm");
        if (outFile.is_open()) {
            outFile << "P3\n" << IMAGE_WIDTH << ' ' << IMAGE_HEIGHT << "\n255\n";
        } else {
            std::cerr << "Unable to open file";
        }

        // parse buffer data
        vector<Triangle> sceneTriangles =
            constructSceneTriangles(vertexBuffer, indexBuffer, normalBuffer);
        vector<vector<int>> trianglesPerBox = assignTriangles(sceneTriangles, leftCorner, rightCorner, BOXDIMENSION);

        ////// Convert above array into flat array for Cuda ///////
        vector<int> cellStart(numCells + 1, 0);
        vector<int> flatTriangleIdx;

        // build offsets
        for (int i = 0; i < numCells; i++) {
            cellStart[i + 1] = cellStart[i] + trianglesPerBox[i].size();
        }

        flatTriangleIdx.resize(cellStart[numCells]);

        vector<int> offset = cellStart;

        // fill
        for (int i = 0; i < numCells; i++) {
            for (int triIdx : trianglesPerBox[i]) {

                if (triIdx < 0 || triIdx >= (int)sceneTriangles.size()) {
                    printf("BAD TRI IDX: %d\n", triIdx);
                    continue;
                }

                flatTriangleIdx[offset[i]++] = triIdx;
            }
        }
        //////////////////////////////////////////////////////////////

        /*
        CUDA PARALLELIZED SECTION START
        */ 

        // cuda malloc and cuda free are very expensive minimize use
        int numTriangles = sceneTriangles.size();
        if ((size_t)numTriangles > triCapacity) {
            if (d_sceneTriangles) {cudaFree(d_sceneTriangles);}
            triCapacity = numTriangles;
            cudaMalloc(&d_sceneTriangles, triCapacity * sizeof(Triangle));
        }
        if (flatTriangleIdx.size() > flatCapacity) {
            if (d_cellTriangles) cudaFree(d_cellTriangles);
            flatCapacity = flatTriangleIdx.size();
            cudaMalloc(&d_cellTriangles, flatCapacity * sizeof(int));
        }


        // copy
        cudaMemcpy(d_sceneTriangles, sceneTriangles.data(),
                numTriangles * sizeof(Triangle),
                cudaMemcpyHostToDevice);
        cudaMemcpy(d_cellStart, cellStart.data(),
                (numCells + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_cellTriangles, flatTriangleIdx.data(),
                flatTriangleIdx.size() * sizeof(int), cudaMemcpyHostToDevice);


        // launch config
        dim3 blockSize(16, 16);
        dim3 gridSize((IMAGE_WIDTH + 15) / 16,
                    (IMAGE_HEIGHT + 15) / 16);



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
            cameraPos
        );

        cudaError_t err = cudaDeviceSynchronize();  // <-- THIS catches runtime errors
        if (err != cudaSuccess) {
            std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        }

        std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;

        // copy back
        cudaMemcpy(rayColors, d_rayColors,
                IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(Color),
                cudaMemcpyDeviceToHost);
        
        /*
        CUDA PARALLELIZED SECTION END
        */ 

        for (int r = 0; r < IMAGE_HEIGHT; r++) {
            for (int c = 0; c < IMAGE_WIDTH; c++) {
                write_Color(outFile, getAntialiasedColor(r, c, rayColors), true);
            }
        }

        outFile.close();
        printf("\tImage rendered\n");
        printf("Complete.\n");

    }

    cudaFree(d_rayColors);
    cudaFree(d_sceneTriangles);
    cudaFree(d_cellStart);
    cudaFree(d_cellTriangles);
    
    
    delete[] rayColors;

    if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror("clock gettime");}		
    time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;
    
    printf("Execution time = %f sec\n",time);	

    return 0;
}
