
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


#include "rayTrace_gpu.h"
#include "color.h"
#include "March.h"
#include "SPH.h"

#define FRAMES 20
#define BOXDIMENSION 2

using std::vector;



// Sphere is centered at (500, 500, 500). Camera is along the z axis
//Phong shading

Color ambient(const HitRecord& pos, SceneConstants scene) {
    if (!pos.hit) return scene.dark;
    return scene.surfaceColor * AMBIENT * scene.light.brightness();
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


// testing function, thnx AI, slightly improved with AlanAI
void generateSphere(
    vector<Point3>& vertexBuffer,
    vector<uint32_t>& indexBuffer,
    vector<Vec3>& normalBuffer
) {
    int latSteps = 20;
    int lonSteps = 20;

    double cx = 500.0, cy = 500.0, cz = 500.0;
    double radius = 200.0;

    for (int i = 0; i <= latSteps; i++) {
        double theta = PI * i / latSteps;

        for (int j = 0; j <= lonSteps; j++) {
            double phi = 2.0 * PI * j / lonSteps;

            double x = cx + radius * sin(theta) * cos(phi);
            double y = cy + radius * sin(theta) * sin(phi);
            double z = cz + radius * cos(theta);

            vertexBuffer.push_back(Point3{x,y,z});

            double nx = x - cx;
            double ny = y - cy;
            double nz = z - cz;

            double len = sqrt(nx * nx + ny * ny + nz * nz);
            

            normalBuffer.push_back(Vec3{nx, ny, nz}/len);
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

// testing function part 2 with parameter to create a video
void generateSphere(
    vector<Point3>& vertexBuffer,
    vector<uint32_t>& indexBuffer,
    vector<Vec3>& normalBuffer,
    int frame
) {
    int latSteps = 20;
    int lonSteps = 20;

    double cx = 500.0;
    double baseCy = 500.0;
    double cz = 500.0;
    double radius = 200.0;

    // move down each frame
    double cy = baseCy + frame * -5; 

    for (int i = 0; i <= latSteps; i++) {
        double theta = PI * i / latSteps;

        for (int j = 0; j <= lonSteps; j++) {
            double phi = 2.0 * PI * j / lonSteps;

            double x = cx + radius * sin(theta) * cos(phi);
            double y = cy + radius * sin(theta) * sin(phi);
            double z = cz + radius * cos(theta);

            vertexBuffer.push_back(Point3{x,y,z});

            double nx = x - cx;
            double ny = y - cy;
            double nz = z - cz;

            double len = sqrt(nx*nx + ny*ny + nz*nz);
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
                      int height, SceneConstants scene) {
    



    // compute r & c as a function of the thread and block idx
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r >= height || c >= width) return;
    int idx = r * width + c;

    // variables for ray tracing
    // FOR CAMERA AT UPPER RIGHT
    Point3 cameraPos = rightCorner; 

    double imagePlaneDistance = 1;
    double theta = PI/4;
    double planeHeight = 2 * imagePlaneDistance * tan(theta/2);
    double planeWidth = ((double)IMAGE_WIDTH / IMAGE_HEIGHT) * planeHeight;

    double u = (c + 0.5) / IMAGE_WIDTH;
    double v = (r + 0.5) / IMAGE_HEIGHT;

    double m_x = (u - 0.5) * planeWidth;
    double m_y = (0.5 - v) * planeHeight;
    double m_z = imagePlaneDistance;

    // redefine camera axises
    Vec3 cameraForward = (sceneCenter - cameraPos).normalized(); // points the camera to the center of the scene
    Vec3 sceneUp(0, 1, 0);
    Vec3 cameraRight = (cross(cameraForward, sceneUp)).normalized(); 
    Vec3 cameraUp = cross(cameraRight, cameraForward);

    camera at top right
    Point3 m = cameraPos + cameraRight * m_x + cameraUp * m_y + cameraForward * m_z;
    Vec3 dir = m - cameraPos;
    Ray ray(cameraPos, dir);
    
    // ex: rightCorner = (1000, 1000, 1000), sceneCenter = (500, 500, 500), calculate leftCorner to be (0, 0, 0)

    // Original version
    HitRecord best;
    best.hit = false;

    for (int i = 0; i < numTriangles; i++) {
        HitRecord h = mollerTrumbore(ray, sceneTriangles[i]);

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
    Point3 leftCorner  = boxCorners[0];   // {BMIN, BMIN, BMIN}
    Point3 rightCorner = boxCorners[6];   // {BMAX, BMAX, BMAX}

    // num sub-steps per rendered frame
    constexpr int SUBSTEPS = 20;


    vector<Point3> vertexBuffer;
    vector<uint32_t> indexBuffer;
    vector<Vec3> normalBuffer;
    Color* rayColors = new Color[IMAGE_HEIGHT * IMAGE_WIDTH];

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

	    buildScalarField(sim.particles);
	    marchCubes(vertexBuffer, indexBuffer, vertexBuffer);
        printf("\tbuffers created\n");

        // file output information
        std::ofstream outFile("frames/image" + std::to_string(frame) + ".ppm");
        if (outFile.is_open()) {
            outFile << "P3\n" << IMAGE_WIDTH << ' ' << IMAGE_HEIGHT << "\n255\n";
        } else {
            std::cerr << "Unable to open file";
        }

        // fill buffers with data from test function
        generateSphere(vertexBuffer, indexBuffer, normalBuffer, frame);
        
        // place camera
        Point3 sceneCenter = leftCorner + ((rightCorner-leftCorner)/2);//TODO RN computeCameraPosition(leftCorner, rightCorner);


        // parse buffer data
        vector<Triangle> sceneTriangles =
            constructSceneTriangles(vertexBuffer, indexBuffer, normalBuffer);
        
        vector<vector<int>> trianglesPerBox = assignTriangles(sceneTriangles, leftCorner, rightCorner, BOXDIMENSION);

        ////// Convert above array into flat array for Cuda ///////
        int numCells = BOXDIMENSION * BOXDIMENSION * BOXDIMENSION;

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
        int numTriangles = sceneTriangles.size();
        Color* d_rayColors;
        Triangle* d_sceneTriangles;
        int* d_cellStart;
        int* d_cellTriangles;

        // allocate
        cudaMalloc(&d_rayColors, IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(Color));
        cudaMalloc(&d_sceneTriangles, numTriangles * sizeof(Triangle));
        cudaMalloc(&d_cellStart, (numCells + 1) * sizeof(int));
        cudaMalloc(&d_cellTriangles, flatTriangleIdx.size() * sizeof(int));


        // copy
        cudaMemcpy(d_sceneTriangles, sceneTriangles.data(),
                numTriangles * sizeof(Triangle),
                cudaMemcpyHostToDevice);
        cudaMemcpy(d_cellStart, cellStart.data(),
                (numCells + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_cellTriangles, flatTriangleIdx.data(),
                flatTriangleIdx.size() * sizeof(int), cudaMemcpyHostToDevice);


        // declare constants
        SceneConstants scene = {
            Light(Point3(200.0, 100.0, 100.0), Color{1,1,1}, 1.0f),
            Color{0.05, 0.15, 0.4},
            Color{0.0, 0.0, 0.0}
        };

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
            scene
        );

        cudaError_t err = cudaDeviceSynchronize();  // <-- THIS catches runtime errors
        if (err != cudaSuccess) {
            std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        }

        cudaDeviceSynchronize();
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
    
    
    delete[] rayColors;

    if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror("clock gettime");}		
    time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;
    
    printf("Execution time = %f sec\n",time);	

    return 0;
}