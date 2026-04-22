
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <string>
#include <cuda_runtime.h>


#include "rayTrace_gpu.h"
#include "color.h"



#define FRAMES 50

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
    Vec3 R = reflection(L, N);

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

__device__
Color colorPixel(int r, int c,
                 const Point3& cameraPos, const Point3& sceneCenter,
                 const Triangle* sceneTriangles,
                 int numTriangles, SceneConstants scene) {

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
    Vec3 cameraUp    = cross(cameraRight, cameraForward);

    Point3 m = cameraPos + cameraRight * m_x + cameraUp * m_y + cameraForward * m_z;

    Vec3 dir = m - cameraPos;

    Ray ray(cameraPos, dir);

    HitRecord h = findIntersectingTriangle(ray, sceneTriangles, numTriangles);
    return phong(h, cameraPos, scene);
}

__global__
void computeRayColors(Color* rayColors,
                      Point3 cameraPos,
                      Point3 sceneCenter,
                      Triangle* sceneTriangles,
                      int numTriangles,
                      int width,
                      int height, SceneConstants scene) {

    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    if (r >= height || c >= width) return;

    int idx = r * width + c;

    rayColors[idx] = colorPixel(r, c, cameraPos, sceneCenter, sceneTriangles, numTriangles, scene);
}



int main() {
    vector<Point3> vertexBuffer;
    vector<uint32_t> indexBuffer;
    vector<Vec3> normalBuffer;
    Color* rayColors = new Color[IMAGE_HEIGHT * IMAGE_WIDTH];

    for(int frame = 0; frame < FRAMES; frame++){

        // clear buffers for every new frame
        vertexBuffer.clear();
        indexBuffer.clear();
        normalBuffer.clear();

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
        Point3 leftCorner(0, 0, 0);
        Point3 rightCorner(1000, 1000, 1000);
        Point3 sceneCenter = leftCorner + ((rightCorner-leftCorner)/2);//TODO RN computeCameraPosition(leftCorner, rightCorner);


        // parse buffer data
        vector<Triangle> sceneTriangles =
            constructSceneTriangles(vertexBuffer, indexBuffer, normalBuffer);

        /*
        CUDA PARALLELIZED SECTION START
        */ 
        int numTriangles = sceneTriangles.size();
        Color* d_rayColors;
        Triangle* d_sceneTriangles;

        // allocate
        cudaMalloc(&d_rayColors, IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(Color));
        cudaMalloc(&d_sceneTriangles, numTriangles * sizeof(Triangle));

        // copy triangles to GPU
        cudaMemcpy(d_sceneTriangles, sceneTriangles.data(),
                numTriangles * sizeof(Triangle),
                cudaMemcpyHostToDevice);

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
            rightCorner,
            sceneCenter,
            d_sceneTriangles,
            numTriangles,
            IMAGE_WIDTH,
            IMAGE_HEIGHT,
            scene
        );

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cout << "CUDA launch error: "
                    << cudaGetErrorString(err) << std::endl;
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

    }
    
    
    delete[] rayColors;

    return 0;
}