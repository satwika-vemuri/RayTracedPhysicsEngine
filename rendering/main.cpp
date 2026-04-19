
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <string>

#include "vec3.h"
#include "light.h"
#include "hitRecord.h"
#include "color.h"
#include "triangle.h"
#include "rayTrace.h"
#include "consts.h"


#define FRAMES 50

using std::vector;



// Sphere is centered at (500, 500, 500). Camera is along the z axis
//Phong shading

Color ambient(const HitRecord& pos) {
    if (!pos.hit) return DARK;
    return SURFACE_COLOR * AMBIENT * LIGHT.brightness();
}

Color phong(const HitRecord& pos, const Point3& cameraPos) {
    if (!pos.hit) return DARK;

    Vec3 N = pos.interpolatedNormal();
    Vec3 L = (LIGHT.coords() - pos.point).normalized();
    Vec3 V = (cameraPos - pos.point).normalized();
    Vec3 R = reflection(L, N);

    double diffuse = std::max(0.0, N.dot(L));
    double spec = std::pow(std::max(0.0, R.dot(V)), SHININESS);

    Color ambientTerm = SURFACE_COLOR * AMBIENT * LIGHT.brightness();
    Color diffuseTerm = SURFACE_COLOR * LIGHT.color() * LIGHT.brightness() * diffuse;
    Color specularTerm = LIGHT.color() * LIGHT.brightness() * REFLECTIVENESS * spec;

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

int main() {
    vector<Point3> vertexBuffer;
    vector<uint32_t> indexBuffer;
    vector<Vec3> normalBuffer;
    Color* rayColors = new Color[IMAGE_HEIGHT * IMAGE_WIDTH];

    for(int frame = 0; frame < FRAMES; frame++){

        vertexBuffer.clear();
        indexBuffer.clear();
        normalBuffer.clear();

        // file output information
        std::ofstream outFile("frames/image" + std::to_string(frame) + ".ppm");

        generateSphere(vertexBuffer, indexBuffer, normalBuffer, frame);

        Point3 leftCorner(0, 0, 0);
        Point3 rightCorner(1000, 1000, 1000);

        vector<Triangle> sceneTriangles =
            constructSceneTriangles(vertexBuffer, indexBuffer, normalBuffer);

        Point3 cameraPos = computeCameraPosition(leftCorner, rightCorner);

        TriangleGrid intersections =
            findSurfaceIntersections(cameraPos, sceneTriangles);

        
        if (outFile.is_open()) {
            outFile << "P3\n" << IMAGE_WIDTH << ' ' << IMAGE_HEIGHT << "\n255\n";
        } else {
            std::cerr << "Unable to open file";
        }

        // Cache colors to enable anti-aliasing
        
        for (int r = 0; r < IMAGE_HEIGHT; r++) {
            for (int c = 0; c < IMAGE_WIDTH; c++) {
                HitRecord hitRec = intersections[r*IMAGE_WIDTH + c];
                rayColors[r * IMAGE_WIDTH + c] = phong(hitRec, cameraPos);
            }
        }

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