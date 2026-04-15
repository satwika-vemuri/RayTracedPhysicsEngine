#include <cstdlib>
#include <algorithm>
#include <array>
#include <vector>
#include <iostream>
#include <cmath>
#include "rt.h"

using namespace std;

const double M_PI = 2 * acos(0.0);

TriangleGrid g;

// function from chatgpt to generate a sphere
#include <vector>
#include <cmath>
using namespace std;

void generateSphere(
    vector<vector<double>>& vertexBuffer,
    vector<double>& indexBuffer,
    vector<vector<double>>& normalBuffer
) {
    int latSteps = 20;
    int lonSteps = 20;

    double cx = 500, cy = 500, cz = 500;
    double radius = 200;

    // ---- STEP 1: vertices + normals ----
    for (int i = 0; i <= latSteps; i++) {
        double theta = M_PI * i / latSteps;

        for (int j = 0; j <= lonSteps; j++) {
            double phi = 2 * M_PI * j / lonSteps;

            double x = cx + radius * sin(theta) * cos(phi);
            double y = cy + radius * sin(theta) * sin(phi);
            double z = cz + radius * cos(theta);

            // vertex
            vertexBuffer.push_back({x,y,z});

            // normal (unit vector from center)
            double nx = x - cx;
            double ny = y - cy;
            double nz = z - cz;

            double len = sqrt(nx*nx + ny*ny + nz*nz);

            normalBuffer.push_back({(nx / len * 1000),(ny / len * 1000),(nz / len * 1000)});
        }
    }

    // ---- STEP 2: indices (3 per triangle) ----
    int stride = lonSteps + 1;

    for (int i = 0; i < latSteps; i++) {
        for (int j = 0; j < lonSteps; j++) {

            int v1 = i * stride + j;
            int v2 = v1 + 1;
            int v3 = (i + 1) * stride + j;
            int v4 = v3 + 1;

            // triangle 1
            indexBuffer.push_back(v1);
            indexBuffer.push_back(v3);
            indexBuffer.push_back(v2);

            // triangle 2
            indexBuffer.push_back(v2);
            indexBuffer.push_back(v3);
            indexBuffer.push_back(v4);
        }
    }
}

int main(){
    vector<vector<double>> vertexBuffer;
    vector<double> indexBuffer;
    vector<vector<double>> normalBuffer;

    // Generate test sphere
    generateSphere(vertexBuffer, indexBuffer, normalBuffer);

    // Bounding box (make sure sphere fits inside)
    Point3 leftCorner(0, 0, 0);
    Point3 rightCorner(1000, 1000, 1000);

    vector<Triangle*> sceneTriangles =
        constructSceneTriangles(vertexBuffer, indexBuffer, normalBuffer);

    Point3* cameraPos = computeCameraPosition(leftCorner, rightCorner);
    cout << flush;

    TriangleGrid* intersections = findSurfaceIntersections(cameraPos, sceneTriangles);
    // TODO Alan
    /*
    intersections is a 2D array (800x600) with an intersection object per pixel
    You can take this output and feed it into your code
    The output you see in the console is a small subset of the intersection points for some of the pixels
    */

    return 1;
}

