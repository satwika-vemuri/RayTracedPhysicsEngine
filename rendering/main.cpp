
#include <cmath>
#include <algorithm>
#include <iostream>

#include "vec3.h"
#include "light.h"
#include "hitRecord.h"
#include "color.h"
#include "triangle.h"

using std::vector;



// Shading Constants
// X = left/right, Y = up/down, Z = depth
// Sphere is centered at (500, 500, 500). Camera is along the z axis
const Point3 CAMERAPOS(500.0f, 500.0f, -1500.0f);

const Light  LIGHT(Point3(200.0f, -800.0f, 900.0f), Color{1.0f, 1.0f, 1.0f}, 1.0f);
const Color  SURFACE_COLOR  = {0.05f, 0.15f, 0.4f};  // blue water
const Color  DARK           = {0.0f, 0.0f, 0.0f};
const float  AMBIENT        = 0.3f;
const float  REFLECTIVENESS = 0.9f;
const float  SHININESS      = 60.0f;

//Phong shading

Color lambertian(const HitRecord& pos) {
    if (!pos.is_hit()) return DARK;
    Vec3 N = pos.normal().normalized();
    Vec3 L = (LIGHT.coords() - pos.coords()).normalized();
    float diffuse = std::max(0.0, N.dot(L));
    return SURFACE_COLOR * LIGHT.color() * LIGHT.brightness() * diffuse;
}

Color ambient(const HitRecord& pos) {
    if (!pos.is_hit()) return DARK;
    return SURFACE_COLOR * AMBIENT * LIGHT.brightness();
}

Color specular(const HitRecord& pos) {
    if (!pos.is_hit()) return DARK;
    Vec3 V = (CAMERAPOS - pos.coords()).normalized();
    Vec3 N = pos.normal().normalized();
    Vec3 L = (LIGHT.coords() - pos.coords()).normalized();
    Vec3 R = reflection(L, N);
    return LIGHT.color() * LIGHT.brightness() * REFLECTIVENESS *
           pow(std::max(0.0, R.dot(V)), SHININESS);
}

Color phong(const HitRecord& pos) {
    return specular(pos) + ambient(pos) + lambertian(pos);
}



// testing function, thnx AI
void generateSphere(
    vector<vector<int>>& vertexBuffer,
    vector<int>& indexBuffer,
    vector<vector<int>>& normalBuffer
) {
    int latSteps = 20;
    int lonSteps = 20;

    double cx = 500.0, cy = 500.0, cz = 500.0;
    double radius = 200.0;

    for (int i = 0; i <= latSteps; i++) {
        double theta = M_PI * i / latSteps;

        for (int j = 0; j <= lonSteps; j++) {
            double phi = 2.0 * M_PI * j / lonSteps;

            double x = cx + radius * sin(theta) * cos(phi);
            double y = cy + radius * sin(theta) * sin(phi);
            double z = cz + radius * cos(theta);

            vertexBuffer.push_back({
                (int)x,
                (int)y,
                (int)z
            });

            double nx = x - cx;
            double ny = y - cy;
            double nz = z - cz;

            double len = sqrt(nx * nx + ny * ny + nz * nz);

            normalBuffer.push_back({
                (int)(nx / len * 1000.0),
                (int)(ny / len * 1000.0),
                (int)(nz / len * 1000.0)
            });
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


// No Phong Shading currently, I'll add it tomorrow.
int main() {
    vector<vector<int>> vertexBuffer;
    vector<int> indexBuffer;
    vector<vector<int>> normalBuffer;

    generateSphere(vertexBuffer, indexBuffer, normalBuffer);

    Point3 leftCorner(0, 0, 0);
    Point3 rightCorner(1000, 1000, 1000);

    vector<Triangle*> sceneTriangles =
        constructSceneTriangles(vertexBuffer, indexBuffer, normalBuffer);

    Point3* cameraPos = computeCameraPosition(leftCorner, rightCorner);

    TriangleGrid* intersections =
        findSurfaceIntersections(cameraPos, sceneTriangles);

    std::cout << "P3\n" << IMAGE_WIDTH << ' ' << IMAGE_HEIGHT << "\n255\n";

    for (int r = 0; r < IMAGE_HEIGHT; r++) {
        for (int c = 0; c < IMAGE_WIDTH; c++) {
            Intersection* hit = (*intersections)[r][c];
            HitRecord rec = HitRecord::toHitRecord(hit);
            write_Color(std::cout, phong(rec));
        }
    }

    delete cameraPos;

    for (int r = 0; r < IMAGE_HEIGHT; r++) {
        for (int c = 0; c < IMAGE_WIDTH; c++) {
            delete (*intersections)[r][c];
        }
    }
    delete intersections;

    for (Triangle* tri : sceneTriangles) {
        delete tri;
    }

    return 0;
}