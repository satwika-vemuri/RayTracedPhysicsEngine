
#include <cmath>
#include <algorithm>
#include <iostream>

#include "vec3.h"
#include "light.h"
#include "hitRecord.h"
#include "color.h"
#include "triangle.h"
#include "rayTrace.h"

using std::vector;



// Shading Constants
// X = left/right, Y = up/down, Z = depth
// Sphere is centered at (500, 500, 500). Camera is along the z axis


const Light  LIGHT(Point3(200.0f, 100.0f, 100.0f), Color{1.0f, 1.0f, 1.0f}, 1.0f);
const Color  SURFACE_COLOR  = {0.05f, 0.15f, 0.4f};  // blue water
const Color  DARK           = {0.0f, 0.0f, 0.0f};
const float  AMBIENT        = 0.3f;
const float  REFLECTIVENESS = 0.9f;
const float  SHININESS      = 60.0f;
// Reverted the name of this since MPI is used in many libraries
const double PI = 2 * acos(0.0);


//Phong shading

Color lambertian(const HitRecord& pos) {
    if (!pos.hit) return DARK;
    Vec3 N = pos.interpolatedNormal();
    Vec3 L = (LIGHT.coords() - pos.point).normalized();
    double diffuse = std::max(0.0, N.dot(L));
    return SURFACE_COLOR * LIGHT.color() * LIGHT.brightness() * diffuse;
}

Color ambient(const HitRecord& pos) {
    if (!pos.hit) return DARK;
    return SURFACE_COLOR * AMBIENT * LIGHT.brightness();
}

Color specular(const HitRecord& pos, const Point3& cameraPos) {
    if (!pos.hit) return DARK;
    Vec3 V = (cameraPos - pos.point).normalized();
    Vec3 N = pos.interpolatedNormal();
    Vec3 L = (LIGHT.coords() - pos.point).normalized();
    Vec3 R = reflection(L, N);
    return LIGHT.color() * LIGHT.brightness() * REFLECTIVENESS *
           pow(std::max(0.0, R.dot(V)), SHININESS);
}

Color phong(const HitRecord& pos, const Point3& cameraPos) {
    return specular(pos, cameraPos) + ambient(pos) + lambertian(pos);
}



// testing function, thnx AI
void generateSphere(
    vector<vector<double>>& vertexBuffer,
    vector<double>& indexBuffer,
    vector<vector<double>>& normalBuffer
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

            vertexBuffer.push_back({x,y,z});

            double nx = x - cx;
            double ny = y - cy;
            double nz = z - cz;

            double len = sqrt(nx * nx + ny * ny + nz * nz);

            normalBuffer.push_back({(nx / len * 1000.0),(ny / len * 1000.0),(nz / len * 1000.0)});
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
    std::vector<Color*> consideredColors(4); // always 5 max
    // all adjacent rays are considered
    consideredColors[0] = (r - 1 >= 0 ? &rayColors[(r - 1) * IMAGE_WIDTH + c] : nullptr);
    consideredColors[1] = (r + 1 < IMAGE_HEIGHT ? &rayColors[(r + 1) * IMAGE_WIDTH + c] : nullptr);
    consideredColors[2] = (c - 1 >= 0 ? &rayColors[r * IMAGE_WIDTH + (c - 1)] : nullptr);
    consideredColors[3] = (c + 1 < IMAGE_WIDTH ? &rayColors[r * IMAGE_WIDTH + (c + 1)] : nullptr);
    // average these with weight of main ray as .5 and weight of the sum
    // of all other rays as .5
    Color final_col;
    for (auto* color : consideredColors) {
        if (color != nullptr) final_col = final_col + *color;
    }
    final_col = final_col/4.0;
    final_col = final_col * 0.5 + rayColors[r * IMAGE_WIDTH + c] * 0.5;
    return final_col;
}


// No Phong Shading currently, I'll add it tomorrow.
int main() {
    vector<vector<double>> vertexBuffer;
    vector<double> indexBuffer;
    vector<vector<double>> normalBuffer;

    generateSphere(vertexBuffer, indexBuffer, normalBuffer);

    Point3 leftCorner(0, 0, 0);
    Point3 rightCorner(1000, 1000, 1000);

    vector<Triangle*> sceneTriangles =
        constructSceneTriangles(vertexBuffer, indexBuffer, normalBuffer);

    Point3 cameraPos = computeCameraPosition(leftCorner, rightCorner);

    TriangleGrid intersections =
        findSurfaceIntersections(cameraPos, sceneTriangles);

    std::cout << "P3\n" << IMAGE_WIDTH << ' ' << IMAGE_HEIGHT << "\n255\n";

    // Cache colors to enable anti-aliasing
    Color* rayColors = new Color[IMAGE_HEIGHT * IMAGE_WIDTH];
    for (int r = 0; r < IMAGE_HEIGHT; r++) {
        for (int c = 0; c < IMAGE_WIDTH; c++) {
            HitRecord hitRec = intersections[r][c];
            rayColors[r * IMAGE_WIDTH + c] = phong(hitRec, cameraPos);
        }
    }

    for (int r = 0; r < IMAGE_HEIGHT; r++) {
        for (int c = 0; c < IMAGE_WIDTH; c++) {
            write_Color(std::cout, getAntialiasedColor(r, c, rayColors));
        }
    }

    for (Triangle* tri : sceneTriangles) {
        delete tri;
    }

    
    delete[] rayColors;

    return 0;
}