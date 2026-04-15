#pragma once

#include <cstdlib>
#include <algorithm>
#include <array>
#include <vector>

#include "triangle.h"
#include "vec3.h"


constexpr int IMAGE_WIDTH = 600;
constexpr int IMAGE_HEIGHT = 800;

struct Intersection {
    Triangle* tri;
    Point3 intersection;

    Intersection() : tri(nullptr), intersection() {}  
    Intersection(Triangle* t, Point3 p) : tri(t), intersection(p) {}
};

using TriangleGrid = std::array<std::array<Intersection*, IMAGE_WIDTH>, IMAGE_HEIGHT>;


struct HitInfo {
    bool hit;
    Point3 point;
    double distance;

    HitInfo(bool h, Point3 p, double d) : hit(h), point(p), distance(d) {}
    HitInfo() : hit(false), point(), distance(-1.0) {}
};

std::vector<Triangle*> constructSceneTriangles(std::vector<std::vector<double>>& vertexBuffer, 
                                                std::vector<double>& indexBuffer, 
                                                std::vector<std::vector<double>>& normalBuffer);

Point3* computeCameraPosition(Point3 leftCorner, Point3 rightCorner);
TriangleGrid* findSurfaceIntersections(Point3* cameraPos, std::vector<Triangle*>& sceneTriangles);
