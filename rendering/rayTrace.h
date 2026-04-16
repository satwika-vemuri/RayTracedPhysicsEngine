#pragma once

#include <cstdlib>
#include <algorithm>
#include <array>
#include <vector>

#include "triangle.h"
#include "vec3.h"
#include "hitRecord.h"


constexpr int IMAGE_WIDTH = 600;
constexpr int IMAGE_HEIGHT = 800;


using TriangleGrid = std::array<std::array<HitRecord, IMAGE_WIDTH>, IMAGE_HEIGHT>;

std::vector<Triangle*> constructSceneTriangles(std::vector<std::vector<double>>& vertexBuffer, 
                                                std::vector<double>& indexBuffer, 
                                                std::vector<std::vector<double>>& normalBuffer);

Point3 computeCameraPosition(Point3 leftCorner, Point3 rightCorner);
TriangleGrid findSurfaceIntersections(const Point3& cameraPos, const std::vector<Triangle*>& sceneTriangles);
