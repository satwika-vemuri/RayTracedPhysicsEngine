#pragma once

#include <cstdint>
#include <cstdlib>
#include <vector>

#include "triangle.h"
#include "vec3.h"
#include "ray_serial.h"
#include "hitRecord_serial.h"





using TriangleGrid = std::vector<HitRecord>;

std::vector<Triangle> constructSceneTriangles( const std::vector<Point3>& vertexBuffer, 
                                                const std::vector<uint32_t>& indexBuffer, 
                                                const std::vector<Vec3>& normalBuffer);

Point3 computeCameraPosition(const Point3& leftCorner,const Point3& rightCorner);
HitRecord findIntersectingTriangle(const Ray& ray, const std::vector<Triangle>& sceneTriangles);

