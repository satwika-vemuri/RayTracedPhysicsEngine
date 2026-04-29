#pragma once

#include <cstdint>
#include <cstdlib>
#include <vector>

#include "triangle.h"
#include "vec3.h"
#include "ray_serial.h"
#include "hitRecord_serial.h"
#include "consts.h"


// ── Triangle construction ────────────────────────────────────────────────────

std::vector<Triangle> constructSceneTriangles(
    const std::vector<Point3>& vertexBuffer,
    const std::vector<uint32_t>& indexBuffer,
    const std::vector<Vec3>& normalBuffer);

Point3 computeCameraPosition(const Point3& leftCorner, const Point3& rightCorner);

// ── Brute-force intersection (kept for comparison benchmarks) ────────────────

HitRecord mollerTrumbore(const Ray& ray, const Triangle& tri);

HitRecord findIntersectingTriangle(const Ray& ray,
                                   const std::vector<Triangle>& sceneTriangles);

// ── AABB-based triangle-to-voxel assignment (mirrors parallel assignTriangles)

std::vector<std::vector<int>> assignTriangles(
    std::vector<Triangle>& sceneTriangles,
    const Point3& leftCorner,
    const Point3& rightCorner,
    int boxDimension);

// ── 3D DDA grid traversal (CPU equivalent of findIntersectingCubes kernel) ──

bool rayIntersectsAABB(const Ray& ray,
                       const Point3& minB, const Point3& maxB,
                       double& tmin_out, double& tmax_out);

int findIntersectingCubes(const Ray& ray,
                          const Point3& leftCorner,
                          const Point3& rightCorner,
                          int boxDim,
                          int* cells,
                          int maxCells);

// ── Per-voxel triangle lookup using CSR layout (mirrors GPU findIntersectingTriangle)

HitRecord findIntersectingTriangleInVoxel(const Ray& ray,
                                          const std::vector<int>& cellTriangles,
                                          const std::vector<Triangle>& sceneTriangles,
                                          const std::vector<int>& cellStart,
                                          int boxNumber);
