#pragma once

#include <cstdint>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

#include "triangle.h"
#include "vec3.h"
#include "ray.h"
#include "hitRecord.h"
#include "consts.h"

// Host functions - implemented in src/parallel/rayTrace.cu
std::vector<std::vector<int>> assignTriangles(
    std::vector<Triangle>& sceneTriangles,
    const Point3& leftCorner,
    const Point3& rightCorner,
    int boxDimension
);

std::vector<Triangle> constructSceneTriangles(
    const std::vector<Point3>& vertexBuffer,
    const std::vector<uint32_t>& indexBuffer,
    const std::vector<Vec3>& normalBuffer);

Point3 computeCameraPosition(const Point3& leftCorner, const Point3& rightCorner);

// Device-side Moller-Trumbore intersection
inline __device__
HitRecord mollerTrumbore(const Ray& ray, const Triangle& tri) {
    const double EPSILON = 1e-8;

    double e1x = tri.v2.x - tri.v1.x;
    double e1y = tri.v2.y - tri.v1.y;
    double e1z = tri.v2.z - tri.v1.z;

    double e2x = tri.v3.x - tri.v1.x;
    double e2y = tri.v3.y - tri.v1.y;
    double e2z = tri.v3.z - tri.v1.z;

    double pvecX = ray.direction.y * e2z - ray.direction.z * e2y;
    double pvecY = ray.direction.z * e2x - ray.direction.x * e2z;
    double pvecZ = ray.direction.x * e2y - ray.direction.y * e2x;

    double det = e1x * pvecX + e1y * pvecY + e1z * pvecZ;
    if (std::fabs(det) < EPSILON) return HitRecord();

    double invDet = 1.0 / det;

    double tvecX = ray.origin.x - tri.v1.x;
    double tvecY = ray.origin.y - tri.v1.y;
    double tvecZ = ray.origin.z - tri.v1.z;

    double u = (tvecX * pvecX + tvecY * pvecY + tvecZ * pvecZ) * invDet;
    if (u < 0.0 || u > 1.0) return HitRecord();

    double qvecX = tvecY * e1z - tvecZ * e1y;
    double qvecY = tvecZ * e1x - tvecX * e1z;
    double qvecZ = tvecX * e1y - tvecY * e1x;

    double baryV = (ray.direction.x * qvecX + ray.direction.y * qvecY + ray.direction.z * qvecZ) * invDet;
    if (baryV < 0.0 || u + baryV > 1.0) return HitRecord();

    double rayT = (e2x * qvecX + e2y * qvecY + e2z * qvecZ) * invDet;
    if (rayT < EPSILON) return HitRecord();

    Point3 intersection(
        ray.origin.x + rayT * ray.direction.x,
        ray.origin.y + rayT * ray.direction.y,
        ray.origin.z + rayT * ray.direction.z);

    return HitRecord(&tri, intersection, rayT, u, baryV);
}

// Device-side triangle search (raw pointer + count for GPU)
inline __device__
HitRecord findIntersectingTriangle(const Ray& ray,
                                   int* cellTriangles,
                                   Triangle* sceneTriangles,
                                   int* cellStart,
                                   int boxNumber) {
    double closestDistance = INFINITY;
    HitRecord closestHit;
    closestHit.hit = false;
    int start = cellStart[boxNumber];
    int end   = cellStart[boxNumber + 1];

    // go through all triangles for appropriate box
    for(int i = start; i < end; i++) {
        int triIdx = cellTriangles[i];
        Triangle tri = sceneTriangles[triIdx];

        HitRecord hit = mollerTrumbore(ray, tri);
        if (hit.hit && hit.distance < closestDistance) {
            closestDistance = hit.distance;
            closestHit = hit;
        }
    }

    return closestHit;
}

inline __device__
bool rayIntersectsAABB(const Ray& ray,
                       const Point3& minB, const Point3& maxB,
                       float& tmin_out, float& tmax_out)
{
    float tmin = -INFINITY;
    float tmax =  INFINITY;

    // X axis
    if (fabsf(ray.direction.x) < 1e-8f) {
        if (ray.origin.x < minB.x || ray.origin.x > maxB.x) return false;
    } else {
        float t1 = (minB.x - ray.origin.x) / ray.direction.x;
        float t2 = (maxB.x - ray.origin.x) / ray.direction.x;
        if (t1 > t2) { float tmp = t1; t1 = t2; t2 = tmp; }
        tmin = fmaxf(tmin, t1);
        tmax = fminf(tmax, t2);
        if (tmax < tmin) return false;
    }

    // Y axis
    if (fabsf(ray.direction.y) < 1e-8f) {
        if (ray.origin.y < minB.y || ray.origin.y > maxB.y) return false;
    } else {
        float t1 = (minB.y - ray.origin.y) / ray.direction.y;
        float t2 = (maxB.y - ray.origin.y) / ray.direction.y;
        if (t1 > t2) { float tmp = t1; t1 = t2; t2 = tmp; }
        tmin = fmaxf(tmin, t1);
        tmax = fminf(tmax, t2);
        if (tmax < tmin) return false;
    }

    // Z axis
    if (fabsf(ray.direction.z) < 1e-8f) {
        if (ray.origin.z < minB.z || ray.origin.z > maxB.z) return false;
    } else {
        float t1 = (minB.z - ray.origin.z) / ray.direction.z;
        float t2 = (maxB.z - ray.origin.z) / ray.direction.z;
        if (t1 > t2) { float tmp = t1; t1 = t2; t2 = tmp; }
        tmin = fmaxf(tmin, t1);
        tmax = fminf(tmax, t2);
        if (tmax < tmin) return false;
    }

    tmin_out = tmin;
    tmax_out = tmax;
    return tmax >= 0.0f;
}

// ex: rightCorner = (1000, 1000, 1000), leftCorner = (0, 0, 0)
// Function assumes that leftCorner and rightCorner define a CUBE (same size dimensions of x,y,z)
// heavily modified with AI
inline __device__
int findIntersectingCubes(const Ray& ray,
                          const Point3& leftCorner,
                          const Point3& rightCorner,
                          int boxDim,
                          int* cells,
                          int maxCells)
{
    float tmin, tmax;

    // 1. intersect ray with full grid AABB
    if (!rayIntersectsAABB(ray, leftCorner, rightCorner, tmin, tmax)) {
        return 0;
    }

    float t = (tmin > 1e-4f) ? tmin : tmax;
    if (t < 0.0f) return 0;

    Vec3 interval = (rightCorner - leftCorner) / boxDim;

    // 2. entry point into grid
    Point3 p = ray.origin + ray.direction * t;

    float ox = ray.origin.x;
    float oy = ray.origin.y;
    float oz = ray.origin.z;

    float dx = ray.direction.x;
    float dy = ray.direction.y;
    float dz = ray.direction.z;

    // 3. compute starting voxel
    int x = (int)((p.x - leftCorner.x) / interval.x);
    int y = (int)((p.y - leftCorner.y) / interval.y);
    int z = (int)((p.z - leftCorner.z) / interval.z);

    // clamp to grid
    x = max(0, min(boxDim - 1, x));
    y = max(0, min(boxDim - 1, y));
    z = max(0, min(boxDim - 1, z));

    // 4. step direction
    int stepX = (dx > 0) ? 1 : -1;
    int stepY = (dy > 0) ? 1 : -1;
    int stepZ = (dz > 0) ? 1 : -1;

    // 5. voxel boundary in each axis
    float voxelX = leftCorner.x + (x + (stepX > 0)) * interval.x;
    float voxelY = leftCorner.y + (y + (stepY > 0)) * interval.y;
    float voxelZ = leftCorner.z + (z + (stepZ > 0)) * interval.z;

    // 6. correct DDA tMax (based on entry point, NOT origin)
    float tMaxX = (dx != 0) ? (voxelX - p.x) / dx : 1e30f;
    float tMaxY = (dy != 0) ? (voxelY - p.y) / dy : 1e30f;
    float tMaxZ = (dz != 0) ? (voxelZ - p.z) / dz : 1e30f;

    float tDeltaX = (dx != 0) ? interval.x / fabsf(dx) : 1e30f;
    float tDeltaY = (dy != 0) ? interval.y / fabsf(dy) : 1e30f;
    float tDeltaZ = (dz != 0) ? interval.z / fabsf(dz) : 1e30f;

    // 7. traversal
    int count = 0;

    for (int i = 0; i < maxCells; i++) {

        int idx = x + y * boxDim + z * boxDim * boxDim;
        cells[count++] = idx;

        if (count >= maxCells) break;

        // step to next voxel
        if (tMaxX < tMaxY) {
            if (tMaxX < tMaxZ) {
                x += stepX;
                tMaxX += tDeltaX;
            } else {
                z += stepZ;
                tMaxZ += tDeltaZ;
            }
        } else {
            if (tMaxY < tMaxZ) {
                y += stepY;
                tMaxY += tDeltaY;
            } else {
                z += stepZ;
                tMaxZ += tDeltaZ;
            }
        }

        // exit grid
        if (x < 0 || x >= boxDim ||
            y < 0 || y >= boxDim ||
            z < 0 || z >= boxDim)
            break;
    }

    return count;
}
