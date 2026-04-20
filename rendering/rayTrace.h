#pragma once

#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <array>
#include <vector>
#include <cuda_runtime.h>


#include "triangle.h"
#include "vec3.h"
#include "ray.h"
#include "hitRecord.h"
#include "consts.h"
#pragma once






using TriangleGrid = std::vector<HitRecord>;

std::vector<Triangle> constructSceneTriangles( const std::vector<Point3>& vertexBuffer, 
                                                const std::vector<uint32_t>& indexBuffer, 
                                                const std::vector<Vec3>& normalBuffer);

Point3 computeCameraPosition(const Point3& leftCorner,const Point3& rightCorner);
 
inline __device__ 
HitRecord mollerTrumbore(const Ray& ray, const Triangle& tri) {
    // cout << "mt w/ point: " 
    //     << "(" << ray->p.x << "," 
    //     << "" << ray->p.y << "," 
    //     << "" << ray->p.z << "," 
    //     << ")" << " direction:"  
    //     << "(" << ray->x << "," 
    //     << "" << ray->y << "," 
    //     << "" << ray->z << ")" << " triangle"
    //     << "(" << tri->v1.x << "," 
    //     << "" << tri->v1.y << "," 
    //     << "" << tri->v1.z << "," 
    //     << ")"<< "(" << tri->v2.x << "," 
    //     << "" << tri->v2.y << "," 
    //     << "" << tri->v2.z << "," 
    //     << ")"<< "(" << tri->v3.x << "," 
    //     << "" << tri->v3.y << "," 
    //     << "" << tri->v3.z << "," 
    //     << ")" << endl;

    const double EPSILON = 1e-8;

    // Triangle edge vectors
    // (v2-v1)
    double e1x = tri.v2.x - tri.v1.x;
    double e1y = tri.v2.y - tri.v1.y;
    double e1z = tri.v2.z - tri.v1.z;

    // (v3-v1)
    double e2x = tri.v3.x - tri.v1.x;
    double e2y = tri.v3.y - tri.v1.y;
    double e2z = tri.v3.z - tri.v1.z;

    // pvec = ray.direction x E2
    double pvecX = ray.direction.y * e2z - ray.direction.z * e2y;
    double pvecY = ray.direction.z * e2x - ray.direction.x * e2z;
    double pvecZ = ray.direction.x * e2y - ray.direction.y * e2x;

    // det = E1 · pvec
    double det = e1x * pvecX + e1y * pvecY + e1z * pvecZ;

    // no hit (ray is close to parallel to triangle)
    //cout << "det: " << det << endl;
    if (std::fabs(det) < EPSILON) {
        return HitRecord();
    }

    double invDet = 1.0 / det;

    // tvec = ray.origin - V1
    double tvecX = ray.origin.x - tri.v1.x;
    double tvecY = ray.origin.y - tri.v1.y;
    double tvecZ = ray.origin.z - tri.v1.z;

    // u barycentric coordinate
    double u = (tvecX * pvecX + tvecY * pvecY + tvecZ * pvecZ) * invDet;
    //cout << "u: " << u << endl;
    if (u < 0.0 || u > 1.0) { // no intersection
        return HitRecord();
    }

    // qvec = tvec x E1
    double qvecX = tvecY * e1z - tvecZ * e1y;
    double qvecY = tvecZ * e1x - tvecX * e1z;
    double qvecZ = tvecX * e1y - tvecY * e1x;

    // v barycentric coordinate
    double baryV = (ray.direction.x * qvecX + ray.direction.y * qvecY + ray.direction.z * qvecZ) * invDet;
    //cout << "v: " << baryV << endl;
    
    if (baryV < 0.0 || u + baryV > 1.0) {  // no intersection
        return HitRecord();
    }

    // ray parameter t
    double rayT = (e2x * qvecX + e2y * qvecY + e2z * qvecZ) * invDet;
    //cout << "t: " << rayT << endl;
    if (rayT < EPSILON) { // no intersection
        return HitRecord();
    }


    // Compute intersection point
    Point3 intersection(
        ray.origin.x + rayT * ray.direction.x,
        ray.origin.y + rayT * ray.direction.y,
        ray.origin.z + rayT * ray.direction.z
    );

    return HitRecord(&tri, intersection, rayT,  u, baryV );
}


inline __device__
HitRecord findIntersectingTriangle(const Ray& ray,
                                    const Triangle* sceneTriangles,
                                    int numTriangles) {
    double closestDistance = INFINITY;
    HitRecord closestHit;
    closestHit.hit = false;

    for (int i = 0; i < numTriangles; i++) {
        HitRecord hit = mollerTrumbore(ray, sceneTriangles[i]);

        if (hit.hit && hit.distance < closestDistance) {
            closestDistance = hit.distance;
            closestHit = hit;
        }
    }

    return closestHit;
}