
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cmath>

#include "vec3.h"
#include "triangle.h"
#include "hitRecord_serial.h"
#include "ray_serial.h"
#include "consts.h"

using std::vector;

vector<Triangle> constructSceneTriangles(const vector<Point3>& vertexBuffer, const vector<uint32_t>& indexBuffer, const vector<Vec3>& normalBuffer){
    vector<Triangle> sceneTriangles;
    sceneTriangles.reserve(indexBuffer.size()/3);
    

    for(size_t i = 0; i+3 <= indexBuffer.size(); i+=3){
        Point3 v1 = vertexBuffer[indexBuffer[i]];
        Point3 v2 = vertexBuffer[indexBuffer[i + 1]];
        Point3 v3 = vertexBuffer[indexBuffer[i + 2]];

        Vec3 n1 = normalBuffer[indexBuffer[i]];
        Vec3 n2 = normalBuffer[indexBuffer[i + 1]];
        Vec3 n3 = normalBuffer[indexBuffer[i + 2]];
        //cout << "creating triangle" << endl;
        Triangle t = Triangle(v1, v2, v3);
        t.n1 = n1;
        t.n2 = n2;
        t.n3 = n3;
        sceneTriangles.push_back(t);
    }

    return sceneTriangles;
}

// assumes that we are positioning the camera further "back" on the z axis
Point3 computeCameraPosition(const Point3& leftCorner, const Point3& rightCorner){
    double D = std::max(std::max(std::abs(rightCorner.x - leftCorner.x),std::abs(rightCorner.y - leftCorner.y)),std::abs(rightCorner.z - leftCorner.z));
    
    // cameraPos = center + (0, 0, -2D)
    double x = (leftCorner.x + rightCorner.x)/2; 
    double y = (leftCorner.y + rightCorner.y)/2;
    double z = (leftCorner.z + rightCorner.z)/2 - 2*D; 
    

    return Point3(x, y, z);
    
}

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


HitRecord findIntersectingTriangle(const Ray& ray, const vector<Triangle>& sceneTriangles){
    double closestDistance = INFINITY;
    HitRecord closestHit;

    for(size_t i = 0; i < sceneTriangles.size(); i++){
        HitRecord hit = mollerTrumbore(ray, sceneTriangles[i]);
        // if(hit.hit){
        //     cout << "TRUE" << endl;
        // }

        if (hit.hit && hit.distance < closestDistance) {
            closestDistance = hit.distance;
            closestHit = hit;
            // cout << "chose hit.point"
            //         << "(" << intersection.x << "," 
            //         << "" << intersection.y << "," 
            //         << "" << intersection.z << "," 
            //         << ")" << endl;
        } 
    }

    return closestHit;
}


vector<vector<int>> assignTriangles(vector<Triangle>& sceneTriangles,
                                    const Point3& leftCorner,
                                    const Point3& rightCorner,
                                    int boxDimension) {
    int numCells = boxDimension * boxDimension * boxDimension;
    vector<vector<int>> trianglesPerBox(numCells);

    Vec3 intervalSize = (rightCorner - leftCorner) / boxDimension;

    for (int i = 0; i < (int)sceneTriangles.size(); i++) {
        Triangle& t = sceneTriangles[i];

        const Point3& a = t.v1;
        const Point3& b = t.v2;
        const Point3& c = t.v3;

        double minx = std::min(a.x, std::min(b.x, c.x));
        double miny = std::min(a.y, std::min(b.y, c.y));
        double minz = std::min(a.z, std::min(b.z, c.z));

        double maxx = std::max(a.x, std::max(b.x, c.x));
        double maxy = std::max(a.y, std::max(b.y, c.y));
        double maxz = std::max(a.z, std::max(b.z, c.z));

        constexpr double eps = 1e-5;

        int x_start = (int)std::floor((minx - leftCorner.x) / intervalSize.x);
        int y_start = (int)std::floor((miny - leftCorner.y) / intervalSize.y);
        int z_start = (int)std::floor((minz - leftCorner.z) / intervalSize.z);

        int x_end = (int)std::floor((maxx - leftCorner.x) / intervalSize.x - eps);
        int y_end = (int)std::floor((maxy - leftCorner.y) / intervalSize.y - eps);
        int z_end = (int)std::floor((maxz - leftCorner.z) / intervalSize.z - eps);

        x_start = std::clamp(x_start, 0, boxDimension - 1);
        y_start = std::clamp(y_start, 0, boxDimension - 1);
        z_start = std::clamp(z_start, 0, boxDimension - 1);

        x_end = std::clamp(x_end, 0, boxDimension - 1);
        y_end = std::clamp(y_end, 0, boxDimension - 1);
        z_end = std::clamp(z_end, 0, boxDimension - 1);

        for (int x = x_start; x <= x_end; x++) {
            for (int y = y_start; y <= y_end; y++) {
                for (int z = z_start; z <= z_end; z++) {
                    int idx = x + y * boxDimension + z * boxDimension * boxDimension;
                    trianglesPerBox[idx].push_back(i);
                }
            }
        }
    }

    return trianglesPerBox;
}


bool rayIntersectsAABB(const Ray& ray,
                       const Point3& minB, const Point3& maxB,
                       double& tmin_out, double& tmax_out) {
    double tmin = -std::numeric_limits<double>::infinity();
    double tmax =  std::numeric_limits<double>::infinity();

    // X slab
    if (std::fabs(ray.direction.x) < 1e-8) {
        if (ray.origin.x < minB.x || ray.origin.x > maxB.x) return false;
    } else {
        double t1 = (minB.x - ray.origin.x) / ray.direction.x;
        double t2 = (maxB.x - ray.origin.x) / ray.direction.x;
        if (t1 > t2) std::swap(t1, t2);
        tmin = std::max(tmin, t1);
        tmax = std::min(tmax, t2);
        if (tmax < tmin) return false;
    }

    // Y slab
    if (std::fabs(ray.direction.y) < 1e-8) {
        if (ray.origin.y < minB.y || ray.origin.y > maxB.y) return false;
    } else {
        double t1 = (minB.y - ray.origin.y) / ray.direction.y;
        double t2 = (maxB.y - ray.origin.y) / ray.direction.y;
        if (t1 > t2) std::swap(t1, t2);
        tmin = std::max(tmin, t1);
        tmax = std::min(tmax, t2);
        if (tmax < tmin) return false;
    }

    // Z slab
    if (std::fabs(ray.direction.z) < 1e-8) {
        if (ray.origin.z < minB.z || ray.origin.z > maxB.z) return false;
    } else {
        double t1 = (minB.z - ray.origin.z) / ray.direction.z;
        double t2 = (maxB.z - ray.origin.z) / ray.direction.z;
        if (t1 > t2) std::swap(t1, t2);
        tmin = std::max(tmin, t1);
        tmax = std::min(tmax, t2);
        if (tmax < tmin) return false;
    }

    tmin_out = tmin;
    tmax_out = tmax;
    return tmax >= 0.0;
}


int findIntersectingCubes(const Ray& ray,
                          const Point3& leftCorner,
                          const Point3& rightCorner,
                          int boxDim,
                          int* cells,
                          int maxCells) {
    double tmin, tmax;

    if (!rayIntersectsAABB(ray, leftCorner, rightCorner, tmin, tmax))
        return 0;

    double t = std::max(0.0, tmin);

    Vec3 interval = (rightCorner - leftCorner) / boxDim;

    Point3 p = ray.origin + ray.direction * t;

    double dx = ray.direction.x;
    double dy = ray.direction.y;
    double dz = ray.direction.z;

    int x = (int)((p.x - leftCorner.x) / interval.x);
    int y = (int)((p.y - leftCorner.y) / interval.y);
    int z = (int)((p.z - leftCorner.z) / interval.z);

    x = std::max(0, std::min(boxDim - 1, x));
    y = std::max(0, std::min(boxDim - 1, y));
    z = std::max(0, std::min(boxDim - 1, z));

    int stepX = (dx > 0) ? 1 : ((dx < 0) ? -1 : 0);
    int stepY = (dy > 0) ? 1 : ((dy < 0) ? -1 : 0);
    int stepZ = (dz > 0) ? 1 : ((dz < 0) ? -1 : 0);

    double voxelX = leftCorner.x + (x + (stepX > 0)) * interval.x;
    double voxelY = leftCorner.y + (y + (stepY > 0)) * interval.y;
    double voxelZ = leftCorner.z + (z + (stepZ > 0)) * interval.z;

    double tMaxX = (dx != 0) ? (voxelX - p.x) / dx : 1e30;
    double tMaxY = (dy != 0) ? (voxelY - p.y) / dy : 1e30;
    double tMaxZ = (dz != 0) ? (voxelZ - p.z) / dz : 1e30;

    double tDeltaX = (dx != 0) ? interval.x / std::fabs(dx) : 1e30;
    double tDeltaY = (dy != 0) ? interval.y / std::fabs(dy) : 1e30;
    double tDeltaZ = (dz != 0) ? interval.z / std::fabs(dz) : 1e30;

    int count = 0;
    const double tEps = 1e-6;

    for (int i = 0; i < maxCells; i++) {
        int idx = x + y * boxDim + z * boxDim * boxDim;
        cells[count++] = idx;

        if (count >= maxCells) break;

        double nextT = std::min(tMaxX, std::min(tMaxY, tMaxZ));

        if (std::fabs(tMaxX - nextT) <= tEps) { x += stepX; tMaxX += tDeltaX; }
        if (std::fabs(tMaxY - nextT) <= tEps) { y += stepY; tMaxY += tDeltaY; }
        if (std::fabs(tMaxZ - nextT) <= tEps) { z += stepZ; tMaxZ += tDeltaZ; }

        if (x < 0 || x >= boxDim ||
            y < 0 || y >= boxDim ||
            z < 0 || z >= boxDim)
            break;
    }

    return count;
}


HitRecord findIntersectingTriangleInVoxel(const Ray& ray,
                                          const vector<int>& cellTriangles,
                                          const vector<Triangle>& sceneTriangles,
                                          const vector<int>& cellStart,
                                          int boxNumber) {
    double closestDistance = std::numeric_limits<double>::infinity();
    HitRecord closestHit;
    closestHit.hit = false;

    int start = cellStart[boxNumber];
    int end   = cellStart[boxNumber + 1];

    for (int i = start; i < end; i++) {
        int triIdx = cellTriangles[i];
        HitRecord hit = mollerTrumbore(ray, sceneTriangles[triIdx]);
        if (hit.hit && hit.distance < closestDistance) {
            closestDistance = hit.distance;
            closestHit = hit;
        }
    }

    return closestHit;
}
