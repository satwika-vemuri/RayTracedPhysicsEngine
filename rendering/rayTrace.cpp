
#include <cstdint>
#include <vector>
#include <algorithm>

#include "rayTrace.h"
#include "hitRecord.h"
#include "ray.h"
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
