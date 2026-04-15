#include <cstddef>
#include <cstdlib>
#include <algorithm>
#include <array>
#include <vector>
#include <cmath>
#include <iostream>

using namespace std;

const int IMAGE_WIDTH = 600;
const int IMAGE_HEIGHT = 800;
const double PI = 2 * acos(0.0);

class Point3;
class Triangle;
struct Intersection;

using TriangleGrid = std::array<std::array<Intersection*, IMAGE_WIDTH>, IMAGE_HEIGHT>;

class Point3 {
public:
    double x;
    double y;
    double z;
    Point3() : x(0), y(0), z(0) {} 
    Point3(double a, double b, double c) : x(a), y(b), z(c) {}
};

class Vector3 { // defined by a start point and direction
public:
    Point3 p;
    double dx;
    double dy; 
    double dz;

    
    Vector3(): p(Point3(0, 0, 0)), dx(0), dy(0), dz(0){
    }
    
    Vector3(double x, double y, double z): Vector3(Point3(0, 0, 0), x, y, z){}
    // automatically normalize direction
    Vector3(Point3 a, double x, double y, double z) : p(a), dx(x), dy(y), dz(z) {
        double length = sqrt(pow(dx, 2) + pow(dy, 2) + pow(dz, 2));
        dx /= length;
        dy /= length;
        dz /= length;
    }
};

class Triangle{
public:
    Point3 v1;
    Point3 v2;
    Point3 v3;

    Vector3 n1; // normal at point v1
    Vector3 n2; // normal at point v2
    Vector3 n3; // normal at point v3
    Vector3 triangleNormal;
    Triangle(Point3 a, Point3 b, Point3 c): v1(a), v2(b), v3(c) {}
};

struct HitInfo {
    bool hit;
    Point3 point;
    double distance;

    HitInfo(): hit(false), point(), distance(-1.0) {}
    HitInfo(bool h, Point3 p, double d) : hit(h), point(p), distance(d) {}
};

struct Intersection {
    Triangle* tri;
    Point3 intersection;

    Intersection() : tri(nullptr), intersection() {} 
    Intersection(Triangle* t, Point3 p) : tri(t), intersection(p) {}
};

/**
 * @brief converts three buffers into a vector of triangles to represent the scene
 *
 * @param vertexBuffer list of points in 3D space: [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 2, 0]]
 * @param indexBuffer vertex indices of triangles (from vertex buffer): [1, 2, 4, 0, 1, 3] means that 
 *                    indices [1, 2, 4] form one triangle, indices [0, 1, 3] form another triangle
 * @param normalBuffer normal vector at each vertex (normalBuffer[0] = normal for vertex with index n in vertexBuffer)
 */
vector<Triangle*> constructSceneTriangles(vector<vector<int>> vertexBuffer, vector<int> indexBuffer, vector<vector<int>> normalBuffer){
    vector<Triangle*> sceneTriangles;

    for(size_t i = 0; i < indexBuffer.size()-2; i+=3){
        //cout <<"a index: " << i  << " iB size: " << indexBuffer.size() << " iB value:" << indexBuffer[i] << " vB size: " << vertexBuffer.size() << endl;
        Point3 v1(vertexBuffer[indexBuffer[i]][0], vertexBuffer[indexBuffer[i]][1], vertexBuffer[indexBuffer[i]][2]);
        Point3 v2(vertexBuffer[indexBuffer[i+1]][0], vertexBuffer[indexBuffer[i+1]][1], vertexBuffer[indexBuffer[i+1]][2]);
        Point3 v3(vertexBuffer[indexBuffer[i+2]][0], vertexBuffer[indexBuffer[i+2]][1], vertexBuffer[indexBuffer[i+2]][2]);

        //cout <<"b index: " << i  << " iB size: " << indexBuffer.size() << "iB value:" << indexBuffer[i] << " vB size: " << vertexBuffer.size() << endl;
        Vector3 n1(normalBuffer[indexBuffer[i]][0], normalBuffer[indexBuffer[i]][1], normalBuffer[indexBuffer[i]][2]);
        Vector3 n2(normalBuffer[indexBuffer[i+1]][0], normalBuffer[indexBuffer[i+1]][1], normalBuffer[indexBuffer[i+1]][2]);
        Vector3 n3(normalBuffer[indexBuffer[i+2]][0], normalBuffer[indexBuffer[i+2]][1], normalBuffer[indexBuffer[i+2]][2]);
        Vector3 tn((n1.dx+n2.dx+n3.dx)/3, (n1.dy+n2.dy+n3.dy)/3, (n1.dz+n2.dz+n3.dz)/3);

        //cout << "creating triangle" << endl;
        Triangle* t = new Triangle(v1, v2, v3);
        t->n1 = n1;
        t->n2 = n2;
        t->n3 = n3;
        t->triangleNormal = tn;
        sceneTriangles.push_back(t);
    }

    return sceneTriangles;
}

// assumes that we are positioning the camera further "back" on the z axis
Point3* computeCameraPosition(Point3 leftCorner, Point3 rightCorner){
    double D = max(max(abs(rightCorner.x - leftCorner.x),abs(rightCorner.y - leftCorner.y)),abs(rightCorner.z - leftCorner.z));
    
    // cameraPos = center + (0, -2D, 0)
    double x = (leftCorner.x + rightCorner.x)/2; 
    double y = (leftCorner.y + rightCorner.y)/2;
    double z = (leftCorner.z + rightCorner.z)/2 - 2*D; 
    

    return new Point3(x, y, z);
    
}

HitInfo mollerTrumbore(Vector3* ray, Triangle* tri) {
    // cout << "mt w/ point: " 
    //     << "(" << ray->p.x << "," 
    //     << "" << ray->p.y << "," 
    //     << "" << ray->p.z << "," 
    //     << ")" << " direction:"  
    //     << "(" << ray->dx << "," 
    //     << "" << ray->dy << "," 
    //     << "" << ray->dz << ")" << " triangle"
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
    double e1x = tri->v2.x - tri->v1.x;
    double e1y = tri->v2.y - tri->v1.y;
    double e1z = tri->v2.z - tri->v1.z;

    // (v3-v1)
    double e2x = tri->v3.x - tri->v1.x;
    double e2y = tri->v3.y - tri->v1.y;
    double e2z = tri->v3.z - tri->v1.z;

    // pvec = ray.direction x E2
    double pvecX = ray->dy * e2z - ray->dz * e2y;
    double pvecY = ray->dz * e2x - ray->dx * e2z;
    double pvecZ = ray->dx * e2y - ray->dy * e2x;

    // det = E1 · pvec
    double det = e1x * pvecX + e1y * pvecY + e1z * pvecZ;

    // no hit (ray is close to parallel to triangle)
    //cout << "det: " << det << endl;
    if (fabs(det) < EPSILON) {
        return HitInfo(false, Point3(0, 0, 0), -1);
    }

    double invDet = 1.0 / det;

    // tvec = ray.origin - V1
    double tvecX = ray->p.x - tri->v1.x;
    double tvecY = ray->p.y - tri->v1.y;
    double tvecZ = ray->p.z - tri->v1.z;

    // u barycentric coordinate
    double u = (tvecX * pvecX + tvecY * pvecY + tvecZ * pvecZ) * invDet;
    //cout << "u: " << u << endl;
    if (u < 0.0 || u > 1.0) { // no intersection
        return HitInfo(false, Point3(0, 0, 0), -1);
    }

    // qvec = tvec x E1
    double qvecX = tvecY * e1z - tvecZ * e1y;
    double qvecY = tvecZ * e1x - tvecX * e1z;
    double qvecZ = tvecX * e1y - tvecY * e1x;

    
    
    

    // v barycentric coordinate
    double baryV = (ray->dx * qvecX + ray->dy * qvecY + ray->dz * qvecZ) * invDet;
    //cout << "v: " << baryV << endl;
    
    if (baryV < 0.0 || u + baryV > 1.0) {  // no intersection
        return HitInfo(false, Point3(0, 0, 0), -1);
    }

    // ray parameter t
    double rayT = (e2x * qvecX + e2y * qvecY + e2z * qvecZ) * invDet;
    //cout << "t: " << rayT << endl;
    if (rayT < EPSILON) { // no intersection
        return HitInfo(false, Point3(0, 0, 0), -1);
    }


    // Compute intersection point
    Point3 intersection(
        ray->p.x + rayT * ray->dx,
        ray->p.y + rayT * ray->dy,
        ray->p.z + rayT * ray->dz
    );

    return HitInfo(true, intersection, rayT);
}

Intersection* findIntersectingTriangle(Vector3 ray, vector<Triangle*> sceneTriangles){
    double closestDistance = INFINITY;
    Triangle* tri = nullptr;
    Point3 intersection(0, 0, 0);

    for(size_t i = 0; i < sceneTriangles.size(); i++){
        HitInfo hit = mollerTrumbore(&ray, sceneTriangles[i]);
        // if(hit.hit){
        //     cout << "TRUE" << endl;
        // }

        if (hit.hit && hit.distance < closestDistance) {
            closestDistance = hit.distance;
            tri = sceneTriangles[i];
            intersection = hit.point;
            // cout << "chose hit.point"
            //         << "(" << intersection.x << "," 
            //         << "" << intersection.y << "," 
            //         << "" << intersection.z << "," 
            //         << ")" << endl;
        }
        
    }

    return new Intersection(tri, intersection);

}

TriangleGrid* findSurfaceIntersections(Point3* cameraPos, vector<Triangle*> sceneTriangles){

    // define image plane metrics
    double imagePlaneDistance = 1;
    double theta = PI/4; // 45 degree FOV
    double planeHeight = 2*imagePlaneDistance*tan(theta/2);
    double planeWidth =  ((double)IMAGE_WIDTH / IMAGE_HEIGHT) * planeHeight;
    
    TriangleGrid* triangleIntersections = new TriangleGrid();
    for(int r = 0; r < IMAGE_HEIGHT; r++){
        for(int c = 0; c < IMAGE_WIDTH; c++){
            
            /*
            for each pixel, figure out which point on the image plane the pixel maps to
            let's call the point the pixel maps to m
            r/IMAGE_WIDTH = m_x/planeWidth; c/IMAGE_HEIGHT = m_z/planeHeight
            */



            double u = (c + 0.5) / IMAGE_WIDTH;
            double v = (r + 0.5) / IMAGE_HEIGHT;

            // map to [-0.5, 0.5]
            double m_x = (u - 0.5) * planeWidth;
            double m_y = (0.5 - v) * planeHeight;  // flip here
            double m_z = imagePlaneDistance;

            Point3 m(cameraPos->x + m_x, cameraPos->y + m_y, m_z); // world-space point on image plane
            Vector3 ray(*cameraPos, m.x - cameraPos->x, m.y - cameraPos->y,m.z - cameraPos->z); // ray direction = m - camera
            (*triangleIntersections)[r][c] = findIntersectingTriangle(ray, sceneTriangles);
            if (r % 100 == 0 && c % 100 == 0) {
                cout << "Intersection of (" << r << "," << c << "): "
                    << "(" << (*triangleIntersections)[r][c]->intersection.x << "," 
                    << "" << (*triangleIntersections)[r][c]->intersection.y << "," 
                    << "" << (*triangleIntersections)[r][c]->intersection.z << "" 
                    << ")" << endl;
            }
            
        }
    }

    
    return triangleIntersections;
}