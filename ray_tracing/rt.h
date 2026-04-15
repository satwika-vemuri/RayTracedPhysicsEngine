#include <cstdlib>
#include <algorithm>
#include <array>
#include <vector>
using namespace std;
constexpr int IMAGE_WIDTH = 600;
constexpr int IMAGE_HEIGHT = 800;

class Point3;
class Triangle;
struct Intersection;

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

    Vector3(): p(Point3(0, 0, 0)), dx(0), dy(0), dz(0){}
    Vector3(double x, double y, double z): p(Point3(0, 0, 0)), dx(x), dy(y), dz(z){}
    Vector3(Point3 a, double x, double y, double z) : p(a), dx(x), dy(y), dz(z) {}
};


struct Intersection {
    Triangle* tri;
    Point3 intersection;

    Intersection() : tri(nullptr), intersection() {}  
    Intersection(Triangle* t, Point3 p) : tri(t), intersection(p) {}
};

using TriangleGrid = std::array<std::array<Intersection*, IMAGE_WIDTH>, IMAGE_HEIGHT>;


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

    HitInfo(bool h, Point3 p, double d) : hit(h), point(p), distance(d) {}
};



vector<Triangle*> constructSceneTriangles(vector<vector<double>> vertexBuffer, vector<double> indexBuffer, vector<vector<double>> normalBuffer);
Point3* computeCameraPosition(Point3 leftCorner, Point3 rightCorner);
TriangleGrid* findSurfaceIntersections(Point3* cameraPos, vector<Triangle*> sceneTriangles);
