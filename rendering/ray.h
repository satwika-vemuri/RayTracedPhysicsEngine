#pragma once
#include "vec3.h"

struct Ray {
    Point3 origin;
    Vec3 direction;

    Ray() : origin(), direction() {}
    Ray(const Point3& o, const Vec3& d) : origin(o), direction(d.normalized()) {}

};

// class Vector3 { // defined by a start point and direction
// public:
//     Point3 p;
//     double dx;
//     double dy; 
//     double dz;

    
//     Vector3(): p(Point3(0, 0, 0)), dx(0), dy(0), dz(0){
//     }
    
//     Vector3(double x, double y, double z): Vector3(Point3(0, 0, 0), x, y, z){}
//     // automatically normalize direction
//     Vector3(Point3 a, double x, double y, double z) : p(a), dx(x), dy(y), dz(z) {
//         double length = sqrt(pow(dx, 2) + pow(dy, 2) + pow(dz, 2));
//         dx /= length;
//         dy /= length;
//         dz /= length;
//     }
// };