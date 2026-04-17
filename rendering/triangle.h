#pragma once
#include "vec3.h"

class Triangle{
public:
    Point3 v1;
    Point3 v2;
    Point3 v3;

    Vec3 n1; // normal at point v1
    Vec3 n2; // normal at point v2
    Vec3 n3; // normal at point v3
    Triangle(Point3 a, Point3 b, Point3 c): v1(a), v2(b), v3(c) {}
};