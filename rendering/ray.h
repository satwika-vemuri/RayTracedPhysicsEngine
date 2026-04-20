#pragma once
#include "vec3.h"

struct Ray {
    Point3 origin;
    Vec3 direction;

    __host__ __device__
    Ray() : origin(), direction() {}

    __host__ __device__
    Ray(const Point3& o, const Vec3& d)
        : origin(o), direction(d.normalized()) {}
};