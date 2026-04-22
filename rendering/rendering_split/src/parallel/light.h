#pragma once

#include "vec3.h"
using Color = Vec3;

class Light {
public:
    __host__ __device__
    Light() {}

    __host__ __device__
    Light(Point3 coords, Color color, float brightness)
        : coords_(coords), color_(color), brightness_(brightness) {}

    __host__ __device__
    Point3 coords() const { return coords_; }

    __host__ __device__
    Color color() const { return color_; }

    __host__ __device__
    float brightness() const { return brightness_; }

private:
    Point3 coords_;
    Color color_;
    float brightness_;
};