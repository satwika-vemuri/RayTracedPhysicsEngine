#pragma once

#include "vec3.h"

using Color = Vec3;

class Light {
public:
    Light() : coords_(), color_(), brightness_(0.0f) {}
    Light(Point3 c, Color col, float b) : coords_(c), color_(col), brightness_(b) {}
    Point3 coords() const { return coords_; }
    Color color() const { return color_; }
    float brightness() const { return brightness_; }
private:
    Point3 coords_;
    Color color_;
    float brightness_;
};