#pragma once

#include "vec3.h"

#include <iostream>

using Color = Vec3;

inline double clamp(double x, double min, double max) {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}  

inline void write_Color(std::ostream& out, const Color& pixel_Color) {
    auto r = clamp(pixel_Color.x, 0.0f, 1.0f);
    auto g = clamp(pixel_Color.y, 0.0f, 1.0f);
    auto b = clamp(pixel_Color.z, 0.0f, 1.0f);

    // Translate the [0,1] component values to the byte range [0,255].
    int rbyte = int(255.999 * r);
    int gbyte = int(255.999 * g);
    int bbyte = int(255.999 * b);

    out << rbyte << ' ' << gbyte << ' ' << bbyte << '\n';
}

