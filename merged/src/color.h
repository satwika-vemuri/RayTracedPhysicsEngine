#pragma once

#include "vec3.h"

#include <iostream>

using Color = Vec3;

inline double clamp(double x, double min, double max) {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}  

inline void pack_Color(unsigned char* out, const Color& c, bool gamma_correction = false) {
    double r = clamp(c.x, 0.0, 1.0);
    double g = clamp(c.y, 0.0, 1.0);
    double b = clamp(c.z, 0.0, 1.0);
    if (gamma_correction) { r = std::sqrt(r); g = std::sqrt(g); b = std::sqrt(b); }
    out[0] = static_cast<unsigned char>(255.999 * r);
    out[1] = static_cast<unsigned char>(255.999 * g);
    out[2] = static_cast<unsigned char>(255.999 * b);
}

inline void write_Color(std::ostream& out, const Color& pixel_Color, bool gamma_correction = false) {
    auto r = clamp(pixel_Color.x, 0.0f, 1.0f);
    auto g = clamp(pixel_Color.y, 0.0f, 1.0f);
    auto b = clamp(pixel_Color.z, 0.0f, 1.0f);

    if(gamma_correction) {
        r = std::sqrt(r);
        g = std::sqrt(g);
        b = std::sqrt(b);
    }
    
    int rbyte = int(255.999 * r);
    int gbyte = int(255.999 * g);
    int bbyte = int(255.999 * b);

    out << rbyte << ' ' << gbyte << ' ' << bbyte << '\n';
}

