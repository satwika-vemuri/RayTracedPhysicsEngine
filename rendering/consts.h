#pragma once

#include "color.h"
#include "light.h"

// Image constants
inline constexpr int IMAGE_WIDTH = 600;
inline constexpr int IMAGE_HEIGHT = 800;

// Math constants
inline constexpr double PI = 3.14159265358979323846;

// Shading constants
inline constexpr double AMBIENT = 0.3;
inline constexpr double REFLECTIVENESS = 0.9;
inline constexpr double SHININESS = 60.0;

// Scene constants
inline const Light LIGHT(Point3(200.0, 100.0, 100.0), Color{1.0, 1.0, 1.0}, 1.0f);
inline const Color SURFACE_COLOR{0.05, 0.15, 0.4};
inline const Color DARK{0.0, 0.0, 0.0};