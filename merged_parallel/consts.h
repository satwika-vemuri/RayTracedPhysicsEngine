#pragma once

#include "light.h"

using Color = Vec3;

// image constants
static const int IMAGE_WIDTH = 600;
static const int IMAGE_HEIGHT = 800;

// math constants
static const double PI = 3.14159265358979323846;

// shading constants
static const double AMBIENT = 0.3;
static const double REFLECTIVENESS = 0.9;
static const double SHININESS = 60.0;

// Scene constants (no globals)
struct SceneConstants {
    Light light;
    Color surfaceColor;
    Color dark;
};

//inline const Light LIGHT(Point3(200.0, 100.0, 100.0), Color{1.0, 1.0, 1.0}, 1.0f);
//inline const Color SURFACE_COLOR{0.05, 0.15, 0.4};
//inline const Color DARK{0.0, 0.0, 0.0};