#pragma once

#include "color.h"
#include "light.h"

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