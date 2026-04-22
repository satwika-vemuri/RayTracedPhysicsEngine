#pragma once
#include "consts.h"
#include "hitRecord_serial.h"
#include <algorithm>

inline Color phong(const HitRecord& h, const Point3& cameraPos,
                   const SceneConstants& scene)
{
    if (!h.hit) return scene.dark;

    Vec3 N = h.interpolatedNormal();
    Vec3 L = (scene.light.coords() - h.point).normalized();
    Vec3 V = (cameraPos - h.point).normalized();
    Vec3 R = reflection(L, N);   // from particle_sim/Vec3.h

    double diffuse = std::max(0.0, N.dot(L));
    double spec    = std::pow(std::max(0.0, R.dot(V)), SHININESS);

    Color amb  = scene.surfaceColor * AMBIENT * scene.light.brightness();
    Color diff = scene.surfaceColor * scene.light.color() * scene.light.brightness() * diffuse;
    Color spec_ = scene.light.color() * scene.light.brightness() * REFLECTIVENESS * spec;

    return amb + diff + spec_;
}



