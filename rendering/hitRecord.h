#pragma once
#include "vec3.h"
#include "rayTrace.h"

class HitRecord{
    public:
        HitRecord() {}
        HitRecord(Point3 coords, Vec3 normal, bool hit) : 
                coords_(coords), normal_(normal), hit_(hit) {}
        
        const Point3& coords() const {return coords_;}
        const Vec3& normal() const {return normal_;}
        bool is_hit() const {return hit_;}

        static HitRecord toHitRecord(const Intersection* hit) {
            if (hit == nullptr || hit->tri == nullptr) {
                return HitRecord(Point3(0,0,0), Vec3(0,0,0), false);
            }

            return HitRecord(
                hit->intersection,
                hit->tri->triangleNormal.normalized(),
                true
            );
        }
        
    private:
        Point3 coords_;
        Vec3 normal_;
        bool hit_;
};