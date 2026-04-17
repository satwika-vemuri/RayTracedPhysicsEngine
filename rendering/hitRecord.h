#pragma once
#include "vec3.h"
#include "triangle.h"

struct HitRecord{
    public:

        bool hit;
        const Triangle* tri;
        double distance;
        double u;
        double v;
        Point3 point;
        
        HitRecord(): hit(false), tri(nullptr), distance(-1.0), u(0.0), v(0.0), point(Point3(0,0,0)) {}

        HitRecord(const Triangle* t, const Point3& p, double dist, double bu, double bv)
        : hit(true), tri(t), distance(dist), u(bu), v(bv), point(p) {}


        double w() const {
        return 1.0 - u - v;
        }

        Vec3 interpolatedNormal() const {
            if(!hit || !tri) {return Vec3(0,0,0);}

            return (tri->n1 * w() +
                    tri->n2 * u + 
                    tri->n3 * v).normalized();
        }


};