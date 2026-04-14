#pragma once

#include "vec3.h"
#include "color.h"

class Light {
    public:
        Light() {}
        Light(Point3 coords, Color color, float brightness) :
                coords_(coords), color_(color), brightness_(brightness) {}

        const Point3& coords() const {return coords_;}
        const Color& color() const {return color_;}
        const float& brightness() const {return brightness_;} 

    private:
        Point3 coords_;
        Color color_;
        float brightness_;
        
};
