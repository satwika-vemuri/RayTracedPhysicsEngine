#pragma once
#include <cmath>

struct Vec3 {
    double x, y, z;

    Vec3() : x(0.0), y(0.0), z(0.0) {}
    Vec3(double x, double y, double z) : x(x), y(y), z(z) {}

    Vec3 operator+(const Vec3& lhs) const { return {x+lhs.x, y+lhs.y, z+lhs.z}; }
    Vec3 operator-(const Vec3& lhs) const { return {x-lhs.x, y-lhs.y, z-lhs.z}; }
    Vec3 operator*(double s) const { return {x*s, y*s, z*s}; }
    Vec3 operator/(double s) const { return {x/s, y/s, z/s}; }
    Vec3 operator-() const { return {-x, -y, -z}; }
    Vec3& operator+=(const Vec3& lhs)
    {
        x+=lhs.x; y+=lhs.y; z+=lhs.z;
        return *this;
    }
    Vec3& operator-=(const Vec3& lhs)
    {
        x-=lhs.x; y-=lhs.y; z-=lhs.z; 
        return *this;
    }
    Vec3& operator*=(double s)
    {
        x*=s; y*=s; z*=s;
        return *this;
    }
    double dot(const Vec3& lhs) const { return x*lhs.x + y*lhs.y + z*lhs.z; }
    double length2() const { return x*x + y*y + z*z; }
    double length() const { return std::sqrt(x*x + y*y + z*z); }

    Vec3 normalized() const {
        double l = length();
        return l > (1e-8f) ? (*this / l) : Vec3{};
    }
};

inline Vec3 operator*(double s, const Vec3& v) { return v * s; }
