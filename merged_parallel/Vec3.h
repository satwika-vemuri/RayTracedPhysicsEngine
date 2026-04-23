#pragma once
#include <cmath>

// allow vec3 methods to run on both CPU and GPU
#ifndef __CUDACC__
#define HD
#else
#define HD __host__ __device__
#endif

struct Vec3 {
    double x, y, z;

    HD Vec3() : x(0.0), y(0.0), z(0.0) {}
    HD Vec3(double x, double y, double z) : x(x), y(y), z(z) {}

    HD Vec3 operator+(const Vec3& lhs) const { return {x+lhs.x, y+lhs.y, z+lhs.z}; }
    HD Vec3 operator-(const Vec3& lhs) const { return {x-lhs.x, y-lhs.y, z-lhs.z}; }
    HD Vec3 operator*(double s) const { return {x*s, y*s, z*s}; }
    HD Vec3 operator/(double s) const { return {x/s, y/s, z/s}; }
    HD Vec3 operator-() const { return {-x, -y, -z}; }

    HD Vec3& operator+=(const Vec3& lhs) { x += lhs.x; y += lhs.y; z += lhs.z; return *this; }
    HD Vec3& operator-=(const Vec3& lhs) { x -= lhs.x; y -= lhs.y; z -= lhs.z; return *this; }
    HD Vec3& operator*=(double s) { x *= s; y *= s; z *= s; return *this; }

    HD double dot(const Vec3& lhs) const { return x*lhs.x + y*lhs.y + z*lhs.z; }
    HD double length2() const { return x*x + y*y + z*z; }
    HD double length() const { return std::sqrt(x*x + y*y + z*z); }

    HD Vec3 normalized() const {
        double l = length();
        return l > 1e-8 ? (*this / l) : Vec3(0.0, 0.0, 0.0);
    }
};

HD inline Vec3 operator*(double s, const Vec3& v) { return v * s; }
HD inline Vec3 operator*(const Vec3& u, const Vec3& v) { return {u.x*v.x, u.y*v.y, u.z*v.z}; }

HD inline Vec3 cross(const Vec3& u, const Vec3& v) {
    return {
        u.y*v.z - u.z*v.y,
        u.z*v.x - u.x*v.z,
        u.x*v.y - u.y*v.x
    };
}

HD inline Vec3 reflect(const Vec3& v, const Vec3& n) {
    return 2.0 * v.dot(n) * n - v;
}

using Point3 = Vec3;
