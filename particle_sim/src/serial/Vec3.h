#pragma once
#include <cmath>

struct Vec3 {
    float x, y, z;

    Vec3() : x(0.0f), y(0.0f), z(0.0f) {}
    Vec3(float x, float y, float z) : x(x), y(y), z(z) {}

    Vec3 operator+(const Vec3& lhs) const { return {x+lhs.x, y+lhs.y, z+lhs.z}; }
    Vec3 operator-(const Vec3& lhs) const { return {x-lhs.x, y-lhs.y, z-lhs.z}; }
    Vec3 operator*(float s) const { return {x*s, y*s, z*s}; }
    Vec3 operator/(float s) const { return {x/s, y/s, z/s}; }
    Vec3 operator-() const { return {-x, -y, -z}; }

    Vec3& operator+=(const Vec3& lhs) { x += lhs.x; y += lhs.y; z += lhs.z; return *this; }
    Vec3& operator-=(const Vec3& lhs) { x -= lhs.x; y -= lhs.y; z -= lhs.z; return *this; }
    Vec3& operator*=(float s) { x *= s; y *= s; z *= s; return *this; }

    float dot(const Vec3& lhs) const { return x*lhs.x + y*lhs.y + z*lhs.z; }
    float length2() const { return x*x + y*y + z*z; }
    float length() const { return sqrtf(x*x + y*y + z*z); }

    Vec3 normalized() const {
        float l = length();
        return l > 1e-8f ? (*this / l) : Vec3(0.0f, 0.0f, 0.0f);
    }
};

inline Vec3 operator*(float s, const Vec3& v) { return v * s; }
inline Vec3 operator*(const Vec3& u, const Vec3& v) { return {u.x*v.x, u.y*v.y, u.z*v.z}; }

inline Vec3 cross(const Vec3& u, const Vec3& v) {
    return {
        u.y*v.z - u.z*v.y,
        u.z*v.x - u.x*v.z,
        u.x*v.y - u.y*v.x
    };
}

inline Vec3 reflect(const Vec3& v, const Vec3& n) {
    return 2.0f * v.dot(n) * n - v;
}

using Point3 = Vec3;
