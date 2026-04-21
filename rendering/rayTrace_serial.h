#pragma once

#include <cmath>
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <limits>
#include <string>
#include <vector>
#include <chrono>

// Reuse the CUDA-free Vec3/Point3 from particle_sim (structurally identical
// to rendering/vec3.h, just without __host__/__device__ decorators).
#include "../particle_sim/src/serial/Vec3.h"

using Color = Vec3;

// ---------- constants ----------
static const int IMAGE_WIDTH = 600;
static const int IMAGE_HEIGHT = 800;
static const double RT_PI = 3.14159265358979323846;
static const double RT_AMBIENT = 0.3;
static const double RT_REFLECTIVENESS = 0.9;
static const double RT_SHININESS = 60.0;

// ---------- Light ----------
class Light {
public:
    Light() : coords_(), color_(), brightness_(0.0f) {}
    Light(Point3 c, Color col, float b) : coords_(c), color_(col), brightness_(b) {}
    Point3 coords() const { return coords_; }
    Color color() const { return color_; }
    float brightness() const { return brightness_; }
private:
    Point3 coords_;
    Color color_;
    float brightness_;
};

// ---------- Scene constants ----------
struct SceneConstants {
    Light light;
    Color surfaceColor;
    Color dark;
};

// ---------- Triangle ----------
struct Triangle {
    Point3 v1, v2, v3;
    Vec3 n1, n2, n3;
    Triangle(Point3 a, Point3 b, Point3 c) : v1(a), v2(b), v3(c) {}
};

// ---------- HitRecord ----------
struct HitRecord {
    bool hit;
    const Triangle* tri;
    double distance;
    double u, v;
    Point3 point;

    HitRecord() : hit(false), tri(nullptr), distance(-1.0), u(0), v(0), point() {}
    HitRecord(const Triangle* t, Point3 p, double d, double bu, double bv)
        : hit(true), tri(t), distance(d), u(bu), v(bv), point(p) {}

    double w() const { return 1.0 - u - v; }

    Vec3 interpolatedNormal() const {
        if (!hit || !tri) return Vec3(0, 0, 0);
        return (tri->n1 * w() + tri->n2 * u + tri->n3 * v).normalized();
    }
};

// ---------- Ray ----------
struct Ray {
    Point3 origin;
    Vec3 direction;
    Ray(const Point3& o, const Vec3& d) : origin(o), direction(d.normalized()) {}
};

// ---------- constructSceneTriangles ----------
inline std::vector<Triangle> constructSceneTriangles(
    const std::vector<Point3>& vertexBuffer,
    const std::vector<uint32_t>& indexBuffer,
    const std::vector<Vec3>& normalBuffer)
{
    std::vector<Triangle> tris;
    tris.reserve(indexBuffer.size() / 3);
    for (size_t i = 0; i + 3 <= indexBuffer.size(); i += 3) {
        Triangle t(vertexBuffer[indexBuffer[i]],
                   vertexBuffer[indexBuffer[i+1]],
                   vertexBuffer[indexBuffer[i+2]]);
        t.n1 = normalBuffer[indexBuffer[i]];
        t.n2 = normalBuffer[indexBuffer[i+1]];
        t.n3 = normalBuffer[indexBuffer[i+2]];
        tris.push_back(t);
    }
    return tris;
}

// ---------- computeCameraPosition ----------
inline Point3 computeCameraPosition(const Point3& leftCorner, const Point3& rightCorner) {
    double D = std::max({std::abs(rightCorner.x - leftCorner.x),
                         std::abs(rightCorner.y - leftCorner.y),
                         std::abs(rightCorner.z - leftCorner.z)});
    return Point3((leftCorner.x + rightCorner.x) / 2.0,
                  (leftCorner.y + rightCorner.y) / 2.0,
                  (leftCorner.z + rightCorner.z) / 2.0 - 2.0 * D);
}

// ---------- mollerTrumbore ----------
inline HitRecord mollerTrumbore(const Ray& ray, const Triangle& tri) {
    const double EPSILON = 1e-8;

    double e1x = tri.v2.x - tri.v1.x, e1y = tri.v2.y - tri.v1.y, e1z = tri.v2.z - tri.v1.z;
    double e2x = tri.v3.x - tri.v1.x, e2y = tri.v3.y - tri.v1.y, e2z = tri.v3.z - tri.v1.z;

    double pvecX = ray.direction.y * e2z - ray.direction.z * e2y;
    double pvecY = ray.direction.z * e2x - ray.direction.x * e2z;
    double pvecZ = ray.direction.x * e2y - ray.direction.y * e2x;

    double det = e1x * pvecX + e1y * pvecY + e1z * pvecZ;
    if (std::fabs(det) < EPSILON) return HitRecord();

    double invDet = 1.0 / det;
    double tvecX  = ray.origin.x - tri.v1.x;
    double tvecY  = ray.origin.y - tri.v1.y;
    double tvecZ  = ray.origin.z - tri.v1.z;

    double u = (tvecX * pvecX + tvecY * pvecY + tvecZ * pvecZ) * invDet;
    if (u < 0.0 || u > 1.0) return HitRecord();

    double qvecX = tvecY * e1z - tvecZ * e1y;
    double qvecY = tvecZ * e1x - tvecX * e1z;
    double qvecZ = tvecX * e1y - tvecY * e1x;

    double baryV = (ray.direction.x * qvecX + ray.direction.y * qvecY + ray.direction.z * qvecZ) * invDet;
    if (baryV < 0.0 || u + baryV > 1.0) return HitRecord();

    double rayT = (e2x * qvecX + e2y * qvecY + e2z * qvecZ) * invDet;
    if (rayT < EPSILON) return HitRecord();

    Point3 intersection(ray.origin.x + rayT * ray.direction.x,
                        ray.origin.y + rayT * ray.direction.y,
                        ray.origin.z + rayT * ray.direction.z);
    return HitRecord(&tri, intersection, rayT, u, baryV);
}

// ---------- findIntersectingTriangle ----------
inline HitRecord findIntersectingTriangle(const Ray& ray,
                                          const std::vector<Triangle>& tris)
{
    double    closest = std::numeric_limits<double>::infinity();
    HitRecord best;
    for (const Triangle& t : tris) {
        HitRecord h = mollerTrumbore(ray, t);
        if (h.hit && h.distance < closest) {
            closest = h.distance;
            best    = h;
        }
    }
    return best;
}

// ---------- phong ----------
inline Color phong(const HitRecord& h, const Point3& cameraPos,
                   const SceneConstants& scene)
{
    if (!h.hit) return scene.dark;

    Vec3 N = h.interpolatedNormal();
    Vec3 L = (scene.light.coords() - h.point).normalized();
    Vec3 V = (cameraPos - h.point).normalized();
    Vec3 R = reflect(L, N);   // from particle_sim/Vec3.h

    double diffuse = std::max(0.0, N.dot(L));
    double spec    = std::pow(std::max(0.0, R.dot(V)), RT_SHININESS);

    Color amb  = scene.surfaceColor * RT_AMBIENT        * scene.light.brightness();
    Color diff = scene.surfaceColor * scene.light.color() * scene.light.brightness() * diffuse;
    Color spec_ = scene.light.color() * scene.light.brightness() * RT_REFLECTIVENESS * spec;

    return amb + diff + spec_;
}

// ---------- colorPixel ----------
inline Color colorPixel(int r, int c, const Point3& cameraPos,
                        const std::vector<Triangle>& tris,
                        const SceneConstants& scene)
{
    double imagePlaneDistance = 1.0;
    double theta      = RT_PI / 4.0;
    double planeHeight = 2.0 * imagePlaneDistance * std::tan(theta / 2.0);
    double planeWidth  = ((double)IMAGE_WIDTH / IMAGE_HEIGHT) * planeHeight;

    double u  = (c + 0.5) / IMAGE_WIDTH;
    double v  = (r + 0.5) / IMAGE_HEIGHT;
    double mx = (u - 0.5) * planeWidth;
    double my = (0.5 - v) * planeHeight;
    double mz = imagePlaneDistance;

    Point3 m(cameraPos.x + mx, cameraPos.y + my, cameraPos.z + mz);
    Ray ray(cameraPos, Vec3(mx, my, mz));

    HitRecord h = findIntersectingTriangle(ray, tris);
    return phong(h, cameraPos, scene);
}

// ---------- getAntialiasedColor ----------
inline Color getAntialiasedColor(int r, int c, Color* rayColors) {
    int valid = 0;
    Color sum;

    if (r - 1 >= 0) { sum += rayColors[(r-1)*IMAGE_WIDTH + c]; ++valid; }
    if (r + 1 < IMAGE_HEIGHT) { sum += rayColors[(r+1)*IMAGE_WIDTH + c]; ++valid; }
    if (c - 1 >= 0) { sum += rayColors[r*IMAGE_WIDTH + (c-1)]; ++valid; }
    if (c + 1 < IMAGE_WIDTH) { sum += rayColors[r*IMAGE_WIDTH + (c+1)]; ++valid; }

    return (sum / valid) * 0.5 + rayColors[r * IMAGE_WIDTH + c] * 0.5;
}

// ---------- write_Color (gamma-corrected PPM output) ----------
inline void write_Color(std::ostream& out, const Color& col, bool gamma = true) {
    auto clamp = [](double x) { return x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x); };
    double r = clamp(col.x), g = clamp(col.y), b = clamp(col.z);
    if (gamma) { r = std::sqrt(r); g = std::sqrt(g); b = std::sqrt(b); }
    out << int(255.999 * r) << ' ' << int(255.999 * g) << ' ' << int(255.999 * b) << '\n';
}

// ---------- rayTrace ----------
// Renders one frame from the given mesh buffers and writes frames/image{frame}.ppm
inline void rayTrace(
    const std::vector<Point3>& vertexBuffer,
    const std::vector<uint32_t>& indexBuffer,
    const std::vector<Vec3>& normalBuffer,
    int frame,
    const Point3& leftCorner,
    const Point3& rightCorner)
{
    std::ofstream outFile("frames/image" + std::to_string(frame) + ".ppm");
    if (!outFile.is_open()) {
        fprintf(stderr, "rayTrace: could not open output file for frame %d\n", frame);
        return;
    }
    outFile << "P3\n" << IMAGE_WIDTH << ' ' << IMAGE_HEIGHT << "\n255\n";

    SceneConstants scene = {
        Light(Point3(200.0, 100.0, 100.0), Color{1, 1, 1}, 1.0f),
        Color{0.05, 0.15, 0.4},
        Color{0.0, 0.0, 0.0}
    };

    Point3 cameraPos = computeCameraPosition(leftCorner, rightCorner);
    std::vector<Triangle> tris = constructSceneTriangles(vertexBuffer, indexBuffer, normalBuffer);

    Color* rayColors = new Color[IMAGE_WIDTH * IMAGE_HEIGHT];

    for (int r = 0; r < IMAGE_HEIGHT; ++r) {
        for (int c = 0; c < IMAGE_WIDTH; ++c) {
            rayColors[r * IMAGE_WIDTH + c] = colorPixel(r, c, cameraPos, tris, scene);
        }
        // printf("not stuck. just taking time... (getting pixel colors for row %d)\n", r);
    }

    for (int r = 0; r < IMAGE_HEIGHT; ++r) {
        for (int c = 0; c < IMAGE_WIDTH; ++c) {
            write_Color(outFile, getAntialiasedColor(r, c, rayColors));
        }
        // printf("not stuck. just taking time... (writing pixel colors to outfile for row %d)\n", r);
    }

    delete[] rayColors;
    outFile.close();
}
