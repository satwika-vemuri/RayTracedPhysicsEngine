#include <SFML/Graphics.hpp>
#include <vector>
#include <algorithm>
#include <string>
#include <sstream>
#include <iomanip>
#include <cmath>

#include "March.h"
#include "SPH.h"
#include "rayTraceTest.h"

// camera
struct Camera {
    double theta = 0.4; // rotation around Y
    double phi = 0.0; // rotation around X
    double dist  = 4.0; // distance from origin
    double focal = 620.0; // focal length

    // world to camera space via two rotations
    Vec3 toCam(Vec3 p) const {
        double ct = std::cos(theta), st = std::sin(theta);
        double x1 = p.x * ct + p.z * st;
        double y1 = p.y;
        double z1 = -p.x * st + p.z * ct;
        
        double cp = std::cos(phi), sp = std::sin(phi);
        double x2 = x1;
        double y2 = y1 * cp - z1 * sp;
        double z2 = y1 * sp + z1 * cp;
        return {x2, y2, z2};
    }

    // perspective projection
    sf::Vector2f project(Vec3 c, double W, double H) const {
        double dz = dist + c.z;
        if (dz < 0.05) dz = 0.05;
        return { W * 0.5 + focal * c.x / dz, H * 0.5 - focal * c.y / dz };
    }

    double screenR(double worldR, Vec3 c) const {
        double dz = dist + c.z;
        if (dz < 0.05) dz = 0.05;
        return focal * worldR / dz;
    }
};

void drawBox(sf::RenderWindow& win, const Camera& cam, double W, double H,
    const Vec3 boxCorners[8], const int boxEdges[12][2]) {
    sf::VertexArray lines(sf::Lines, 24);
    for (int e = 0; e < 12; ++e) {
        Vec3 a = boxCorners[boxEdges[e][0]];
        Vec3 b = boxCorners[boxEdges[e][1]];
        sf::Color col = sf::Color::White;
        lines[e*2] = sf::Vertex(cam.project(cam.toCam(a), W, H), col);
        lines[e*2+1] = sf::Vertex(cam.project(cam.toCam(b), W, H), col);
    }
    win.draw(lines);
}

// particle rendering
struct ParticleRenderEntry {
    sf::Vector2f pos;
    double depth;
    double radius;
};

void drawParticles(sf::RenderWindow& win, const Camera& cam,
    const std::vector<Particle>& particles, double W, double H)
{
    if (particles.empty()) return;

    std::vector<ParticleRenderEntry> entries;
    entries.reserve(particles.size());

    // particle render attributes
    for (const auto& p : particles) {
        Vec3 cp = cam.toCam(p.pos);
        double r = std::max(2., cam.screenR(SPH::H * 0.5, cp));
        entries.push_back({ cam.project(cp, W, H), cp.z, r});
    }

    // sort back to front so closer particles paint over far ones
    std::sort(entries.begin(), entries.end(),
        [](const ParticleRenderEntry& a, const ParticleRenderEntry& b){
            return a.depth < b.depth;
        });

    sf::CircleShape glow, core;
    glow.setPointCount(10);
    core.setPointCount(10);

    for (const auto& e : entries) {
        // blue glow
        double glowRadius = e.radius * 1.8; // 1.8 is the extra glow area
        glow.setRadius(glowRadius);
        glow.setOrigin(glowRadius, glowRadius);
        glow.setPosition(e.pos);
        sf::Color gc = sf::Color::Blue;
        gc.a = 50;
        glow.setFillColor(gc);
        win.draw(glow);

        // solid black core
        core.setRadius(e.radius);
        core.setOrigin(e.radius, e.radius);
        core.setPosition(e.pos);
        core.setFillColor(sf::Color::Black);
        win.draw(core);
    }
}

void drawMesh(sf::RenderWindow& win, const Camera& cam,
              const std::vector<std::vector<double>>& vertexBuffer,
              const std::vector<int>& indexBuffer,
              double W, double H)
{
    if (vertexBuffer.empty()) return;

    sf::VertexArray lines(sf::Lines);
    sf::Color wireColor(0, 200, 255, 120);

    for (int t = 0; t < (int)indexBuffer.size(); t += 3) {
        int i0 = indexBuffer[t];
        int i1 = indexBuffer[t + 1];
        int i2 = indexBuffer[t + 2];

        auto toScreen = [&](int idx) -> sf::Vector2f {
            Vec3 world = {
                vertexBuffer[idx][0],
                vertexBuffer[idx][1],
                vertexBuffer[idx][2]
            };
            return cam.project(cam.toCam(world), W, H);
        };

        sf::Vector2f p0 = toScreen(i0);
        sf::Vector2f p1 = toScreen(i1);
        sf::Vector2f p2 = toScreen(i2);

        lines.append(sf::Vertex(p0, wireColor));
        lines.append(sf::Vertex(p1, wireColor));

        lines.append(sf::Vertex(p1, wireColor));
        lines.append(sf::Vertex(p2, wireColor));

        lines.append(sf::Vertex(p2, wireColor));
        lines.append(sf::Vertex(p0, wireColor));
    }

    win.draw(lines);
}
// main
int main() {
    constexpr unsigned WIN_W = 1280, WIN_H = 720;

    /*
    sf::RenderWindow window(
        sf::VideoMode(WIN_W, WIN_H),
        "SPH Simulator",
        sf::Style::Default
    );
    window.setFramerateLimit(60);
    */

    SPH sim;
    printf("Number of particles: %ld\n", sim.particles.size());
    // the box outline
    static const Vec3 boxCorners[8] = {
    {sim.BMIN,sim.BMIN,sim.BMIN},
    {sim.BMAX,sim.BMIN,sim.BMIN},
    {sim.BMAX, sim.BMAX,sim.BMIN},
    {sim.BMIN, sim.BMAX,sim.BMIN},
    {sim.BMIN,sim.BMIN, sim.BMAX},
    {sim.BMAX,sim.BMIN, sim.BMAX},
    {sim.BMAX, sim.BMAX, sim.BMAX},
    {sim.BMIN, sim.BMAX, sim.BMAX}
    };
    static constexpr int boxEdges[12][2] = {
    {0,1},{1,2},{2,3},{3,0}, {4,5},{5,6},{6,7},{7,4}, {0,4},{1,5},{2,6},{3,7}
    };

    

    Camera cam;

    // state variables
    bool paused = false;
    sf::Vector2i prevMouse;

    // num sub-steps per rendered frame
    // 5 × Dt(0.001 s) = 0.005 s of simulated time per visual frame
    constexpr int SUBSTEPS = 5;

    std::vector<std::vector<double>> vertexBuffer;
    std::vector<int> indexBuffer;
    std::vector<std::vector<double>> normalBuffer;

    // main loop
    for (int i = 0; i < 60; ++i) {
        /*
        sf::Event ev;
        while (window.pollEvent(ev)) {
            switch (ev.type) {

            case sf::Event::Closed:
                window.close();
                break;

            case sf::Event::KeyPressed:
                switch (ev.key.code) {
                case sf::Keyboard::Escape:
                    window.close();
                    break;
                case sf::Keyboard::Space:
                    paused = !paused;
                    break;
                case sf::Keyboard::R:
                    sim.reset();
                    break;
                default:
                    break;
                }
                break;

            default:
                break;
            }
        }
        */

        // physics
        if (!paused)
            for (int s = 0; s < SUBSTEPS; ++s) sim.step();

	vertexBuffer.clear();
	indexBuffer.clear();
	normalBuffer.clear();

	buildScalarField(sim.particles);
	marchCubes(vertexBuffer, indexBuffer, normalBuffer);

    static int iq = 0;

    traceAndShade(vertexBuffer, indexBuffer, normalBuffer,
    sim.BMIN,sim.BMIN,sim.BMIN, sim.BMAX,sim.BMAX,sim.BMAX, iq);

    printf("iteration: %d\n", i);
    ++i;

    /*
        // render
        window.clear(sf::Color(8, 12, 22));
        drawBox(window, cam, (double)WIN_W, (double)WIN_H, boxCorners, boxEdges);
        drawParticles(window, cam, sim.particles, (double)WIN_W, (double)WIN_H);
	drawMesh(window, cam, vertexBuffer, indexBuffer, (double)WIN_W, (double)WIN_H);
        window.display();
    }

    */
    }

    return 0;
}
