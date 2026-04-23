/*
 * Kasra Farsoudi 2026
 */
#include "March.h"
#include "SPH.h"
#include "SpatialHash.h"

#include <cstdint>
#include <vector>

// TODO: Refactor w more helpers + make more readable

// Density Grid
double scalar[GRID_N + 1][GRID_N + 1][GRID_N + 1];

/*
 * Before generating the mesh:
 * Finding how "full of water" every point in space is
 * We compute a density value at each grid corner
 */
void buildScalarField(const std::vector<Particle> &particles) {
        // Spatial Hash for quickly finding nearby particles at given vertex
        // corner
        SpatialHash hash(SPH::H);
        for (int idx = 0; idx < particles.size(); ++idx)
                hash.insert(idx, particles[idx].pos);

        std::vector<int> nbrs; // Reused buffer for neighbor indices

        // Visit every grid corner (i, j, k)
        for (int i = 0; i <= GRID_N; ++i) {
                for (int j = 0; j <= GRID_N; ++j) {
                        for (int k = 0; k <= GRID_N; ++k) {

                                // Convert grid index → world-space 3D position
                                Vec3 p = {SPH::BMIN + i * CELL,
                                          SPH::BMIN + j * CELL,
                                          SPH::BMIN + k * CELL};

                                // Find particles within radius H of p
                                hash.query(p, SPH::H, nbrs);

                                // Sum up each nearby particle's contribution
                                // using W_poly6.
                                double density = 0.0;
                                for (int pidx : nbrs) {
                                        double r2 =
                                            (p - particles[pidx].pos).length2();
                                        // W_poly6 is defined as: K_POLY6 * (H²-
                                        // r²)³ when r²<H², else 0. resuse of
                                        // same formula and constant K_POLY6
                                        // from SPH.h
                                        if (r2 < SPH::H2) {
                                                double d = SPH::H2 - r2;
                                                density +=
                                                    SPH::K_POLY6 * d * d * d;
                                        }
                                }
                                scalar[i][j][k] = density;
                        }
                }
        }
	zeroifyBoundary();
}
/*
 * Part 2-5 of Mesh Generation
 */
void marchCubes(std::vector<Vec3>& vertexBuffer,
                std::vector<uint32_t>& indexBuffer,
                std::vector<Vec3>& normalBuffer) {

        /*
         * Now that we have a density of every corner of the grid, find which
         * corners of each cube are inside the surface (density above isovalue).
         * This info is stored in a single integer because it has 8 bits for 8
         * corners. This integer then maps to the edge table for easy lookup.
         */
        for (int i = 0; i < GRID_N; ++i) {
                for (int j = 0; j < GRID_N; ++j) {
                        for (int k = 0; k < GRID_N; ++k) {

                                // Read density at all 8 corners of this cube to
                                // d[0..7]
                                double d[8];
                                readDensity(d, i, j, k);

                                // Build cubeIndex: set bit N if corner N
                                // is inside surface. Meaning if d[n] >
                                // ISOVALUE, that corner is in the water
                                int cubeIndex = 0;
                                for (int n = 0; n < 8; ++n)
                                        if (d[n] > ISOVALUE)
                                                cubeIndex |=
                                                    (1 << n); // set the nth bit

                                // 0 = all corners outside (no surface here),
                                // skip 255 = all corners inside (fully
                                // submerged), also no surface
                                if (cubeIndex == 0 || cubeIndex == 255)
                                        continue;

                                if (edgeTable[cubeIndex] == 0)
                                        continue; // no edges crossed, skip

                                // Array of 8 corners as Vec3 Positions in 3d
                                // Space read to c[0..7]
                                Vec3 c[8];
                                cornersTo3d(c, i, j, k);

                                Vec3 edgeVerts[12]; // world-space position of
                                                    // crossing on each edge
                                // check edgeTable to see which of the
                                // 12 edges are crossed edgeTable[cubeIndex] is
                                // a 12-bit integer where bit N=1 means edge N
                                // is crossed
                                // For each of the 12 possible edges, compute
                                // the crossing point IF the edge is flagged.
                                // The & operator checks individual bits:
                                if (edgeTable[cubeIndex] &
                                    1) // bit 0 = edge between corner 0 and 1
                                        edgeVerts[0] =
                                            interp(c[0], d[0], c[1], d[1]);
                                if (edgeTable[cubeIndex] &
                                    2) // bit 1 = edge between corner 1 and 2
                                        edgeVerts[1] =
                                            interp(c[1], d[1], c[2], d[2]);
                                if (edgeTable[cubeIndex] &
                                    4) // bit 2 = edge between corner 2 and 3
                                        edgeVerts[2] =
                                            interp(c[2], d[2], c[3], d[3]);
                                if (edgeTable[cubeIndex] & 8)
                                        edgeVerts[3] =
                                            interp(c[3], d[3], c[0], d[0]);
                                if (edgeTable[cubeIndex] & 16)
                                        edgeVerts[4] =
                                            interp(c[4], d[4], c[5], d[5]);
                                if (edgeTable[cubeIndex] & 32)
                                        edgeVerts[5] =
                                            interp(c[5], d[5], c[6], d[6]);
                                if (edgeTable[cubeIndex] & 64)
                                        edgeVerts[6] =
                                            interp(c[6], d[6], c[7], d[7]);
                                if (edgeTable[cubeIndex] & 128)
                                        edgeVerts[7] =
                                            interp(c[7], d[7], c[4], d[4]);
                                if (edgeTable[cubeIndex] & 256)
                                        edgeVerts[8] =
                                            interp(c[0], d[0], c[4], d[4]);
                                if (edgeTable[cubeIndex] & 512)
                                        edgeVerts[9] =
                                            interp(c[1], d[1], c[5], d[5]);
                                if (edgeTable[cubeIndex] & 1024)
                                        edgeVerts[10] =
                                            interp(c[2], d[2], c[6], d[6]);
                                if (edgeTable[cubeIndex] & 2048)
                                        edgeVerts[11] =
                                            interp(c[3], d[3], c[7], d[7]);

                                for (int t = 0; triTable[cubeIndex][t] != -1;
                                     t += 3) {

                                        // Look up which edge crossing points
                                        // form this triangle
                                        Vec3 v0 =
                                            edgeVerts[triTable[cubeIndex][t]];
                                        Vec3 v1 = edgeVerts[triTable[cubeIndex]
                                                                    [t + 1]];
                                        Vec3 v2 = edgeVerts[triTable[cubeIndex]
                                                                    [t + 2]];

                                        // Record the base index before we add
                                        // new vertices
                                        uint32_t base = (uint32_t)vertexBuffer.size();

                                        vertexBuffer.push_back(v0);
                                        vertexBuffer.push_back(v1);
                                        vertexBuffer.push_back(v2);

                                        // Push 3 indices pointing to those
                                        // vertices next stage in pipeline
                                        // reads: indexBuffer[i], [i+1], [i+2] =
                                        // one triangle
                                        indexBuffer.push_back(base);
                                        indexBuffer.push_back(base + 1);
                                        indexBuffer.push_back(base + 2);

                                        normalBuffer.push_back(gradientNormal(v0));
                                        normalBuffer.push_back(gradientNormal(v1));
                                        normalBuffer.push_back(gradientNormal(v2));
                                }
                        }
                }
        }
}
