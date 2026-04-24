/*
 * Kasra Farsoudi 2026
 */
#include "March.h"
#include "SPH.h"
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <cstdint>
#include <vector>

// Lookup tables in constant memory (broadcast read to all threads)
__constant__ int d_edgeTable[256];
__constant__ int d_triTable[256][16];

// Density Grid (device side)
static double*   d_scalar    = nullptr;
static int*      d_triCounts = nullptr;
static int*      d_triOffsets= nullptr;
static Vec3*     d_verts     = nullptr;
static uint32_t* d_idx       = nullptr;
static Vec3*     d_norms     = nullptr;
static int       d_maxTris   = 0;

// call once before the main loop to upload tables + alloc persistent buffers
void initMarchTables() {
    cudaMemcpyToSymbol(d_edgeTable, edgeTable, sizeof(edgeTable));
    cudaMemcpyToSymbol(d_triTable,  triTable,  sizeof(triTable));
    cudaMalloc(&d_scalar,     (GRID_N+1)*(GRID_N+1)*(GRID_N+1) * sizeof(double));
    cudaMalloc(&d_triCounts,  GRID_N*GRID_N*GRID_N * sizeof(int));
    cudaMalloc(&d_triOffsets, GRID_N*GRID_N*GRID_N * sizeof(int));
}

// Density Grid (host side - kept for CPU helpers in March.h)
double scalar[GRID_N + 1][GRID_N + 1][GRID_N + 1];

/*
 * Device version of gradientNormal - same central difference logic
 * as the CPU version in March.h but reads from d_scalar
 */
__device__ static Vec3 d_gradNormal(Vec3 p, const double* sc) {
    int S = GRID_N + 1;
    #define GI(x) max(0, min((int)((x - SPH::BMIN) / CELL), GRID_N))
    double gx = sc[GI(p.x+CELL)*S*S + GI(p.y)*S + GI(p.z)] - sc[GI(p.x-CELL)*S*S + GI(p.y)*S + GI(p.z)];
    double gy = sc[GI(p.x)*S*S + GI(p.y+CELL)*S + GI(p.z)] - sc[GI(p.x)*S*S + GI(p.y-CELL)*S + GI(p.z)];
    double gz = sc[GI(p.x)*S*S + GI(p.y)*S + GI(p.z+CELL)] - sc[GI(p.x)*S*S + GI(p.y)*S + GI(p.z-CELL)];
    #undef GI
    return (-Vec3(gx, gy, gz)).normalized();
}

/*
 * Before generating the mesh:
 * Finding how "full of water" every point in space is
 * We compute a density value at each grid corner
 * one thread per grid corner
 */
__global__ void buildScalarField_kernel(const Particle* particles, int n, GpuHash hash, double* sc) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i > GRID_N || j > GRID_N || k > GRID_N) return;

    // Convert grid index -> world-space 3D position
    Vec3 p = {SPH::BMIN + i * CELL, SPH::BMIN + j * CELL, SPH::BMIN + k * CELL};
    double density = 0.0;

    int ix = (int)floor(p.x / hash.cellSize);
    int iy = (int)floor(p.y / hash.cellSize);
    int iz = (int)floor(p.z / hash.cellSize);

    // Sum up each nearby particle's contribution using W_poly6
    #pragma unroll
    for (int dx = -1; dx <= 1; ++dx) {
        #pragma unroll
        for (int dy = -1; dy <= 1; ++dy) {
            #pragma unroll
            for (int dz = -1; dz <= 1; ++dz) {
                int pidx = hash.d_head[hash.cellKey(ix+dx, iy+dy, iz+dz)];
                while (pidx != -1) {
                    double r2 = (p - particles[pidx].pos).length2();
                    if (r2 < SPH::H2) { double d = SPH::H2 - r2; density += SPH::K_POLY6*d*d*d; }
                    pidx = hash.d_next[pidx];
                }
            }
        }
    }

    sc[i*(GRID_N+1)*(GRID_N+1) + j*(GRID_N+1) + k] = density;
}

// one thread per boundary face cell
__global__ void zeroifyBoundary_kernel(double* sc) {
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y * blockDim.y + threadIdx.y;
    if (a > GRID_N || b > GRID_N) return;
    int N = GRID_N, S = N+1;
    sc[0*S*S+a*S+b] = sc[N*S*S+a*S+b] = 0.0;
    sc[a*S*S+0*S+b] = sc[a*S*S+N*S+b] = 0.0;
    sc[a*S*S+b*S+0] = sc[a*S*S+b*S+N] = 0.0;
}

/*
 * Part 1 of GPU marchCubes:
 * Count how many triangles each cube emits so we can do a prefix scan
 * and give each cube a safe write offset (no race conditions)
 * one thread per cube
 */
__global__ void countTriangles_kernel(const double* sc, int* counts) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= GRID_N || j >= GRID_N || k >= GRID_N) return;

    // Read density at all 8 corners of this cube
    int S = GRID_N+1;
    double d[8] = {
        sc[ i   *S*S+ j   *S+ k  ], sc[(i+1)*S*S+ j   *S+ k  ],
        sc[(i+1)*S*S+ j   *S+(k+1)], sc[ i   *S*S+ j   *S+(k+1)],
        sc[ i   *S*S+(j+1)*S+ k  ], sc[(i+1)*S*S+(j+1)*S+ k  ],
        sc[(i+1)*S*S+(j+1)*S+(k+1)], sc[ i   *S*S+(j+1)*S+(k+1)]
    };

    // Build cubeIndex: set bit N if corner N is inside surface
    int cubeIndex = 0;
    for (int n = 0; n < 8; ++n) if (d[n] > ISOVALUE) cubeIndex |= (1<<n);

    int cnt = 0;
    if (cubeIndex != 0 && cubeIndex != 255 && d_edgeTable[cubeIndex] != 0)
        for (int t = 0; d_triTable[cubeIndex][t] != -1; t += 3) cnt++;

    counts[i*GRID_N*GRID_N + j*GRID_N + k] = cnt;
}

/*
 * Part 2 of GPU marchCubes:
 * Each cube writes its triangles to its pre-computed offset - no races
 * one thread per cube
 */
__global__ void emitTriangles_kernel(const double* sc, const int* offsets, Vec3* ov, uint32_t* oi, Vec3* on) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= GRID_N || j >= GRID_N || k >= GRID_N) return;

    // Read density at all 8 corners of this cube to d[0..7]
    int S = GRID_N+1;
    double d[8] = {
        sc[ i   *S*S+ j   *S+ k  ], sc[(i+1)*S*S+ j   *S+ k  ],
        sc[(i+1)*S*S+ j   *S+(k+1)], sc[ i   *S*S+ j   *S+(k+1)],
        sc[ i   *S*S+(j+1)*S+ k  ], sc[(i+1)*S*S+(j+1)*S+ k  ],
        sc[(i+1)*S*S+(j+1)*S+(k+1)], sc[ i   *S*S+(j+1)*S+(k+1)]
    };

    // Build cubeIndex: set bit N if corner N is inside surface
    int cubeIndex = 0;
    for (int n = 0; n < 8; ++n) if (d[n] > ISOVALUE) cubeIndex |= (1<<n);

    // 0 = all corners outside, 255 = all inside - no surface either way
    if (cubeIndex == 0 || cubeIndex == 255 || d_edgeTable[cubeIndex] == 0) return;

    // Array of 8 corners as Vec3 positions in 3d space
    Vec3 c[8] = {
        {SPH::BMIN+ i   *CELL, SPH::BMIN+ j   *CELL, SPH::BMIN+ k   *CELL},
        {SPH::BMIN+(i+1)*CELL, SPH::BMIN+ j   *CELL, SPH::BMIN+ k   *CELL},
        {SPH::BMIN+(i+1)*CELL, SPH::BMIN+ j   *CELL, SPH::BMIN+(k+1)*CELL},
        {SPH::BMIN+ i   *CELL, SPH::BMIN+ j   *CELL, SPH::BMIN+(k+1)*CELL},
        {SPH::BMIN+ i   *CELL, SPH::BMIN+(j+1)*CELL, SPH::BMIN+ k   *CELL},
        {SPH::BMIN+(i+1)*CELL, SPH::BMIN+(j+1)*CELL, SPH::BMIN+ k   *CELL},
        {SPH::BMIN+(i+1)*CELL, SPH::BMIN+(j+1)*CELL, SPH::BMIN+(k+1)*CELL},
        {SPH::BMIN+ i   *CELL, SPH::BMIN+(j+1)*CELL, SPH::BMIN+(k+1)*CELL}
    };

    Vec3 edgeVerts[12]; // world-space position of crossing on each edge
    if (d_edgeTable[cubeIndex] &    1) edgeVerts[0]  = interp(c[0],d[0],c[1],d[1]);
    if (d_edgeTable[cubeIndex] &    2) edgeVerts[1]  = interp(c[1],d[1],c[2],d[2]);
    if (d_edgeTable[cubeIndex] &    4) edgeVerts[2]  = interp(c[2],d[2],c[3],d[3]);
    if (d_edgeTable[cubeIndex] &    8) edgeVerts[3]  = interp(c[3],d[3],c[0],d[0]);
    if (d_edgeTable[cubeIndex] &   16) edgeVerts[4]  = interp(c[4],d[4],c[5],d[5]);
    if (d_edgeTable[cubeIndex] &   32) edgeVerts[5]  = interp(c[5],d[5],c[6],d[6]);
    if (d_edgeTable[cubeIndex] &   64) edgeVerts[6]  = interp(c[6],d[6],c[7],d[7]);
    if (d_edgeTable[cubeIndex] &  128) edgeVerts[7]  = interp(c[7],d[7],c[4],d[4]);
    if (d_edgeTable[cubeIndex] &  256) edgeVerts[8]  = interp(c[0],d[0],c[4],d[4]);
    if (d_edgeTable[cubeIndex] &  512) edgeVerts[9]  = interp(c[1],d[1],c[5],d[5]);
    if (d_edgeTable[cubeIndex] & 1024) edgeVerts[10] = interp(c[2],d[2],c[6],d[6]);
    if (d_edgeTable[cubeIndex] & 2048) edgeVerts[11] = interp(c[3],d[3],c[7],d[7]);

    // Record the base index before we add new vertices
    int base = offsets[i*GRID_N*GRID_N + j*GRID_N + k] * 3;

    for (int t = 0; d_triTable[cubeIndex][t] != -1; t += 3) {
        // Look up which edge crossing points form this triangle
        Vec3 v0 = edgeVerts[d_triTable[cubeIndex][t]];
        Vec3 v1 = edgeVerts[d_triTable[cubeIndex][t+1]];
        Vec3 v2 = edgeVerts[d_triTable[cubeIndex][t+2]];

        int vb = base + t;
        ov[vb] = v0; ov[vb+1] = v1; ov[vb+2] = v2;

        // Push 3 indices pointing to those vertices
        oi[vb] = vb; oi[vb+1] = vb+1; oi[vb+2] = vb+2;

        on[vb] = d_gradNormal(v0, sc); on[vb+1] = d_gradNormal(v1, sc); on[vb+2] = d_gradNormal(v2, sc);
    }
}

/*
 * Before generating the mesh:
 * GPU version of buildScalarField - takes device pointers directly
 * so no H->D copy needed after SPH step
 */
void buildScalarField(Particle* d_particles, int n, int* d_hashHead, int* d_hashNext) {
    GpuHash hash{ d_hashHead, d_hashNext, SPH::H };

    constexpr int BLOCK = 8;
    int numBlocks = (GRID_N + BLOCK) / BLOCK;
    buildScalarField_kernel<<<dim3(numBlocks,numBlocks,numBlocks), dim3(BLOCK,BLOCK,BLOCK)>>>(d_particles, n, hash, d_scalar);
    zeroifyBoundary_kernel<<<dim3((GRID_N+15)/16,(GRID_N+15)/16), dim3(16,16)>>>(d_scalar);
}

/*
 * Part 2-5 of Mesh Generation
 * GPU version - count -> prefix scan -> emit
 */
void marchCubes(std::vector<Vec3>& vertexBuffer, std::vector<uint32_t>& indexBuffer, std::vector<Vec3>& normalBuffer) {
    constexpr int BLOCK = 8;
    int numBlocks = (GRID_N + BLOCK - 1) / BLOCK;
    dim3 block(BLOCK, BLOCK, BLOCK);
    dim3 grid(numBlocks, numBlocks, numBlocks);

    countTriangles_kernel<<<grid, block>>>(d_scalar, d_triCounts);

    // prefix scan gives each cube its write offset - no race conditions
    thrust::exclusive_scan(
        thrust::device_ptr<int>(d_triCounts),
        thrust::device_ptr<int>(d_triCounts + GRID_N*GRID_N*GRID_N),
        thrust::device_ptr<int>(d_triOffsets));

    int lastCnt, lastOff;
    cudaMemcpy(&lastCnt, d_triCounts  + GRID_N*GRID_N*GRID_N - 1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&lastOff, d_triOffsets + GRID_N*GRID_N*GRID_N - 1, sizeof(int), cudaMemcpyDeviceToHost);
    int total = lastOff + lastCnt;
    if (total <= 0) return;

    if (total > d_maxTris) {
        cudaFree(d_verts); cudaFree(d_idx); cudaFree(d_norms);
        cudaMalloc(&d_verts, total*3*sizeof(Vec3));
        cudaMalloc(&d_idx,   total*3*sizeof(uint32_t));
        cudaMalloc(&d_norms, total*3*sizeof(Vec3));
        d_maxTris = total;
    }

    emitTriangles_kernel<<<grid, block>>>(d_scalar, d_triOffsets, d_verts, d_idx, d_norms);
    cudaDeviceSynchronize();

    int nv = total*3;
    vertexBuffer.resize(nv); indexBuffer.resize(nv); normalBuffer.resize(nv);
    cudaMemcpy(vertexBuffer.data(), d_verts, nv*sizeof(Vec3),     cudaMemcpyDeviceToHost);
    cudaMemcpy(indexBuffer.data(),  d_idx,   nv*sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(normalBuffer.data(), d_norms, nv*sizeof(Vec3),     cudaMemcpyDeviceToHost);
}
