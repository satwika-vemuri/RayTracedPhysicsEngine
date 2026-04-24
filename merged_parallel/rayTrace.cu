
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#include "rayTrace_gpu.h"

using std::vector;

#include <vector>
#include <algorithm>
using std::vector;

// to enforce a range that we want
inline int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

vector<vector<int>> assignTriangles(vector<Triangle>& sceneTriangles,const Point3& leftCorner,
                                    const Point3& rightCorner,int boxDimension)
{
    int numCells = boxDimension * boxDimension * boxDimension;
    vector<vector<int>> trianglesPerBox(numCells);

    Vec3 intervalSize = (rightCorner - leftCorner) / boxDimension;

    for (int i = 0; i < (int)sceneTriangles.size(); i++) {

        Triangle& t = sceneTriangles[i];

        const Point3& a = t.v1;
        const Point3& b = t.v2;
        const Point3& c = t.v3;

        float minx = fminf(a.x, fminf(b.x, c.x));
        float miny = fminf(a.y, fminf(b.y, c.y));
        float minz = fminf(a.z, fminf(b.z, c.z));

        float maxx = fmaxf(a.x, fmaxf(b.x, c.x));
        float maxy = fmaxf(a.y, fmaxf(b.y, c.y));
        float maxz = fmaxf(a.z, fmaxf(b.z, c.z));

        // convert bounds to indices
        int x_start = (int)floor((minx - leftCorner.x) / intervalSize.x);
        int y_start = (int)floor((miny - leftCorner.y) / intervalSize.y);
        int z_start = (int)floor((minz - leftCorner.z) / intervalSize.z);

        int x_end = (int)ceil((maxx - leftCorner.x) / intervalSize.x);
        int y_end = (int)ceil((maxy - leftCorner.y) / intervalSize.y);
        int z_end = (int)ceil((maxz - leftCorner.z) / intervalSize.z);

        // assign to grind
        x_start = clampi(x_start, 0, boxDimension - 1);
        y_start = clampi(y_start, 0, boxDimension - 1);
        z_start = clampi(z_start, 0, boxDimension - 1);

        x_end = clampi(x_end, 0, boxDimension - 1);
        y_end = clampi(y_end, 0, boxDimension - 1);
        z_end = clampi(z_end, 0, boxDimension - 1);

        // IMPORTANT: expand by 1 voxel padding
        x_start = max(0, x_start - 1);
        y_start = max(0, y_start - 1);
        z_start = max(0, z_start - 1);

        x_end = min(boxDimension - 1, x_end + 1);
        y_end = min(boxDimension - 1, y_end + 1);
        z_end = min(boxDimension - 1, z_end + 1);

        // insert triangle into all overlapping cells
        for (int x = x_start; x <= x_end; x++) {
            for (int y = y_start; y <= y_end; y++) {
                for (int z = z_start; z <= z_end; z++) {

                    int idx = x + y * boxDimension + z * boxDimension * boxDimension;
                    trianglesPerBox[idx].push_back(i);
                }
            }
        }
    }

    return trianglesPerBox;
}

vector<Triangle> constructSceneTriangles(const vector<Point3>& vertexBuffer, const vector<uint32_t>& indexBuffer, const vector<Vec3>& normalBuffer){
    vector<Triangle> sceneTriangles;
    sceneTriangles.reserve(indexBuffer.size()/3);
    

    for(size_t i = 0; i+3 <= indexBuffer.size(); i+=3){
        Point3 v1 = vertexBuffer[indexBuffer[i]];
        Point3 v2 = vertexBuffer[indexBuffer[i + 1]];
        Point3 v3 = vertexBuffer[indexBuffer[i + 2]];

        Vec3 n1 = normalBuffer[indexBuffer[i]];
        Vec3 n2 = normalBuffer[indexBuffer[i + 1]];
        Vec3 n3 = normalBuffer[indexBuffer[i + 2]];
        //cout << "creating triangle" << endl;
        Triangle t = Triangle(v1, v2, v3);
        t.n1 = n1;
        t.n2 = n2;
        t.n3 = n3;
        sceneTriangles.push_back(t);
    }

    return sceneTriangles;
}

// assumes that we are positioning the camera further "back" on the z axis
Point3 computeCameraPosition(const Point3& leftCorner, const Point3& rightCorner){
    float D = std::max(std::max(std::abs(rightCorner.x - leftCorner.x),std::abs(rightCorner.y - leftCorner.y)),std::abs(rightCorner.z - leftCorner.z));
    
    // cameraPos = center + (0, 0, -2D)
    float x = (leftCorner.x + rightCorner.x)/2; 
    float y = (leftCorner.y + rightCorner.y)/2;
    float z = (leftCorner.z + rightCorner.z)/2 - 2*D; 
    

    return Point3(x, y, z);
    
}