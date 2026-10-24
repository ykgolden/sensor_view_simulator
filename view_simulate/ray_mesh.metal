// One thread per ray. Same intersection as ray_mesh.cu.
#include <metal_stdlib>
using namespace metal;

struct Uniforms {
    int n_rays;
    int n_faces;
    float clearance;
    float pad;
};

kernel void ray_kernel(
    device const float* origins [[buffer(0)]],
    device const float* dirs [[buffer(1)]],
    device const float* v0 [[buffer(2)]],
    device const float* e1 [[buffer(3)]],
    device const float* e2 [[buffer(4)]],
    constant Uniforms& uniforms [[buffer(5)]],
    device float* out [[buffer(6)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= (uint)uniforms.n_rays) {
        return;
    }
    float ox = origins[index * 3];
    float oy = origins[index * 3 + 1];
    float oz = origins[index * 3 + 2];
    float dx = dirs[index * 3];
    float dy = dirs[index * 3 + 1];
    float dz = dirs[index * 3 + 2];
    float best = INFINITY;
    int n_faces = uniforms.n_faces;
    float clearance = uniforms.clearance;
    for (int face = 0; face < n_faces; ++face) {
        float e2x = e2[face * 3];
        float e2y = e2[face * 3 + 1];
        float e2z = e2[face * 3 + 2];
        float px = dy * e2z - dz * e2y;
        float py = dz * e2x - dx * e2z;
        float pz = dx * e2y - dy * e2x;
        float e1x = e1[face * 3];
        float e1y = e1[face * 3 + 1];
        float e1z = e1[face * 3 + 2];
        float det = e1x * px + e1y * py + e1z * pz;
        if (fabs(det) < 1e-8f) {
            continue;
        }
        float inv = 1.0f / det;
        float tvx = ox - v0[face * 3];
        float tvy = oy - v0[face * 3 + 1];
        float tvz = oz - v0[face * 3 + 2];
        float u = (tvx * px + tvy * py + tvz * pz) * inv;
        if (u < -1e-5f || u > 1.0f + 1e-5f) {
            continue;
        }
        float qx = tvy * e1z - tvz * e1y;
        float qy = tvz * e1x - tvx * e1z;
        float qz = tvx * e1y - tvy * e1x;
        float v = (dx * qx + dy * qy + dz * qz) * inv;
        if (v < -1e-5f || u + v > 1.0f + 1e-5f) {
            continue;
        }
        float distance = (e2x * qx + e2y * qy + e2z * qz) * inv;
        if (distance >= clearance && distance < best) {
            best = distance;
        }
    }
    out[index] = best;
}
