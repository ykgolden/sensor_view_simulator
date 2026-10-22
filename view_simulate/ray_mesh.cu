// One thread per ray. Faces stay on the device between calls that reuse the same host buffers.
#include <cuda_runtime.h>
#include <math.h>

static const float* g_host_v0 = nullptr;
static const float* g_host_e1 = nullptr;
static const float* g_host_e2 = nullptr;
static int g_faces = -1;
static float* g_v0 = nullptr;
static float* g_e1 = nullptr;
static float* g_e2 = nullptr;

__global__ void ray_kernel(
    const float* __restrict__ origins,
    const float* __restrict__ dirs,
    const float* __restrict__ v0,
    const float* __restrict__ e1,
    const float* __restrict__ e2,
    int n_rays,
    int n_faces,
    float clearance,
    float* __restrict__ out) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n_rays) {
        return;
    }
    float ox = origins[index * 3];
    float oy = origins[index * 3 + 1];
    float oz = origins[index * 3 + 2];
    float dx = dirs[index * 3];
    float dy = dirs[index * 3 + 1];
    float dz = dirs[index * 3 + 2];
    float best = INFINITY;
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
        if (fabsf(det) < 1e-8f) {
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

static void release_faces() {
    cudaFree(g_v0);
    cudaFree(g_e1);
    cudaFree(g_e2);
    g_v0 = g_e1 = g_e2 = nullptr;
    g_host_v0 = g_host_e1 = g_host_e2 = nullptr;
    g_faces = -1;
}

extern "C" int ray_mesh_distances(
    const float* origins,
    const float* dirs,
    const float* v0,
    const float* e1,
    const float* e2,
    int n_rays,
    int n_faces,
    float clearance,
    float* out) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 1 || n_rays <= 0) {
        return 1;
    }
    cudaError_t err = cudaSuccess;
    float* d_origins = nullptr;
    float* d_dirs = nullptr;
    float* d_out = nullptr;
    size_t ray_bytes = (size_t)n_rays * 3 * sizeof(float);
    size_t face_bytes = (size_t)n_faces * 3 * sizeof(float);
    if (g_host_v0 != v0 || g_host_e1 != e1 || g_host_e2 != e2 || g_faces != n_faces) {
        release_faces();
        if (n_faces > 0) {
            if ((err = cudaMalloc(&g_v0, face_bytes)) || (err = cudaMalloc(&g_e1, face_bytes)) ||
                (err = cudaMalloc(&g_e2, face_bytes)) ||
                (err = cudaMemcpy(g_v0, v0, face_bytes, cudaMemcpyHostToDevice)) ||
                (err = cudaMemcpy(g_e1, e1, face_bytes, cudaMemcpyHostToDevice)) ||
                (err = cudaMemcpy(g_e2, e2, face_bytes, cudaMemcpyHostToDevice))) {
                release_faces();
                return 2;
            }
            g_host_v0 = v0;
            g_host_e1 = e1;
            g_host_e2 = e2;
            g_faces = n_faces;
        }
    }
    if ((err = cudaMalloc(&d_origins, ray_bytes)) || (err = cudaMalloc(&d_dirs, ray_bytes)) ||
        (err = cudaMalloc(&d_out, (size_t)n_rays * sizeof(float))) ||
        (err = cudaMemcpy(d_origins, origins, ray_bytes, cudaMemcpyHostToDevice)) ||
        (err = cudaMemcpy(d_dirs, dirs, ray_bytes, cudaMemcpyHostToDevice))) {
        cudaFree(d_origins);
        cudaFree(d_dirs);
        cudaFree(d_out);
        return 3;
    }
    int threads = 128;
    int blocks = (n_rays + threads - 1) / threads;
    ray_kernel<<<blocks, threads>>>(d_origins, d_dirs, g_v0, g_e1, g_e2, n_rays, n_faces, clearance, d_out);
    err = cudaGetLastError();
    if (!err) {
        err = cudaDeviceSynchronize();
    }
    if (!err) {
        err = cudaMemcpy(out, d_out, (size_t)n_rays * sizeof(float), cudaMemcpyDeviceToHost);
    }
    cudaFree(d_origins);
    cudaFree(d_dirs);
    cudaFree(d_out);
    return err ? 4 : 0;
}

extern "C" int gpu_device_name(char* buffer, int length) {
    if (buffer == nullptr || length < 2) {
        return 1;
    }
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        buffer[0] = '\0';
        return 2;
    }
    int i = 0;
    for (; prop.name[i] != '\0' && i < length - 1; ++i) {
        buffer[i] = prop.name[i];
    }
    buffer[i] = '\0';
    return 0;
}
