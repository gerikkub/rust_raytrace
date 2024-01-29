
#include "rust/cxx.h"
#include "cuda_raytrace/src/main.rs.h"

#include <cfloat>
#include <iostream>
#include <unistd.h>

#include <cuda_runtime.h>

__device__ float vec_dot(float a[3], float b[3]) {
    return a[0] * b[0] +
           a[1] * b[1] +
           a[2] * b[2];
}

__device__ void vec_add(float a[3], float b[3], float out[3]) {
    out[0] = a[0] + b[0];
    out[1] = a[1] + b[1];
    out[2] = a[2] + b[2];
}

__device__ void vec_sub(float a[3], float b[3], float out[3]) {
    out[0] = a[0] - b[0];
    out[1] = a[1] - b[1];
    out[2] = a[2] - b[2];
}

__device__ void vec_mult(float a[3], const float b, float out[3]) {
    out[0] = a[0] * b;
    out[1] = a[1] * b;
    out[2] = a[2] * b;
}


__global__ void cuda_triangle_intersect(int* triidxs, float* r, float* incenters, float* norms, float* sides, float* dists, float* ts, bool* hit_edges) {


    // CudaTriangle* tri = &tris[blockDim.x];
    unsigned int _idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = triidxs[_idx];
    int idx3 = triidxs[_idx] * 3;
    int idx9 = triidxs[_idx] * 9;

    float incenter[3] = {incenters[idx3], incenters[idx3+1], incenters[idx3+2]};
    float ra[3] = {r[0], r[1], r[2]};
    float ru[3] = {r[3], r[4], r[5]};
    float norm[3] = {norms[idx3], norms[idx3+1], norms[idx3+2]};
    float side[3][3] = {{sides[idx9 + 0], sides[idx9 + 1], sides[idx9 + 2]},
                        {sides[idx9 + 3], sides[idx9 + 4], sides[idx9 + 5]},
                        {sides[idx9 + 6], sides[idx9 + 7], sides[idx9 + 8]}};
    float dist[3] = {dists[idx3], dists[idx3+1], dists[idx3+2]};

    float a[3];
    vec_sub(incenter, ra, a);
    float t = vec_dot(norm, a) / vec_dot(norm, ru);
    float b[3];
    float p[3];
    vec_mult(ru, t, b);
    vec_add(ra, b, p);
    float ip[3];
    vec_sub(p, incenter, ip);

    // float t = vec_dot(tri->norm, vec_sub(tri->incenter, r.a)) / vec_dot(tri->norm, r.u);
    // CudaVec3 p = vec_add(r.a, vec_mult(r.u, t));
    // CudaVec3 ip = vec_sub(p, tri->incenter);

    bool hit_edge = false;

    bool outside = false;
    for (int jdx = 0; jdx < 3; jdx++) {
        // float dist = vec_dot(ip, tri->sides[idx]);
        float jdist = vec_dot(ip, side[jdx]);
        if (jdist > dist[jdx]) {
            outside = true;
        } else if (jdist > dist[jdx] * 0.95) {
            hit_edge = true;
        }
    }

    ts[_idx] = outside ? -1 : t;
    hit_edges[_idx] = hit_edge;
}

#define cudaAssert(err) \
do { \
    if (err != cudaSuccess) { \
        std::cout << "Cuda Error: " << cudaGetErrorString(err) << ": " << __LINE__ << std::endl; \
        assert(err == cudaSuccess); \
    } \
} while(0)
//#define cudaAssert(err) assert((void(cudaGetErrorString(err)), err == cudaSuccess))

uint64_t gettime_nanos(int clock) {
    struct timespec t;
    int stat = clock_gettime(clock, &t);
    assert(stat == 0);
    return t.tv_sec * (1000*1000*1000) + t.tv_nsec;
}

unsigned int iDivUp(unsigned int a, unsigned int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

static float* h_incenters = nullptr;
static float* h_norms = nullptr;
static float* h_sides = nullptr;
static float* h_dists = nullptr;

static float* d_incenters = nullptr;
static float* d_norms = nullptr;
static float* d_sides = nullptr;
static float* d_dists = nullptr;

void preload_triangles_cuda(rust::Vec<::CudaTriangle> const& tris) {

    cudaError_t err;

    h_incenters = new float[tris.size()*3];
    h_norms = new float[tris.size()*3];
    h_sides = new float[tris.size()*3*3];
    h_dists = new float[tris.size()*3];
    for (int idx = 0; idx < tris.size(); idx++) {
        h_incenters[3*idx] = tris[idx].incenter.v[0];
        h_incenters[3*idx+1] = tris[idx].incenter.v[1];
        h_incenters[3*idx+2] = tris[idx].incenter.v[2];

        h_norms[3*idx] = tris[idx].norm.v[0];
        h_norms[3*idx+1] = tris[idx].norm.v[1];
        h_norms[3*idx+2] = tris[idx].norm.v[2];

        h_sides[9*idx] = tris[idx].sides[0].v[0];
        h_sides[9*idx+1] = tris[idx].sides[0].v[1];
        h_sides[9*idx+2] = tris[idx].sides[0].v[2];
        h_sides[9*idx+3] = tris[idx].sides[1].v[0];
        h_sides[9*idx+4] = tris[idx].sides[1].v[1];
        h_sides[9*idx+5] = tris[idx].sides[1].v[2];
        h_sides[9*idx+6] = tris[idx].sides[2].v[0];
        h_sides[9*idx+7] = tris[idx].sides[2].v[1];
        h_sides[9*idx+8] = tris[idx].sides[2].v[2];

        h_dists[3*idx] = tris[idx].side_lens[0];
        h_dists[3*idx+1] = tris[idx].side_lens[1];
        h_dists[3*idx+2] = tris[idx].side_lens[2];
    }
    d_incenters = nullptr;
    err = cudaMalloc((void**)&d_incenters, tris.size()*3*sizeof(float));
    cudaAssert(err);
    err = cudaMemcpy(d_incenters, h_incenters, tris.size()*3*sizeof(float), cudaMemcpyHostToDevice);
    cudaAssert(err);

    d_norms = nullptr;
    err = cudaMalloc((void**)&d_norms, tris.size()*3*sizeof(float));
    cudaAssert(err);
    err = cudaMemcpy(d_norms, h_norms, tris.size()*3*sizeof(float), cudaMemcpyHostToDevice);
    cudaAssert(err);

    d_sides = nullptr;
    err = cudaMalloc((void**)&d_sides, tris.size()*9*sizeof(float));
    cudaAssert(err);
    err = cudaMemcpy(d_sides, h_sides, tris.size()*9*sizeof(float), cudaMemcpyHostToDevice);
    cudaAssert(err);

    d_dists = nullptr;
    err = cudaMalloc((void**)&d_dists, tris.size()*3*sizeof(float));
    cudaAssert(err);
    err = cudaMemcpy(d_dists, h_dists, tris.size()*3*sizeof(float), cudaMemcpyHostToDevice);
    cudaAssert(err);
}

CudaColor project_ray_cuda(CudaRay const & r,
                           rust::Vec<std::size_t> const & tris,
                           std::size_t ignore_objid,
                           std::size_t depth,
                           std::array<std::uint64_t, 3> &runtimes) {
    

    (void)depth;
    (void)ignore_objid;
    cudaError_t err;

    CudaColor background = { 128, 200, 255 };

    if (tris.size() == 0) {
        return background;
    }

    unsigned int numThreads = min((unsigned int)tris.size(), 256);
    unsigned int numBlocks = iDivUp(tris.size(), numThreads);
    unsigned int ecount = numThreads*numBlocks;

    // std::size_t max = 0;
    // for (auto i : tris) {
    //     if (i > max) {
    //         max = i;
    //     }
    // }
    // std::cout << "Max: " << max << " Count: " << tris.size() << std::endl;

    uint64_t t1 = gettime_nanos(CLOCK_THREAD_CPUTIME_ID);

    int* d_triidxs = nullptr;
    err = cudaMalloc((void**)& d_triidxs, tris.size()*sizeof(int));
    cudaAssert(err);
    err = cudaMemcpy(d_triidxs, tris.data(), tris.size()*sizeof(int), cudaMemcpyHostToDevice);

    float h_r[6] = {r.a.v[0], r.a.v[1], r.a.v[2],
                    r.u.v[0], r.u.v[1], r.u.v[2]};
    float* d_r = nullptr;
    err = cudaMalloc((void**)& d_r, 6*sizeof(float));
    cudaAssert(err);
    err = cudaMemcpy(d_r, h_r, 6*sizeof(float), cudaMemcpyHostToDevice);
    cudaAssert(err);

    float* cuda_ts = nullptr;
    err = cudaMalloc((void**)&cuda_ts, ecount*sizeof(float));
    cudaAssert(err);

    bool* cuda_hit_edges = nullptr;
    err = cudaMalloc((void**)&cuda_hit_edges, ecount*sizeof(bool));
    cudaAssert(err);


    uint64_t t2 = gettime_nanos(CLOCK_THREAD_CPUTIME_ID);
    uint64_t t1_mono = gettime_nanos(CLOCK_MONOTONIC);

// __global__ void cuda_triangle_intersect(float* r, float* incenters, float* norms, float* sides, float* dists, float* ts, bool* hit_edges) {
    cuda_triangle_intersect<<<numBlocks, numThreads>>>(d_triidxs, d_r, d_incenters, d_norms, d_sides, d_dists, cuda_ts, cuda_hit_edges);
    err = cudaGetLastError();
    cudaAssert(err);

    uint64_t t2_mono = gettime_nanos(CLOCK_MONOTONIC);
    uint64_t t3 = gettime_nanos(CLOCK_THREAD_CPUTIME_ID);


    float* ts = new float[ecount];
    bool* hit_edges = new bool[ecount];

    err = cudaMemcpy(ts, cuda_ts, (ecount)*sizeof(float), cudaMemcpyDeviceToHost);
    cudaAssert(err);
    err = cudaFree(cuda_ts);
    cudaAssert(err);

    err = cudaMemcpy(hit_edges, cuda_hit_edges, (ecount)*sizeof(bool), cudaMemcpyDeviceToHost);
    cudaAssert(err);
    err = cudaFree(cuda_hit_edges);
    cudaAssert(err);

    int min_idx = -1;
    float min_ts = FLT_MAX;
    for (std::size_t idx = 0; idx < tris.size(); idx++) {
        if (ts[idx] > 0. && ts[idx] < min_ts) {
            min_ts = ts[idx];
            min_idx = idx;
        }
    }

    uint64_t t4 = gettime_nanos(CLOCK_THREAD_CPUTIME_ID);

    runtimes[0] = t2-t1;
    runtimes[1] = t2_mono-t1_mono;
    runtimes[2] = t4-t3;

    CudaColor ret;

    if (min_idx >= 0) {
        if (hit_edges[min_idx]) {
            ret = CudaColor {
                0, 0, 0
            };
        } else {
            ret = CudaColor {
                255, 0, 0
            };
        }
    } else {
        ret = background;
    }

    delete[] ts;
    delete[] hit_edges;

    return ret;
}

// CudaColor project_ray_cpp(CudaRay const & r,
//                           rust::Vec<::CudaTriangle> const & tris,
//                           std::size_t ignore_objid,
//                           std::size_t depth) {
//     (void)depth;
    
//     float min_t = FLT_MAX;
//     CudaTriangle* min_tri = nullptr;
//     bool min_hit_edge = false;
//     for (auto tri : tris) {
//         if (tri.id == ignore_objid) {
//             continue;
//         }

//         float t = vec_dot(tri.norm, vec_sub(tri.incenter, r.a)) / vec_dot(tri.norm, r.u);
//         if (t < 0 || t > min_t) {
//             continue;
//         }
//         CudaVec3 p = vec_add(r.a, vec_mult(r.u, t));
//         CudaVec3 ip = vec_sub(p, tri.incenter);

//         bool hit_edge = false;

//         bool outside = false;
//         for (int idx = 0; idx < 3; idx++) {
//             float dist = vec_dot(ip, tri.sides[idx]);
//             if (dist > tri.side_lens[idx]) {
//                 outside = true;
//                 break;
//             } else if (dist > tri.side_lens[idx] * 0.95) {
//                 hit_edge = true;
//             }
//         }

//         if (outside) {
//             continue;
//         }

//         min_t = t;
//         min_tri = &tri;
//         min_hit_edge = hit_edge;
//     }

//     if (min_tri != nullptr) {
//         if (min_hit_edge) {
//             return CudaColor {
//                 0, 0, 0
//             };
//         } else {
//             return CudaColor {
//                 255, 0, 0
//             };
//         }
//     } else {
//         return CudaColor {
//             128, 200, 255
//         };
//     }
// }