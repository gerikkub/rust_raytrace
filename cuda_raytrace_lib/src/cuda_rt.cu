
#include "rust/cxx.h"
#include "cuda_raytrace_lib/src/cuda_raytrace.rs.h"

#include <limits>
#include <chrono>

#include <cuda_runtime.h>

// Memory Layout

// Triangle
// - Incenter (Vec3)
// - Normal (Vec3)
// - Sides (Vec3)
// - Number (U32)

// Possible kernels
// - Take one ray and N triangles. Find the closest hit
// - Take one triangle and N rays. Record the hit times for all triangles
//   - Then look for the best hit for a given ray. CAS at a global memory location?
// - Take N rays and M triangles. Perform hit detection for N rays at once
//   - Each thread block corresponds to 1 ray and up to 1024 triangles
//   - How to access triangles effectively
//     - Effectively a random access lookup. Might be okay?
//   - Rays stay in global memory
//   - Each kernel invocation calculates one level set of rays cast
//     - Invoke multiple times to for reflections
//   - At the end of all invocations, rays are an array of triangle hits
//     - Colors still need to be mixed

enum {
    INCENTER_0_OFF = 0,
    INCENTER_1_OFF = 1,
    INCENTER_2_OFF = 2,

    NORM_0_OFF = 3,
    NORM_1_OFF = 4,
    NORM_2_OFF = 5,

    SIDE_0_0_OFF = 6,
    SIDE_0_1_OFF = 7,
    SIDE_0_2_OFF = 8,

    SIDE_1_0_OFF = 9,
    SIDE_1_1_OFF = 10,
    SIDE_1_2_OFF = 11,

    SIDE_2_0_OFF = 12,
    SIDE_2_1_OFF = 13,
    SIDE_2_2_OFF = 14,

    NUM_OFF = 15,

    TRILIST_MAX = 16,
};

enum {
    RAY_P_0_OFF = 0,
    RAY_P_1_OFF = 1,
    RAY_P_2_OFF = 2,

    RAY_V_0_OFF = 3,
    RAY_V_1_OFF = 4,
    RAY_V_2_OFF = 5,

    RAYLIST_MAX = 6,
};

enum {
    TRI_T_OFF = 0,
    TRI_NUM_OFF = 1
};

__device__ float fetch_thread_value_f32(float* base, unsigned int off) {
    return base[off*blockDim.x + threadIdx.x];
}

__device__ uint32_t fetch_thread_value_u32(float* base, unsigned int off) {
    uint32_t* u32_base = reinterpret_cast<uint32_t*>(base);
    return u32_base[off*blockDim.x + threadIdx.x];
}

__device__ void write_thread_value_f32(float* base, unsigned int off, float val) {
    base[off*blockDim.x + threadIdx.x] = val;
}

__device__ void write_thread_value_u32(float* base, unsigned int off, uint32_t val) {
    uint32_t* u32_base = reinterpret_cast<uint32_t*>(base);
    u32_base[off*blockDim.x + threadIdx.x] = val;
}

__device__ float3 fetch_thread_vec_f32(float* base, unsigned int off) {
    float3 v;
    v.x = fetch_thread_value_f32(base, off);
    v.y = fetch_thread_value_f32(base, off + 1);
    v.z = fetch_thread_value_f32(base, off + 2);
    return v;
}

__device__ float fetch_ray_value_f32(float* base, unsigned int off) {
    return base[off*gridDim.x + blockIdx.x];
}

__device__ float3 fetch_ray_vec_f32(float* base, unsigned int off) {
    float3 v;
    v.x = fetch_ray_value_f32(base, off);
    v.y = fetch_ray_value_f32(base, off + 1);
    v.z = fetch_ray_value_f32(base, off + 2);
    return v;
}


__device__ float vec_dot(const float3 a, const float3 b) {
    float t = a.x * b.x;
    t = fmaf(a.y, b.y, t);
    return fmaf(a.z, b.z, t);
}

__device__ float3 vec_add(const float3 a, const float3 b) {
    float3 o;
    o.x = a.x + b.x;
    o.y = a.y + b.y;
    o.z = a.z + b.z;
    return o;
}

__device__ float3 vec_sub(const float3 a, const float3 b) {
    float3 o;
    o.x = a.x - b.x;
    o.y = a.y - b.y;
    o.z = a.z - b.z;
    return o;
}

__device__ float3 vec_scale(const float3 a, const float b) {
    float3 o;
    o.x = a.x * b;
    o.y = a.y * b;
    o.z = a.z * b;
    return o;
}


__global__ void cuda_triangle_intersect(float* rays,
                                        float* tripointer_full,
                                        uint32_t* ray_hitnums,
                                        float* ray_hittimes) {

    extern __shared__ float thread_temp[];
    float* hit_times = thread_temp;
    uint32_t* hit_nums = reinterpret_cast<uint32_t*>(thread_temp + blockDim.x);

    float* tris = &tripointer_full[blockIdx.x * TRILIST_MAX * blockDim.x];

    bool hit = false;
    float t = 0;
    uint32_t tri_num = fetch_thread_value_u32(tris, NUM_OFF);

    hit_nums[threadIdx.x] = tri_num;

    // if (threadIdx.x < 4) {
    //     printf("tri_num: %d %d %d\n", blockIdx.x, threadIdx.x, tri_num);
    // }

    if (tri_num != 0) {
        // Offset ray intersection by triangle center
        float3 incenter = fetch_thread_vec_f32(tris, INCENTER_0_OFF);
        // if (tri_num == 858) {
        //     printf("Incenter %f %f %f\n",
        //            incenter.x, incenter.y, incenter.z);
        // }

        float3 ray_p = fetch_ray_vec_f32(rays, RAY_P_0_OFF);
        // if (tri_num == 858) {
        //     printf("Ray P %f %f %f\n",
        //            ray_p.x, ray_p.y, ray_p.z);
        // }

        float3 a = vec_sub(incenter, ray_p);

        // Calculate ray hit time with plane
        float3 norm = fetch_thread_vec_f32(tris, NORM_0_OFF);
        // if (tri_num == 858) {
        //     printf("norm %f %f %f\n",
        //            norm.x, norm.y, norm.z);
        // }
        float t0 = vec_dot(norm, a);
        float3 ray_v = fetch_ray_vec_f32(rays, RAY_V_0_OFF);
        // if (tri_num == 858) {
        //     printf("Ray V %f %f %f\n",
        //            ray_v.x, ray_v.y, ray_v.z);
        // }
        float t1 = vec_dot(norm, ray_v);
        t = fdividef(t0, t1);

        // Calculate hit point in plane
        float3 b = vec_scale(ray_v, t);
        float3 p = vec_add(ray_p, b);
        float3 ip = vec_sub(p, incenter);

        // Calculate hit vectors for all three triangle sides
        float3 side_0 = fetch_thread_vec_f32(tris, SIDE_0_0_OFF);
        float side_0_len = norm3df(side_0.x, side_0.y, side_0.z);
        float3 side_0_norm = vec_scale(side_0, __frcp_rn(side_0_len));
        float d_0 = vec_dot(ip, side_0_norm);
        hit = d_0 < side_0_len;
        // if (blockIdx.x == 0 && threadIdx.x == 1) {
        // }
        // TODO: Could early exit if all threads fail hit detection

        float3 side_1 = fetch_thread_vec_f32(tris, SIDE_1_0_OFF);
        float side_1_len = norm3df(side_1.x, side_1.y, side_1.z);
        float3 side_1_norm = vec_scale(side_1, __frcp_rn(side_1_len));
        float d_1 = vec_dot(ip, side_1_norm);
        hit = hit && (d_1 < side_1_len);
        // if (blockIdx.x == 0 && threadIdx.x == 1) {
        // }
        // TODO: Could early exit if all threads fail hit detection

        float3 side_2 = fetch_thread_vec_f32(tris, SIDE_2_0_OFF);
        float side_2_len = norm3df(side_2.x, side_2.y, side_2.z);
        float3 side_2_norm = vec_scale(side_2, __frcp_rn(side_2_len));
        float d_2 = vec_dot(ip, side_2_norm);
        hit = hit && (d_2 < side_2_len);
        // if (blockIdx.x == 0 && threadIdx.x == 1) {
        // }

        // if (hit && t > 0.) {
        //     printf("CT hit %d %d %d %f\n", blockIdx.x, threadIdx.x, tri_num, t);
        //     printf("Ray: (%f %f %f) (%f %f %f)\n",
        //            ray_p.x, ray_p.y, ray_p.z, 
        //            ray_v.x, ray_v.y, ray_v.z);
        //     printf("Tri: (%f %f %f) (%f %f %f)\n",
        //            incenter.x, incenter.y, incenter.z, 
        //            norm.x, norm.y, norm.z);
        //     printf("Side 0 (%f %f %f)\n",
        //            side_0.x, side_0.y, side_0.z);
        //     printf(" len: %f\n", side_0_len);
        //     printf(" d: %f\n", d_0);
        //     printf("Side 1 (%f %f %f)\n",
        //            side_1.x, side_1.y, side_1.z);
        //     printf(" len: %f\n", side_1_len);
        //     printf(" d: %f\n", d_1);
        //     printf("Side 2 (%f %f %f)\n",
        //            side_2.x, side_2.y, side_2.z);
        //     printf(" len: %f\n", side_2_len);
        //     printf(" d: %f\n", d_2);
        // }
    }


    // Filter out negative times and misses
    if (!hit || t < 0.) {
        t = FLT_MAX;
    }

    // if (threadIdx.x < 4) {
    //     if (hit) {
    //         printf("CUDA Thread %d %d %d Hit\n",
    //                blockIdx.x, threadIdx.x, tri_num);
    //     } else {
    //         printf("CUDA Thread %d %d %d Miss\n",
    //                blockIdx.x, threadIdx.x, tri_num);
    //     }
    // }

    // Place hit times in shared memory
    hit_times[threadIdx.x] = t;
    __syncthreads();

    // Find minimum hit time in the thread block
    // Requires log2(1024) == 10 steps
    for (int i = 1; i < __ffs(blockDim.x); i++) {
        if (threadIdx.x >= (blockDim.x >> i)) {
            return;
        }

        uint32_t cmp_offset = blockDim.x >> i;
        float t1 = hit_times[threadIdx.x];
        float t2 = hit_times[threadIdx.x + cmp_offset];

        uint32_t t1_num = hit_nums[threadIdx.x];
        uint32_t t2_num = hit_nums[threadIdx.x + cmp_offset];

        // if (threadIdx.x == (864 - 512)) {
        //     printf("Compare %d %f to %d %f\n",
        //            t1_num, t1,
        //            t2_num, t2);
        // }

        // if (t1_num == 858 || t2_num == 858) {
        //     printf("Compare %d %f to %d %f\n",
        //            t1_num, t1,
        //            t2_num, t2);
        // }

        float tmin = fminf(t1, t2);
        uint32_t tmin_num;
        if (tmin == t1) {
            tmin_num = t1_num;
        } else {
            tmin_num = t2_num;
        }

        hit_times[threadIdx.x] = tmin;
        hit_nums[threadIdx.x] = tmin_num;

        __syncthreads();
    }

    // Write the minimum hit time to ray_results
    if (threadIdx.x == 0) {
        // printf("CUDA Hit %d %f\n", hit_nums[0], hit_times[0]);
        if (hit_times[0] == FLT_MAX) {
            // printf("CUDA Miss %d\n", blockIdx.x);
            ray_hitnums[blockIdx.x] = 0;
        } else {
            // printf("CUDA Hit %d %d %f\n", blockIdx.x, hit_nums[0], hit_times[0]);
            ray_hitnums[blockIdx.x] = hit_nums[0];
            ray_hittimes[blockIdx.x] = hit_times[0];
        }
    }
}

void exec_cuda_raytrace(rust::Vec<CudaTriangle> const & alltris,
                        rust::Vec<CudaRay> const & rays,
                        rust::Vec<uint32_t> const & tris,
                        const uint32_t trilist_stride,
                        const uint32_t stream_num,
                        rust::Slice<std::uint32_t> hit_nums_out,
                        rust::Slice<float> hit_times_out,
                        std::array<std::uint64_t, 4> &runtimes_out) {

    auto t0 = std::chrono::high_resolution_clock::now();

    // Build memory layout for cuda execution
    float* raylist = new float[RAYLIST_MAX * rays.size()];
    float* trilist = new float[TRILIST_MAX * trilist_stride * rays.size()];

    for (int idx = 0; idx < rays.size(); idx++) {
        raylist[RAY_P_0_OFF * rays.size() + idx] = rays[idx].a.v[0];
        raylist[RAY_P_1_OFF * rays.size() + idx] = rays[idx].a.v[1];
        raylist[RAY_P_2_OFF * rays.size() + idx] = rays[idx].a.v[2];

        raylist[RAY_V_0_OFF * rays.size() + idx] = rays[idx].u.v[0];
        raylist[RAY_V_1_OFF * rays.size() + idx] = rays[idx].u.v[1];
        raylist[RAY_V_2_OFF * rays.size() + idx] = rays[idx].u.v[2];
    }

    for (int ridx = 0; ridx < rays.size(); ridx++) {
        int roff = ridx * TRILIST_MAX * trilist_stride;
        for (int idx = 0; idx < trilist_stride; idx++) {
            int tri_idx = ridx * trilist_stride + idx;
            trilist[roff + INCENTER_0_OFF * trilist_stride + idx] = alltris[tris[tri_idx]].incenter.v[0];
            trilist[roff + INCENTER_1_OFF * trilist_stride + idx] = alltris[tris[tri_idx]].incenter.v[1];
            trilist[roff + INCENTER_2_OFF * trilist_stride + idx] = alltris[tris[tri_idx]].incenter.v[2];

            trilist[roff + NORM_0_OFF * trilist_stride + idx] = alltris[tris[tri_idx]].norm.v[0];
            trilist[roff + NORM_1_OFF * trilist_stride + idx] = alltris[tris[tri_idx]].norm.v[1];
            trilist[roff + NORM_2_OFF * trilist_stride + idx] = alltris[tris[tri_idx]].norm.v[2];

            trilist[roff + SIDE_0_0_OFF * trilist_stride + idx] = alltris[tris[tri_idx]].sides[0].v[0];
            trilist[roff + SIDE_0_1_OFF * trilist_stride + idx] = alltris[tris[tri_idx]].sides[0].v[1];
            trilist[roff + SIDE_0_2_OFF * trilist_stride + idx] = alltris[tris[tri_idx]].sides[0].v[2];

            trilist[roff + SIDE_1_0_OFF * trilist_stride + idx] = alltris[tris[tri_idx]].sides[1].v[0];
            trilist[roff + SIDE_1_1_OFF * trilist_stride + idx] = alltris[tris[tri_idx]].sides[1].v[1];
            trilist[roff + SIDE_1_2_OFF * trilist_stride + idx] = alltris[tris[tri_idx]].sides[1].v[2];

            trilist[roff + SIDE_2_0_OFF * trilist_stride + idx] = alltris[tris[tri_idx]].sides[2].v[0];
            trilist[roff + SIDE_2_1_OFF * trilist_stride + idx] = alltris[tris[tri_idx]].sides[2].v[1];
            trilist[roff + SIDE_2_2_OFF * trilist_stride + idx] = alltris[tris[tri_idx]].sides[2].v[2];

            *reinterpret_cast<uint32_t*>(&trilist[roff + NUM_OFF * trilist_stride + idx]) = tris[tri_idx];
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    float* cuda_raylist;
    cudaMalloc(&cuda_raylist, rays.size() * RAYLIST_MAX * sizeof(float));
    float* cuda_trilist;
    cudaMalloc(&cuda_trilist, rays.size() * trilist_stride * TRILIST_MAX * sizeof(float));

    cudaMemcpy(cuda_raylist, raylist,
               rays.size() * RAYLIST_MAX * sizeof(float),
               cudaMemcpyHostToDevice);

    cudaMemcpy(cuda_trilist, trilist,
               rays.size() * trilist_stride * TRILIST_MAX * sizeof(float),
               cudaMemcpyHostToDevice);

    uint32_t* cuda_ray_hitnums;
    cudaMalloc(&cuda_ray_hitnums, rays.size() * sizeof(uint32_t));

    float* cuda_ray_hittimes;
    cudaMalloc(&cuda_ray_hittimes, rays.size() * sizeof(float));

    auto t2 = std::chrono::high_resolution_clock::now();

    cuda_triangle_intersect<<<rays.size(),
                              trilist_stride,
                              trilist_stride * 2 * sizeof(float)>>>
                            (cuda_raylist, cuda_trilist, cuda_ray_hitnums, cuda_ray_hittimes);

    auto t3 = std::chrono::high_resolution_clock::now();

    cudaMemcpy(hit_nums_out.data(), cuda_ray_hitnums,
               rays.size() * sizeof(uint32_t),
               cudaMemcpyDeviceToHost);

    cudaMemcpy(hit_times_out.data(), cuda_ray_hittimes,
               rays.size() * sizeof(float),
               cudaMemcpyDeviceToHost);

    // printf("Hit nums: ");
    // for (int i = 0; i < rays.size(); i++) {
    //     printf("%d ", hit_nums_out[i]);
    // }
    // printf("\n");

    cudaFree(cuda_raylist);
    cudaFree(cuda_trilist);
    cudaFree(cuda_ray_hitnums);

    delete[] raylist;
    delete[] trilist;

    auto t4 = std::chrono::high_resolution_clock::now();

    uint64_t t_host_mem = std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count();
    uint64_t t_host_dev_copy = std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count();
    uint64_t t_cuda_exec = std::chrono::duration_cast<std::chrono::nanoseconds>(t3-t2).count();
    uint64_t t_dev_host_copy = std::chrono::duration_cast<std::chrono::nanoseconds>(t4-t3).count();

    runtimes_out[0] = t_host_mem;
    runtimes_out[1] = t_host_dev_copy;
    runtimes_out[2] = t_cuda_exec;
    runtimes_out[3] = t_dev_host_copy;
}