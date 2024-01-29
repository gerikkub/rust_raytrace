
#pragma once
#include "rust/cxx.h"
// #include "cuda_raytrace/src/main.rs.h"

struct CudaColor;
struct CudaRay;
struct CudaTriangle;

void preload_triangles_cuda(rust::Vec<::CudaTriangle> const& tris);

CudaColor project_ray_cuda(CudaRay const & r,
                           rust::Vec<std::size_t> const & ctris,
                           std::size_t maxdepth,
                           std::size_t depth,
                           std::array<std::uint64_t, 3> &runtimes);