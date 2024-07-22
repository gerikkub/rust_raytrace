
#pragma once
#include "rust/cxx.h"

struct CudaRay;
struct CudaTriangle;

void exec_cuda_raytrace(rust::Vec<CudaTriangle> const & alltris,
                        rust::Vec<CudaRay> const & rays,
                        rust::Vec<uint32_t> const & tris,
                        const uint32_t trilist_stride,
                        const uint32_t stream_num,
                        rust::Slice<std::uint32_t> hit_nums,
                        std::array<std::uint64_t, 4> &runtimes);