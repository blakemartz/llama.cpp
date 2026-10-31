#include "mmv-mxfp8.cuh"

// Rows of MXFP8 blocks are 33 bytes per 32 values, so blocks are not 4-byte aligned; the kernel needs
// ne00 % 128 == 0 so that every row starts 4-byte aligned.

#define MMV_MXFP8_MAX_WARPS 8

// Four E4M3 codes (little-endian in q) -> four floats. Exact: every E4M3 value is an F16 value.
static __device__ __forceinline__ float4 mxfp8_decode4(const uint32_t q) {
#if !defined(GGML_USE_HIP)
    const __half2_raw lo = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t) (q & 0xFFFF), __NV_E4M3);
    const __half2_raw hi = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t) (q >> 16),    __NV_E4M3);
    const float2 a = __half22float2(*(const half2 *) &lo);
    const float2 b = __half22float2(*(const half2 *) &hi);
    return make_float4(a.x, a.y, b.x, b.y);
#else
    float v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t x = (q >> (8*i)) & 0xFF;
        const int e = (x >> 3) & 0xF, m = x & 0x7;
        const float f = e == 0 ? ldexpf((float) m, -9) : ldexpf((float) (8 + m), e - 10);
        v[i] = (x & 0x80) ? -f : f;
    }
    return make_float4(v[0], v[1], v[2], v[3]);
#endif // !defined(GGML_USE_HIP)
}

// Lane mapping: a row is ne00/4 four-value words in groups of P (P divides 8, so a group never leaves its block);
// the T threads that share a row take groups t, t+T, ... so the activation loads of a warp are consecutive
// (coalesced). P > 1 loads and applies each block scale once per P words and reads P+1 aligned words for P. A word's four E4M3 codes straddle two aligned 32-bit
// words (blocks are 33 bytes); both are read straight from global memory (L1 absorbs the overlap) and joined with
// a funnel shift. The inner loop is unrolled 8x so the loads issue before their first use. A row group of R rows
// reuses every activation load. splitk: all warps of the block share the R rows and split K (T = 32*nwarps),
// otherwise each warp has its own rows (T = 32).
template <int ncols_dst, int R, bool splitk, int P>
static __global__ void mul_mat_vec_mxfp8(
        const char * __restrict__ x, const float * __restrict__ y, float * __restrict__ dst,
        const int ncols_x, const int nrows_x,
        const int64_t stride_row_x, const int64_t stride_col_y, const int64_t stride_col_dst,
        const int channel_ratio, const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst,
        const int sample_ratio, const int64_t stride_sample_x, const int64_t stride_sample_y, const int64_t stride_sample_dst) {
    const int lane    = threadIdx.x;
    const int nwarps  = blockDim.y;
    const int t       = splitk ? threadIdx.y*WARP_SIZE + lane : lane;
    const int T       = splitk ? nwarps*WARP_SIZE : WARP_SIZE;
    const int row0    = splitk ? blockIdx.x*R : (blockIdx.x*nwarps + threadIdx.y)*R;
    const int channel = blockIdx.y;
    const int sample  = blockIdx.z;
    if (row0 >= nrows_x) {
        return;                                                   // uniform per block when splitk
    }

    const char * xb = x + (sample/sample_ratio)*stride_sample_x + (channel/channel_ratio)*stride_channel_x;
    y   += sample*stride_sample_y   + channel*stride_channel_y;
    dst += sample*stride_sample_dst + channel*stride_channel_dst;

    const uint32_t * xr[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        xr[r] = (const uint32_t *) (xb + min(row0 + r, nrows_x - 1)*stride_row_x); // duplicated last row: not stored
    }

    const int nwq = ncols_x/4;                                   // four-value words per row
    const int ngq = nwq/P;                                       // P-word groups per row (P divides 8)
    float acc[R][ncols_dst] = {{0.0f}};

    for (int g0 = 0; g0 < ngq; g0 += 8*T) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int g = g0 + i*T + t;
            if (g >= ngq) {
                break;
            }
            const int q = g*P;                                    // first word of the group
            float4 a[ncols_dst][P];
#pragma unroll
            for (int col = 0; col < ncols_dst; ++col) {
#pragma unroll
                for (int j = 0; j < P; ++j) {
                    a[col][j] = *(const float4 *) (y + col*stride_col_y + 4*(q + j));
                }
            }
            const int bo = (q >> 3)*(int) sizeof(block_mxfp8);   // block byte offset in the row; scale byte first
            const int eo = bo + 1 + 4*(q & 7);                    // the group's first E4M3 code
            const int sh = 8*(eo & 3);
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const float d = ggml_cuda_e8m0_to_fp32((uint8_t) ((__ldg(xr[r] + (bo >> 2)) >> (8*(bo & 3))) & 0xFF));
                uint32_t w[P + 1];
#pragma unroll
                for (int j = 0; j < P; ++j) {
                    w[j] = __ldg(xr[r] + (eo >> 2) + j);
                }
                w[P] = (eo & 3) ? __ldg(xr[r] + (eo >> 2) + P) : 0; // aligned: never read past the row
                float s[ncols_dst] = {0.0f};
#pragma unroll
                for (int j = 0; j < P; ++j) {
                    const float4 v = mxfp8_decode4(__funnelshift_r(w[j], w[j + 1], sh));
#pragma unroll
                    for (int col = 0; col < ncols_dst; ++col) {
                        s[col] += v.x*a[col][j].x + v.y*a[col][j].y + v.z*a[col][j].z + v.w*a[col][j].w;
                    }
                }
#pragma unroll
                for (int col = 0; col < ncols_dst; ++col) {
                    acc[r][col] += d*s[col];
                }
            }
        }
    }

    __shared__ float red[MMV_MXFP8_MAX_WARPS][R][ncols_dst];
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int col = 0; col < ncols_dst; ++col) {
            float v = warp_reduce_sum<WARP_SIZE>(acc[r][col]);
            if (splitk) {
                if (lane == 0) {
                    red[threadIdx.y][r][col] = v;
                }
            } else if (lane == 0 && row0 + r < nrows_x) {
                dst[col*stride_col_dst + row0 + r] = v;
            }
        }
    }
    if (splitk) {
        __syncthreads();
        const int k = threadIdx.y*WARP_SIZE + lane;               // one thread per (row, col)
        if (k < R*ncols_dst) {
            const int r = k / ncols_dst, col = k % ncols_dst;
            float v = 0.0f;
            for (int w = 0; w < nwarps; ++w) {
                v += red[w][r][col];
            }
            if (row0 + r < nrows_x) {
                dst[col*stride_col_dst + row0 + r] = v;
            }
        }
    }
}

bool ggml_cuda_should_use_mmv_mxfp8(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    return src0->type == GGML_TYPE_MXFP8 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
        src1->ne[1] <= MMV_MXFP8_MAX_COLS &&
        src0->ne[0] % 128 == 0 && src0->nb[1] % 4 == 0 && src0->nb[2] % 4 == 0 && src0->nb[3] % 4 == 0 &&
        ((uintptr_t) src0->data) % 4 == 0 &&
        src1->nb[0] == sizeof(float) && src1->nb[1] % 16 == 0 && src1->nb[2] % 16 == 0 && src1->nb[3] % 16 == 0 &&
        ((uintptr_t) src1->data) % 16 == 0 &&
        dst->nb[0] == sizeof(float) &&
        src1->ne[2] % src0->ne[2] == 0 && src1->ne[3] % src0->ne[3] == 0 &&
        src1->ne[2] <= 65535 && src1->ne[3] <= 65535;
}

struct mmv_mxfp8_cfg { int R, W; bool splitk; int P; };

// Defaults from the 2026-10-30 sweep at this model's dense shapes (m x k = 32768x1280, 5120x8192, 1280x5120,
// 5120x2304; n = 1 and 6). Ada: 2 warps x 4 rows, 4 words per lane at n = 1 (sum 42 vs 90 us for the first
// lane-per-word kernel; Q8_0 mmvq 51), 2 words per lane at n > 1. GB10 (DRAM-bound at these sizes): 4 warps x
// 4 rows, 4 words per lane (n = 1: 404 vs 486 us, Q8_0 391). MXFP8_MMV_CFG="R,W,S[,P]" (rows per group, warps per
// block, split-K 0/1, words per lane 1/2/4) overrides both for tuning.
static mmv_mxfp8_cfg mmv_mxfp8_get_cfg(const int ncols_dst, const int cc) {
    static const mmv_mxfp8_cfg env = [] {
        mmv_mxfp8_cfg c = {0, 0, false, 1};
        if (const char * e = getenv("MXFP8_MMV_CFG")) {
            int r = 2, w = 4, sk = 0, p = 1;
            if (sscanf(e, "%d,%d,%d,%d", &r, &w, &sk, &p) >= 3 && (r == 1 || r == 2 || r == 4) && w >= 1 && w <= MMV_MXFP8_MAX_WARPS &&
                (p == 1 || p == 2 || p == 4)) {
                c = {r, w, sk != 0, p};
            }
        }
        return c;
    }();
    if (env.R != 0) {
        return env;
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_BLACKWELL) {
        return {4, 4, true, 4};
    }
    return ncols_dst == 1 ? mmv_mxfp8_cfg{4, 2, true, 4} : mmv_mxfp8_cfg{4, 2, true, 2};
}

template <int ncols_dst, int R, bool splitk, int P>
static void launch_mul_mat_vec_mxfp8_rk(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, int W, cudaStream_t stream) {
    const int64_t nrows = src0->ne[1];
    const int64_t rows_per_block = splitk ? R : R*W;
    const dim3 grid((nrows + rows_per_block - 1)/rows_per_block, src1->ne[2], src1->ne[3]);
    const dim3 block(WARP_SIZE, W, 1);
    mul_mat_vec_mxfp8<ncols_dst, R, splitk, P><<<grid, block, 0, stream>>>(
        (const char *) src0->data, (const float *) src1->data, (float *) dst->data,
        src0->ne[0], nrows,
        src0->nb[1], src1->nb[1]/sizeof(float), dst->nb[1]/sizeof(float),
        src1->ne[2]/src0->ne[2], src0->nb[2], src1->nb[2]/sizeof(float), dst->nb[2]/sizeof(float),
        src1->ne[3]/src0->ne[3], src0->nb[3], src1->nb[3]/sizeof(float), dst->nb[3]/sizeof(float));
    CUDA_CHECK(cudaGetLastError());
}

template <int ncols_dst, int R, bool splitk>
static void launch_mul_mat_vec_mxfp8_p(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const mmv_mxfp8_cfg & c, cudaStream_t stream) {
    switch (c.P) {
        case 1: launch_mul_mat_vec_mxfp8_rk<ncols_dst, R, splitk, 1>(src0, src1, dst, c.W, stream); break;
        case 2: launch_mul_mat_vec_mxfp8_rk<ncols_dst, R, splitk, 2>(src0, src1, dst, c.W, stream); break;
        case 4: launch_mul_mat_vec_mxfp8_rk<ncols_dst, R, splitk, 4>(src0, src1, dst, c.W, stream); break;
        default: GGML_ABORT("fatal error");
    }
}

template <int ncols_dst>
static void launch_mul_mat_vec_mxfp8(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const int cc, cudaStream_t stream) {
    const mmv_mxfp8_cfg c = mmv_mxfp8_get_cfg(ncols_dst, cc);
    switch (c.R*2 + c.splitk) {
        case 2: launch_mul_mat_vec_mxfp8_p<ncols_dst, 1, false>(src0, src1, dst, c, stream); break;
        case 3: launch_mul_mat_vec_mxfp8_p<ncols_dst, 1, true >(src0, src1, dst, c, stream); break;
        case 4: launch_mul_mat_vec_mxfp8_p<ncols_dst, 2, false>(src0, src1, dst, c, stream); break;
        case 5: launch_mul_mat_vec_mxfp8_p<ncols_dst, 2, true >(src0, src1, dst, c, stream); break;
        case 8: launch_mul_mat_vec_mxfp8_p<ncols_dst, 4, false>(src0, src1, dst, c, stream); break;
        case 9: launch_mul_mat_vec_mxfp8_p<ncols_dst, 4, true >(src0, src1, dst, c, stream); break;
        default: GGML_ABORT("fatal error");
    }
}
void ggml_cuda_mul_mat_vec_mxfp8(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_should_use_mmv_mxfp8(src0, src1, dst));
    cudaStream_t stream = ctx.stream();
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    switch (src1->ne[1]) {
        case 1: launch_mul_mat_vec_mxfp8<1>(src0, src1, dst, cc, stream); break;
        case 2: launch_mul_mat_vec_mxfp8<2>(src0, src1, dst, cc, stream); break;
        case 3: launch_mul_mat_vec_mxfp8<3>(src0, src1, dst, cc, stream); break;
        case 4: launch_mul_mat_vec_mxfp8<4>(src0, src1, dst, cc, stream); break;
        case 5: launch_mul_mat_vec_mxfp8<5>(src0, src1, dst, cc, stream); break;
        case 6: launch_mul_mat_vec_mxfp8<6>(src0, src1, dst, cc, stream); break;
        case 7: launch_mul_mat_vec_mxfp8<7>(src0, src1, dst, cc, stream); break;
        case 8: launch_mul_mat_vec_mxfp8<8>(src0, src1, dst, cc, stream); break;
        default: GGML_ABORT("fatal error");
    }
}
