#include "common.cuh"

// MXFP8 (E4M3 + one E8M0 scale per 32) x F32 matrix-vector product for up to MMV_MXFP8_MAX_COLS columns of src1.
// The weights are decoded exactly and the activations stay F32 (no q8_1 activation quantization), so the only
// rounding is the F32 accumulation. DeepSeek-V4.1 ships its dense projections in this format.
#define MMV_MXFP8_MAX_COLS 8

bool ggml_cuda_should_use_mmv_mxfp8(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);

void ggml_cuda_mul_mat_vec_mxfp8(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
