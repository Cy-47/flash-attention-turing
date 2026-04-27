#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <vector>

namespace {

constexpr int kThreads = 256;
constexpr int kDecodeThreads = 128;

template <int NThreads>
__inline__ __device__ float block_sum(float val) {
    __shared__ float smem[NThreads];
    smem[threadIdx.x] = val;
    __syncthreads();
    for (int stride = NThreads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] += smem[threadIdx.x + stride];
        }
        __syncthreads();
    }
    return smem[0];
}

template <int NThreads>
__inline__ __device__ float block_max(float val) {
    __shared__ float smem[NThreads];
    smem[threadIdx.x] = val;
    __syncthreads();
    for (int stride = NThreads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    return smem[0];
}

__device__ __forceinline__ bool is_pow2_u32(const unsigned int x) {
    return x != 0u && (x & (x - 1u)) == 0u;
}

__device__ __forceinline__ int floor_log2_pow2_u32(const unsigned int x) {
    return 31 - __clz(x);
}

template <int Headdim>
__device__ __forceinline__ float q_dot_k_half2(
    const float *q_vec, const half *__restrict__ k_row_base) {
    static_assert(Headdim % 2 == 0, "Headdim must be even for half2 vectorization");
    float acc = 0.0f;
    #pragma unroll
    for (int d = 0; d < Headdim; d += 2) {
        const half2 k2 = *reinterpret_cast<const half2 *>(k_row_base + d);
        const float2 kf = __half22float2(k2);
        acc += q_vec[d] * kf.x + q_vec[d + 1] * kf.y;
    }
    return acc;
}

__device__ __forceinline__ int paged_table_idx(
    const int pos, const int page_block_size, const bool pbs_is_pow2, const int pbs_lg) {
    if (pbs_is_pow2) {
        return pos >> pbs_lg;
    }
    return pos / page_block_size;
}

__device__ __forceinline__ int paged_page_offset(
    const int pos, const int page_block_size, const bool pbs_is_pow2, const int pbs_lg) {
    if (pbs_is_pow2) {
        return pos & (page_block_size - 1);
    }
    return pos - (pos / page_block_size) * page_block_size;
}

__device__ __forceinline__ void paged_advance_in_split(
    int &table_idx,
    int &page_offset,
    int &physical_block,
    const int pos,
    const int split_end,
    const int next_stride,
    const int batch,
    const int *block_table,
    const int64_t block_table_batch_stride,
    const int page_block_size,
    const bool pbs_is_pow2,
    const int pbs_lg) {
    if (pbs_is_pow2) {
        const unsigned int mask = static_cast<unsigned int>(page_block_size - 1);
        int next_off = page_offset + next_stride;
        if (static_cast<unsigned int>(next_off) > mask) {
            const unsigned int u = static_cast<unsigned int>(next_off);
            const int n_wrap = static_cast<int>(u >> pbs_lg);
            table_idx += n_wrap;
            next_off = static_cast<int>(u & mask);
        }
        page_offset = next_off;
    } else {
        const int carry = page_offset + next_stride;
        table_idx += carry / page_block_size;
        page_offset = carry - (carry / page_block_size) * page_block_size;
    }
    if (pos + next_stride < split_end) {
        physical_block = block_table[int64_t(batch) * block_table_batch_stride + table_idx];
    }
}

__device__ __forceinline__ int64_t dense_kv_offset(
    const int batch,
    const int pos,
    const int head,
    const int dim,
    const int64_t batch_stride,
    const int64_t row_stride,
    const int64_t head_stride) {
    return int64_t(batch) * batch_stride + int64_t(pos) * row_stride
        + int64_t(head) * head_stride + dim;
}

__device__ __forceinline__ int64_t paged_kv_offset(
    const int logical_batch,
    const int pos,
    const int head,
    const int dim,
    const int *block_table,
    const int64_t block_table_batch_stride,
    const int page_block_size,
    const int64_t block_stride,
    const int64_t page_stride,
    const int64_t head_stride) {
    const int table_idx = pos / page_block_size;
    const int page_offset = pos - table_idx * page_block_size;
    const int physical_block = block_table[int64_t(logical_batch) * block_table_batch_stride + table_idx];
    return int64_t(physical_block) * block_stride + int64_t(page_offset) * page_stride
        + int64_t(head) * head_stride + dim;
}

template <int Headdim>
__global__ void kvcache_split_kernel_legacy(
    const half *__restrict__ q,
    const half *__restrict__ k_cache,
    const half *__restrict__ v_cache,
    const int *__restrict__ cache_seqlens,
    const int *__restrict__ cache_batch_idx,
    const int *__restrict__ block_table,
    half *__restrict__ out,
    float *__restrict__ lse,
    float *__restrict__ out_accum,
    float *__restrict__ lse_accum,
    const int batch_size,
    const int seqlen_q,
    const int seqlen_k_max,
    const int nheads,
    const int nheads_k,
    const int num_splits,
    const int page_block_size,
    const int64_t q_batch_stride,
    const int64_t q_row_stride,
    const int64_t q_head_stride,
    const int64_t k_batch_stride,
    const int64_t k_row_stride,
    const int64_t k_head_stride,
    const int64_t v_batch_stride,
    const int64_t v_row_stride,
    const int64_t v_head_stride,
    const int64_t o_batch_stride,
    const int64_t o_row_stride,
    const int64_t o_head_stride,
    const int64_t block_table_batch_stride,
    const float softmax_scale,
    const bool is_causal) {
    const int row_idx = blockIdx.x;
    const int split_idx = blockIdx.y;
    const int batch = row_idx / (nheads * seqlen_q);
    const int rem = row_idx - batch * nheads * seqlen_q;
    const int head = rem / seqlen_q;
    const int q_pos = rem - head * seqlen_q;
    const int kv_head = head / (nheads / nheads_k);
    const int cache_batch = cache_batch_idx == nullptr ? batch : cache_batch_idx[batch];
    const int seqlen_k = cache_seqlens[batch];
    const int split_start = (seqlen_k * split_idx) / num_splits;
    const int split_end = (seqlen_k * (split_idx + 1)) / num_splits;
    const int causal_limit = q_pos + seqlen_k - seqlen_q;
    const bool pbs_is_pow2 = is_pow2_u32(static_cast<unsigned int>(page_block_size));
    const int pbs_lg = pbs_is_pow2 ? floor_log2_pow2_u32(static_cast<unsigned int>(page_block_size)) : 0;
    __shared__ float q_vec[Headdim];
    if (threadIdx.x < Headdim) {
        q_vec[threadIdx.x] = __half2float(q[int64_t(batch) * q_batch_stride
            + int64_t(q_pos) * q_row_stride + int64_t(head) * q_head_stride + threadIdx.x]);
    }
    __syncthreads();

    float local_max = -FLT_MAX;
    for (int pos = split_start + threadIdx.x; pos < split_end; pos += blockDim.x) {
        if (!is_causal || pos <= causal_limit) {
            float score = 0.0f;
            int64_t k_base = 0;
            if (block_table == nullptr) {
                k_base = int64_t(cache_batch) * k_batch_stride + int64_t(pos) * k_row_stride
                    + int64_t(kv_head) * k_head_stride;
            } else {
                const int table_idx = paged_table_idx(pos, page_block_size, pbs_is_pow2, pbs_lg);
                const int page_offset = paged_page_offset(pos, page_block_size, pbs_is_pow2, pbs_lg);
                const int physical_block =
                    block_table[int64_t(batch) * block_table_batch_stride + table_idx];
                k_base = int64_t(physical_block) * k_batch_stride + int64_t(page_offset) * k_row_stride
                    + int64_t(kv_head) * k_head_stride;
            }
            score = q_dot_k_half2<Headdim>(q_vec, k_cache + k_base);
            local_max = fmaxf(local_max, score * softmax_scale);
        }
    }
    const float row_max = block_max<kThreads>(local_max);

    float local_sum = 0.0f;
    if (row_max != -FLT_MAX) {
        for (int pos = split_start + threadIdx.x; pos < split_end; pos += blockDim.x) {
            if (!is_causal || pos <= causal_limit) {
                float score = 0.0f;
                int64_t k_base = 0;
                if (block_table == nullptr) {
                    k_base = int64_t(cache_batch) * k_batch_stride + int64_t(pos) * k_row_stride
                        + int64_t(kv_head) * k_head_stride;
                } else {
                    const int table_idx = paged_table_idx(pos, page_block_size, pbs_is_pow2, pbs_lg);
                    const int page_offset = paged_page_offset(pos, page_block_size, pbs_is_pow2, pbs_lg);
                    const int physical_block =
                        block_table[int64_t(batch) * block_table_batch_stride + table_idx];
                    k_base = int64_t(physical_block) * k_batch_stride + int64_t(page_offset) * k_row_stride
                        + int64_t(kv_head) * k_head_stride;
                }
                score = q_dot_k_half2<Headdim>(q_vec, k_cache + k_base);
                local_sum += expf(score * softmax_scale - row_max);
            }
        }
    }
    const float row_sum = block_sum<kThreads>(local_sum);
    const float row_lse = row_sum == 0.0f ? -INFINITY : row_max + logf(row_sum);

    if (threadIdx.x == 0 && num_splits > 1) {
        lse_accum[int64_t(split_idx) * batch_size * nheads * seqlen_q + row_idx] = row_lse;
    }

    for (int dim = 0; dim < Headdim; ++dim) {
        float local_out = 0.0f;
        if (row_sum != 0.0f) {
            for (int pos = split_start + threadIdx.x; pos < split_end; pos += blockDim.x) {
                if (!is_causal || pos <= causal_limit) {
                    float score = 0.0f;
                    int64_t k_base = 0;
                    int64_t v_base = 0;
                    if (block_table == nullptr) {
                        k_base = int64_t(cache_batch) * k_batch_stride + int64_t(pos) * k_row_stride
                            + int64_t(kv_head) * k_head_stride;
                        v_base = int64_t(cache_batch) * v_batch_stride + int64_t(pos) * v_row_stride
                            + int64_t(kv_head) * v_head_stride;
                    } else {
                        const int table_idx = paged_table_idx(pos, page_block_size, pbs_is_pow2, pbs_lg);
                        const int page_offset = paged_page_offset(pos, page_block_size, pbs_is_pow2, pbs_lg);
                        const int physical_block =
                            block_table[int64_t(batch) * block_table_batch_stride + table_idx];
                        k_base = int64_t(physical_block) * k_batch_stride + int64_t(page_offset) * k_row_stride
                            + int64_t(kv_head) * k_head_stride;
                        v_base = int64_t(physical_block) * v_batch_stride + int64_t(page_offset) * v_row_stride
                            + int64_t(kv_head) * v_head_stride;
                    }
                    score = q_dot_k_half2<Headdim>(q_vec, k_cache + k_base);
                    const float p = expf(score * softmax_scale - row_lse);
                    local_out += p * __half2float(v_cache[v_base + dim]);
                }
            }
        }
        const float dim_out = block_sum<kThreads>(local_out);
        if (threadIdx.x == 0) {
            if (num_splits == 1) {
                out[int64_t(batch) * o_batch_stride + int64_t(q_pos) * o_row_stride
                    + int64_t(head) * o_head_stride + dim] = __float2half_rn(dim_out);
                if (dim == 0) {
                    lse[int64_t(batch) * nheads * seqlen_q + int64_t(head) * seqlen_q + q_pos] =
                        row_lse == -INFINITY ? 0.0f : row_lse;
                }
            } else {
                out_accum[(int64_t(split_idx) * batch_size * nheads * seqlen_q + row_idx) * Headdim + dim] =
                    dim_out;
            }
        }
    }
}

template <int Headdim>
__global__ void kvcache_split_kernel(
    const half *__restrict__ q,
    const half *__restrict__ k_cache,
    const half *__restrict__ v_cache,
    const int *__restrict__ cache_seqlens,
    const int *__restrict__ cache_batch_idx,
    const int *__restrict__ block_table,
    half *__restrict__ out,
    float *__restrict__ lse,
    float *__restrict__ out_accum,
    float *__restrict__ lse_accum,
    const int batch_size,
    const int seqlen_q,
    const int seqlen_k_max,
    const int nheads,
    const int nheads_k,
    const int num_splits,
    const int page_block_size,
    const int64_t q_batch_stride,
    const int64_t q_row_stride,
    const int64_t q_head_stride,
    const int64_t k_batch_stride,
    const int64_t k_row_stride,
    const int64_t k_head_stride,
    const int64_t v_batch_stride,
    const int64_t v_row_stride,
    const int64_t v_head_stride,
    const int64_t o_batch_stride,
    const int64_t o_row_stride,
    const int64_t o_head_stride,
    const int64_t block_table_batch_stride,
    const float softmax_scale,
    const bool is_causal) {
    extern __shared__ float scores[];
    const int row_idx = blockIdx.x;
    const int split_idx = blockIdx.y;
    const int batch = row_idx / (nheads * seqlen_q);
    const int rem = row_idx - batch * nheads * seqlen_q;
    const int head = rem / seqlen_q;
    const int q_pos = rem - head * seqlen_q;
    const int kv_head = head / (nheads / nheads_k);
    const int cache_batch = cache_batch_idx == nullptr ? batch : cache_batch_idx[batch];
    const int seqlen_k = cache_seqlens[batch];
    const int split_start = (seqlen_k * split_idx) / num_splits;
    const int split_end = (seqlen_k * (split_idx + 1)) / num_splits;
    const int split_len = split_end - split_start;
    const int causal_limit = q_pos + seqlen_k - seqlen_q;
    const bool pbs_is_pow2 = is_pow2_u32(static_cast<unsigned int>(page_block_size));
    const int pbs_lg = pbs_is_pow2 ? floor_log2_pow2_u32(static_cast<unsigned int>(page_block_size)) : 0;
    __shared__ float q_vec[Headdim];
    if (threadIdx.x < Headdim) {
        q_vec[threadIdx.x] = __half2float(q[int64_t(batch) * q_batch_stride
            + int64_t(q_pos) * q_row_stride + int64_t(head) * q_head_stride + threadIdx.x]);
    }
    __syncthreads();

    if (split_len <= 0) {
        if (threadIdx.x == 0) {
            if (num_splits > 1) {
                lse_accum[int64_t(split_idx) * batch_size * nheads * seqlen_q + row_idx] = -INFINITY;
            } else {
                lse[int64_t(batch) * nheads * seqlen_q + int64_t(head) * seqlen_q + q_pos] = 0.0f;
            }
        }
        if (threadIdx.x < Headdim && num_splits == 1) {
            out[int64_t(batch) * o_batch_stride + int64_t(q_pos) * o_row_stride
                + int64_t(head) * o_head_stride + threadIdx.x] = __float2half_rn(0.0f);
        }
        if (threadIdx.x < Headdim && num_splits > 1) {
            out_accum[(int64_t(split_idx) * batch_size * nheads * seqlen_q + row_idx) * Headdim + threadIdx.x] = 0.0f;
        }
        return;
    }

    float local_max = -FLT_MAX;
    int table_idx = 0;
    int page_offset = 0;
    int physical_block = -1;
    if (block_table != nullptr) {
        const int start_pos = split_start + threadIdx.x;
        table_idx = paged_table_idx(start_pos, page_block_size, pbs_is_pow2, pbs_lg);
        page_offset = paged_page_offset(start_pos, page_block_size, pbs_is_pow2, pbs_lg);
        if (start_pos < split_end) {
            physical_block = block_table[int64_t(batch) * block_table_batch_stride + table_idx];
        }
    }
    for (int pos = split_start + threadIdx.x; pos < split_end; pos += blockDim.x) {
        float score = -INFINITY;
        if (!is_causal || pos <= causal_limit) {
            int64_t k_base = 0;
            if (block_table == nullptr) {
                k_base = int64_t(cache_batch) * k_batch_stride + int64_t(pos) * k_row_stride
                    + int64_t(kv_head) * k_head_stride;
            } else {
                k_base = int64_t(physical_block) * k_batch_stride + int64_t(page_offset) * k_row_stride
                    + int64_t(kv_head) * k_head_stride;
            }
            const float dot = q_dot_k_half2<Headdim>(q_vec, k_cache + k_base);
            score = dot * softmax_scale;
        }
        scores[pos - split_start] = score;
        local_max = fmaxf(local_max, score);
        if (block_table != nullptr) {
            paged_advance_in_split(
                table_idx, page_offset, physical_block, pos, split_end, blockDim.x, batch, block_table,
                block_table_batch_stride, page_block_size, pbs_is_pow2, pbs_lg);
        }
    }
    const float row_max = block_max<kThreads>(local_max);

    float local_sum = 0.0f;
    for (int pos = split_start + threadIdx.x; pos < split_end; pos += blockDim.x) {
        const int idx = pos - split_start;
        const float score = scores[idx];
        if (score != -INFINITY) {
            const float p = expf(score - row_max);
            scores[idx] = p;
            local_sum += p;
        } else {
            scores[idx] = 0.0f;
        }
    }
    const float row_sum = block_sum<kThreads>(local_sum);
    const float row_lse = row_sum == 0.0f ? -INFINITY : row_max + logf(row_sum);
    const float inv_row_sum = row_sum == 0.0f ? 0.0f : 1.0f / row_sum;

    if (threadIdx.x == 0 && num_splits > 1) {
        lse_accum[int64_t(split_idx) * batch_size * nheads * seqlen_q + row_idx] = row_lse;
    }

    for (int d = 0; d < Headdim; d += 2) {
        float local_out0 = 0.0f;
        float local_out1 = 0.0f;
        int v_table_idx = 0;
        int v_page_offset = 0;
        int v_physical_block = -1;
        if (block_table != nullptr) {
            const int start_pos = split_start + threadIdx.x;
            v_table_idx = paged_table_idx(start_pos, page_block_size, pbs_is_pow2, pbs_lg);
            v_page_offset = paged_page_offset(start_pos, page_block_size, pbs_is_pow2, pbs_lg);
            if (start_pos < split_end) {
                v_physical_block = block_table[int64_t(batch) * block_table_batch_stride + v_table_idx];
            }
        }
        for (int pos = split_start + threadIdx.x; pos < split_end; pos += blockDim.x) {
            const float p = scores[pos - split_start] * inv_row_sum;
            int64_t v_base = 0;
            if (block_table == nullptr) {
                v_base = int64_t(cache_batch) * v_batch_stride + int64_t(pos) * v_row_stride
                    + int64_t(kv_head) * v_head_stride;
            } else {
                v_base = int64_t(v_physical_block) * v_batch_stride + int64_t(v_page_offset) * v_row_stride
                    + int64_t(kv_head) * v_head_stride;
            }
            const half2 v2 = *reinterpret_cast<const half2 *>(v_cache + v_base + d);
            const float2 vf = __half22float2(v2);
            local_out0 += p * vf.x;
            local_out1 += p * vf.y;
            if (block_table != nullptr) {
                paged_advance_in_split(
                    v_table_idx, v_page_offset, v_physical_block, pos, split_end, blockDim.x, batch, block_table,
                    block_table_batch_stride, page_block_size, pbs_is_pow2, pbs_lg);
            }
        }
        const float out0 = block_sum<kThreads>(local_out0);
        const float out1 = block_sum<kThreads>(local_out1);
        if (threadIdx.x == 0) {
            if (num_splits == 1) {
                out[int64_t(batch) * o_batch_stride + int64_t(q_pos) * o_row_stride
                    + int64_t(head) * o_head_stride + d] = __float2half_rn(out0);
                out[int64_t(batch) * o_batch_stride + int64_t(q_pos) * o_row_stride
                    + int64_t(head) * o_head_stride + d + 1] = __float2half_rn(out1);
                if (d == 0) {
                    lse[int64_t(batch) * nheads * seqlen_q + int64_t(head) * seqlen_q + q_pos] =
                        row_lse == -INFINITY ? 0.0f : row_lse;
                }
            } else {
                out_accum[(int64_t(split_idx) * batch_size * nheads * seqlen_q + row_idx) * Headdim + d] = out0;
                out_accum[(int64_t(split_idx) * batch_size * nheads * seqlen_q + row_idx) * Headdim + d + 1] = out1;
            }
        }
    }
}

template <int Headdim>
__global__ void kvcache_decode_split_kernel(
    const half *__restrict__ q,
    const half *__restrict__ k_cache,
    const half *__restrict__ v_cache,
    const int *__restrict__ cache_seqlens,
    const int *__restrict__ cache_batch_idx,
    const int *__restrict__ block_table,
    half *__restrict__ out,
    float *__restrict__ lse,
    float *__restrict__ out_accum,
    float *__restrict__ lse_accum,
    const int batch_size,
    const int nheads,
    const int nheads_k,
    const int num_splits,
    const int page_block_size,
    const int64_t q_batch_stride,
    const int64_t q_row_stride,
    const int64_t q_head_stride,
    const int64_t k_batch_stride,
    const int64_t k_row_stride,
    const int64_t k_head_stride,
    const int64_t v_batch_stride,
    const int64_t v_row_stride,
    const int64_t v_head_stride,
    const int64_t o_batch_stride,
    const int64_t o_row_stride,
    const int64_t o_head_stride,
    const int64_t block_table_batch_stride,
    const float softmax_scale) {
    extern __shared__ float scores[];
    const int row_idx = blockIdx.x;
    const int split_idx = blockIdx.y;
    const int batch = row_idx / nheads;
    const int head = row_idx - batch * nheads;
    const int kv_head = head / (nheads / nheads_k);
    const int cache_batch = cache_batch_idx == nullptr ? batch : cache_batch_idx[batch];
    const int seqlen_k = cache_seqlens[batch];
    const int split_start = (seqlen_k * split_idx) / num_splits;
    const int split_end = (seqlen_k * (split_idx + 1)) / num_splits;
    const int split_len = split_end - split_start;
    const bool pbs_is_pow2 = is_pow2_u32(static_cast<unsigned int>(page_block_size));
    const int pbs_lg = pbs_is_pow2 ? floor_log2_pow2_u32(static_cast<unsigned int>(page_block_size)) : 0;

    if (split_len <= 0) {
        if (threadIdx.x == 0) {
            lse_accum[int64_t(split_idx) * batch_size * nheads + row_idx] = -INFINITY;
            if (num_splits == 1) {
                lse[int64_t(batch) * nheads + head] = 0.0f;
            }
        }
        if (threadIdx.x < Headdim) {
            if (num_splits == 1) {
                out[int64_t(batch) * o_batch_stride + int64_t(0) * o_row_stride
                    + int64_t(head) * o_head_stride + threadIdx.x] = __float2half_rn(0.0f);
            } else {
                out_accum[(int64_t(split_idx) * batch_size * nheads + row_idx) * Headdim + threadIdx.x] = 0.0f;
            }
        }
        return;
    }

    __shared__ float q_vec[Headdim];
    if (threadIdx.x < Headdim) {
        q_vec[threadIdx.x] = __half2float(q[int64_t(batch) * q_batch_stride
            + int64_t(0) * q_row_stride + int64_t(head) * q_head_stride + threadIdx.x]);
    }
    __syncthreads();

    // K phase: parallel over positions; one q_dot per key (no per-key full-CTA sum).
    float local_max = -FLT_MAX;
    int table_idx_k = 0;
    int page_offset_k = 0;
    int physical_block_k = -1;
    if (block_table != nullptr) {
        const int start_pos = split_start + threadIdx.x;
        table_idx_k = paged_table_idx(start_pos, page_block_size, pbs_is_pow2, pbs_lg);
        page_offset_k = paged_page_offset(start_pos, page_block_size, pbs_is_pow2, pbs_lg);
        if (start_pos < split_end) {
            physical_block_k = block_table[int64_t(batch) * block_table_batch_stride + table_idx_k];
        }
    }
    for (int pos = split_start + threadIdx.x; pos < split_end; pos += kDecodeThreads) {
        int64_t k_base = 0;
        if (block_table == nullptr) {
            k_base = int64_t(cache_batch) * k_batch_stride + int64_t(pos) * k_row_stride
                + int64_t(kv_head) * k_head_stride;
        } else {
            k_base = int64_t(physical_block_k) * k_batch_stride + int64_t(page_offset_k) * k_row_stride
                + int64_t(kv_head) * k_head_stride;
        }
        const float dot = q_dot_k_half2<Headdim>(q_vec, k_cache + k_base);
        const float sc = dot * softmax_scale;
        scores[pos - split_start] = sc;
        local_max = fmaxf(local_max, sc);
        if (block_table != nullptr) {
            paged_advance_in_split(
                table_idx_k, page_offset_k, physical_block_k, pos, split_end, kDecodeThreads, batch, block_table,
                block_table_batch_stride, page_block_size, pbs_is_pow2, pbs_lg);
        }
    }
    const float row_max = block_max<kDecodeThreads>(local_max);

    float local_sum = 0.0f;
    for (int pos = split_start + threadIdx.x; pos < split_end; pos += kDecodeThreads) {
        const int idx = pos - split_start;
        const float s = scores[idx];
        const float p = expf(s - row_max);
        scores[idx] = p;
        local_sum += p;
    }
    const float row_sum = block_sum<kDecodeThreads>(local_sum);
    const float row_lse = row_sum == 0.0f ? -INFINITY : row_max + logf(row_sum);
    const float inv_row_sum = row_sum == 0.0f ? 0.0f : 1.0f / row_sum;

    if (threadIdx.x == 0 && num_splits > 1) {
        lse_accum[int64_t(split_idx) * batch_size * nheads + row_idx] = row_lse;
    }
    if (num_splits == 1 && row_lse == -INFINITY) {
        if (threadIdx.x < Headdim) {
            out[int64_t(batch) * o_batch_stride + int64_t(0) * o_row_stride
                + int64_t(head) * o_head_stride + threadIdx.x] = __float2half_rn(0.0f);
        }
        if (threadIdx.x == 0) {
            lse[int64_t(batch) * nheads + head] = 0.0f;
        }
        return;
    }
    if (num_splits > 1 && row_lse == -INFINITY) {
        if (threadIdx.x < Headdim) {
            out_accum[(int64_t(split_idx) * batch_size * nheads + row_idx) * Headdim + threadIdx.x] = 0.0f;
        }
        return;
    }

    // V: half2 per position for pairs of dims; two block_sums per pair.
    for (int d = 0; d < Headdim; d += 2) {
        float local_out0 = 0.0f;
        float local_out1 = 0.0f;
        int v_table_idx = 0;
        int v_page_offset = 0;
        int v_physical_block = -1;
        if (block_table != nullptr) {
            const int start_pos = split_start + threadIdx.x;
            v_table_idx = paged_table_idx(start_pos, page_block_size, pbs_is_pow2, pbs_lg);
            v_page_offset = paged_page_offset(start_pos, page_block_size, pbs_is_pow2, pbs_lg);
            if (start_pos < split_end) {
                v_physical_block = block_table[int64_t(batch) * block_table_batch_stride + v_table_idx];
            }
        }
        for (int pos = split_start + threadIdx.x; pos < split_end; pos += kDecodeThreads) {
            const int idx = pos - split_start;
            const float p = scores[idx] * inv_row_sum;
            int64_t v_base = 0;
            if (block_table == nullptr) {
                v_base = int64_t(cache_batch) * v_batch_stride + int64_t(pos) * v_row_stride
                    + int64_t(kv_head) * v_head_stride;
            } else {
                v_base = int64_t(v_physical_block) * v_batch_stride + int64_t(v_page_offset) * v_row_stride
                    + int64_t(kv_head) * v_head_stride;
            }
            const half2 v2 = *reinterpret_cast<const half2 *>(v_cache + v_base + d);
            const float2 vf = __half22float2(v2);
            local_out0 += p * vf.x;
            local_out1 += p * vf.y;
            if (block_table != nullptr) {
                paged_advance_in_split(
                    v_table_idx, v_page_offset, v_physical_block, pos, split_end, kDecodeThreads, batch, block_table,
                    block_table_batch_stride, page_block_size, pbs_is_pow2, pbs_lg);
            }
        }
        const float acc0 = block_sum<kDecodeThreads>(local_out0);
        const float acc1 = block_sum<kDecodeThreads>(local_out1);
        if (threadIdx.x == 0) {
            if (num_splits == 1) {
                out[int64_t(batch) * o_batch_stride + int64_t(0) * o_row_stride
                    + int64_t(head) * o_head_stride + d] = __float2half_rn(acc0);
                out[int64_t(batch) * o_batch_stride + int64_t(0) * o_row_stride
                    + int64_t(head) * o_head_stride + d + 1] = __float2half_rn(acc1);
                if (d == 0) {
                    lse[int64_t(batch) * nheads + head] = row_lse;
                }
            } else {
                out_accum[(int64_t(split_idx) * batch_size * nheads + row_idx) * Headdim + d] = acc0;
                out_accum[(int64_t(split_idx) * batch_size * nheads + row_idx) * Headdim + d + 1] = acc1;
            }
        }
    }
}

template <int Headdim>
__global__ void kvcache_combine_kernel(
    const float *__restrict__ out_accum,
    const float *__restrict__ lse_accum,
    half *__restrict__ out,
    float *__restrict__ lse,
    const int rows,
    const int num_splits,
    const int batch_size,
    const int seqlen_q,
    const int nheads,
    const int64_t o_batch_stride,
    const int64_t o_row_stride,
    const int64_t o_head_stride) {
    const int row_idx = blockIdx.x;
    const int dim = threadIdx.x;
    if (row_idx >= rows || dim >= Headdim) {
        return;
    }

    __shared__ float s_lse_total;
    if (threadIdx.x == 0) {
        float lse_max = -INFINITY;
        for (int split = 0; split < num_splits; ++split) {
            lse_max = fmaxf(lse_max, lse_accum[int64_t(split) * rows + row_idx]);
        }
        float lse_sum = 0.0f;
        for (int split = 0; split < num_splits; ++split) {
            lse_sum += expf(lse_accum[int64_t(split) * rows + row_idx] - lse_max);
        }
        s_lse_total = (lse_sum == 0.0f || lse_max == -INFINITY) ? -INFINITY : lse_max + logf(lse_sum);
    }
    __syncthreads();
    const float lse_total = s_lse_total;

    float out_val = 0.0f;
    if (lse_total != -INFINITY) {
        for (int split = 0; split < num_splits; ++split) {
            const float weight = expf(lse_accum[int64_t(split) * rows + row_idx] - lse_total);
            out_val += weight * out_accum[(int64_t(split) * rows + row_idx) * Headdim + dim];
        }
    }

    const int batch = row_idx / (nheads * seqlen_q);
    const int rem = row_idx - batch * nheads * seqlen_q;
    const int head = rem / seqlen_q;
    const int q_pos = rem - head * seqlen_q;
    out[int64_t(batch) * o_batch_stride + int64_t(q_pos) * o_row_stride
        + int64_t(head) * o_head_stride + dim] = __float2half_rn(out_val);
    if (threadIdx.x == 0) {
        lse[int64_t(batch) * nheads * seqlen_q + int64_t(head) * seqlen_q + q_pos] =
            lse_total == -INFINITY ? 0.0f : lse_total;
    }
}

template <int Headdim>
__global__ void kvcache_combine_kernel_legacy(
    const float *__restrict__ out_accum,
    const float *__restrict__ lse_accum,
    half *__restrict__ out,
    float *__restrict__ lse,
    const int rows,
    const int num_splits,
    const int batch_size,
    const int seqlen_q,
    const int nheads,
    const int64_t o_batch_stride,
    const int64_t o_row_stride,
    const int64_t o_head_stride) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * Headdim;
    if (idx >= total) {
        return;
    }
    const int row_idx = idx / Headdim;
    const int dim = idx - row_idx * Headdim;

    float lse_max = -INFINITY;
    for (int split = 0; split < num_splits; ++split) {
        lse_max = fmaxf(lse_max, lse_accum[int64_t(split) * rows + row_idx]);
    }
    float lse_sum = 0.0f;
    for (int split = 0; split < num_splits; ++split) {
        lse_sum += expf(lse_accum[int64_t(split) * rows + row_idx] - lse_max);
    }
    const float lse_total = (lse_sum == 0.0f || lse_max == -INFINITY) ? -INFINITY : lse_max + logf(lse_sum);
    float out_val = 0.0f;
    if (lse_total != -INFINITY) {
        for (int split = 0; split < num_splits; ++split) {
            const float weight = expf(lse_accum[int64_t(split) * rows + row_idx] - lse_total);
            out_val += weight * out_accum[(int64_t(split) * rows + row_idx) * Headdim + dim];
        }
    }

    const int batch = row_idx / (nheads * seqlen_q);
    const int rem = row_idx - batch * nheads * seqlen_q;
    const int head = rem / seqlen_q;
    const int q_pos = rem - head * seqlen_q;
    out[int64_t(batch) * o_batch_stride + int64_t(q_pos) * o_row_stride
        + int64_t(head) * o_head_stride + dim] = __float2half_rn(out_val);
    if (dim == 0) {
        lse[int64_t(batch) * nheads * seqlen_q + int64_t(head) * seqlen_q + q_pos] =
            lse_total == -INFINITY ? 0.0f : lse_total;
    }
}

__global__ void paged_append_kernel(
    half *__restrict__ k_cache,
    half *__restrict__ v_cache,
    const half *__restrict__ k_new,
    const half *__restrict__ v_new,
    const int *__restrict__ cache_seqlens,
    const int *__restrict__ block_table,
    const int batch_size,
    const int seqlen_new,
    const int nheads_k,
    const int head_dim,
    const int page_block_size,
    const int64_t k_cache_block_stride,
    const int64_t k_cache_row_stride,
    const int64_t k_cache_head_stride,
    const int64_t v_cache_block_stride,
    const int64_t v_cache_row_stride,
    const int64_t v_cache_head_stride,
    const int64_t k_new_batch_stride,
    const int64_t k_new_row_stride,
    const int64_t k_new_head_stride,
    const int64_t v_new_batch_stride,
    const int64_t v_new_row_stride,
    const int64_t v_new_head_stride,
    const int64_t block_table_batch_stride) {
    const int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (head_dim % 2 == 0) {
        const int h2 = head_dim / 2;
        const int64_t elems_per_batch = int64_t(seqlen_new) * nheads_k * h2;
        const int64_t total = int64_t(batch_size) * elems_per_batch;
        if (idx >= total) {
            return;
        }
        const bool pbs_is_pow2 = is_pow2_u32(static_cast<unsigned int>(page_block_size));
        const int pbs_lg = pbs_is_pow2 ? floor_log2_pow2_u32(static_cast<unsigned int>(page_block_size)) : 0;
        const int batch = static_cast<int>(idx / elems_per_batch);
        const int64_t rem0 = idx - int64_t(batch) * elems_per_batch;
        const int token = static_cast<int>(rem0 / (nheads_k * h2));
        const int64_t rem1 = rem0 - int64_t(token) * nheads_k * h2;
        const int kv_head = static_cast<int>(rem1 / h2);
        const int dim_pair = static_cast<int>(rem1 - int64_t(kv_head) * h2);
        const int dim = dim_pair * 2;

        const int logical_pos = cache_seqlens[batch] + token;
        const int table_idx = paged_table_idx(logical_pos, page_block_size, pbs_is_pow2, pbs_lg);
        const int page_offset = paged_page_offset(logical_pos, page_block_size, pbs_is_pow2, pbs_lg);
        const int physical_block = block_table[int64_t(batch) * block_table_batch_stride + table_idx];

        const int64_t src_k = int64_t(batch) * k_new_batch_stride + int64_t(token) * k_new_row_stride
            + int64_t(kv_head) * k_new_head_stride + dim;
        const int64_t src_v = int64_t(batch) * v_new_batch_stride + int64_t(token) * v_new_row_stride
            + int64_t(kv_head) * v_new_head_stride + dim;
        const int64_t dst_k = int64_t(physical_block) * k_cache_block_stride + int64_t(page_offset) * k_cache_row_stride
            + int64_t(kv_head) * k_cache_head_stride + dim;
        const int64_t dst_v = int64_t(physical_block) * v_cache_block_stride + int64_t(page_offset) * v_cache_row_stride
            + int64_t(kv_head) * v_cache_head_stride + dim;
        *reinterpret_cast<half2 *>(k_cache + dst_k) = *reinterpret_cast<const half2 *>(k_new + src_k);
        *reinterpret_cast<half2 *>(v_cache + dst_v) = *reinterpret_cast<const half2 *>(v_new + src_v);
    } else {
        const int64_t elems_per_batch = int64_t(seqlen_new) * nheads_k * head_dim;
        const int64_t total = int64_t(batch_size) * elems_per_batch;
        if (idx >= total) {
            return;
        }
        const bool pbs_is_pow2 = is_pow2_u32(static_cast<unsigned int>(page_block_size));
        const int pbs_lg = pbs_is_pow2 ? floor_log2_pow2_u32(static_cast<unsigned int>(page_block_size)) : 0;
        const int batch = static_cast<int>(idx / elems_per_batch);
        const int64_t rem0 = idx - int64_t(batch) * elems_per_batch;
        const int token = static_cast<int>(rem0 / (nheads_k * head_dim));
        const int64_t rem1 = rem0 - int64_t(token) * nheads_k * head_dim;
        const int kv_head = static_cast<int>(rem1 / head_dim);
        const int dim = static_cast<int>(rem1 - int64_t(kv_head) * head_dim);

        const int logical_pos = cache_seqlens[batch] + token;
        const int table_idx = paged_table_idx(logical_pos, page_block_size, pbs_is_pow2, pbs_lg);
        const int page_offset = paged_page_offset(logical_pos, page_block_size, pbs_is_pow2, pbs_lg);
        const int physical_block = block_table[int64_t(batch) * block_table_batch_stride + table_idx];

        const int64_t src_k = int64_t(batch) * k_new_batch_stride + int64_t(token) * k_new_row_stride
            + int64_t(kv_head) * k_new_head_stride + dim;
        const int64_t src_v = int64_t(batch) * v_new_batch_stride + int64_t(token) * v_new_row_stride
            + int64_t(kv_head) * v_new_head_stride + dim;
        const int64_t dst_k = int64_t(physical_block) * k_cache_block_stride + int64_t(page_offset) * k_cache_row_stride
            + int64_t(kv_head) * k_cache_head_stride + dim;
        const int64_t dst_v = int64_t(physical_block) * v_cache_block_stride + int64_t(page_offset) * v_cache_row_stride
            + int64_t(kv_head) * v_cache_head_stride + dim;
        k_cache[dst_k] = k_new[src_k];
        v_cache[dst_v] = v_new[src_v];
    }
}

int ceil_div_int(const int a, const int b) {
    return (a + b - 1) / b;
}

bool split_is_eligible(const int num_n_blocks, const int num_splits) {
    return num_splits == 1
        || ceil_div_int(num_n_blocks, num_splits) != ceil_div_int(num_n_blocks, num_splits - 1);
}

int num_splits_efficiency_heuristic(
    const int row_blocks,
    const int num_sms,
    const int num_n_blocks,
    const int max_splits) {
    if (row_blocks >= int(0.8f * float(num_sms))) {
        return 1;
    }
    const int capped_splits = std::min({max_splits, num_sms, num_n_blocks, 128});
    float max_efficiency = 0.0f;
    float efficiency[128] = {};
    for (int split = 1; split <= capped_splits; ++split) {
        if (!split_is_eligible(num_n_blocks, split)) {
            continue;
        }
        const float waves = float(row_blocks * split) / float(num_sms);
        const float eff = waves / std::ceil(waves);
        efficiency[split - 1] = eff;
        max_efficiency = std::max(max_efficiency, eff);
    }
    for (int split = 1; split <= capped_splits; ++split) {
        if (!split_is_eligible(num_n_blocks, split)) {
            continue;
        }
        if (efficiency[split - 1] >= 0.95f * max_efficiency) {
            return split;
        }
    }
    return 1;
}

int choose_num_splits(
    const int requested,
    const int seqlen_k,
    const int head_dim,
    const int seqlen_q,
    const int batch_size,
    const int nheads,
    const int num_sms,
    const bool is_paged) {
    if (requested > 0) {
        return requested;
    }
    (void)is_paged;
    const int block_n = head_dim <= 64 ? 256 : 128;
    const int n_blocks = (seqlen_k + block_n - 1) / block_n;
    if (n_blocks <= 1) {
        return 1;
    }

    if (seqlen_q > 1) {
        return 1;
    }

    // Below roughly 2k tokens for h64 (or 1k for h128), split-KV overhead
    // dominates on SM75 even when occupancy looks low.
    if (n_blocks < 8) {
        return 1;
    }

    // Match upstream's split-KV intuition: split only when the output-row
    // parallelism cannot fill the SMs well.  The split kernels use 64-row M
    // blocks for inference, so query chunks contribute row blocks naturally.
    const int num_m_blocks = ceil_div_int(seqlen_q, 64);
    const int row_blocks = batch_size * nheads * num_m_blocks;
    const int occupancy_splits = num_splits_efficiency_heuristic(row_blocks, num_sms, n_blocks, 128);
    if (row_blocks <= num_sms / 3) {
        return std::max(occupancy_splits, std::min(6, n_blocks));
    }
    return occupancy_splits;
}

}  // namespace

void run_kvcache_paged_append(
    at::Tensor k_cache,
    at::Tensor v_cache,
    at::Tensor k_new,
    at::Tensor v_new,
    at::Tensor cache_seqlens,
    at::Tensor block_table) {
    const int batch_size = k_new.size(0);
    const int seqlen_new = k_new.size(1);
    if (seqlen_new == 0) {
        return;
    }
    const int nheads_k = k_new.size(2);
    const int head_dim = k_new.size(3);
    const int page_block_size = k_cache.size(1);
    const int64_t elems_per =
        (head_dim % 2 == 0)
            ? (int64_t(seqlen_new) * nheads_k * (head_dim / 2))
            : (int64_t(seqlen_new) * nheads_k * head_dim);
    const int64_t total = int64_t(batch_size) * elems_per;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    paged_append_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<half *>(k_cache.data_ptr()),
        reinterpret_cast<half *>(v_cache.data_ptr()),
        reinterpret_cast<const half *>(k_new.data_ptr()),
        reinterpret_cast<const half *>(v_new.data_ptr()),
        cache_seqlens.data_ptr<int>(),
        block_table.data_ptr<int>(),
        batch_size, seqlen_new, nheads_k, head_dim, page_block_size,
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        k_new.stride(0), k_new.stride(1), k_new.stride(2),
        v_new.stride(0), v_new.stride(1), v_new.stride(2),
        block_table.stride(0));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<at::Tensor> run_mha_fwd_kvcache_native(
    at::Tensor q,
    at::Tensor k_cache,
    at::Tensor v_cache,
    at::Tensor cache_seqlens,
    at::Tensor cache_batch_idx,
    bool has_cache_batch_idx,
    at::Tensor block_table,
    bool has_block_table,
    const float softmax_scale,
    bool is_causal,
    int requested_num_splits) {
    const int batch_size = q.size(0);
    const int seqlen_q = q.size(1);
    const int nheads = q.size(2);
    const int head_dim = q.size(3);
    const int nheads_k = k_cache.size(2);
    TORCH_CHECK(head_dim == 64 || head_dim == 128,
                "Turing FlashAttention supports head_dim 64 or 128");
    const int seqlen_k_max = has_block_table ? block_table.size(1) * k_cache.size(1) : k_cache.size(1);
    const int rows = batch_size * nheads * seqlen_q;
    const cudaDeviceProp *prop = at::cuda::getCurrentDeviceProperties();
    int num_splits =
        choose_num_splits(
            requested_num_splits, seqlen_k_max, head_dim, seqlen_q,
            batch_size, nheads, prop->multiProcessorCount, has_block_table);
    // Score/decode split kernels: dynamic `scores[split_len]` + static `q_vec[Headdim]`; stay within 48 KiB
    // per block. Bump split count (up to 128) so a single split never needs more than the limit, avoiding
    // a multi-second legacy fallback when users request e.g. num_splits=1 on long paged context.
    constexpr int kMaxSplitsClamp = 128;
    constexpr size_t kKvcacheSmemPerBlock = 48 * 1024;
    const auto kvcache_scorebuf_smem_bytes = [&](int ns) -> size_t {
        if (ns <= 0) {
            return 0u;
        }
        const int sl = (seqlen_k_max + ns - 1) / ns;
        return size_t(sl) * sizeof(float) + size_t(head_dim) * sizeof(float);
    };
    while (num_splits < kMaxSplitsClamp && kvcache_scorebuf_smem_bytes(num_splits) > kKvcacheSmemPerBlock) {
        num_splits += 1;
    }
    const char *legacy_env = std::getenv("FLASH_TURING_LEGACY_COMBINE");
    const bool use_legacy_combine = legacy_env != nullptr && legacy_env[0] == '1';

    auto out = torch::empty_like(q);
    auto lse = torch::empty({batch_size, nheads, seqlen_q}, q.options().dtype(torch::kFloat32));
    auto out_accum = num_splits > 1
        ? torch::empty({num_splits, rows, head_dim}, q.options().dtype(torch::kFloat32))
        : torch::empty({0}, q.options().dtype(torch::kFloat32));
    auto lse_accum = torch::empty({num_splits, rows}, q.options().dtype(torch::kFloat32));

    const int *cache_batch_idx_ptr = has_cache_batch_idx ? cache_batch_idx.data_ptr<int>() : nullptr;
    const int *block_table_ptr = has_block_table ? block_table.data_ptr<int>() : nullptr;
    const int page_block_size = has_block_table ? k_cache.size(1) : 1;
    const int64_t block_table_batch_stride = has_block_table ? block_table.stride(0) : 0;

    dim3 grid(rows, num_splits);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const bool wants_decode_kernel = seqlen_q == 1 && is_causal;
    const int split_len_max = (seqlen_k_max + num_splits - 1) / num_splits;
    const size_t score_dynamic_bytes = size_t(split_len_max) * sizeof(float);
    // Keep launches within the safe per-block limit to avoid cudaErrorInvalidValue.
    constexpr size_t kMaxDecodeSmemBytes = 48 * 1024;
    constexpr size_t kMaxSplitSmemBytes = 48 * 1024;
    const bool can_use_decode_kernel = wants_decode_kernel
        && (size_t(head_dim) * sizeof(float) + score_dynamic_bytes) <= kMaxDecodeSmemBytes;
    const bool can_use_split_scores_kernel =
        (size_t(head_dim) * sizeof(float) + score_dynamic_bytes) <= kMaxSplitSmemBytes;
    if (head_dim == 64) {
        if (can_use_decode_kernel) {
            dim3 decode_grid(batch_size * nheads, num_splits);
            kvcache_decode_split_kernel<64><<<decode_grid, kDecodeThreads, score_dynamic_bytes, stream>>>(
                reinterpret_cast<const half *>(q.data_ptr()),
                reinterpret_cast<const half *>(k_cache.data_ptr()),
                reinterpret_cast<const half *>(v_cache.data_ptr()),
                cache_seqlens.data_ptr<int>(),
                cache_batch_idx_ptr,
                block_table_ptr,
                reinterpret_cast<half *>(out.data_ptr()),
                lse.data_ptr<float>(),
                out_accum.numel() == 0 ? nullptr : out_accum.data_ptr<float>(),
                lse_accum.data_ptr<float>(),
                batch_size, nheads, nheads_k, num_splits, page_block_size,
                q.stride(0), q.stride(1), q.stride(2),
                k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
                v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
                out.stride(0), out.stride(1), out.stride(2),
                block_table_batch_stride, softmax_scale);
        } else if (can_use_split_scores_kernel) {
            kvcache_split_kernel<64><<<grid, kThreads, score_dynamic_bytes, stream>>>(
                reinterpret_cast<const half *>(q.data_ptr()),
                reinterpret_cast<const half *>(k_cache.data_ptr()),
                reinterpret_cast<const half *>(v_cache.data_ptr()),
                cache_seqlens.data_ptr<int>(),
                cache_batch_idx_ptr,
                block_table_ptr,
                reinterpret_cast<half *>(out.data_ptr()),
                lse.data_ptr<float>(),
                out_accum.numel() == 0 ? nullptr : out_accum.data_ptr<float>(),
                lse_accum.data_ptr<float>(),
                batch_size, seqlen_q, seqlen_k_max, nheads, nheads_k, num_splits, page_block_size,
                q.stride(0), q.stride(1), q.stride(2),
                k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
                v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
                out.stride(0), out.stride(1), out.stride(2),
                block_table_batch_stride, softmax_scale, is_causal);
        } else {
            kvcache_split_kernel_legacy<64><<<grid, kThreads, 0, stream>>>(
                reinterpret_cast<const half *>(q.data_ptr()),
                reinterpret_cast<const half *>(k_cache.data_ptr()),
                reinterpret_cast<const half *>(v_cache.data_ptr()),
                cache_seqlens.data_ptr<int>(),
                cache_batch_idx_ptr,
                block_table_ptr,
                reinterpret_cast<half *>(out.data_ptr()),
                lse.data_ptr<float>(),
                out_accum.numel() == 0 ? nullptr : out_accum.data_ptr<float>(),
                lse_accum.data_ptr<float>(),
                batch_size, seqlen_q, seqlen_k_max, nheads, nheads_k, num_splits, page_block_size,
                q.stride(0), q.stride(1), q.stride(2),
                k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
                v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
                out.stride(0), out.stride(1), out.stride(2),
                block_table_batch_stride, softmax_scale, is_causal);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        if (num_splits > 1) {
            if (use_legacy_combine) {
                const int threads = 256;
                const int blocks = (rows * 64 + threads - 1) / threads;
                kvcache_combine_kernel_legacy<64><<<blocks, threads, 0, stream>>>(
                    out_accum.data_ptr<float>(), lse_accum.data_ptr<float>(),
                    reinterpret_cast<half *>(out.data_ptr()), lse.data_ptr<float>(), rows, num_splits,
                    batch_size, seqlen_q, nheads, out.stride(0), out.stride(1), out.stride(2));
            } else {
                const int threads = 64;
                const int blocks = rows;
                kvcache_combine_kernel<64><<<blocks, threads, 0, stream>>>(
                    out_accum.data_ptr<float>(), lse_accum.data_ptr<float>(),
                    reinterpret_cast<half *>(out.data_ptr()), lse.data_ptr<float>(), rows, num_splits,
                    batch_size, seqlen_q, nheads, out.stride(0), out.stride(1), out.stride(2));
            }
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    } else {
        if (can_use_decode_kernel) {
            dim3 decode_grid(batch_size * nheads, num_splits);
            kvcache_decode_split_kernel<128><<<decode_grid, kDecodeThreads, score_dynamic_bytes, stream>>>(
                reinterpret_cast<const half *>(q.data_ptr()),
                reinterpret_cast<const half *>(k_cache.data_ptr()),
                reinterpret_cast<const half *>(v_cache.data_ptr()),
                cache_seqlens.data_ptr<int>(),
                cache_batch_idx_ptr,
                block_table_ptr,
                reinterpret_cast<half *>(out.data_ptr()),
                lse.data_ptr<float>(),
                out_accum.numel() == 0 ? nullptr : out_accum.data_ptr<float>(),
                lse_accum.data_ptr<float>(),
                batch_size, nheads, nheads_k, num_splits, page_block_size,
                q.stride(0), q.stride(1), q.stride(2),
                k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
                v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
                out.stride(0), out.stride(1), out.stride(2),
                block_table_batch_stride, softmax_scale);
        } else if (can_use_split_scores_kernel) {
            kvcache_split_kernel<128><<<grid, kThreads, score_dynamic_bytes, stream>>>(
                reinterpret_cast<const half *>(q.data_ptr()),
                reinterpret_cast<const half *>(k_cache.data_ptr()),
                reinterpret_cast<const half *>(v_cache.data_ptr()),
                cache_seqlens.data_ptr<int>(),
                cache_batch_idx_ptr,
                block_table_ptr,
                reinterpret_cast<half *>(out.data_ptr()),
                lse.data_ptr<float>(),
                out_accum.numel() == 0 ? nullptr : out_accum.data_ptr<float>(),
                lse_accum.data_ptr<float>(),
                batch_size, seqlen_q, seqlen_k_max, nheads, nheads_k, num_splits, page_block_size,
                q.stride(0), q.stride(1), q.stride(2),
                k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
                v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
                out.stride(0), out.stride(1), out.stride(2),
                block_table_batch_stride, softmax_scale, is_causal);
        } else {
            kvcache_split_kernel_legacy<128><<<grid, kThreads, 0, stream>>>(
                reinterpret_cast<const half *>(q.data_ptr()),
                reinterpret_cast<const half *>(k_cache.data_ptr()),
                reinterpret_cast<const half *>(v_cache.data_ptr()),
                cache_seqlens.data_ptr<int>(),
                cache_batch_idx_ptr,
                block_table_ptr,
                reinterpret_cast<half *>(out.data_ptr()),
                lse.data_ptr<float>(),
                out_accum.numel() == 0 ? nullptr : out_accum.data_ptr<float>(),
                lse_accum.data_ptr<float>(),
                batch_size, seqlen_q, seqlen_k_max, nheads, nheads_k, num_splits, page_block_size,
                q.stride(0), q.stride(1), q.stride(2),
                k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
                v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
                out.stride(0), out.stride(1), out.stride(2),
                block_table_batch_stride, softmax_scale, is_causal);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        if (num_splits > 1) {
            if (use_legacy_combine) {
                const int threads = 256;
                const int blocks = (rows * 128 + threads - 1) / threads;
                kvcache_combine_kernel_legacy<128><<<blocks, threads, 0, stream>>>(
                    out_accum.data_ptr<float>(), lse_accum.data_ptr<float>(),
                    reinterpret_cast<half *>(out.data_ptr()), lse.data_ptr<float>(), rows, num_splits,
                    batch_size, seqlen_q, nheads, out.stride(0), out.stride(1), out.stride(2));
            } else {
                const int threads = 128;
                const int blocks = rows;
                kvcache_combine_kernel<128><<<blocks, threads, 0, stream>>>(
                    out_accum.data_ptr<float>(), lse_accum.data_ptr<float>(),
                    reinterpret_cast<half *>(out.data_ptr()), lse.data_ptr<float>(), rows, num_splits,
                    batch_size, seqlen_q, nheads, out.stride(0), out.stride(1), out.stride(2));
            }
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    }

    return {out, lse};
}
