#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
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
                const int table_idx = pos / page_block_size;
                const int page_offset = pos - table_idx * page_block_size;
                const int physical_block =
                    block_table[int64_t(batch) * block_table_batch_stride + table_idx];
                k_base = int64_t(physical_block) * k_batch_stride + int64_t(page_offset) * k_row_stride
                    + int64_t(kv_head) * k_head_stride;
            }
            #pragma unroll
            for (int dim = 0; dim < Headdim; ++dim) {
                score += q_vec[dim] * __half2float(k_cache[k_base + dim]);
            }
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
                    const int table_idx = pos / page_block_size;
                    const int page_offset = pos - table_idx * page_block_size;
                    const int physical_block =
                        block_table[int64_t(batch) * block_table_batch_stride + table_idx];
                    k_base = int64_t(physical_block) * k_batch_stride + int64_t(page_offset) * k_row_stride
                        + int64_t(kv_head) * k_head_stride;
                }
                #pragma unroll
                for (int dim = 0; dim < Headdim; ++dim) {
                    score += q_vec[dim] * __half2float(k_cache[k_base + dim]);
                }
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
                        const int table_idx = pos / page_block_size;
                        const int page_offset = pos - table_idx * page_block_size;
                        const int physical_block =
                            block_table[int64_t(batch) * block_table_batch_stride + table_idx];
                        k_base = int64_t(physical_block) * k_batch_stride + int64_t(page_offset) * k_row_stride
                            + int64_t(kv_head) * k_head_stride;
                        v_base = int64_t(physical_block) * v_batch_stride + int64_t(page_offset) * v_row_stride
                            + int64_t(kv_head) * v_head_stride;
                    }
                    #pragma unroll
                    for (int kdim = 0; kdim < Headdim; ++kdim) {
                        score += q_vec[kdim] * __half2float(k_cache[k_base + kdim]);
                    }
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

    if (split_len <= 0) {
        if (threadIdx.x == 0) {
            lse_accum[int64_t(split_idx) * batch_size * nheads + row_idx] = -INFINITY;
            if (num_splits == 1) {
                lse[int64_t(batch) * nheads + head] = 0.0f;
            }
        }
        if (threadIdx.x < Headdim) {
            if (num_splits == 1) {
                out[int64_t(batch) * o_batch_stride + int64_t(head) * o_head_stride + threadIdx.x] = __float2half_rn(0.0f);
            } else {
                out_accum[(int64_t(split_idx) * batch_size * nheads + row_idx) * Headdim + threadIdx.x] = 0.0f;
            }
        }
        return;
    }

    const int dim = threadIdx.x;
    float q_val = 0.0f;
    if (dim < Headdim) {
        q_val = __half2float(q[int64_t(batch) * q_batch_stride
            + int64_t(0) * q_row_stride + int64_t(head) * q_head_stride + dim]);
    }

    int table_idx_k = split_start / page_block_size;
    int page_offset_k = split_start - table_idx_k * page_block_size;
    int physical_block_k = -1;
    if (block_table != nullptr && split_len > 0) {
        physical_block_k = block_table[int64_t(batch) * block_table_batch_stride + table_idx_k];
    }
    for (int pos_idx = 0; pos_idx < split_len; ++pos_idx) {
        float dot = 0.0f;
        if (dim < Headdim) {
            int64_t k_base = 0;
            if (block_table == nullptr) {
                const int pos = split_start + pos_idx;
                k_base = int64_t(cache_batch) * k_batch_stride + int64_t(pos) * k_row_stride
                    + int64_t(kv_head) * k_head_stride;
            } else {
                k_base = int64_t(physical_block_k) * k_batch_stride + int64_t(page_offset_k) * k_row_stride
                    + int64_t(kv_head) * k_head_stride;
            }
            dot = q_val * __half2float(k_cache[k_base + dim]);
        }
        const float score = block_sum<kDecodeThreads>(dot);
        if (threadIdx.x == 0) {
            scores[pos_idx] = score * softmax_scale;
        }
        __syncthreads();
        if (block_table != nullptr) {
            page_offset_k += 1;
            if (page_offset_k == page_block_size && pos_idx + 1 < split_len) {
                table_idx_k += 1;
                page_offset_k = 0;
                physical_block_k = block_table[int64_t(batch) * block_table_batch_stride + table_idx_k];
            }
        }
    }

    float row_max = -INFINITY;
    if (threadIdx.x == 0) {
        for (int i = 0; i < split_len; ++i) {
            row_max = fmaxf(row_max, scores[i]);
        }
    }
    __shared__ float s_row_max;
    if (threadIdx.x == 0) {
        s_row_max = row_max;
    }
    __syncthreads();
    row_max = s_row_max;

    __shared__ float s_row_lse;
    if (threadIdx.x == 0) {
        float row_sum = 0.0f;
        for (int i = 0; i < split_len; ++i) {
            scores[i] = expf(scores[i] - row_max);
            row_sum += scores[i];
        }
        s_row_lse = row_sum == 0.0f ? -INFINITY : row_max + logf(row_sum);
        if (num_splits > 1) {
            lse_accum[int64_t(split_idx) * batch_size * nheads + row_idx] = s_row_lse;
        }
    }
    __syncthreads();

    if (dim < Headdim) {
        float out_val = 0.0f;
        int table_idx_v = split_start / page_block_size;
        int page_offset_v = split_start - table_idx_v * page_block_size;
        int physical_block_v = -1;
        if (block_table != nullptr && split_len > 0) {
            physical_block_v = block_table[int64_t(batch) * block_table_batch_stride + table_idx_v];
        }
        for (int pos_idx = 0; pos_idx < split_len; ++pos_idx) {
            int64_t v_base = 0;
            if (block_table == nullptr) {
                const int pos = split_start + pos_idx;
                v_base = int64_t(cache_batch) * v_batch_stride + int64_t(pos) * v_row_stride
                    + int64_t(kv_head) * v_head_stride;
            } else {
                v_base = int64_t(physical_block_v) * v_batch_stride + int64_t(page_offset_v) * v_row_stride
                    + int64_t(kv_head) * v_head_stride;
            }
            out_val += scores[pos_idx] * __half2float(v_cache[v_base + dim]);
            if (block_table != nullptr) {
                page_offset_v += 1;
                if (page_offset_v == page_block_size && pos_idx + 1 < split_len) {
                    table_idx_v += 1;
                    page_offset_v = 0;
                    physical_block_v = block_table[int64_t(batch) * block_table_batch_stride + table_idx_v];
                }
            }
        }
        if (s_row_lse != -INFINITY) {
            out_val *= expf(row_max - s_row_lse);
        } else {
            out_val = 0.0f;
        }
        if (num_splits == 1) {
            out[int64_t(batch) * o_batch_stride + int64_t(0) * o_row_stride
                + int64_t(head) * o_head_stride + dim] = __float2half_rn(out_val);
        } else {
            out_accum[(int64_t(split_idx) * batch_size * nheads + row_idx) * Headdim + dim] = out_val;
        }
    }
    if (threadIdx.x == 0 && num_splits == 1) {
        lse[int64_t(batch) * nheads + head] = s_row_lse == -INFINITY ? 0.0f : s_row_lse;
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

int choose_num_splits(
    const int requested,
    const int seqlen_k,
    const int head_dim,
    const int seqlen_q,
    const bool is_paged) {
    if (requested > 0) {
        return requested;
    }
    const int block_n = head_dim <= 64 ? 256 : 128;
    const int n_blocks = (seqlen_k + block_n - 1) / block_n;
    if (n_blocks <= 1) {
        return 1;
    }
    const int seqlen_k_bucket = seqlen_k >= 16384 ? 2 : (seqlen_k >= 8192 ? 1 : 0);
    const int head_dim_bucket = head_dim <= 64 ? 0 : 1;
    if (seqlen_q > 1) {
        // Chunked queries have enough intrinsic parallelism from seqlen_q.
        return 1;
    }
    if (is_paged) {
        // Policy table: [seqlen_k_bucket][head_dim_bucket]
        // Buckets: seqlen_k {<8k, 8k-16k, >=16k}, head_dim {<=64, >64}
        static constexpr int paged_decode_policy[3][2] = {
            {1, 1},
            {4, 4},
            {4, 4},
        };
        return std::min(paged_decode_policy[seqlen_k_bucket][head_dim_bucket], n_blocks);
    }
    static constexpr int contiguous_decode_policy[3][2] = {
        {1, 1},
        {1, 1},
        {2, 2},
    };
    return std::min(contiguous_decode_policy[seqlen_k_bucket][head_dim_bucket], n_blocks);
}

}  // namespace

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
    const int seqlen_k_max = has_block_table ? block_table.size(1) * k_cache.size(1) : k_cache.size(1);
    const int rows = batch_size * nheads * seqlen_q;
    int num_splits =
        choose_num_splits(requested_num_splits, seqlen_k_max, head_dim, seqlen_q, has_block_table);
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
    const size_t decode_smem_bytes = size_t(split_len_max) * sizeof(float);
    // Decode kernel stores one score per split token in dynamic shared memory.
    // Keep launches within the safe per-block limit to avoid cudaErrorInvalidValue.
    constexpr size_t kMaxDecodeSmemBytes = 48 * 1024;
    const bool can_use_decode_kernel = wants_decode_kernel && decode_smem_bytes <= kMaxDecodeSmemBytes;
    if (head_dim == 64) {
        if (can_use_decode_kernel) {
            dim3 decode_grid(batch_size * nheads, num_splits);
            kvcache_decode_split_kernel<64><<<decode_grid, kDecodeThreads, decode_smem_bytes, stream>>>(
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
        } else {
            kvcache_split_kernel<64><<<grid, kThreads, 0, stream>>>(
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
            kvcache_decode_split_kernel<128><<<decode_grid, kDecodeThreads, decode_smem_bytes, stream>>>(
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
        } else {
            kvcache_split_kernel<128><<<grid, kThreads, 0, stream>>>(
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
