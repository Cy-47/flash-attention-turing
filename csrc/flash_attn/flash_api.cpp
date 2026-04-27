#include <torch/extension.h>
#include <pybind11/pytypes.h>
#include "flash.h"
#include "static_switch.h"

namespace py = pybind11;

namespace {

inline void check_supported_head_dim(const int head_size)
{
    TORCH_CHECK(head_size == 64 || head_size == 128,
                "Turing FlashAttention supports head_dim 64 or 128");
}

} // namespace

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
    int requested_num_splits);

void run_kvcache_paged_append(
    at::Tensor k_cache,
    at::Tensor v_cache,
    at::Tensor k_new,
    at::Tensor v_new,
    at::Tensor cache_seqlens,
    at::Tensor block_table);

void set_params_fprop(Flash_fwd_params &params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      // device pointers
                      const at::Tensor q,
                      const at::Tensor k,
                      const at::Tensor v,
                      at::Tensor out,
                      at::Tensor l,
                      // void *softmax_lse_d,
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      void *seqused_k_d,
                      float softmax_scale,
                      bool is_causal)
{

    // Reset the parameters
    params = {};

    // Set the pointers and strides.
    params.q_ptr = reinterpret_cast<half_t *>(q.data_ptr());
    params.k_ptr = reinterpret_cast<half_t *>(k.data_ptr());
    params.v_ptr = reinterpret_cast<half_t *>(v.data_ptr());
    params.o_ptr = reinterpret_cast<half_t *>(out.data_ptr());

    // Softmax sum
    params.l_ptr = reinterpret_cast<float *>(l.data_ptr());

    // All stride are in elements, not bytes.
    // params.q_row_stride = q.stride(-3);
    // params.k_row_stride = k.stride(-3);
    // params.v_row_stride = v.stride(-3);
    // params.q_head_stride = q.stride(-2);
    // params.k_head_stride = k.stride(-2);
    // params.v_head_stride = v.stride(-2);

    // params.o_row_stride = out.stride(-3);
    // params.o_head_stride = out.stride(-2);

    // if (cu_seqlens_q_d == nullptr) {
    //     params.q_batch_stride = q.stride(0);
    //     params.k_batch_stride = k.stride(0);
    //     params.v_batch_stride = v.stride(0);
    //     params.o_batch_stride = out.stride(0);
    // }

    // Set the dimensions.
    params.b = b;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.h = h;
    params.h_k = h_k;
    params.h_h_k_ratio = h / h_k;
    params.d = d;
    params.softmax_scale = softmax_scale;
    params.is_causal = is_causal;
    params.cu_seqlens_q = static_cast<int *>(cu_seqlens_q_d);
    params.cu_seqlens_k = static_cast<int *>(cu_seqlens_k_d);
    params.seqused_k = static_cast<int *>(seqused_k_d);
    params.page_block_size = 1;
    params.num_splits = 1;
}

void set_params_dgrad(Flash_bwd_params &params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      // device pointers
                      const at::Tensor q,
                      const at::Tensor k,
                      const at::Tensor v,
                      const at::Tensor out,
                      const at::Tensor l,
                      const at::Tensor dout,
                      at::Tensor dq,
                      at::Tensor dk,
                      at::Tensor dv,
                      at::Tensor do_o,
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      void *seqused_k_d,
                      // void *softmax_lse_d,
                      float softmax_scale,
                      bool is_causal)
{

    set_params_fprop(params,
                     b,
                     seqlen_q,
                     seqlen_k,
                     h,
                     h_k,
                     d,
                     q, k, v, out, l,
                     cu_seqlens_q_d,
                     cu_seqlens_k_d,
                     seqused_k_d,
                     softmax_scale,
                     is_causal);

    params.do_o_ptr = reinterpret_cast<float *>(do_o.data_ptr());
    params.do_ptr = reinterpret_cast<half_t *>(dout.data_ptr());

    params.dq_ptr = reinterpret_cast<half_t *>(dq.data_ptr());
    params.dk_ptr = reinterpret_cast<half_t *>(dk.data_ptr());
    params.dv_ptr = reinterpret_cast<half_t *>(dv.data_ptr());

    // params.do_row_stride = dout.stride(-3);
    // params.do_head_stride = dout.stride(-2);

    // params.dq_row_stride = dq.stride(-3);
    // params.dk_row_stride = dk.stride(-3);
    // params.dv_row_stride = dv.stride(-3);
    // params.dq_head_stride = dq.stride(-2);
    // params.dk_head_stride = dk.stride(-2);
    // params.dv_head_stride = dv.stride(-2);

    // if (cu_seqlens_q_d == nullptr) {
    //     params.do_batch_stride = dout.stride(0);
    //     params.dq_batch_stride = dq.stride(0);
    //     params.dk_batch_stride = dk.stride(0);
    //     params.dv_batch_stride = dv.stride(0);
    // }
}

void run_mha_fwd(Flash_fwd_params &params)
{
    HEADDIM_SWITCH(params.d, [&]
                   { BOOL_SWITCH(params.is_causal, Is_causal, [&]
                                 { run_mha_fwd_<kHeadDim, Is_causal>(params); }); });
}

void run_mha_bwd(Flash_bwd_params &params)
{
    HEADDIM_SWITCH(params.d, [&]
                   { BOOL_SWITCH(params.is_causal, Is_causal, [&]
                                 { run_mha_bwd_<kHeadDim, Is_causal>(params); }); });
}

std::vector<at::Tensor>
mha_fwd(at::Tensor q,
        at::Tensor k,
        at::Tensor v,
        //             int batch_size,
        //             int seq_len,
        //             int num_heads,
        //             int head_dim,
        const float softmax_scale,
        bool is_causal)
{
    auto device = q.device();

    const auto sizes = q.sizes();

    int batch_size = sizes[0];
    int seqlen_q = sizes[1];
    int num_heads = sizes[2];
    int head_size = sizes[3];

    int seqlen_k = k.size(1);
    int num_heads_k = k.size(2);

    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q, k, v must be rank-4 tensors");
    TORCH_CHECK(k.size(0) == batch_size && v.size(0) == batch_size, "k/v batch size must match q");
    TORCH_CHECK(v.size(1) == seqlen_k, "k and v seqlen_k must match");
    TORCH_CHECK(v.size(2) == num_heads_k, "k and v num_heads must match");
    TORCH_CHECK(k.size(3) == head_size && v.size(3) == head_size, "q/k/v head_dim must match");
    TORCH_CHECK(num_heads % num_heads_k == 0, "num_heads_q must be divisible by num_heads_k for GQA/MQA");
    check_supported_head_dim(head_size);

    at::Tensor o = torch::zeros(q.sizes(), q.options().dtype(torch::kFloat16));

    std::vector<int64_t> size = {batch_size, num_heads, seqlen_q};
    at::Tensor l = torch::zeros(size, q.options().dtype(torch::kFloat32).device(device));

    TORCH_CHECK(o.is_cuda(), "Tensor o is not on CUDA");

    //    half_t* q_ptr = reinterpret_cast<half_t*>(q.data_ptr());
    //    half_t* k_ptr = reinterpret_cast<half_t*>(k.data_ptr());
    //    half_t* v_ptr = reinterpret_cast<half_t*>(v.data_ptr());
    //    half_t* o_ptr = reinterpret_cast<half_t*>(o.data_ptr());
    //
    //    float* l_ptr = reinterpret_cast<float*>(l.data_ptr());

    Flash_fwd_params params;
    set_params_fprop(params,
                     batch_size,
                     seqlen_q,
                     seqlen_k,
                     num_heads,
                     num_heads_k,
                     head_size,
                     q, k, v, o, l,
                     nullptr,
                     nullptr,
                     nullptr,
                     softmax_scale,
                     is_causal);

    // std::cout << "Q ptr: " << q.data_ptr() << "\n";
    // std::cout << "K ptr: " << k.data_ptr() << "\n";
    // std::cout << "V ptr: " << v.data_ptr() << "\n";
    // std::cout << "O ptr: " << o.data_ptr() << "\n";

    run_mha_fwd(params);

    return {o, l};
}

std::vector<at::Tensor>
mha_bwd(at::Tensor q,
        at::Tensor k,
        at::Tensor v,
        at::Tensor out,
        at::Tensor l,
        at::Tensor dout,
        //        int batch_size,
        //        int seq_len,
        //        int num_heads,
        //        int head_dim,
        const float softmax_scale,
        bool is_causal)
{

    const auto sizes = q.sizes();

    int batch_size = sizes[0];
    int seqlen_q = sizes[1];
    int num_heads = sizes[2];
    int head_size = sizes[3];

    int seqlen_k = k.size(1);
    int num_heads_k = k.size(2);

    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q, k, v must be rank-4 tensors");
    TORCH_CHECK(out.dim() == 4 && dout.dim() == 4, "out and dout must be rank-4 tensors");
    TORCH_CHECK(k.size(0) == batch_size && v.size(0) == batch_size, "k/v batch size must match q");
    TORCH_CHECK(v.size(1) == seqlen_k, "k and v seqlen_k must match");
    TORCH_CHECK(v.size(2) == num_heads_k, "k and v num_heads must match");
    TORCH_CHECK(k.size(3) == head_size && v.size(3) == head_size, "q/k/v head_dim must match");
    TORCH_CHECK(out.sizes() == q.sizes() && dout.sizes() == q.sizes(), "out and dout must match q shape");
    TORCH_CHECK(num_heads % num_heads_k == 0, "num_heads_q must be divisible by num_heads_k for GQA/MQA");
    check_supported_head_dim(head_size);

    at::Tensor dq = torch::zeros(q.sizes(), q.options().dtype(torch::kFloat16));
    at::Tensor dk = torch::zeros(k.sizes(), k.options().dtype(torch::kFloat16));
    at::Tensor dv = torch::zeros(v.sizes(), v.options().dtype(torch::kFloat16));

    at::Tensor dk_expanded, dv_expanded;
    if (num_heads != num_heads_k)
    {
        dk_expanded = torch::zeros({batch_size, seqlen_k, num_heads, head_size}, k.options().dtype(torch::kFloat16));
        dv_expanded = torch::zeros({batch_size, seqlen_k, num_heads, head_size}, v.options().dtype(torch::kFloat16));
    }
    else
    {
        dk_expanded = dk;
        dv_expanded = dv;
    }

    at::Tensor do_o = torch::zeros(l.sizes(), l.options());

    Flash_bwd_params params;

    set_params_dgrad(params,
                     batch_size,
                     seqlen_q,
                     seqlen_k,
                     num_heads,
                     num_heads_k,
                     head_size,
                     q,
                     k,
                     v,
                     out,
                     l,
                     dout,
                     dq,
                     dk_expanded,
                     dv_expanded,
                     do_o,
                     nullptr,
                     nullptr,
                     nullptr,
                     softmax_scale,
                     is_causal);

    run_mha_bwd(params);

    if (num_heads != num_heads_k)
    {
        torch::sum_out(
            dk,
            torch::reshape(dk_expanded, {batch_size, seqlen_k, num_heads_k, num_heads / num_heads_k, head_size}),
            {3});
        torch::sum_out(
            dv,
            torch::reshape(dv_expanded, {batch_size, seqlen_k, num_heads_k, num_heads / num_heads_k, head_size}),
            {3});
    }

    return {dq, dk, dv};
}

std::vector<at::Tensor>
mha_varlen_fwd(at::Tensor q,
               at::Tensor k,
               at::Tensor v,
               at::Tensor &cu_seqlens_q,
               at::Tensor &cu_seqlens_k,
               const int max_seqlen_q,
               const int max_seqlen_k,
               const float softmax_scale,
               bool is_causal)
{
    TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3, "q, k, v must be rank-3 packed tensors");
    TORCH_CHECK(cu_seqlens_q.is_cuda() && cu_seqlens_k.is_cuda(), "cu_seqlens_q/cu_seqlens_k must be CUDA tensors");
    TORCH_CHECK(cu_seqlens_q.scalar_type() == torch::kInt32 && cu_seqlens_k.scalar_type() == torch::kInt32,
                "cu_seqlens_q/cu_seqlens_k must be int32 tensors");
    TORCH_CHECK(cu_seqlens_q.is_contiguous() && cu_seqlens_k.is_contiguous(),
                "cu_seqlens_q/cu_seqlens_k must be contiguous");
    TORCH_CHECK(cu_seqlens_q.dim() == 1 && cu_seqlens_k.dim() == 1, "cu_seqlens_q/cu_seqlens_k must be rank-1");
    TORCH_CHECK(cu_seqlens_q.numel() >= 2 && cu_seqlens_k.numel() >= 2,
                "cu_seqlens_q/cu_seqlens_k must have at least 2 elements");
    const int batch_size = cu_seqlens_q.numel() - 1;
    TORCH_CHECK(cu_seqlens_k.numel() == batch_size + 1,
                "cu_seqlens_k must have shape [batch_size + 1] with cumulative offsets");
    TORCH_CHECK(k.size(0) == v.size(0), "k and v total tokens must match");
    TORCH_CHECK(k.size(1) == v.size(1), "k and v num_heads must match");
    TORCH_CHECK(k.size(2) == v.size(2), "k and v head_dim must match");
    TORCH_CHECK(q.size(2) == k.size(2), "q/k/v head_dim must match");
    TORCH_CHECK(q.size(1) % k.size(1) == 0, "num_heads_q must be divisible by num_heads_k for GQA/MQA");

    const int num_heads = q.size(1);
    const int num_heads_k = k.size(1);
    const int head_size = q.size(2);
    check_supported_head_dim(head_size);

    at::Tensor out = torch::zeros_like(q);
    at::Tensor l = torch::zeros({batch_size, num_heads, max_seqlen_q}, q.options().dtype(torch::kFloat32));

    // at::Tensor o = torch::zeros(q.sizes(), q.options().dtype(torch::kFloat16));

    // std::vector<int64_t> size = {batch_size, num_heads, seqlen_q};
    // at::Tensor l = torch::zeros(size, q.options().dtype(torch::kFloat32).device(device));

    // TORCH_CHECK(o.is_cuda(), "Tensor o is not on CUDA");

    Flash_fwd_params params;
    set_params_fprop(
        params,
        batch_size,
        max_seqlen_q,
        max_seqlen_k,
        num_heads,
        num_heads_k,
        head_size,
        q, k, v, out, l,
        cu_seqlens_q.data_ptr(),
        cu_seqlens_k.data_ptr(),
        nullptr,
        softmax_scale,
        is_causal);

    run_mha_fwd(params);

    return {out, l};
}

std::vector<at::Tensor>
mha_varlen_bwd(at::Tensor q,
               at::Tensor k,
               at::Tensor v,
               at::Tensor out,
               at::Tensor l,
               at::Tensor dout,
               at::Tensor cu_seqlens_q,
               at::Tensor cu_seqlens_k,
               const int max_seqlen_q,
               const int max_seqlen_k,
               const float softmax_scale,
               bool is_causal)
{
    TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3, "q, k, v must be rank-3 packed tensors");
    TORCH_CHECK(cu_seqlens_q.is_cuda() && cu_seqlens_k.is_cuda(), "cu_seqlens_q/cu_seqlens_k must be CUDA tensors");
    TORCH_CHECK(cu_seqlens_q.scalar_type() == torch::kInt32 && cu_seqlens_k.scalar_type() == torch::kInt32,
                "cu_seqlens_q/cu_seqlens_k must be int32 tensors");
    TORCH_CHECK(cu_seqlens_q.is_contiguous() && cu_seqlens_k.is_contiguous(),
                "cu_seqlens_q/cu_seqlens_k must be contiguous");
    TORCH_CHECK(cu_seqlens_q.dim() == 1 && cu_seqlens_k.dim() == 1, "cu_seqlens_q/cu_seqlens_k must be rank-1");
    TORCH_CHECK(cu_seqlens_q.numel() >= 2 && cu_seqlens_k.numel() >= 2,
                "cu_seqlens_q/cu_seqlens_k must have at least 2 elements");
    const int64_t batch_size = cu_seqlens_q.numel() - 1;
    TORCH_CHECK(cu_seqlens_k.numel() == batch_size + 1,
                "cu_seqlens_k must have shape [batch_size + 1] with cumulative offsets");
    TORCH_CHECK(k.size(0) == v.size(0), "k and v total tokens must match");
    TORCH_CHECK(k.size(1) == v.size(1), "k and v num_heads must match");
    TORCH_CHECK(k.size(2) == v.size(2), "k and v head_dim must match");
    TORCH_CHECK(q.size(2) == k.size(2), "q/k/v head_dim must match");
    TORCH_CHECK(q.size(1) % k.size(1) == 0, "num_heads_q must be divisible by num_heads_k for GQA/MQA");

    TORCH_CHECK(out.sizes() == q.sizes(), "out must match q shape");
    TORCH_CHECK(dout.sizes() == q.sizes(), "dout must match q shape");
    TORCH_CHECK(l.dim() == 3, "l must be rank-3 for varlen_bwd");

    const int64_t num_heads = q.size(1);
    const int64_t num_heads_k = k.size(1);
    const int64_t head_size = q.size(2);
    const int64_t total_k = k.size(0);
    check_supported_head_dim(head_size);
    TORCH_CHECK(l.size(0) == batch_size && l.size(1) == num_heads && l.size(2) == max_seqlen_q,
                "l must have shape [batch_size, nheads_q, max_seqlen_q]");

    at::Tensor dq = torch::zeros_like(q);
    at::Tensor dk = torch::zeros_like(k);
    at::Tensor dv = torch::zeros_like(v);
    at::Tensor dk_expanded = dk;
    at::Tensor dv_expanded = dv;
    if (num_heads != num_heads_k)
    {
        dk_expanded = torch::zeros({total_k, num_heads, head_size}, k.options().dtype(torch::kFloat16));
        dv_expanded = torch::zeros({total_k, num_heads, head_size}, v.options().dtype(torch::kFloat16));
    }
    at::Tensor do_o = torch::zeros(l.sizes(), l.options());

    Flash_bwd_params params;
    set_params_dgrad(
        params,
        batch_size,
        max_seqlen_q,
        max_seqlen_k,
        num_heads,
        num_heads_k,
        head_size,
        q, k, v, out, l, dout,
        dq, dk_expanded, dv_expanded, do_o,
        cu_seqlens_q.data_ptr(),
        cu_seqlens_k.data_ptr(),
        nullptr,
        softmax_scale,
        is_causal);

    run_mha_bwd(params);

    if (num_heads != num_heads_k)
    {
        torch::sum_out(
            dk,
            torch::reshape(dk_expanded, {total_k, num_heads_k, num_heads / num_heads_k, head_size}),
            {2});
        torch::sum_out(
            dv,
            torch::reshape(dv_expanded, {total_k, num_heads_k, num_heads / num_heads_k, head_size}),
            {2});
    }

    return {dq, dk, dv};
}

std::vector<at::Tensor>
mha_fwd_kvcache(at::Tensor q,
                at::Tensor k_cache,
                at::Tensor v_cache,
                py::object k_obj,
                py::object v_obj,
                at::Tensor cache_seqlens,
                py::object rotary_cos_obj,
                py::object rotary_sin_obj,
                py::object cache_batch_idx_obj,
                py::object cache_leftpad_obj,
                py::object block_table_obj,
                py::object alibi_slopes_obj,
                py::object out_obj,
                const float softmax_scale,
                bool is_causal,
                int window_size_left,
                int window_size_right,
                const float softcap,
                bool rotary_interleaved,
                int num_splits)
{
    (void)rotary_interleaved;
    TORCH_CHECK(rotary_cos_obj.is_none() && rotary_sin_obj.is_none(),
                "Turing KV-cache does not support rotary embeddings yet");
    TORCH_CHECK(cache_leftpad_obj.is_none(), "Turing KV-cache does not support cache_leftpad yet");
    TORCH_CHECK(alibi_slopes_obj.is_none(), "Turing KV-cache does not support ALiBi yet");
    TORCH_CHECK(out_obj.is_none(), "Turing KV-cache does not support a preallocated output tensor yet");
    TORCH_CHECK(window_size_left == -1 && window_size_right == -1,
                "Turing KV-cache does not support local window attention yet");
    TORCH_CHECK(softcap == 0.0f, "Turing KV-cache does not support softcap yet");
    TORCH_CHECK(num_splits >= 0, "num_splits must be non-negative");

    TORCH_CHECK(q.is_cuda() && k_cache.is_cuda() && v_cache.is_cuda(), "q, k_cache, v_cache must be CUDA tensors");
    TORCH_CHECK(q.scalar_type() == torch::kFloat16 && k_cache.scalar_type() == torch::kFloat16 && v_cache.scalar_type() == torch::kFloat16,
                "q, k_cache, v_cache must be float16 tensors");
    TORCH_CHECK(q.dim() == 4 && k_cache.dim() == 4 && v_cache.dim() == 4,
                "q, k_cache, v_cache must be rank-4 tensors");
    TORCH_CHECK(k_cache.size(1) == v_cache.size(1), "k_cache and v_cache seqlen must match");
    TORCH_CHECK(k_cache.size(2) == v_cache.size(2), "k_cache and v_cache num_heads must match");
    TORCH_CHECK(q.size(3) == k_cache.size(3) && q.size(3) == v_cache.size(3),
                "q, k_cache, v_cache head_dim must match");
    TORCH_CHECK(q.size(2) % k_cache.size(2) == 0,
                "num_heads_q must be divisible by num_heads_k for GQA/MQA");
    check_supported_head_dim(q.size(3));
    TORCH_CHECK(q.stride(-1) == 1 && k_cache.stride(-1) == 1 && v_cache.stride(-1) == 1,
                "q, k_cache, v_cache must have contiguous last dimension");
    TORCH_CHECK(cache_seqlens.is_cuda(), "cache_seqlens must be a CUDA tensor");
    TORCH_CHECK(cache_seqlens.scalar_type() == torch::kInt32, "cache_seqlens must be an int32 tensor");
    TORCH_CHECK(cache_seqlens.is_contiguous(), "cache_seqlens must be contiguous");
    TORCH_CHECK(cache_seqlens.dim() == 1 && cache_seqlens.numel() == q.size(0),
                "cache_seqlens must have shape [batch_size]");

    const int batch_size = q.size(0);
    const int seqlen_q = q.size(1);
    const int num_heads = q.size(2);
    const int num_heads_k = k_cache.size(2);
    const int head_size = q.size(3);
    const bool has_block_table = !block_table_obj.is_none();
    at::Tensor block_table;
    int page_block_size = 1;
    int seqlen_cache = k_cache.size(1);
    int batch_size_cache = k_cache.size(0);
    if (has_block_table)
    {
        block_table = block_table_obj.cast<at::Tensor>();
        TORCH_CHECK(block_table.is_cuda(), "block_table must be a CUDA tensor");
        TORCH_CHECK(block_table.scalar_type() == torch::kInt32, "block_table must be an int32 tensor");
        TORCH_CHECK(block_table.stride(-1) == 1, "block_table must have contiguous last dimension");
        TORCH_CHECK(block_table.dim() == 2 && block_table.size(0) == batch_size,
                    "block_table must have shape [batch_size, max_num_blocks_per_seq]");
        page_block_size = k_cache.size(1);
        TORCH_CHECK(page_block_size % 256 == 0, "Paged KV cache block size must be divisible by 256");
        seqlen_cache = block_table.size(1) * page_block_size;
        batch_size_cache = batch_size;
    }

    at::Tensor cache_batch_idx;
    const bool has_cache_batch_idx = !cache_batch_idx_obj.is_none();
    if (has_cache_batch_idx)
    {
        TORCH_CHECK(!has_block_table, "Paged KV cache does not support cache_batch_idx");
        cache_batch_idx = cache_batch_idx_obj.cast<at::Tensor>();
        TORCH_CHECK(cache_batch_idx.is_cuda(), "cache_batch_idx must be a CUDA tensor");
        TORCH_CHECK(cache_batch_idx.scalar_type() == torch::kInt32, "cache_batch_idx must be an int32 tensor");
        TORCH_CHECK(cache_batch_idx.is_contiguous(), "cache_batch_idx must be contiguous");
        TORCH_CHECK(cache_batch_idx.dim() == 1 && cache_batch_idx.numel() == batch_size,
                    "cache_batch_idx must have shape [batch_size]");
    }
    else
    {
        TORCH_CHECK(batch_size == batch_size_cache,
                    "q and cache batch size must match when cache_batch_idx is not provided");
    }

    at::Tensor k;
    at::Tensor v;
    const bool has_k = !k_obj.is_none();
    const bool has_v = !v_obj.is_none();
    TORCH_CHECK(has_k == has_v, "k and v must either both be provided or both be None");
    int seqlen_new = 0;
    if (has_k)
    {
        k = k_obj.cast<at::Tensor>();
        v = v_obj.cast<at::Tensor>();
        TORCH_CHECK(k.is_cuda() && v.is_cuda(), "k and v must be CUDA tensors");
        TORCH_CHECK(k.scalar_type() == torch::kFloat16 && v.scalar_type() == torch::kFloat16,
                    "k and v must be float16 tensors");
        TORCH_CHECK(k.dim() == 4 && v.dim() == 4, "k and v must be rank-4 tensors");
        TORCH_CHECK(k.size(0) == batch_size && v.size(0) == batch_size, "k/v batch size must match q");
        TORCH_CHECK(k.size(1) == v.size(1), "k and v seqlen must match");
        TORCH_CHECK(k.size(2) == num_heads_k && v.size(2) == num_heads_k,
                    "k/v num_heads must match the cache num_heads");
        TORCH_CHECK(k.size(3) == head_size && v.size(3) == head_size, "k/v head_dim must match q");
        TORCH_CHECK(k.stride(-1) == 1 && v.stride(-1) == 1, "k and v must have contiguous last dimension");
        seqlen_new = k.size(1);
    }

    at::Tensor cache_seqlens_end = has_k ? cache_seqlens + seqlen_new : cache_seqlens;
    TORCH_CHECK(cache_seqlens_end.max().item<int>() <= seqlen_cache,
                "cache capacity is insufficient for the requested append length");

    at::Tensor cache_batch_idx_cpu;
    int *cache_batch_ptr = nullptr;
    if (has_cache_batch_idx)
    {
        cache_batch_idx_cpu = cache_batch_idx.to(torch::TensorOptions().device(torch::kCPU));
        cache_batch_ptr = cache_batch_idx_cpu.data_ptr<int>();
        for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx)
        {
            TORCH_CHECK(cache_batch_ptr[batch_idx] >= 0 && cache_batch_ptr[batch_idx] < batch_size_cache,
                        "cache_batch_idx entries must be within cache batch bounds");
        }
    }

    if (has_k && !has_block_table)
    {
        at::Tensor cache_seqlens_cpu = cache_seqlens.to(torch::TensorOptions().device(torch::kCPU));
        auto cache_ptr = cache_seqlens_cpu.data_ptr<int>();
        for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx)
        {
            const int start = cache_ptr[batch_idx];
            const int cache_row = has_cache_batch_idx ? cache_batch_ptr[batch_idx] : batch_idx;
            k_cache[cache_row].narrow(0, start, seqlen_new).copy_(k[batch_idx]);
            v_cache[cache_row].narrow(0, start, seqlen_new).copy_(v[batch_idx]);
        }
    }
    else if (has_k)
    {
        run_kvcache_paged_append(k_cache, v_cache, k, v, cache_seqlens, block_table);
    }

    // Use the native KV-cache path when:
    //   (a) paged layout (block_table present) — always native,
    //   (b) caller requested explicit splits > 1,
    //   (c) decode (seqlen_q == 1) — native runs choose_num_splits() and
    //       dispatches the decode-specialised kernel; the upstream tiled FA
    //       kernel is not efficient for single-token queries.
    // For contiguous chunk (seqlen_q > 1, no explicit splits), the upstream
    // tiled FlashAttention kernel is faster and remains the default.
    const bool use_native_kvcache = has_block_table || num_splits > 1 || seqlen_q == 1;
    if (use_native_kvcache)
    {
        at::Tensor empty;
        return run_mha_fwd_kvcache_native(
            q,
            k_cache,
            v_cache,
            cache_seqlens_end,
            has_cache_batch_idx ? cache_batch_idx : empty,
            has_cache_batch_idx,
            has_block_table ? block_table : empty,
            has_block_table,
            softmax_scale,
            is_causal,
            num_splits);
    }

    at::Tensor k_cache_used = has_cache_batch_idx
                                  ? k_cache.index_select(0, cache_batch_idx.to(torch::kLong))
                                  : k_cache;
    at::Tensor v_cache_used = has_cache_batch_idx
                                  ? v_cache.index_select(0, cache_batch_idx.to(torch::kLong))
                                  : v_cache;

    at::Tensor out = torch::zeros_like(q);
    at::Tensor l = torch::zeros({batch_size, num_heads, seqlen_q}, q.options().dtype(torch::kFloat32));

    Flash_fwd_params params;
    set_params_fprop(
        params,
        batch_size,
        seqlen_q,
        seqlen_cache,
        num_heads,
        num_heads_k,
        head_size,
        q, k_cache_used, v_cache_used, out, l,
        nullptr,
        nullptr,
        cache_seqlens_end.data_ptr(),
        softmax_scale,
        is_causal);

    run_mha_fwd(params);

    return {out, l};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("fwd", &mha_fwd, "Forward pass");
    m.def("bwd", &mha_bwd, "Backward pass");
    m.def("varlen_fwd", &mha_varlen_fwd, "Varlen forward pass");
    m.def("varlen_bwd", &mha_varlen_bwd, "Varlen backward pass");
    m.def("fwd_kvcache", &mha_fwd_kvcache, "KV-cache forward pass");
}
