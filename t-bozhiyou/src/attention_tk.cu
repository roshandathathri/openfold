// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <stdio.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include "kernel_forward.h"
#include "gemm_kernel_utils.h"
#include "transform/bias_broadcast.h"
#include "kernel_forward.cuh"

template <typename arch,
          typename scalar_t,
          typename torch_scalar_t,
          template <typename, typename, typename>
          class Broadcast1_,
          template <typename, typename, typename>
          class Broadcast2_>
typename std::enable_if<!CheckArch<arch, scalar_t>::value>::type attention_impl_template(
    torch::Tensor& q,
    torch::Tensor& k,
    torch::Tensor& v,
    torch::Tensor& bias1,
    torch::Tensor& bias2,
    torch::Tensor& o,
    float* lse_ptr)
{
    EVOFORMER_CHECK(false, "Unsupported GPU and data type combination")
}

template <typename arch,
          typename scalar_t,
          typename torch_scalar_t,
          template <typename, typename, typename>
          class Broadcast1_,
          template <typename, typename, typename>
          class Broadcast2_>
typename std::enable_if<CheckArch<arch, scalar_t>::value>::type attention_impl_template(
    torch::Tensor& q,
    torch::Tensor& k,
    torch::Tensor& v,
    torch::Tensor& bias1,
    torch::Tensor& bias2,
    torch::Tensor& o,
    float* lse_ptr)
{ 
    // Attention definition goes here, replaced with BroadcastType1 and
    // BroadcastType2
    using Attention = TKAttentionKernel<scalar_t, /* scalar_t */
                                      arch,     /* ArchTag */
                                      true,     /* Memory is aligned */
                                      64,
                                      64,
                                      true,
                                      true, /* Supports bias */
                                      Broadcast1_,
                                      Broadcast2_>;

    // static_assert(!Attention::kNeedsOutputAccumulatorBuffer,
    //               "This test does not support output accumulator buffer");
    int head_size = q.size(-1);
    int head_number = q.size(-2);
    int seq_length = q.size(-3);
    auto q_view = q.view({-1, seq_length, head_number, head_size});
    auto k_view = k.view({-1, seq_length, head_number, head_size});
    auto v_view = v.view({-1, seq_length, head_number, head_size});
    auto o_view = o.view({-1, seq_length, head_number, head_size});
    int batch_size = q_view.size(0);
    auto q_ptr = reinterpret_cast<scalar_t*>(q.data_ptr<torch_scalar_t>());
    auto k_ptr = reinterpret_cast<scalar_t*>(k.data_ptr<torch_scalar_t>());
    auto v_ptr = reinterpret_cast<scalar_t*>(v.data_ptr<torch_scalar_t>());
    auto o_ptr = reinterpret_cast<scalar_t*>(o.data_ptr<torch_scalar_t>());

    auto bias1_ptr = reinterpret_cast<scalar_t*>(bias1.data_ptr<torch_scalar_t>());
    auto bias2_ptr = reinterpret_cast<scalar_t*>(bias2.data_ptr<torch_scalar_t>());

    typename Attention::Params p;
    {  // set parameters
        p.query_ptr = q_ptr;
        p.key_ptr = k_ptr;
        p.value_ptr = v_ptr;
        p.output_ptr = o_ptr;
        p.scale = 1.0f / sqrt(float(head_size));

        p.bias1_ptr = bias1_ptr;
        p.bias2_ptr = bias2_ptr;


        p.num_heads = head_number;
        p.num_batches = batch_size;
        p.num_queries = seq_length;
        p.num_keys = seq_length;

        // All tensors are in BMHK shapes
        p.q_strideH = q_view.stride(-2);
        p.k_strideH = k_view.stride(-2);
        p.v_strideH = v_view.stride(-2);
        p.o_strideH = o_view.stride(-2);
        p.q_strideM = q_view.stride(-3);
        p.k_strideM = k_view.stride(-3);
        p.v_strideM = v_view.stride(-3);
        p.o_strideM = o_view.stride(-3);
        p.q_strideB = q_view.stride(-4);
        p.k_strideB = k_view.stride(-4);
        p.v_strideB = v_view.stride(-4);
        p.o_strideB = o_view.stride(-4);

        p.b1_strideB = bias1.stride(-4);

        p.N = q.size(1);
        p.b2_strideB = bias2.stride(0);
        p.b2_strideH = bias2.stride(2);
        p.b2_strideQ = bias2.stride(-2);
    }

    constexpr auto kernel_fn = attention_kernel_batched_impl<Attention>;

    // {
    //     int mem_size;
    //     cudaDeviceGetAttribute(&mem_size, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
    //     printf("Max shared memory  optin: %lu\n", mem_size);
    //     cudaDeviceGetAttribute(&mem_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    //     printf("Max shared memory: %lu\n", mem_size);
    //     cudaDeviceGetAttribute(&mem_size, cudaDevAttrMaxThreadsPerBlock, 0);
    //     printf("Max threads per block: %d\n", mem_size);
    //     cudaDeviceGetAttribute(&mem_size, cudaDevAttrMaxBlockDimX, 0);
    //     printf("Max block dim x: %d\n", mem_size);
    //     cudaDeviceGetAttribute(&mem_size, cudaDevAttrMaxRegistersPerBlock, 0);
    //     printf("Max registers per block: %d\n", mem_size);
    // }
    // printf("Initial CUDA error: %s\n", cudaGetErrorString(cudaGetLastError()));
    unsigned long smem_bytes = 16*1024;
    cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    // printf("cudaFuncSetAttribute CUDA error: %s\n", cudaGetErrorString(cudaGetLastError()));

    // printf("Batch size: %d, seq length: %d, head number: %d, head size: %d\n", batch_size, seq_length, head_number, head_size);
    // printf("Using dtype %s\n", typeid(scalar_t).name());

    // cudaDeviceSynchronize();
    // printf("<<<(%d, %d, %d), (%d, %d, %d), %d>>>\n",
    //     p.getBlocksGrid().x, p.getBlocksGrid().y, p.getBlocksGrid().z,
    //     p.getThreadsGrid().x, p.getThreadsGrid().y, p.getThreadsGrid().z,
    //     smem_bytes);
    kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), smem_bytes>>>(p);
    // cudaDeviceSynchronize();
    // printf("CUDA error: %s\n", cudaGetErrorString(cudaGetLastError()));
    // cudaDeviceSynchronize();
}

#define CODE(scalar_t, torch_scalar_t)                                                          \
    do {                                                                                        \
        if (bias1.size(0) == 0 && bias2.size(0) == 0) {                                         \
            attention_impl_template<ArchTag,                                                    \
                                    scalar_t,                                                   \
                                    torch_scalar_t,                                             \
                                    BroadcastNoLoad,                                            \
                                    BroadcastNoLoad>(q, k, v, bias1, bias2, o, lse_ptr);        \
        } else if (bias1.size(0) == 0) {                                                        \
            attention_impl_template<ArchTag,                                                    \
                                    scalar_t,                                                   \
                                    torch_scalar_t,                                             \
                                    BroadcastNoLoad,                                            \
                                    BroadcastB>(q, k, v, bias1, bias2, o, lse_ptr);             \
        } else if (bias2.size(0) == 0) {                                                        \
            attention_impl_template<ArchTag,                                                    \
                                    scalar_t,                                                   \
                                    torch_scalar_t,                                             \
                                    BroadcastA,                                                 \
                                    BroadcastNoLoad>(q, k, v, bias1, bias2, o, lse_ptr);        \
        } else {                                                                                \
            attention_impl_template<ArchTag, scalar_t, torch_scalar_t, BroadcastA, BroadcastB>( \
                q, k, v, bias1, bias2, o, lse_ptr);                                             \
        }                                                                                       \
    } while (0)

// Function to select and call the correct template based on biases sizes
void attention_impl(torch::Tensor& q,
                    torch::Tensor& k,
                    torch::Tensor& v,
                    torch::Tensor& bias1,
                    torch::Tensor& bias2,
                    torch::Tensor& o,
                    torch::Tensor& lse)
{
    auto lse_ptr = lse.size(0) == 0 ? nullptr : reinterpret_cast<float*>(lse.data_ptr<float>());
    cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
    DISPATCH_ARCHTAG(prop->major * 10 + prop->minor,
                     DISPATCH_TYPES(q, { CODE(kittens::bf16, torch_scalar_t); }));
}
