#include "src/kittens.cuh"

namespace kittens {
template<int _height, int _width, ducks::st_layout::all layout=ducks::st_layout::swizzle>
using st_hf = st<half, _height, _width, layout>; // prelim tests indicate this is fastest default
template<ducks::st_layout::all layout=ducks::st_layout::swizzle>
using st_hf_1x4 = st_hf<1, 4, layout>;
}

using namespace kittens;

#define TK_BIAS1_USE_VEC 0

template <
    // The datatype of Q/K/V
    typename scalar_t_,
    // Architecture we are targeting (eg `cutlass::arch::Sm80`)
    typename ArchTag,
    // If Q/K/V are correctly aligned in memory and we can run a fast kernel
    bool isAligned_,
    int kQueriesPerBlock,
    int kKeysPerBlock_,
    bool kSingleValueIteration_,  // = `value.shape[-1] <= kKeysPerBlock`
    // This is quite slower on V100 for some reason
    // Set to false if you know at compile-time you will never need dropout
    bool kSupportsBias_ = false,
    template <typename, typename, typename> class Broadcast1_ = BroadcastNoLoad,
    template <typename, typename, typename> class Broadcast2_ = BroadcastNoLoad>
struct TKAttentionKernel {
    using scalar_t = scalar_t_;
    using output_t = scalar_t;
    static constexpr int kKeysPerBlock = kKeysPerBlock_;

    static_assert(kQueriesPerBlock % 32 == 0, "");
    static_assert(kKeysPerBlock % 32 == 0, "");
    static constexpr int kNumWarpsPerBlock = kQueriesPerBlock * kKeysPerBlock / (32 * 32);
    static constexpr int kWarpSize = 32;

    // Launch bounds
    static constexpr int kNumThreads = kWarpSize * kNumWarpsPerBlock;
    static constexpr int kMinBlocksPerSm = getWarpsPerSm<scalar_t, ArchTag>() / kNumWarpsPerBlock;

    struct Params {
        // Input tensors
        scalar_t* query_ptr;  // [num_queries, num_heads, head_dim]
        scalar_t* key_ptr;    // [num_keys, num_heads, head_dim]
        scalar_t* value_ptr;  // [num_keys, num_heads, head_dim_value]
        // Parameters for biases
        scalar_t* bias1_ptr = nullptr;
        scalar_t* bias2_ptr = nullptr;
        // Output tensors
        output_t* output_ptr;              // [num_queries, num_heads, head_dim_value]

        // Scale
        float scale;

        // Dimensions for kernel launch configuration
        int32_t num_queries;
        int32_t num_keys;
        int32_t num_batches;
        int32_t num_heads;
        int32_t N;

        // Dimensions/strides
        int32_t head_dim;
        int32_t head_dim_value;

        int32_t q_strideM;
        int32_t k_strideM;
        int32_t v_strideM;
        int32_t o_strideM;

        // Everything below is only used in `advance_to_block`
        // and shouldn't use registers
        int32_t q_strideH;
        int32_t k_strideH;
        int32_t v_strideH;
        int32_t o_strideH;

        int64_t q_strideB;
        int64_t k_strideB;
        int64_t v_strideB;
        int32_t o_strideB;

        int32_t b1_strideB;

        int32_t b2_strideB;
        int32_t b2_strideH;
        int32_t b2_strideQ;


        __host__ dim3 getBlocksGrid() const
        {
            return dim3(ceil_div(num_queries, (int32_t)kQueriesPerBlock), num_heads, num_batches);
        }

        __host__ dim3 getThreadsGrid() const { return dim3(32, kNumWarpsPerBlock, 1); }

        // Moves pointers to what we should process
        // Returns "false" if there is no work to do
        __device__ bool advance_to_block()
        {
            auto query_block_id = blockIdx.x * kQueriesPerBlock;
            auto batch_id = blockIdx.z;
            auto head_id = blockIdx.y;

            // auto lse_dim = ceil_div((int32_t)num_queries, kAlignLSE) * kAlignLSE;

            query_ptr +=
                (query_block_id * q_strideM) + (head_id * q_strideH) + (batch_id * q_strideB);
            key_ptr += (head_id * k_strideH) + (batch_id * k_strideB);
            value_ptr += (head_id * v_strideH) + (batch_id * v_strideB);
            output_ptr +=
                (query_block_id * q_strideM) + (head_id * o_strideH) + (batch_id * o_strideB);

            bias1_ptr += (batch_id * b1_strideB);
            bias2_ptr += (query_block_id * b2_strideQ) + (head_id * b2_strideH) + (int(batch_id / N) * b2_strideB);
            // if (output_accum_ptr != nullptr) {
            //     output_accum_ptr += int64_t(batch_id * num_queries) * (head_dim_value * num_heads);
            // }

            // if (output_accum_ptr != nullptr) {
            //     output_accum_ptr += int64_t(q_start + query_start) * (head_dim_value * num_heads) +
            //                         head_id * head_dim_value;
            // } else {
            //     // Accumulate directly in the destination buffer (eg for f32)
            //     output_accum_ptr = (accum_t*)output_ptr;
            // }

            // if (logsumexp_ptr != nullptr) {
            //     // lse[batch_id, head_id, query_start]
            //     logsumexp_ptr += batch_id * lse_dim * num_heads + head_id * lse_dim + query_start;
            // }

            // num_queries -= query_block_id; // num_queries is now the number of queries left to process
            // // If num_queries == 1, and there is only one key head we're wasting
            // // 15/16th of tensor core compute In that case :
            // //  - we only launch kernels for head_id % kQueriesPerBlock == 0
            // //  - we iterate over heads instead of queries (strideM = strideH)
            // if (num_queries == 1) {
            //     if (head_id % kQueriesPerBlock != 0) return false;
            //     q_strideM = q_strideH;
            //     num_queries = num_heads;
            //     num_heads = 1;  // unused but here for intent
            //     o_strideM = head_dim_value;
            // }

            return true;
        }
    };

    __device__ static void attention_kernel(Params& p){
    const bf16* __restrict__ _q = reinterpret_cast<const bf16*>(p.query_ptr);
    const bf16* __restrict__ _k = reinterpret_cast<const bf16*>(p.key_ptr);
    const bf16* __restrict__ _v = reinterpret_cast<const bf16*>(p.value_ptr);
    const bf16* __restrict__ _b1 = reinterpret_cast<const bf16*>(p.bias1_ptr);
    const bf16* __restrict__ _b2 = reinterpret_cast<const bf16*>(p.bias2_ptr);
    bf16* _o = reinterpret_cast<bf16*>(p.output_ptr);

    // {if (blockIdx.x == 2 && blockIdx.y == 3 && blockIdx.z == 5
    // && threadIdx.x == 0 && threadIdx.y == 1 && threadIdx.z == 0) {
    //     printf("block dim: (%d, %d, %d)\n", gridDim.x, gridDim.y, gridDim.z);
    //     printf("thread dim: (%d, %d, %d)\n", blockDim.x, blockDim.y, blockDim.z);
    //     printf("n: %d\n", kQueriesPerBlock);
    //     printf("q[0]: %f\n", (float)_q[0]);
    //     printf("k[0]: %f\n", (float)_k[0]);
    //     printf("v[0]: %f\n", (float)_v[0]);
    //     printf("o[0]: %f\n", (float)_o[0]);
    //     printf("b2[0]: %f\n", (float)_b2[0]);
    //     printf("N: %d\n", p.N);
    //     printf("b2_strideQ: %d\n", p.b2_strideQ);
    // }}
    auto warpid        = threadIdx.y;

    extern __shared__ alignment_dummy __shm[]; // this is the CUDA shared memory
    shared_allocator al((int*)&__shm[0]);
    
    // K and V live in shared memory -- this is about all that will fit.
    st_bf_1x4<ducks::st_layout::swizzle> (&k_smem)[kNumWarpsPerBlock] = al.allocate<st_bf_1x4<ducks::st_layout::swizzle>, kNumWarpsPerBlock>();
    st_bf_1x4<ducks::st_layout::swizzle> (&v_smem)[kNumWarpsPerBlock] = al.allocate<st_bf_1x4<ducks::st_layout::swizzle>, kNumWarpsPerBlock>();
    //st_bf_1x1<ducks::st_layout::swizzle>::row_vec (&b1_smem)[kNumWarpsPerBlock] = al.allocate<st_bf_1x1<ducks::st_layout::swizzle>::row_vec, kNumWarpsPerBlock>();

    // Initialize all of the register tiles.
    rt_bf_1x4<> q_reg, k_reg, v_reg; // v_reg need to be swapped into col_l
#if TK_BIAS1_USE_VEC
    rt_fl_1x1<>::row_vec b1_vec;
#else
    rt_fl_1x1<> b1_reg;
#endif
    rt_fl_1x1<> b2_reg;
    rt_fl_1x1<> att_block;
    rt_bf_1x1<> att_block_mma;
    rt_fl_1x4<> o_reg;
    rt_fl_1x1<>::col_vec max_vec_last, max_vec; // these are column vectors for the attention block
    rt_fl_1x1<>::col_vec norm_vec_last, norm_vec; // these are column vectors for the attention block

    int qo_blocks = kQueriesPerBlock / (q_reg.rows*kNumWarpsPerBlock), kv_blocks = p.num_keys / (q_reg.rows*kNumWarpsPerBlock);

    // {if (blockIdx.x == 2 && blockIdx.y == 2 && blockIdx.z == 2
    // && threadIdx.x == 0 && threadIdx.y == 1 && threadIdx.z == 0)
    //  printf("qo_blocks: %d, kv_blocks: %d\n", qo_blocks, kv_blocks);}

    for(auto q_blk = 0; q_blk < qo_blocks; q_blk++) {

        // each warp loads its own Q tile of 16x64, and then multiplies by 1/sqrt(d)
        load(q_reg, _q + (q_blk*kNumWarpsPerBlock + warpid)*(q_reg.rows*p.q_strideM), p.q_strideM);
        mul(q_reg, q_reg, __float2bfloat16(p.scale)); // temperature adjustment

        // zero flash attention L, M, and O registers.
        neg_infty(max_vec); // zero registers for the Q chunk
        zero(norm_vec);
        zero(o_reg);

        // iterate over k, v for these q's that have been loaded
        for(auto kv_idx = 0; kv_idx < kv_blocks; kv_idx++) {

            // each warp loads its own chunk of k, v into shared memory
            load(v_smem[warpid], _v + (kv_idx*kNumWarpsPerBlock + warpid)*(v_reg.rows*p.v_strideM), p.v_strideM);
            load(k_smem[warpid], _k + (kv_idx*kNumWarpsPerBlock + warpid)*(k_reg.rows*p.k_strideM), p.k_strideM);
            // load(b1_smem[warpid], _b1 + (kv_idx*kNumWarpsPerBlock + warpid)*b1_reg.outer_dim);
            __syncthreads(); // we need to make sure all memory is loaded before we can begin the compute phase

            // now each warp goes through all of the subtiles, loads them, and then does the flash attention internal alg.
            for(int subtile = 0; subtile < kNumWarpsPerBlock; subtile++) {

                load(k_reg, k_smem[subtile]); // load k from shared into registers

                zero(att_block); // zero 16x16 attention tile
                mma_ABt(att_block, q_reg, k_reg, att_block); // Q@K.T

#if TK_BIAS1_USE_VEC
                load(b1_vec, _b1 + (kv_idx*kNumWarpsPerBlock + subtile)*att_block.cols);
                add_col(att_block, att_block, b1_vec); // add the bias1
#else
                load(b1_reg, _b1 + (kv_idx*kNumWarpsPerBlock + subtile)*b1_reg.cols, 0);
                add(att_block, att_block, b1_reg); // add the bias1
#endif
                load(b2_reg, _b2 + (q_blk*kNumWarpsPerBlock + warpid)*(b2_reg.rows*p.b2_strideQ) + (kv_idx*kNumWarpsPerBlock + subtile)*b2_reg.cols, p.b2_strideQ);
                add(att_block, att_block, b2_reg); // add the bias2

                // 2. Load b1 into reg as 16x16 tile and add
                // 3. Load b1 into reg as 1x16 (row_vec) and modify add to support row_vec of the same type
                // 4. Load b1 into shared_mem as 1x256 or 16x16 (row_vec) before q loop (use all warps/workers) and then load from shared_mem to reg inside subtile loop

                copy(norm_vec_last, norm_vec);
                copy(max_vec_last,  max_vec);

                row_max(max_vec, att_block, max_vec); // accumulate onto the max_vec
                sub_row(att_block, att_block, max_vec); // subtract max from attention -- now all <=0
                exp(att_block, att_block); // exponentiate the block in-place.

                sub(max_vec_last, max_vec_last, max_vec); // subtract new max from old max to find the new normalization.
                exp(max_vec_last, max_vec_last); // exponentiate this vector -- this is what we need to normalize by.
                mul(norm_vec, norm_vec, max_vec_last); // and the norm vec is now normalized.

                row_sum(norm_vec, att_block, norm_vec); // accumulate the new attention block onto the now-rescaled norm_vec
                div_row(att_block, att_block, norm_vec); // now the attention block is correctly normalized

                mul(norm_vec_last, norm_vec_last, max_vec_last); // normalize the previous norm vec according to the new max
                div(norm_vec_last, norm_vec_last, norm_vec); // normalize the previous norm vec according to the new norm

                copy(att_block_mma, att_block); // convert to bf16 for mma_AB

                load(v_reg, v_smem[subtile]); // load v from shared into registers.
                rt_bf_1x4<ducks::rt_layout::col> &v_reg_col = swap_layout_inplace(v_reg); // this is a reference and the call has invalidated v_reg

                mul_row(o_reg, o_reg, norm_vec_last); // normalize o_reg in advance of mma_AB'ing onto it
                mma_AB(o_reg, att_block_mma, v_reg_col, o_reg); // mfma onto o_reg with the local attention@V matmul.
            }
            __syncthreads(); // we need to make sure all warps are done before we can start loading the next kv chunk
        }

        store(_o + (q_blk*kNumWarpsPerBlock + warpid)*(o_reg.rows*p.o_strideM), o_reg, p.o_strideM); // write out o. compiler has an issue with register usage if d is made constexpr q_reg.rows :/
    }
}

};
