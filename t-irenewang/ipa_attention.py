"""
FlashIPA implementation (forward)
===============

This is a Triton implementation of the Invariant Point Attention in Alphafold's structural module. 

Extra Credits:
- Original Triton implementation of the Flash Attention paper 
    Credits: OpenAI kernel team
- Original flash attention paper (https://arxiv.org/abs/2205.14135)
- Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)

"""

import pytest
import torch

import triton
import triton.language as tl
import os
from typing import Tuple, List, Callable, Any, Dict, Sequence, Optional

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


@triton.jit
def _attn_fwd_inner(acc_reg, acc_pv_0,acc_pv_1,acc_pts_0, acc_pts_1, acc_pts_2, l_i, m_i, q,  #
                    K_block_ptr, V_block_ptr, #
                    start_m, qk_scale,  #
                    q0_pts,q1_pts, q2_pts, K_pts_block_0_ptr,K_pts_block_1_ptr,K_pts_block_2_ptr, #
                    V_pts_block_0_ptr, V_pts_block_1_ptr, V_pts_block_2_ptr,  #
                    pv_block_ptr_0,pv_block_ptr_1, pb_block_ptr, #
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    N_CTX: tl.constexpr, fp8_v: tl.constexpr):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, N_CTX
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))

    K_pts_block_0_ptr = tl.advance(K_pts_block_0_ptr, (lo, 0 , 0))
    K_pts_block_1_ptr = tl.advance(K_pts_block_1_ptr, (lo, 0 , 0))
    K_pts_block_2_ptr = tl.advance(K_pts_block_2_ptr, (lo, 0 , 0))

    V_pts_block_0_ptr = tl.advance(V_pts_block_0_ptr, (lo, 0 , 0))
    V_pts_block_1_ptr = tl.advance(V_pts_block_1_ptr, (lo, 0 , 0))
    V_pts_block_2_ptr = tl.advance(V_pts_block_2_ptr, (lo, 0 , 0))

    # pv_block_ptr = tl.advance(pv_block_ptr, (lo, 0 , 0)) 
    pv_block_ptr_0 = tl.advance(pv_block_ptr_0, (lo, 0 , 0))
    pv_block_ptr_1 = tl.advance(pv_block_ptr_1, (lo, 0 , 0))

    pb_block_ptr = tl.advance(pb_block_ptr, (lo, 0)) 

    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(K_block_ptr)

        # add pair bias to qk 
        pb = tl.load(pb_block_ptr)
        qk = tl.dot(q, k) + pb

        # Compute square distance 
        k0_pts = tl.load(K_pts_block_0_ptr)
        k1_pts = tl.load(K_pts_block_1_ptr)
        k2_pts = tl.load(K_pts_block_2_ptr)

        temp0 =(q0_pts.expand_dims(1) - k0_pts.expand_dims(0))
        temp0 = temp0 * temp0
        temp1 =(q1_pts.expand_dims(1) - k1_pts.expand_dims(0))
        temp1 = temp1 * temp1
        temp2 =(q2_pts.expand_dims(1) - k2_pts.expand_dims(0))
        temp2 = temp2 * temp2
        
        sq_dist = tl.sum((temp0 + temp1 + temp2), axis=-1) 
        sq_reduced = tl.sum(sq_dist, axis=-1) 
        # TODO: apply scaling
        qk = qk + sq_reduced

        # -- softmax ----
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        alpha = alpha.to(tl.float32)
        acc_reg = acc_reg * alpha[:, None]
        acc_pv_0 = acc_pv_0 * alpha[:, None]
        acc_pv_1 = acc_pv_1 * alpha[:, None]
        acc_pts_0 = acc_pts_0 * alpha[:, None, None]
        acc_pts_1 = acc_pts_1 * alpha[:, None, None]
        acc_pts_2 = acc_pts_2 * alpha[:, None, None]
        
        # update acc
        v = tl.load(V_block_ptr)
        if fp8_v:
            p = p.to(tl.float8e5)
        else:
            p = p.to(tl.float32)

        acc_reg = tl.dot(p, v, acc_reg, out_dtype=tl.float32)
        
        # apply sq_dist attn
        v_pts_0 = tl.load(V_pts_block_0_ptr)
        v_pts_1 = tl.load(V_pts_block_1_ptr)
        v_pts_2 = tl.load(V_pts_block_2_ptr)
        
        acc_pts_0 = acc_pts_0 + tl.sum(v_pts_0[None, :, :, :] * p.expand_dims(-1).expand_dims(-1), axis=-3)
        acc_pts_1 = acc_pts_1 + tl.sum(v_pts_1[None, :, :, :] * p.expand_dims(-1).expand_dims(-1), axis=-3)
        acc_pts_2 = acc_pts_2 + tl.sum(v_pts_2[None, :, :, :] * p.expand_dims(-1).expand_dims(-1), axis=-3)

        # apply pair value attn 
        pv0= tl.load(pv_block_ptr_0)
        acc_pv_0 = acc_pv_0 + tl.sum(p[:,:,None] * pv0, axis=-2)

        pv1= tl.load(pv_block_ptr_1)
        acc_pv_1 = acc_pv_1 + tl.sum(p[:,:,None] * pv1, axis=-2)
        
        # update m_i and l_i
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))

        K_pts_block_0_ptr = tl.advance(K_pts_block_0_ptr, (BLOCK_N, 0, 0))
        K_pts_block_1_ptr = tl.advance(K_pts_block_1_ptr, (BLOCK_N, 0, 0))
        K_pts_block_2_ptr = tl.advance(K_pts_block_2_ptr, (BLOCK_N, 0, 0))

        V_pts_block_0_ptr = tl.advance(V_pts_block_0_ptr, (BLOCK_N, 0, 0))
        V_pts_block_1_ptr = tl.advance(V_pts_block_1_ptr, (BLOCK_N, 0, 0))
        V_pts_block_2_ptr = tl.advance(V_pts_block_2_ptr, (BLOCK_N, 0, 0))

        pv_block_ptr_0 = tl.advance(pv_block_ptr_0, (0, BLOCK_N, 0))
        pv_block_ptr_1 = tl.advance(pv_block_ptr_1, (0, BLOCK_N, 0))
    
        pb_block_ptr = tl.advance(pb_block_ptr, (0, BLOCK_N)) 

    return acc_reg, acc_pv_0,acc_pv_1, acc_pts_0, acc_pts_1, acc_pts_2, l_i, m_i


# We don't run auto-tuning every time to keep the tutorial fast. Keeping
# the code below and commenting out the equivalent parameters is convenient for
# re-tuning.
configs = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w) \
    for BM in [16, 32, 64]\
    for BN in [16, 32, 64]\
    for s in ([1] if is_hip() else [3, 4, 7])\
    for w in [4, 8]\
]

def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    if BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8:
        return False
    return True


@triton.autotune(list(filter(keep, configs)), key=["N_CTX","HEAD_DIM", "C_Z", "P"]) 
@triton.jit
def _attn_fwd(Q, K, V, sm_scale, M, Out_reg, Out_pv, Out_pts,  #
              Q_pts, #
              K_pts, V_pts,  #
              PV, PB, #
              stride_qz, stride_qh, stride_qm, stride_qk,  #
              stride_kz, stride_kh, stride_kn, stride_kk,  #
              stride_vz, stride_vh, stride_vk, stride_vn,  #
              stride_oz, stride_oh, stride_om, stride_on,  #
              stride_qpts_z, stride_qpts_h, stride_qpts_m, stride_qpts_k, stride_qpts_c,  #
              stride_opvz, stride_opvh, stride_opvm, stride_opvn,
              stride_pb_z, stride_pb_h, stride_pb_m, stride_pb_n,  #
              stride_pv_z, stride_pv_m, stride_pv_n, stride_pv_c, #
              Z, H, N_CTX,  #
              P: tl.constexpr,# IPA
              C_Z: tl.constexpr,  # IPA
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              STAGE: tl.constexpr  #
              ):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H # batch offset
    off_h = off_hz % H # head offset

    # we need different offsets for each tensor of different operations because they 
    # all have differet dimensions

    # offset to point to for vanilla QKV and Output blocks 
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

    # offset for pair bias to correct head and batch
    pb_offset = off_z.to(tl.int64) * stride_pb_z  + off_h.to(tl.int64) * stride_pb_h
    # offset for pair bias to correct batch (no head dimension)
    pv_offset = off_z.to(tl.int64) * stride_pv_z

    # offset for the output of the pair value attention 
    opv_offset = off_z.to(tl.int64) * stride_opvz + off_h.to(tl.int64) * stride_opvh

    # offset for pts QKV and output (square distance)
    pts_offset = off_z.to(tl.int64) * stride_qpts_z + off_h.to(tl.int64) * stride_qpts_h

    cz_int: tl.constexpr = int(C_Z/2)
    # initialize block pointers for vanilla Attention
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    v_order: tl.constexpr = (0, 1) if V.dtype.element_ty == tl.float8e5 else (1, 0)
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=v_order,
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(HEAD_DIM, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )

    # initialize block pointers for pts Attention (Squared distance)
    # QKV tensors are all split in to 3 (one per column)
    Q_pts_block_0_ptr = tl.make_block_ptr(
        base=Q_pts + pts_offset,
        shape=(N_CTX, P, 1),
        strides=(stride_qpts_m, stride_qpts_k, stride_qpts_c),
        offsets=(start_m * BLOCK_M, 0, 0),
        block_shape=(BLOCK_M, P, 1),
        order=(2, 1, 0),
    )

    Q_pts_block_1_ptr = tl.make_block_ptr(
        base=Q_pts + pts_offset + 1,
        shape=(N_CTX, P, 1),
        strides=(stride_qpts_m, stride_qpts_k, stride_qpts_c),
        offsets=(start_m * BLOCK_M, 0, 0),
        block_shape=(BLOCK_M, P, 1),
        order=(2, 1, 0),
    )

    Q_pts_block_2_ptr = tl.make_block_ptr(
        base=Q_pts + pts_offset + 2,
        shape=(N_CTX, P, 1),
        strides=(stride_qpts_m, stride_qpts_k, stride_qpts_c),
        offsets=(start_m * BLOCK_M, 0, 0),
        block_shape=(BLOCK_M, P, 1),
        order=(2, 1, 0),
    )
    V_pts_block_0_ptr = tl.make_block_ptr(
        base=V_pts + pts_offset,
        shape=(N_CTX, P, 1),
        strides=(stride_qpts_m, stride_qpts_k, stride_qpts_c),
        offsets=(0, 0, 0),
        block_shape=(BLOCK_N, P, 1),
        order=(2, 1, 0),
    )
    V_pts_block_1_ptr = tl.make_block_ptr(
        base=V_pts + pts_offset + 1,
        shape=(N_CTX, P, 1),
        strides=(stride_qpts_m, stride_qpts_k, stride_qpts_c),
        offsets=(0, 0, 0),
        block_shape=(BLOCK_N, P, 1),
        order=(2, 1, 0),
    )
    V_pts_block_2_ptr = tl.make_block_ptr(
        base=V_pts + pts_offset + 2,
        shape=(N_CTX, P, 1),
        strides=(stride_qpts_m, stride_qpts_k, stride_qpts_c),
        offsets=(0, 0, 0),
        block_shape=(BLOCK_N, P, 1),
        order=(2, 1, 0),
    )

    K_pts_block_0_ptr = tl.make_block_ptr(
        base=K_pts + pts_offset,
        shape=(N_CTX, P, 1),
        strides=(stride_qpts_m, stride_qpts_k, stride_qpts_c),
        offsets=(0, 0, 0),
        block_shape=(BLOCK_N, P, 1),
        order=(2, 1, 0),
    )


    K_pts_block_1_ptr = tl.make_block_ptr(
        base=K_pts + pts_offset+1 ,
        shape=(N_CTX, P, 1),
        strides=(stride_qpts_m, stride_qpts_k, stride_qpts_c),
        offsets=(0, 0, 0),
        block_shape=(BLOCK_N, P, 1),
        order=(2, 1, 0),
    )

    K_pts_block_2_ptr = tl.make_block_ptr(
        base=K_pts + pts_offset+2 ,
        shape=(N_CTX, P, 1),
        strides=(stride_qpts_m, stride_qpts_k, stride_qpts_c),
        offsets=(0, 0, 0),
        block_shape=(BLOCK_N, P, 1),
        order=(2, 1, 0),
    )

    
    # initialize block pointers for Pair Value Attention
    # PV tensors are all split in to 2 ()
    pv_block_ptr_0 = tl.make_block_ptr(
        base=PV + pv_offset,
        shape=(N_CTX, N_CTX, cz_int),
        strides=(stride_pv_m, stride_pv_n, stride_pv_c),
        offsets=(start_m * BLOCK_M, 0, 0),
        block_shape=(BLOCK_M, BLOCK_N, cz_int),
        order=(2,1,0),
    )

    pv_block_ptr_1 = tl.make_block_ptr(
        base=PV + pv_offset + (1* cz_int),
        shape=(N_CTX, N_CTX, cz_int),
        strides=(stride_pv_m, stride_pv_n, stride_pv_c),
        offsets=(start_m * BLOCK_M, 0, 0),
        block_shape=(BLOCK_M, BLOCK_N, cz_int),
        order=(2,1,0),
    )

    # initialize block pointers for Pair Bias 
    pb_block_ptr = tl.make_block_ptr(
        base=PB + pb_offset,
        shape=(N_CTX, N_CTX),
        strides=(stride_pb_m, stride_pb_n), 
        offsets=(start_m * BLOCK_M, 0), 
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1,0),
    )


    # initialize block pointers for output
    O_block_ptr = tl.make_block_ptr(
        base=Out_reg + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )

    O_pts_block0_ptr = tl.make_block_ptr(
        base=Out_pts + pts_offset,
        shape=(N_CTX, P, 1), 
        strides=(stride_qpts_m, stride_qpts_k, stride_qpts_c),
        offsets=(start_m * BLOCK_M, 0, 0),
        block_shape=(BLOCK_M, P, 1),
        order=(2,1,0),
    )
    O_pts_block1_ptr = tl.make_block_ptr(
        base=Out_pts + pts_offset + 1,
        shape=(N_CTX, P, 1), 
        strides=(stride_qpts_m, stride_qpts_k, stride_qpts_c),
        offsets=(start_m * BLOCK_M, 0, 0),
        block_shape=(BLOCK_M, P, 1),
        order=(2,1,0),
    )
    O_pts_block2_ptr = tl.make_block_ptr(
        base=Out_pts + pts_offset + 2,
        shape=(N_CTX, P, 1), 
        strides=(stride_qpts_m, stride_qpts_k, stride_qpts_c),
        offsets=(start_m * BLOCK_M, 0, 0),
        block_shape=(BLOCK_M, P, 1),
        order=(2,1,0),
    )

    O_pv_block0_ptr = tl.make_block_ptr(
        base=Out_pv + opv_offset,
        shape=(N_CTX, cz_int),
        strides=(stride_opvm, stride_opvn),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, cz_int),
        order=(1, 0),
    )

    O_pv_block1_ptr = tl.make_block_ptr(
        base=Out_pv + opv_offset+ (1* cz_int),
        shape=(N_CTX, cz_int),
        strides=(stride_opvm, stride_opvn),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, cz_int),
        order=(1, 0),
    )

    

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0

    # init accumulators for vanilla attn, pts_attn and pair value attn
    acc_reg = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    acc_pv_0 = tl.zeros([BLOCK_M, cz_int], dtype=tl.float32)
    acc_pv_1 = tl.zeros([BLOCK_M, cz_int], dtype=tl.float32)
    acc_pts_0 = tl.zeros([BLOCK_M, P, 1], dtype=tl.float32)
    acc_pts_1 = tl.zeros([BLOCK_M, P, 1], dtype=tl.float32)
    acc_pts_2 = tl.zeros([BLOCK_M, P, 1], dtype=tl.float32)

    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout

    # load vanilla q
    q = tl.load(Q_block_ptr)

    # load q_pts (squared distance)
    q0_pts = tl.load(Q_pts_block_0_ptr)
    q1_pts = tl.load(Q_pts_block_1_ptr)
    q2_pts = tl.load(Q_pts_block_2_ptr)

    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    

    if STAGE & 1:
        acc_reg, acc_pv_0,acc_pv_1, acc_pts_0, acc_pts_1, acc_pts_2, l_i, m_i = _attn_fwd_inner(acc_reg, acc_pv_0,acc_pv_1, acc_pts_0, acc_pts_1, acc_pts_2, l_i, m_i, q, K_block_ptr, V_block_ptr,  #
                                        start_m, qk_scale,  #
                                        q0_pts,q1_pts, q2_pts, K_pts_block_0_ptr,K_pts_block_1_ptr,K_pts_block_2_ptr,V_pts_block_0_ptr, V_pts_block_1_ptr, V_pts_block_2_ptr, #
                                        pv_block_ptr_0,pv_block_ptr_1, pb_block_ptr, #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        4 - STAGE, offs_m, offs_n, N_CTX, V.dtype.element_ty == tl.float8e5  #
                                        )
    # stage 2: on-band
    if STAGE & 2:
        # barrier makes it easier for compielr to schedule the
        # two loops independently
        acc_reg, acc_pv_0,acc_pv_1,acc_pts_0, acc_pts_1, acc_pts_2, l_i, m_i = _attn_fwd_inner(acc_reg, acc_pv_0,acc_pv_1, acc_pts_0, acc_pts_1, acc_pts_2, l_i, m_i, q, K_block_ptr, V_block_ptr,  #
                                        start_m, qk_scale,  #
                                        q0_pts,q1_pts, q2_pts, K_pts_block_0_ptr,K_pts_block_1_ptr,K_pts_block_2_ptr, V_pts_block_0_ptr, V_pts_block_1_ptr, V_pts_block_2_ptr, #
                                        pv_block_ptr_0,pv_block_ptr_1, pb_block_ptr, #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        2, offs_m, offs_n, N_CTX, V.dtype.element_ty == tl.float8e5  #
                                        )
    # epilogue
    m_i += tl.math.log2(l_i)

    acc_reg = acc_reg / l_i[:, None]
    acc_pv_0 = acc_pv_0 / l_i[:, None]
    acc_pv_1 = acc_pv_1 / l_i[:, None]
    acc_pts_0 = acc_pts_0 / l_i[:, None,None]
    acc_pts_1 = acc_pts_1 / l_i[:, None,None]
    acc_pts_2 = acc_pts_2 / l_i[:, None,None]
    m_ptrs = M + off_hz * N_CTX + offs_m

    tl.store(m_ptrs, m_i)

    # Store output of vanilla attention
    tl.store(O_block_ptr, acc_reg.to(Out_reg.type.element_ty))

    # Store output of Pair Value attention 
    tl.store(O_pv_block0_ptr, acc_pv_0.to(Out_pv.type.element_ty))
    tl.store(O_pv_block1_ptr, acc_pv_1.to(Out_pv.type.element_ty))

    # Store output of pts attention (Squared distnace affinity)
    tl.store(O_pts_block0_ptr, acc_pts_0.to(Out_pts.type.element_ty))
    tl.store(O_pts_block1_ptr, acc_pts_1.to(Out_pts.type.element_ty))
    tl.store(O_pts_block2_ptr, acc_pts_2.to(Out_pts.type.element_ty))


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, q_pts, k_pts, v_pts, pv, pb, causal, sm_scale):
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        o = torch.empty_like(q)
        o_pv = torch.empty((q.shape[0], q.shape[1], q.shape[2], pv.shape[3]), device=q.device, dtype=q.dtype)
        o_pts = torch.empty_like(q_pts)
        o_pts = torch.empty_like(q_pts)

        stage = 3 if causal else 1
        extra_kern_args = {}
        # Tuning for AMD target
        if is_hip():
            waves_per_eu = 3 if HEAD_DIM_K <= 64 else 2
            extra_kern_args = {"waves_per_eu": waves_per_eu, "allow_flush_denorm": True}

        grid = lambda args: (triton.cdiv(q.shape[2], args["BLOCK_M"]), q.shape[0] * q.shape[1], 1)
        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
    
        _attn_fwd[grid](
            q, k, v, sm_scale, M, o, o_pv, o_pts,  #
            #IPA
            q_pts, k_pts, v_pts,  #
            pv, pb,  #
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
            q_pts.stride(0), q_pts.stride(1), q_pts.stride(2), q_pts.stride(3), q_pts.stride(4),  #
            o_pv.stride(0), o_pv.stride(1), o_pv.stride(2), o_pv.stride(3),  #
            pb.stride(0), pb.stride(1), pb.stride(2), pb.stride(3),  #
            pv.stride(0), pv.stride(1), pv.stride(2), pv.stride(3),  #
            q.shape[0], q.shape[1],  #
            N_CTX=q.shape[2],  #
            P=q_pts.shape[3], C_Z=pv.shape[3],  # IPA
            HEAD_DIM=HEAD_DIM_K,  #
            STAGE=stage,  #
            **extra_kern_args)

        ctx.save_for_backward(q, k, v, o, M)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        return o , o_pv, o_pts

attention = _attention.apply


@pytest.mark.parametrize("Z, H, N_CTX, HEAD_DIM", [(2, 2, 1024, 128)])
@pytest.mark.parametrize("causal", [False])
def test_op(Z, H, N_CTX, HEAD_DIM, P, C_Z, causal, dtype=torch.float32):
    torch.manual_seed(20)
    print("Z, H, N_CTX, HEAD_DIM", Z, H, N_CTX, HEAD_DIM)
    q = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    k = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    v = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    q_pts = (torch.empty((Z, H, N_CTX, P, 3), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    k_pts = (torch.empty((Z, H, N_CTX, P, 3), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    v_pts = (torch.empty((Z, H, N_CTX, P, 3), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    pv = (torch.empty((Z, N_CTX, N_CTX, C_Z), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    pb = (torch.empty((Z, H, N_CTX, N_CTX), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    sm_scale = 0.5

    # reference implementation
    o_actual , o_pv_actual, o_pts_actual = ipa_impl(q, k, v, q_pts, k_pts, v_pts, pv, pb, causal, sm_scale)
    
    # triton implementation
    tri_out , tri_pv, tri_pts = attention(q, k, v, q_pts, k_pts, v_pts, pv, pb, causal, sm_scale)

    # compare
    assert torch.allclose(o_actual, tri_out, atol=1e-2, rtol=0)
    assert torch.allclose(o_pv_actual, tri_pv, atol=1e-2, rtol=0)
    assert torch.allclose(o_pts_actual, tri_pts, atol=1e-2, rtol=0)
    print("passed all correctness checks")


try:
    from flash_attn.flash_attn_interface import \
        flash_attn_qkvpacked_func as flash_attn_func
    HAS_FLASH = True
except BaseException:
    HAS_FLASH = False

BATCH, N_HEADS, HEAD_DIM, P, C_Z = 1, 12, 16, 4, 128
# vary seq length for fixed head and batch=4
configs = []
for mode in ["fwd"]:
    for causal in [False]:
        if mode == "bwd" and not causal:
            continue
        configs.append(
            triton.testing.Benchmark(
                x_names=["N_CTX"],
                x_vals=[2**i for i in range(5, 11)],
                line_arg="provider",
                line_vals=["triton-fp32"] + ["Pytorch"],
                line_names=["Triton [FP32]"] +
                ["Pytorch"],
                styles=[("red", "-"), ("blue", "-")],
                ylabel="Runtime (ms)",
                plot_name=f"flashIPA-batch{BATCH}-head{N_HEADS}-d{HEAD_DIM}-{mode}-causal={causal}",
                args={
                    "H": N_HEADS,
                    "BATCH": BATCH,
                    "HEAD_DIM": HEAD_DIM,
                    "P": P,
                    "C_Z": C_Z,
                    "mode": mode,
                    "causal": causal,
                },
            ))


def ipa_impl(q, k, v, q_pts, k_pts, v_pts, pv, pb, causal, sm_scale):


    def permute_final_dims(tensor: torch.Tensor, inds: List[int]):
            zero_index = -1 * len(inds)
            first_inds = list(range(len(tensor.shape[:zero_index])))
            return tensor.permute(first_inds + [zero_index + i for i in inds])

    ##########################

    ##########################
    # Compute attention scores
    ##########################
    # [*, N_res, N_res, H]
    b = pb
    a = torch.matmul(
            q,  # [*, H, N_res, C_hidden]
            k.transpose(2, 3)  # [*, H, C_hidden, N_res]
    )

    # [*, H, N_res, N_res, P_q, 3]
    pt_att = q_pts.unsqueeze(-3) - k_pts.unsqueeze(-4)
    pt_att = pt_att ** 2
    pt_att = sum(torch.unbind(pt_att, dim=-1))
    pt_att = torch.sum(pt_att, dim=-1)
    
    a = (a + pt_att + b)* sm_scale
    a = torch.softmax(a, dim=-1)

    o = torch.matmul(a, v.to(dtype=a.dtype))
    
    #[h, N_res, p, 3] -> [h, 3,N_res, p]
    v_pts = permute_final_dims(v_pts, (0, 3, 1, 2))
    o_pt = [
        torch.matmul(a, v.to(a.dtype))
        for v in torch.unbind(v_pts, dim=-3)
    ]
    o_pt = torch.stack(o_pt, dim=-3)
    
    # [h,3,N_res, p] -> [h,N_res, p, 3] 
    o_pt = permute_final_dims(o_pt, (0, 2, 3, 1))
    #o_pt = r[..., None, None].invert_apply(o_pt)

    # [*, N_res, H, C_z]
    o_pair = torch.matmul(a.transpose(-2, -3), pv.to(dtype=a.dtype))
    o_pair = permute_final_dims(o_pair, (0, 2, 1, 3))
    return o, o_pair, o_pt

@triton.testing.perf_report(configs)
def bench_flash_attention(BATCH, H, N_CTX, HEAD_DIM, P, C_Z, causal, mode, provider, device="cuda"):
    assert mode in ["fwd", "bwd"]
    warmup = 25
    rep = 100
    dtype = torch.float32

    # Test correctness agains pytorchIPA
    test_op(Z=BATCH, H=H, N_CTX=N_CTX, HEAD_DIM=HEAD_DIM, P=P, C_Z=C_Z, causal=causal, dtype=dtype)

    q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    q_pts = torch.randn((BATCH, H, N_CTX, P, 3), dtype=dtype, device=device, requires_grad=True)
    k_pts = torch.randn((BATCH, H, N_CTX, P, 3), dtype=dtype, device=device, requires_grad=True)
    v_pts = torch.randn((BATCH, H, N_CTX, P, 3), dtype=dtype, device=device, requires_grad=True)
    pv = torch.randn((BATCH, N_CTX, N_CTX, C_Z), dtype=dtype, device=device, requires_grad=True)
    pb = torch.randn((BATCH, H, N_CTX, N_CTX), dtype=dtype, device=device, requires_grad=True)
    sm_scale = 1.33
    
    if "triton" in provider:
        fn = lambda: attention(q, k, v, q_pts, k_pts, v_pts, pv, pb, causal, sm_scale)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        try:
            ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        except Exception:
            ms = 0     

    if provider == "Pytorch":
        fn = lambda: ipa_impl(q, k, v, q_pts, k_pts, v_pts, pv, pb, causal, sm_scale)
        try:
            ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        except Exception:
            ms = 0
    return ms


if __name__ == "__main__":
    # only works on post-Ampere GPUs right now
    bench_flash_attention.run(save_path=".", print_data=True)
