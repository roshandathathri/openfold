# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
This script is to test the performance of the DS4Sci_EvoformerAttention op.
To run the script,
1. Clone the CUTLASS repo. E.g. git clone https://github.com/NVIDIA/cutlass.git
2. Specify the CUTLASS_PATH environment variable. E.g. export CUTLASS_PATH=$(pwd)/cutlass
3. Run the script. E.g. python DS4Sci_EvoformerAttention_bench.py
"""
import os
import time
import importlib
from typing import List

import torch
import numpy as np

import contextlib
from torch.nn import functional as F
from deepspeed.accelerator import get_accelerator

from op_builder.builder import CUDAOpBuilder

import tomllib
with open('./config.toml', 'rb') as f:
    GLOBAL_CONFIG = tomllib.load(f)

############################################################################################################
# deepspeed/ops/op_builder/evoformer_attn.py
############################################################################################################

class EvoformerAttnBuilder(CUDAOpBuilder):
    BUILD_VAR = "DS_BUILD_EVOFORMER_ATTN"
    NAME = "tk_evoformer_attn"

    def __init__(self, name=None):
        name = self.NAME if name is None else name
        super().__init__(name=name)
        self.cutlass_path = os.environ.get('CUTLASS_PATH')
        print("CUTLASS_PATH="+repr(self.cutlass_path))

    def absolute_name(self):
        return f'deepspeed.ops.{self.NAME}_op'

    def sources(self):
        src_dir = GLOBAL_CONFIG['src_dir']
        return [f'{src_dir}/attention.cpp', f'{src_dir}/attention_back.cu']

    def load(self, verbose=True):
        if self.name in __class__._loaded_ops:
            return __class__._loaded_ops[self.name]

        from deepspeed.git_version_info import installed_ops, torch_info, accelerator_name
        from deepspeed.accelerator import get_accelerator
        if installed_ops.get(self.name, False) and accelerator_name == get_accelerator()._name:
            # Ensure the op we're about to load was compiled with the same
            # torch/cuda versions we are currently using at runtime.
            self.validate_torch_version(torch_info)
            if torch.cuda.is_available() and isinstance(self, CUDAOpBuilder):
                self.validate_torch_op_version(torch_info)

            op_module = importlib.import_module(self.absolute_name())
            __class__._loaded_ops[self.name] = op_module
            return op_module
        else:
            return self.jit_load(verbose)
        
    def extra_ldflags(self):
        if not self.is_rocm_pytorch():
            return ['-lcurand']
        else:
            return []
        
    def include_paths(self):
        includes = [f'{self.cutlass_path}/include', f'{self.cutlass_path}/tools/util/include']
        return includes
    
    def nvcc_args(self):
        args = super().nvcc_args()
        try:
            import torch
        except ImportError:
            self.warning("Please install torch if trying to pre-compile kernels")
            return args
        major = torch.cuda.get_device_properties(0).major  #ignore-cuda
        minor = torch.cuda.get_device_properties(0).minor  #ignore-cuda
        args.append(f"-DGPU_ARCH={major}{minor}")
        return args

    def jit_load(self, verbose=True):
        if not self.is_compatible(verbose):
            raise RuntimeError(
                f"Unable to JIT load the {self.name} op due to it not being compatible due to hardware/software issue. {self.error_log}"
            )
        try:
            import ninja  # noqa: F401 # type: ignore
        except ImportError:
            raise RuntimeError(f"Unable to JIT load the {self.name} op due to ninja not being installed.")

        if isinstance(self, CUDAOpBuilder) and not self.is_rocm_pytorch():
            self.build_for_cpu = not torch.cuda.is_available()

        self.jit_mode = True
        from torch.utils.cpp_extension import load

        start_build = time.time()
        sources = [os.path.abspath(self.deepspeed_src_path(path)) for path in self.sources()] + [f'./src/attention_tk.cu']
        extra_include_paths = [os.path.abspath(self.deepspeed_src_path(path)) for path in self.include_paths()] + ['./DeepSpeed/csrc/deepspeed4science/evoformer_attn/', './ThunderKittens']

        # Torch will try and apply whatever CCs are in the arch list at compile time,
        # we have already set the intended targets ourselves we know that will be
        # needed at runtime. This prevents CC collisions such as multiple __half
        # implementations. Stash arch list to reset after build.
        torch_arch_list = None
        if "TORCH_CUDA_ARCH_LIST" in os.environ:
            torch_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
            os.environ["TORCH_CUDA_ARCH_LIST"] = ""

        nvcc_args = self.strip_empty_entries(self.nvcc_args()) + ['-std c++20']# + ['-g']
        cxx_args = self.strip_empty_entries(self.cxx_args()) + ['-std=c++20 -fdiagnostics-color=always']# + ['-g', '-ggdb']

        if isinstance(self, CUDAOpBuilder):
            if not self.build_for_cpu and self.enable_bf16:
                cxx_args.append("-DBF16_AVAILABLE")
                nvcc_args.append("-DBF16_AVAILABLE")
                nvcc_args.append("-U__CUDA_NO_BFLOAT16_OPERATORS__")
                nvcc_args.append("-U__CUDA_NO_BFLOAT162_OPERATORS__")
                nvcc_args.append("-U__CUDA_NO_BFLOAT16_CONVERSIONS__")

        if self.is_rocm_pytorch():
            cxx_args.append("-D__HIP_PLATFORM_AMD__=1")
            os.environ["PYTORCH_ROCM_ARCH"] = self.get_rocm_gpu_arch()
            cxx_args.append('-DROCM_WAVEFRONT_SIZE=%s' % self.get_rocm_wavefront_size())

        op_module =  load(name=self.name,
                         sources=self.strip_empty_entries(sources),
                         extra_include_paths=self.strip_empty_entries(extra_include_paths),
                         extra_cflags=cxx_args,
                         extra_cuda_cflags=nvcc_args,
                         extra_ldflags=self.strip_empty_entries(self.extra_ldflags()),
                         verbose=verbose)

        build_duration = time.time() - start_build
        if verbose:
            print(f"Time to load {self.name} op: {build_duration} seconds")

        # Reset arch list so we are not silently removing it for other possible use cases
        if torch_arch_list:
            os.environ["TORCH_CUDA_ARCH_LIST"] = torch_arch_list

        __class__._loaded_ops[self.name] = op_module

        return op_module



############################################################################################################
# deepspeed/ops/deepspeed4science/evoformer_attn.py
############################################################################################################

kernel_ = None


class EvoformerFusedAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, bias1=None, bias2=None):
        """
        q, k, v: are in shape [*, L, H, D]
        """
        bias1_ = bias1.contiguous() if bias1 is not None else torch.tensor([], dtype=q.dtype, device=q.device)
        bias2_ = bias2.contiguous() if bias2 is not None else torch.tensor([], dtype=q.dtype, device=q.device)
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        def _attention(Q, K, V, bias1, bias2):
            assert Q.shape[-3] > 16, "seq_len must be greater than 16"
            O = torch.empty_like(Q, dtype=Q.dtype)
            assert get_accelerator().on_accelerator(Q), "Q must be on cuda"
            assert get_accelerator().on_accelerator(K), "K must be on cuda"
            assert get_accelerator().on_accelerator(V), "V must be on cuda"
            assert get_accelerator().on_accelerator(bias1), "bias1 must be on cuda"
            assert get_accelerator().on_accelerator(bias2), "bias2 must be on cuda"
            global kernel_
            if kernel_ is None:
                kernel_ = EvoformerAttnBuilder().load()
            nheads = Q.shape[-2]
            nq = (Q.shape[-3] + 31) // 32 * 32
            nb = np.prod(Q.shape[:-3])
            lse = torch.empty((nb, nheads, nq), dtype=torch.float32, device=Q.device)
            # print("Q:", Q.view(-1, *Q.shape[-3:])[5, 128, 3, 0])
            # print("K:", K.view(-1, *K.shape[-3:])[5, 0, 3, 0])
            # print("V:", V.view(-1, *V.shape[-3:])[5, 0, 3, 0])
            # print("O:", O.view(-1, *O.shape[-3:])[5, 128, 3, 0])
            kernel_.attention(Q, K, V, bias1, bias2, O, lse)
            return O, lse
        o, lse = _attention(q, k, v, bias1_, bias2_)
        ctx.save_for_backward(q, k, v, o, lse, bias1_, bias2_)
        return o


def DS4Sci_EvoformerAttention(Q, K, V, biases):
    assert len(biases) <= 2

    if (len(biases) == 0):
        biases.append(None)

    if (len(biases) == 1):
        biases.append(None)

    bias_1_shape = lambda x: (x.shape[0], x.shape[1], 1, 1, x.shape[2])
    bias_2_shape = lambda x: (x.shape[0], 1, x.shape[3], x.shape[2], x.shape[2])

    if biases[0] is not None:
        assert biases[0].shape == bias_1_shape(Q), "bias1 shape is incorrect"

    if biases[1] is not None:
        assert biases[1].shape == bias_2_shape(Q), "bias2 shape is incorrect"

    return EvoformerFusedAttention.apply(Q, K, V, biases[0], biases[1])
###

def attention_reference(
        q_input: torch.Tensor,  # [*, Dim_Q, H, C_hid]
        k_input: torch.Tensor,  # [*, Dim_Q, H, C_hid]
        v_input: torch.Tensor,  # [*, Dim_Q, H, C_hid]
        biases: List[torch.Tensor],
        sm_scale: float) -> torch.Tensor:
    # Original shape: [*, Dim_Q, H, C_hid] -> Transpose to: [*, H, Dim_Q, C_hid]
    q = q_input.transpose(-2, -3)
    k = k_input.transpose(-2, -3)
    v = v_input.transpose(-2, -3)

    # Now, q, k, v are in shape: [*, H, Dim_Q, C_hid]

    # Transpose k to shape [*, H, C_hid, Dim_Q]
    k_t = k.transpose(-1, -2)

    # Now, q and k_t are in shapes: [*, H, Dim_Q, C_hid] and [*, H, C_hid, Dim_Q] respectively

    # [*, H, Dim_Q, Dim_Q]
    a = torch.matmul(q, k_t) * sm_scale

    for b in biases:
        a += b

    a = F.softmax(a, dim=-1)

    # Now, a is in shape [*, H, Dim_Q, Dim_Q], v is in shape [*, H, Dim_Q, C_hid]

    # Matmul operation results in [*, H, Dim_Q, C_hid]
    a_v = torch.matmul(a, v)

    # [*, Dim_Q, H, C_hid]
    o = a_v.transpose(-2, -3)

    return o


dtype = torch.bfloat16

N = 256
heads = 4
dim = 64  # 32
seq_len = 256


@contextlib.contextmanager
def cuda_timer(res_list):
    start = get_accelerator().Event(enable_timing=True)
    end = get_accelerator().Event(enable_timing=True)
    start.record()
    yield
    end.record()
    get_accelerator().synchronize()
    res_list.append(start.elapsed_time(end))


def benchmark():
    torch.manual_seed(24)
    ours_fw = []
    ours_bw = []
    baseline_fw = []
    baseline_bw = []
    for batch in range(1, 17):
        Q = torch.randn(batch, N, seq_len, heads, dim, dtype=dtype, device="cuda", requires_grad=True)
        K = torch.randn(batch, N, seq_len, heads, dim, dtype=dtype, device="cuda", requires_grad=True)
        V = torch.randn(batch, N, seq_len, heads, dim, dtype=dtype, device="cuda", requires_grad=True)
        bias1 = torch.randn(batch, N, 1, 1, seq_len, dtype=dtype, device="cuda", requires_grad=False)
        bias2 = torch.randn(batch, 1, heads, seq_len, seq_len, dtype=dtype, device="cuda", requires_grad=True)
        # warm up
        out = DS4Sci_EvoformerAttention(Q, K, V, [bias1, bias2])
        with cuda_timer(ours_fw):
            out = DS4Sci_EvoformerAttention(Q, K, V, [bias1, bias2])
        # d_out = torch.rand_like(out)
        # with cuda_timer(ours_bw):
        #     out.backward(d_out)
        # warm up
        attention_reference(Q, K, V, [bias1, bias2], 1 / (dim**0.5))
        with cuda_timer(baseline_fw):
            ref_out = attention_reference(Q, K, V, [bias1, bias2], 1 / (dim**0.5))
        # with cuda_timer(baseline_bw):
        #     ref_out.backward(d_out)
        # assert torch.allclose(out.to("cpu"), ref_out.to("cpu"), atol=1e-02)

    print(f"batch size\tours (FW)\tbaseline (FW)\tours (BW)\tbaseline (BW)")
    for i in range(len(ours_fw)):
        print(f"{i+1}\t{ours_fw[i]}\t{baseline_fw[i]}")#\t{ours_bw[i]}\t{baseline_bw[i]}")


benchmark()
