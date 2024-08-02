#!/bin/sh
# 1. git submodule update --init --recursive    # Get DeepSpeed and ThunderKittens repos
# 2. alternatively, manually clone the repos and configure the paths to your own path
DS_PATH=./DeepSpeed
TK_PATH=./ThunderKittens

CUTLASS_PATH=~/code/cutlass
MAX_JOBS=0  # for ninja; 0 means unlimited

#############################################
# Experiments
#############################################

# TK attention benchmark
PYTHONPATH="$(pwd):${DS_PATH}" CUTLASS_PATH=${CUTLASS_PATH} MAX_JOBS=$MAX_JOBS python ./tk_evoformer.py

# DeepSpeed evoformer benchmark
# PYTHONPATH="${DS_PATH}" CUTLASS_PATH=${CUTLASS_PATH} MAX_JOBS=$MAX_JOBS python ${DS_PATH}/tests/benchmarks/DS4Sci_EvoformerAttention_bench.py