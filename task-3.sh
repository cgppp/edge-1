#!/bin/bash
# Qwen3 基座 + LoRA 微调，单卡 5090。四个数据集已做到 3.4 步后，用本脚本启动训练。
# 用法：
#   1）改下面 PRE_TRAINED_M、DATA_ID 后执行: bash task.sh
#   2）或传参: DATA_ID=llm/pubdatasets_metamathqa PRE_TRAINED_M=/path/to/Qwen3-0.6B/model.h5 bash task.sh

set -e -o pipefail

# 激活 conda（按你本机路径选一个 source）
source /home/gpchen/lora/transformer-edge/init_conda.sh
conda activate lora-edge

PROJECT_ROOT=/home/gpchen/lora/transformer-edge
cd "$PROJECT_ROOT"
# 保证 Python 能 import 到项目包（cnfg、transformer 等），避免 ModuleNotFoundError
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# 基座权重：Qwen3 的 model.h5（或 .bin），必须事先准备好
export PRE_TRAINED_M="${PRE_TRAINED_M:-/home/common/plm/Qwen/Qwen3-8B/model.h5}"

# 训练数据：选一个数据集目录（与 3.2~3.4 生成的 cache/llm/pubdatasets_* 对应）
export DATA_ID="${DATA_ID:-llm/pubdatasets_nq_closed}"
# 其他可选: llm/pubdatasets_metamathqa, llm/pubdatasets_squad_closed, llm/pubdatasets_toolbench

# LoRA 超参（可选，默认在 cnfg/lora.py 里为 8/16）
export LORA_RANK="${LORA_RANK:-8}"
export LORA_ALPHA="${LORA_ALPHA:-16}"

# 单卡 5090，使用 cuda:0
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "DATA_ID=$DATA_ID PRE_TRAINED_M=$PRE_TRAINED_M LORA_RANK=$LORA_RANK LORA_ALPHA=$LORA_ALPHA"
python adv/train/plm/train_lora_qwen.py
