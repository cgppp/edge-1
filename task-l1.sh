#!/bin/bash
# Qwen3 + HPLSTM（可选 LoRA）：见 adv/train/plm/train_hplstm_qwen.py。
# 用法：
#   1）改下面 PRE_TRAINED_M、DATA_ID 后执行: bash task_1_1.sh
#   2）或传参: DATA_ID=llm/pubdatasets_metamathqa PRE_TRAINED_M=/path/to/model.h5 bash task_1_1.sh
#   3）融合方案: HPLSTM_FUSION=A|B|C（对应 Decoder_1/2/3），默认 A；与 cnfg/lora.py 中 hplstm_fusion 一致时可省略

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
export DATA_ID="${DATA_ID:-llm/pubdatasets_metamathqa}"
# 其他可选: llm/pubdatasets_squad_closed, llm/pubdatasets_nq_closed, llm/pubdatasets_toolbench

# HPLSTM 与 Attention 融合拓扑：A=并行(Decoder_1)，B=串联(Decoder_2，默认)，C=ResHPLSTM 子层(Decoder_3)
export HPLSTM_FUSION="${HPLSTM_FUSION:-B}"

# LoRA 超参（可选；与 cnfg/lora.py 一致，train_hplstm_qwen 会读环境变量覆盖）
# 默认保持空：避免意外启用 LoRA（实现 HPLSTM-only）。
# 如需启用 LoRA，请显式传参：
#   LORA_RANK=8 LORA_ALPHA=16 bash task_1_1.sh
export LORA_RANK="${LORA_RANK:-}"
export LORA_ALPHA="${LORA_ALPHA:-}"

# 单卡 5090，使用 cuda:0
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# 显存紧张时开启 AMP 可减显存（如 Qwen3-8B 单卡 32GiB）：USE_AMP=1
export USE_AMP="${USE_AMP:-0}"

echo "DATA_ID=$DATA_ID PRE_TRAINED_M=$PRE_TRAINED_M HPLSTM_FUSION=$HPLSTM_FUSION LORA_RANK=${LORA_RANK:-<unset>} LORA_ALPHA=${LORA_ALPHA:-<unset>} USE_AMP=$USE_AMP"
python adv/train/plm/train_hplstm_qwen.py
