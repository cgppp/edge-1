#!/bin/bash
# Qwen3 预测 + 可选离线评测（与 task-4.sh 同一数据集：pubdatasets_metamathqa）
# 用法：
#   bash task-p4.sh
#
# 说明：
# - predict.py 读 cache/<DATA_ID>/test.h5（须 mktest.py 生成）
# - MetaMathQA 的 tgt.dev 为长 CoT 文本，与 run_eval 的 math/gsm8k 数字列表 gold 不完全一致，默认跳过评测。
#   若你有对齐的 gold 行：SKIP_EVAL=0 TASK=math GOLD_FILE=... bash task-p4.sh
# - 与 task-4.sh 一致可设：USE_AMP=1（本脚本仅打印提示，训练用；预测由 cnfg 控制）

set -e -o pipefail

source /home/gpchen/lora/transformer-edge/init_conda.sh
conda activate lora-edge

PROJECT_ROOT=/home/gpchen/lora/transformer-edge
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# 与 task-4.sh 默认 DATA_ID 一致
export DATA_ID="${DATA_ID:-llm/pubdatasets_metamathqa}"
_DATA_TAG="${DATA_ID##*/}"

export OUT="${OUT:-expm/llm/${_DATA_TAG}/std/base/pred_last.txt}"
export TOKENIZER="${TOKENIZER:-/home/common/plm/Qwen/Qwen3-8B}"
export MODEL="${MODEL:-expm/llm/${_DATA_TAG}/std/base/merge_last.h5}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "OUT=$OUT"
echo "TOKENIZER=$TOKENIZER"
echo "MODEL=$MODEL"
echo "DATA_ID=$DATA_ID"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

python adv/predict/plm/qwen/predict.py "$OUT" "$TOKENIZER" "$MODEL"

# ---- 离线评测（math/gsm8k 需可解析的数字列表 gold；默认跳过）----
SKIP_EVAL="${SKIP_EVAL:-1}"
if [[ "$SKIP_EVAL" == "0" ]]; then
	TASK="${TASK:-math}"
	PRED_FILE="${PRED_FILE:-$OUT}"
	GOLD_FILE="${GOLD_FILE:-cache/llm/${_DATA_TAG}/tgt.dev.txt}"
	DETAIL_FILE="${DETAIL_FILE:-expm/llm/${_DATA_TAG}/std/base/eval_detail.jsonl}"
	echo "TASK=$TASK PRED_FILE=$PRED_FILE GOLD_FILE=$GOLD_FILE DETAIL_FILE=$DETAIL_FILE"
	python tools/eval/run_eval.py --task "$TASK" --pred "$PRED_FILE" --gold "$GOLD_FILE" --detail "$DETAIL_FILE"
else
	echo "SKIP_EVAL=1: 跳过离线评测（MetaMath tgt.dev 与 run_eval task=math/gsm8k 的 gold 格式通常不对齐）。设 SKIP_EVAL=0 并指定 TASK/GOLD_FILE 可启用。"
fi
