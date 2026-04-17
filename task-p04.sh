#!/bin/bash
# Qwen3 基座模型预测 + 可选离线评测（默认 MetaMathQA）
# 用法：
#   bash task-p04.sh
#   SKIP_EVAL=1 bash task-p04.sh

set -e -o pipefail

source /home/gpchen/lora/transformer-edge/init_conda.sh
conda activate lora-edge

PROJECT_ROOT=/home/gpchen/lora/transformer-edge
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

export DATA_ID="${DATA_ID:-llm/pubdatasets_metamathqa}"
_DATA_TAG="${DATA_ID##*/}"

export OUT="${OUT:-expm/llm/${_DATA_TAG}/std/base/pred_base.txt}"
export TOKENIZER="${TOKENIZER:-/home/common/plm/Qwen/Qwen3-8B}"
export MODEL="${MODEL:-/home/common/plm/Qwen/Qwen3-8B/model.h5}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "OUT=$OUT"
echo "TOKENIZER=$TOKENIZER"
echo "MODEL=$MODEL"
echo "DATA_ID=$DATA_ID"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

python adv/predict/plm/qwen/predict.py "$OUT" "$TOKENIZER" "$MODEL"

SKIP_EVAL="${SKIP_EVAL:-0}"
if [[ "$SKIP_EVAL" == "0" ]]; then
	TASK="${TASK:-math}"
	PRED_FILE="${PRED_FILE:-$OUT}"
	GOLD_FILE="${GOLD_FILE:-cache/llm/${_DATA_TAG}/gold.math.loose.txt}"
	DETAIL_FILE="${DETAIL_FILE:-expm/llm/${_DATA_TAG}/std/base/eval_detail_base.jsonl}"
	echo "TASK=$TASK PRED_FILE=$PRED_FILE GOLD_FILE=$GOLD_FILE DETAIL_FILE=$DETAIL_FILE"
	python tools/eval/run_eval.py --task "$TASK" --pred "$PRED_FILE" --gold "$GOLD_FILE" --detail "$DETAIL_FILE"
else
	echo "SKIP_EVAL=1: 跳过离线评测。"
fi
