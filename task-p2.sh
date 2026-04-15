#!/bin/bash
# Qwen3 预测 + 可选离线评测（与 task-2.sh 同一数据集：pubdatasets_toolbench）
# 用法：
#   bash task-p2.sh
#   SKIP_EVAL=1 bash task-p2.sh
#
# 说明：
# - predict.py 读 cache/<DATA_ID>/test.h5（须 mktest 流程生成）
# - 评测使用 run_eval task=tool（gold 为每行 JSON 数组）；不需要评测时设 SKIP_EVAL=1

set -e -o pipefail

source /home/gpchen/lora/transformer-edge/init_conda.sh
conda activate lora-edge

PROJECT_ROOT=/home/gpchen/lora/transformer-edge
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

export DATA_ID="${DATA_ID:-llm/pubdatasets_toolbench}"
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

SKIP_EVAL="${SKIP_EVAL:-0}"
if [[ "$SKIP_EVAL" == "0" ]]; then
	TASK="${TASK:-tool}"
	PRED_FILE="${PRED_FILE:-$OUT}"
	GOLD_FILE="${GOLD_FILE:-cache/llm/${_DATA_TAG}/tgt.dev.txt}"
	DETAIL_FILE="${DETAIL_FILE:-expm/llm/${_DATA_TAG}/std/base/eval_detail.jsonl}"
	echo "TASK=$TASK PRED_FILE=$PRED_FILE GOLD_FILE=$GOLD_FILE DETAIL_FILE=$DETAIL_FILE"
	python tools/eval/run_eval.py --task "$TASK" --pred "$PRED_FILE" --gold "$GOLD_FILE" --detail "$DETAIL_FILE"
else
	echo "SKIP_EVAL=1: 跳过离线评测。"
fi
