#!/bin/bash
# Qwen3 基座模型预测 + 可选离线评测（默认 ToolBench）
# 用法：
#   bash task-p02.sh
#   SKIP_EVAL=1 bash task-p02.sh

set -e -o pipefail

source /home/gpchen/lora/transformer-edge/init_conda.sh
conda activate lora-edge

PROJECT_ROOT=/home/gpchen/lora/transformer-edge
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

export DATA_ID="${DATA_ID:-llm/pubdatasets_toolbench_benchmark}"
_DATA_TAG="${DATA_ID##*/}"
# 与 task-p2.sh 一致：benchmark 的预测输出写到已有 expm/.../pubdatasets_toolbench/std/base
_TOOLBENCH_EXPM="expm/llm/pubdatasets_toolbench/std/base"
if [[ "$_DATA_TAG" == "pubdatasets_toolbench_benchmark" ]]; then
	export OUT="${OUT:-${_TOOLBENCH_EXPM}/pred_base.txt}"
else
	export OUT="${OUT:-expm/llm/${_DATA_TAG}/std/base/pred_base.txt}"
fi
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
	TASK="${TASK:-tool}"
	PRED_FILE="${PRED_FILE:-$OUT}"
	if [[ "${GOLD_FILE:-}" == "" ]]; then
		if [[ "$_DATA_TAG" == "pubdatasets_toolbench_benchmark" ]]; then
			GOLD_FILE="cache/llm/${_DATA_TAG}/gold.toolbench.benchmark.txt"
		else
			GOLD_FILE="cache/llm/${_DATA_TAG}/tgt.dev.txt"
		fi
	fi
	if [[ "$_DATA_TAG" == "pubdatasets_toolbench_benchmark" ]]; then
		DETAIL_FILE="${DETAIL_FILE:-${_TOOLBENCH_EXPM}/eval_detail_base.jsonl}"
	else
		DETAIL_FILE="${DETAIL_FILE:-expm/llm/${_DATA_TAG}/std/base/eval_detail_base.jsonl}"
	fi
	echo "TASK=$TASK PRED_FILE=$PRED_FILE GOLD_FILE=$GOLD_FILE DETAIL_FILE=$DETAIL_FILE"
	python tools/eval/run_eval.py --task "$TASK" --pred "$PRED_FILE" --gold "$GOLD_FILE" --detail "$DETAIL_FILE"
else
	echo "SKIP_EVAL=1: 跳过离线评测。"
fi
