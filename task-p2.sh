#!/bin/bash
# Qwen3 预测 + 可选离线评测（与 task-2.sh 同一数据集：pubdatasets_toolbench）
# 用法：
#   bash task-p2.sh
#   SKIP_EVAL=1 bash task-p2.sh
#
# 说明：
# - predict.py 读 cache/<DATA_ID>/test.h5（须 mktest 流程生成）
# - 评测使用 run_eval task=tool（gold 为每行 JSON 数组）
# - 若 DATA_ID=llm/pubdatasets_toolbench_benchmark：test 仍读 cache/.../toolbench_benchmark/test.h5；
#   预测输出与 merge 权重默认写到已有目录 expm/llm/pubdatasets_toolbench/std/base（与 task-2 训练产物一致）。

set -e -o pipefail

source /home/gpchen/lora/transformer-edge/init_conda.sh
conda activate lora-edge

PROJECT_ROOT=/home/gpchen/lora/transformer-edge
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

export DATA_ID="${DATA_ID:-llm/pubdatasets_toolbench_benchmark}"
_DATA_TAG="${DATA_ID##*/}"
# benchmark 评测时复用 pubdatasets_toolbench 的 expm 目录（避免 expm/.../toolbench_benchmark/... 未创建导致 predict 写文件失败）
_TOOLBENCH_EXPM="expm/llm/pubdatasets_toolbench/std/base"

if [[ "$_DATA_TAG" == "pubdatasets_toolbench_benchmark" ]]; then
	export OUT="${OUT:-${_TOOLBENCH_EXPM}/pred_last.txt}"
else
	export OUT="${OUT:-expm/llm/${_DATA_TAG}/std/base/pred_last.txt}"
fi
export TOKENIZER="${TOKENIZER:-/home/common/plm/Qwen/Qwen3-8B}"
# 评测 benchmark 时通常仍使用 pubdatasets_toolbench 训练出的模型
if [[ "${MODEL:-}" == "" ]]; then
	if [[ "$_DATA_TAG" == "pubdatasets_toolbench_benchmark" ]]; then
		export MODEL="${_TOOLBENCH_EXPM}/merge_last.h5"
	else
		export MODEL="expm/llm/${_DATA_TAG}/std/base/merge_last.h5"
	fi
else
	export MODEL
fi

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
		DETAIL_FILE="${DETAIL_FILE:-${_TOOLBENCH_EXPM}/eval_detail.jsonl}"
	else
		DETAIL_FILE="${DETAIL_FILE:-expm/llm/${_DATA_TAG}/std/base/eval_detail.jsonl}"
	fi
	echo "TASK=$TASK PRED_FILE=$PRED_FILE GOLD_FILE=$GOLD_FILE DETAIL_FILE=$DETAIL_FILE"
	python tools/eval/run_eval.py --task "$TASK" --pred "$PRED_FILE" --gold "$GOLD_FILE" --detail "$DETAIL_FILE"
else
	echo "SKIP_EVAL=1: 跳过离线评测。"
fi
