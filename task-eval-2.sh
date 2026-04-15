#!/bin/bash
# 离线评测：显式任务名 + 统一 CLI（tools/eval/run_eval.py），规则与 meft raw_evaluate_my 各任务一致。
# 用法：
#   bash task-eval.sh
#   TASK=nq PRED_FILE=... GOLD_FILE=... DETAIL_FILE=... bash task-eval.sh
#
# 任务名 TASK 示例：nq trivia nq_v1 nq_std all_nq medmcqa pubmedqa mmlu gsm8k math tool
# 详见：python tools/eval/run_eval.py --help

set -e -o pipefail

source /home/gpchen/lora/transformer-edge/init_conda.sh
conda activate lora-edge

PROJECT_ROOT=/home/gpchen/lora/transformer-edge
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

TASK="${TASK:-nq_v1}"
PRED_FILE="${PRED_FILE:-expm/llm/pubdatasets_nq_closed/std/base/pred_last.txt}"
GOLD_FILE="${GOLD_FILE:-cache/llm/pubdatasets_nq_closed/tgt.dev.txt}"
DETAIL_FILE="${DETAIL_FILE:-expm/llm/pubdatasets_nq_closed/std/base/eval_detail.jsonl}"

echo "TASK=$TASK PRED_FILE=$PRED_FILE GOLD_FILE=$GOLD_FILE DETAIL_FILE=$DETAIL_FILE"

exec python tools/eval/run_eval.py --task "$TASK" --pred "$PRED_FILE" --gold "$GOLD_FILE" --detail "$DETAIL_FILE"
