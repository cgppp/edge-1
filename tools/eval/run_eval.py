#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
命令行入口：按行对齐的 pred 文件与 gold 文件做离线评测。

- 通过 --task 显式指定任务名（不根据路径猜数据集），再查 TASK_REGISTRY 调用对应 _task_*。
- 每行一条样本；pred/gold 行数不一致时取较短长度并打 WARN。
- 可选 --detail 输出 JSONL，每行一条含 idx、task、pred、gold_line、score 及 scorer 返回的附加字段。

Gold 行格式约定（与 scorers + 数据导出一致时即可复现 raw_evaluate 口径）：
  squad / trivia：多答案列表，如引号列表或 Python list
  nq_v1、nq_std、all_nq：同上
  medmcqa、pubmedqa、mmlu：单行标签
  gsm8k、math：数字列表（JSON 或逗号分隔）
  tool：整行 JSON 数组，元素为 API 描述字符串

使用示例：
  python tools/eval/run_eval.py --task squad \\
    --pred expm/.../pred_last.txt \\
    --gold cache/.../tgt.dev.txt \\
    --detail expm/.../eval_detail.jsonl
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import os
from typing import Any, Callable

from scorers import (
    score_gsm8k,
    score_medmcqa,
    score_mmlu,
    score_nq_std,
    score_nq_v1,
    score_pubmedqa,
    score_squad,
    score_tool,
)

# ---------------------------------------------------------------------------
# Gold 行解析：把「文件里的一行字符串」变成各任务 scorer 需要的 Python 结构
# ---------------------------------------------------------------------------


def parse_gold_list(line: str) -> list[str]:
    """
    解析 SQuAD / NQ 类 gold 行：多种常见格式兼容。

    1) 优先用正则提取单/双引号内片段（适配类似 ['a' 'b'] 的非标准列表）；
    2) 否则尝试 ast.literal_eval 成 list/tuple，元素转 str；
    3) 再失败则整行 strip 作为单元素列表（或空列表）。
    """
    quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", line)
    answers = [a or b for a, b in quoted if (a or b) is not None]
    if answers:
        return answers
    try:
        obj = ast.literal_eval(line)
        if isinstance(obj, (list, tuple)):
            return [str(x) for x in obj]
    except (SyntaxError, ValueError, TypeError):
        pass
    return [line.strip()] if line.strip() else []


def parse_gold_float_list(line: str) -> list[float]:
    """
    解析 GSM8K/Math 的标准答案数列：一行内可为 JSON 数组、literal 列表或逗号分隔数字。
    按顺序尝试 json.loads → ast.literal_eval → 按逗号拆分 float，无法转换的片段跳过。
    """
    line = line.strip()
    try:
        obj = json.loads(line)
        if isinstance(obj, list):
            return [float(x) for x in obj]
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    try:
        obj = ast.literal_eval(line)
        if isinstance(obj, (list, tuple)):
            return [float(x) for x in obj]
    except (SyntaxError, ValueError, TypeError):
        pass
    out: list[float] = []
    for p in line.split(","):
        p = p.strip()
        if not p:
            continue
        try:
            out.append(float(p))
        except ValueError:
            pass
    return out


def parse_gold_tool(line: str) -> list[str]:
    """解析 Tool 任务 gold：整行须为 JSON 数组，元素转为 str；否则抛错由上层捕获并记 0 分。"""
    line = line.strip()
    obj = json.loads(line)
    if not isinstance(obj, list):
        raise ValueError("tool gold line must be a JSON array")
    return [str(x) for x in obj]


# ---------------------------------------------------------------------------
# 单样本适配层：统一签名 (pred, gold_line) -> (分数, extra_dict)，供主循环调用
# ---------------------------------------------------------------------------


def _run_single_label(pred: str, gold_line: str, fn: Callable[[str, str], Any]) -> tuple[int, dict[str, Any]]:
    """
    封装「gold 一整行就是一个标签」的任务：strip 后交给 medmcqa/pubmedqa/mmlu 的 score_*。
    返回 int 分数与 scorer 自带的说明字典。
    """
    label = gold_line.strip()
    score, extra = fn(pred, label)
    return int(score), extra


def _run_gsm(pred: str, gold_line: str) -> tuple[int, dict[str, Any]]:
    """将 gold 行解析为 float 列表后调用 score_gsm8k。"""
    std = parse_gold_float_list(gold_line)
    return score_gsm8k(pred, std)


def _task_squad(p: str, g: str) -> tuple[int | float, dict[str, Any]]:
    """squad：gold 行 -> 字符串列表 -> score_squad。"""
    golds = parse_gold_list(g)
    s, ex = score_squad(p, golds)
    return s, ex


def _task_trivia(p: str, g: str) -> tuple[int | float, dict[str, Any]]:
    """trivia：与 squad 判分规则相同，共用 _task_squad。"""
    return _task_squad(p, g)


def _task_nq_v1(p: str, g: str) -> tuple[int | float, dict[str, Any]]:
    """nq_v1：gold 多候选 -> parse_gold_list -> score_nq_v1（大小写敏感子串）。"""
    golds = parse_gold_list(g)
    return score_nq_v1(p, golds)


def _task_nq_std(p: str, g: str) -> tuple[int | float, dict[str, Any]]:
    """nq_std / all_nq：与 nq_v1 相同列表格式，走 score_nq_std（内部等同 nq_v1）。"""
    golds = parse_gold_list(g)
    return score_nq_std(p, golds)


def _task_medmcqa(p: str, g: str) -> tuple[int | float, dict[str, Any]]:
    return _run_single_label(p, g, score_medmcqa)


def _task_pubmedqa(p: str, g: str) -> tuple[int | float, dict[str, Any]]:
    return _run_single_label(p, g, score_pubmedqa)


def _task_mmlu(p: str, g: str) -> tuple[int | float, dict[str, Any]]:
    return _run_single_label(p, g, score_mmlu)


def _task_gsm8k(p: str, g: str) -> tuple[int | float, dict[str, Any]]:
    return _run_gsm(p, g)


def _task_tool(p: str, g: str) -> tuple[int | float, dict[str, Any]]:
    """tool：gold 行为 JSON 数组 -> parse_gold_tool -> score_tool（返回连续分）。"""
    std = parse_gold_tool(g)
    return score_tool(p, std)


# ---------------------------------------------------------------------------
# 任务名 -> 单样本适配函数；主程序只查表，不做路径推断
# ---------------------------------------------------------------------------

TASK_REGISTRY = {
    "squad": _task_squad,
    "trivia": _task_trivia,
    "nq_v1": _task_nq_v1,
    "nq_std": _task_nq_std,
    "all_nq": _task_nq_std,
    "medmcqa": _task_medmcqa,
    "pubmedqa": _task_pubmedqa,
    "mmlu": _task_mmlu,
    "gsm8k": _task_gsm8k,
    "math": _task_gsm8k,
    "tool": _task_tool,
}


def main() -> None:
    """
    主流程：解析参数；读 pred/gold 全文；按行循环调用 TASK_REGISTRY[--task]；
    累加分数（tool 为 Jaccard 连续分，其余多为 0/1）；打印汇总；可选写 JSONL。

    单条异常时记 0 分并写入 error 字段，避免整批中断。
    """
    parser = argparse.ArgumentParser(description="Offline eval: pred vs gold (raw_evaluate_my-compatible scorers).")
    parser.add_argument(
        "--task",
        required=True,
        choices=sorted(TASK_REGISTRY.keys()),
        help="Task name (explicit; no path inference).",
    )
    parser.add_argument("--pred", required=True, help="Prediction file, one line per sample.")
    parser.add_argument("--gold", required=True, help="Gold file, one line per sample.")
    parser.add_argument("--detail", default="", help="Optional JSONL path for per-example records.")
    args = parser.parse_args()

    pred_path, gold_path = args.pred, args.gold
    with open(pred_path, encoding="utf-8") as f:
        preds = [line.rstrip("\n") for line in f]
    with open(gold_path, encoding="utf-8") as f:
        golds = [line.rstrip("\n") for line in f]

    n = min(len(preds), len(golds))
    if len(preds) != len(golds):
        print(
            f"[WARN] line count mismatch: pred={len(preds)} gold={len(golds)}; using first {n} lines",
            file=sys.stderr,
        )

    scorer = TASK_REGISTRY[args.task]
    aggregate = 0.0
    detail_path = args.detail.strip()
    out_f = None
    if detail_path:
        os.makedirs(os.path.dirname(detail_path) or ".", exist_ok=True)
        out_f = open(detail_path, "w", encoding="utf-8")
    try:
        for i in range(n):
            pred, gold_line = preds[i], golds[i]
            try:
                score, extra = scorer(pred, gold_line)
            except Exception as e:
                score = 0.0
                extra = {"error": str(e)}
            aggregate += float(score)
            rec = {"idx": i, "task": args.task, "pred": pred, "gold_line": gold_line, "score": score, **extra}
            if out_f:
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    finally:
        if out_f:
            out_f.close()

    mean = (aggregate / n) if n else 0.0
    print(f"task={args.task}")
    print(f"total={n}")
    print(f"sum_score={aggregate:.6f}")
    print(f"mean_score={mean:.6f}")
    if args.task == "tool":
        print(f"(tool uses Jaccard mean; same aggregation as raw_evaluate_my)")
    else:
        print(f"accuracy={mean:.6f}")
    if detail_path:
        print(f"detail_saved={detail_path}")


if __name__ == "__main__":
    main()
