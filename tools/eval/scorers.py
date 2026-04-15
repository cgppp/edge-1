# -*- coding: utf-8 -*-
"""
离线判分函数（与 meft/MEFT/raw_evaluate_my.py 各任务逐条逻辑对齐）。

本模块只做「给定模型输出字符串 + 标准答案结构」的纯计算，不加载模型、不读文件。
每个 score_* 返回 (标量分数, 附加信息 dict)，便于 CLI 汇总或写入 JSONL。
"""

from __future__ import annotations

import re
from typing import Any

__all__ = [
    "score_squad",
    "score_nq_v1",
    "score_nq_std",
    "score_medmcqa",
    "score_pubmedqa",
    "score_mmlu",
    "score_gsm8k",
    "score_tool",
]

# ---------------------------------------------------------------------------
# 抽取式 QA：SQuAD / TriviaQA
# 规则：任一候选答案在预测中「忽略大小写」子串命中即 1，否则 0。
# ---------------------------------------------------------------------------


def score_squad(pred: str, gold_answers: list[str]) -> tuple[int, dict[str, Any]]:
    """
    SQuAD / TriviaQA 类任务。

    对 gold_answers 中每个非空串，检查 pred.lower() 是否包含该串的小写形式；
    命中一个即判对。返回 0/1 与 std_ans 副本供日志使用。
    """
    low = pred.lower()
    hit = 0
    for ans in gold_answers:
        if not ans:
            continue
        if ans.lower() in low:
            hit = 1
            break
    return hit, {"hit": hit, "std_ans": gold_answers}


# ---------------------------------------------------------------------------
# Natural Questions：nq_v1 与 all_nq(nq_std)
# nq_v1：字段 short_answers；nq_std：字段 answer，判分规则相同。
# 规则：任一候选为 pred 的「大小写敏感」子串即 1（与 raw 中 `choice in output` 一致）。
# ---------------------------------------------------------------------------


def score_nq_v1(pred: str, short_answers: list[str]) -> tuple[int, dict[str, Any]]:
    """
    NQ v1：按 short_answers 列表从左到右，第一个出现在 pred 中的串记为命中。

    注意：此处为大小写敏感子串，与 score_squad 不同。
    """
    f_ans = None
    flag = 0
    for choice in short_answers:
        if choice in pred:
            flag = 1
            f_ans = choice
            break
    return flag, {"hit": flag, "model_ans": f_ans, "choices": short_answers}


def score_nq_std(pred: str, answers: list[str]) -> tuple[int, dict[str, Any]]:
    """
    all_nq 分支：标准答案字段名为 answer，判分与 nq_v1 相同，直接委托 score_nq_v1。
    """
    return score_nq_v1(pred, answers)


# ---------------------------------------------------------------------------
# 选择题：MedMCQA / PubMedQA / MMLU
# 均依赖模型输出中是否出现约定英文短语或选项格式（正则）。
# ---------------------------------------------------------------------------


def score_medmcqa(pred: str, std_ans: str) -> tuple[int, dict[str, Any]]:
    """
    MedMCQA：预测中需出现「option {答案}」后可选句号/空格/叹号（与 raw 一致）。

    std_ans 一般为单字母选项；正则中对 std_ans 做 re.escape，避免特殊字符误解释。
    """
    flag = 1 if re.search(f"option {re.escape(std_ans)}" + r"[. !]?", pred) else 0
    return flag, {"hit": flag, "std_ans": std_ans}


def score_pubmedqa(pred: str, std_ans: str) -> tuple[int, dict[str, Any]]:
    """
    PubMedQA：预测中需出现「my final answer is {标签}」及句末标点模式（大小写敏感标签）。
    """
    flag = 1 if re.search(f"my final answer is {re.escape(std_ans)}" + r"[. !]?", pred) else 0
    return flag, {"hit": flag, "std_ans": std_ans}


def score_mmlu(pred: str, std_ans: str) -> tuple[int, dict[str, Any]]:
    """
    MMLU：预测中需出现「My final choice is (X|x)」，X 为大写或小写选项字母。
    """
    pat = f"My final choice is ({re.escape(std_ans)}|{re.escape(std_ans.lower())})"
    flag = 1 if re.search(pat, pred) else 0
    return flag, {"hit": flag, "std_ans": std_ans}


# ---------------------------------------------------------------------------
# GSM8K / Math：从固定版式中解析数字并与标准数列逐项比对
# ---------------------------------------------------------------------------


def _gsm_float_list(s: str) -> list[float]:
    """将「逗号分隔数字串」拆成 float 列表，非法片段跳过（与 raw 中 get_float_answers 行为一致）。"""
    out: list[float] = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        try:
            out.append(float(x))
        except ValueError:
            pass
    return out


def score_gsm8k(pred: str, std_ans: list[float]) -> tuple[int, dict[str, Any]]:
    """
    GSM8K/Math：用正则抓取「若干数字 + 换行 + The answer is: 数字,数字,...」中的后半段，
    解析为 float 列表，与 std_ans 等长且逐项误差 ≤1e-6 则判 1。

    若正则不匹配或长度不一致，判 0；异常时保持 flag=0。
    """
    model_ans: list[float] = []
    flag = 0
    try:
        m = re.search(r"[0-9.]+\nThe answer is: ([0-9.,]+)", pred)
        if m:
            model_ans = _gsm_float_list(m.group(1))
        if len(model_ans) == len(std_ans):
            flag = 1
            for a, b in zip(model_ans, std_ans):
                if abs(a - b) > 1e-6:
                    flag = 0
                    break
    except (AttributeError, TypeError):
        pass
    return flag, {"hit": flag, "std_ans": std_ans, "model_ans": model_ans}


# ---------------------------------------------------------------------------
# ToolBench：对 API 描述集合做 Jaccard（连续分数，非 0/1）
# ---------------------------------------------------------------------------


def score_tool(pred: str, std_ans: list[str]) -> tuple[float, dict[str, Any]]:
    """
    ToolBench：将标准答案与预测中的多段 API 描述各自截断为「逗号前 3 段」、转小写，
    再对两个集合算 Jaccard = |交|/|并|；预测侧先去掉前缀「API MAIN INFO: 」，
    再按空行分段。无元素时分数为 0.0。

    返回 [0,1] 浮点分及规范化后的 std_ans 列表（供明细打印）。
    """
    gold = [",".join(ans.split(",")[:3]).lower() for ans in std_ans]
    low_output = pred.replace("API MAIN INFO: ", "")
    parts = low_output.split("\n\n")
    pred_apis = [",".join(api.split(",")[:3]).lower() for api in parts if api.strip()]
    merged = set(pred_apis).union(set(gold))
    inter = set(pred_apis).intersection(set(gold))
    score = (len(inter) / len(merged)) if merged else 0.0
    return score, {"score": score, "std_ans": gold}
