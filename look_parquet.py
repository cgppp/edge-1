#!/usr/bin/env python3
#encoding: utf-8
"""
查看 pubdatasets 下四个数据集的：分片数、总行数、列数、列名、抽样行。
与 `pubdatasets/DATASETS_STRUCTURE_SUMMARY.md` 中的字段约定对照。

依赖：pyarrow；MetaMathQA 另需 `datasets`（`pip install datasets`）。

示例：
  python look_parquet.py --root /home/gpchen/pubdatasets --dataset all
  python look_parquet.py --dataset metamathqa
  python look_parquet.py --dataset natural_questions --config default --split train
  python look_parquet.py --dataset squad --split validation
  python look_parquet.py --dataset toolbench --config default --split train
  python look_parquet.py --dataset toolbench --config benchmark --split g1_instruction
"""
from __future__ import annotations

import argparse
import glob
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pyarrow.parquet as pq

# 与 DATASETS_STRUCTURE_SUMMARY.md §2 对齐的「期望列」（多出的列会标为 extra）
EXPECTED_COLUMNS: Dict[str, Sequence[str]] = {
    "metamathqa": ("query", "response", "type", "original_question"),
    "natural_questions": ("id", "document", "question", "long_answer_candidates", "annotations"),
    "squad": ("id", "title", "context", "question", "answers"),
    "toolbench_data": ("id", "conversations"),
    "toolbench_benchmark": ("query_id", "query", "api_list", "relevant_apis"),
}


def _len_flex(x: Any) -> Any:
    if x is None:
        return 0
    if isinstance(x, list):
        return len(x)
    if isinstance(x, dict):
        return len(x)
    try:
        return len(x)  # numpy ndarray, pyarrow list
    except Exception:
        return "?"


def _trunc(s: Any, n: int = 200) -> str:
    t = repr(s) if not isinstance(s, str) else s
    t = t.replace("\n", " ")
    return t if len(t) <= n else t[: n - 3] + "..."


def _extract_question_text(v: Any) -> str:
    if isinstance(v, dict):
        return str(v.get("text", ""))
    return str(v)


def _extract_first_short_answer(v: Any) -> str:
    if not isinstance(v, list) or not v:
        return ""
    first_ann = v[0]
    if not isinstance(first_ann, dict):
        return ""
    sas = first_ann.get("short_answers")
    if not isinstance(sas, list) or not sas:
        return ""
    first_sa = sas[0]
    if isinstance(first_sa, dict):
        return str(first_sa.get("text", ""))
    return str(first_sa)


def _extract_doc_title(v: Any) -> str:
    if isinstance(v, dict):
        return str(v.get("title", ""))
    return ""


def _extract_toolbench_user_assistant(v: Any) -> Tuple[str, str]:
    user_text, assistant_text = "", ""
    if isinstance(v, dict):
        roles = v.get("from")
        values = v.get("value")
        if isinstance(roles, list) and isinstance(values, list):
            for role, content in zip(roles, values):
                role = str(role).lower()
                content = str(content)
                if role == "user" and not user_text:
                    user_text = content
                if role == "assistant" and not assistant_text:
                    assistant_text = content
        return user_text, assistant_text
    if isinstance(v, list):
        for msg in v:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("from", msg.get("role", ""))).lower()
            content = str(msg.get("value", msg.get("content", "")))
            if role == "user" and not user_text:
                user_text = content
            if role == "assistant" and not assistant_text:
                assistant_text = content
    return user_text, assistant_text


def _schema_columns_parquet(path: str) -> List[str]:
    return list(pq.ParquetFile(path).schema_arrow.names)


def _count_rows_parquet(path: str) -> int:
    return int(pq.ParquetFile(path).metadata.num_rows)


def _sample_rows_parquet(path: str, n: int) -> List[Dict[str, Any]]:
    table = pq.read_table(path, memory_map=True)
    n = max(1, min(n, table.num_rows))
    return table.slice(0, n).to_pylist()


def inspect_metamathqa(root: str, head: int) -> None:
    print("=" * 72)
    print("dataset: metamathqa  (HuggingFace datasets save_to_disk / Arrow)")
    print("path:", os.path.join(root, "metamathqa"))
    try:
        from datasets import load_from_disk
    except ImportError:
        print("SKIP: 需要 pip install datasets")
        return
    path = os.path.join(root, "metamathqa")
    if not os.path.isdir(path) or not os.path.isfile(os.path.join(path, "dataset_dict.json")):
        print("SKIP: 未找到 dataset_dict.json，目录不存在或非 save_to_disk 格式")
        return
    try:
        d = load_from_disk(path)
    except Exception as e:
        print("FAIL load_from_disk:", e)
        return
    split_names = list(d.keys()) if hasattr(d, "keys") else ["train"]
    expected = EXPECTED_COLUMNS["metamathqa"]
    for sn in split_names:
        split = d[sn] if hasattr(d, "keys") else d
        n_rows = len(split)
        cols = list(split.column_names)
        print("-" * 72)
        print("split:", sn)
        print("shards: 1 (single Dataset)")
        print("total rows:", n_rows)
        print("columns count:", len(cols))
        print("columns:", cols)
        exp = set(expected)
        got = set(cols)
        missing = sorted(exp - got)
        extra = sorted(got - exp)
        print("expected columns present:", "YES" if not missing else "NO")
        if missing:
            print("  missing:", missing)
        if extra:
            print("  extra:", extra)
        # 抽样
        k = max(1, min(head, n_rows))
        for i in range(k):
            row = split[i]
            print(f"--- sample row {i} ---")
            print("  query:", _trunc(row.get("query", ""), 220))
            print("  response:", _trunc(row.get("response", ""), 220))
            if "type" in row:
                print("  type:", _trunc(row.get("type"), 120))
            if "original_question" in row:
                print("  original_question:", _trunc(row.get("original_question"), 160))
        ok = n_rows > 0 and "query" in got and "response" in got
        print("basic format check:", "PASS" if ok else "FAIL")


def inspect_parquet_bundle(
    title: str,
    files: List[str],
    expected_key: str,
    head: int,
    preview_fn: Optional[Any] = None,
) -> None:
    print("=" * 72)
    print(title)
    print("matched files:", len(files))
    if not files:
        print("SKIP: no parquet files")
        return
    print("first file:", files[0])

    shard_rows = [(_count_rows_parquet(fp), fp) for fp in files]
    total_rows = sum(r for r, _ in shard_rows)
    base_cols = _schema_columns_parquet(files[0])
    base_set = set(base_cols)

    print("total rows (all shards):", total_rows)
    print("columns count:", len(base_cols))
    print("columns:", base_cols)
    print("rows in first shard:", shard_rows[0][0])

    inconsistent = [fp for fp in files[1:] if _schema_columns_parquet(fp) != base_cols]
    print("schema consistent across shards:", "YES" if not inconsistent else "NO")
    if inconsistent:
        print("  inconsistent examples:", inconsistent[:3])

    expected = set(EXPECTED_COLUMNS[expected_key])
    missing = sorted(expected - base_set)
    extra = sorted(base_set - expected)
    print("expected columns present:", "YES" if not missing else "NO")
    if missing:
        print("  missing expected:", missing)
    if extra:
        print("  extra columns:", extra)

    rows = _sample_rows_parquet(files[0], head)
    print(f"--- sample rows (n={len(rows)}) from first shard ---")
    for i, row in enumerate(rows):
        print(f"[row {i}] id={row.get('id', row.get('query_id'))}")
        if preview_fn:
            preview_fn(i, row)

    min_cols = max(2, len(expected) - 1)
    valid = (not missing) and len(base_cols) >= min_cols and total_rows > 0
    print("basic format check:", "PASS" if valid else "FAIL")


def _preview_nq(i: int, row: Dict[str, Any]) -> None:
    q_text = _extract_question_text(row.get("question"))
    short_ans = _extract_first_short_answer(row.get("annotations"))
    title = _extract_doc_title(row.get("document"))
    lac = row.get("long_answer_candidates")
    ann = row.get("annotations")
    print(f"  question.text={_trunc(q_text, 180)}")
    print(f"  document.title={_trunc(title, 120)}")
    print(f"  first_short_answer={_trunc(short_ans, 120)}")
    print(f"  long_answer_candidates count: {_len_flex(lac)}, annotations count: {_len_flex(ann)}")


def _preview_squad(i: int, row: Dict[str, Any]) -> None:
    print(f"  title={_trunc(row.get('title', ''), 120)}")
    print(f"  question={_trunc(row.get('question', ''), 180)}")
    print(f"  context={_trunc(row.get('context', ''), 180)}")
    print(f"  answers={row.get('answers')}")


def _preview_toolbench_data(i: int, row: Dict[str, Any]) -> None:
    conv = row.get("conversations")
    u, a = _extract_toolbench_user_assistant(conv)
    if isinstance(conv, list):
        clen = len(conv)
    elif isinstance(conv, dict):
        clen = len(conv.get("from", [])) if "from" in conv else len(conv)
    else:
        clen = 0
    print(f"  conversations_len={clen}")
    print(f"  first_user={_trunc(u, 180)}")
    print(f"  first_assistant={_trunc(a, 180)}")


def _preview_toolbench_bench(i: int, row: Dict[str, Any]) -> None:
    print(f"  query_id={row.get('query_id')}")
    print(f"  query={_trunc(row.get('query', ''), 180)}")
    print(f"  api_list={_trunc(row.get('api_list', ''), 120)}")
    print(f"  relevant_apis={_trunc(row.get('relevant_apis', ''), 120)}")


def resolve_nq_pattern(root: str, config: str, split: str) -> str:
    if config == "dev" and split == "train":
        raise SystemExit("natural_questions config=dev 无 train，请用 --split validation")
    return os.path.join(root, "natural_questions", config, f"{split}-*.parquet")


def resolve_squad_pattern(root: str, config: str, split: str) -> str:
    cfg = config or "plain_text"
    name = "train" if split == "train" else "validation"
    return os.path.join(root, "squad", cfg, f"{name}-*.parquet")


def resolve_toolbench_pattern(root: str, config: str, split: str) -> str:
    if config == "default":
        return os.path.join(root, "toolbench-v1", "data", f"{split}-*.parquet")
    # benchmark: g1_instruction-*.parquet 等
    return os.path.join(root, "toolbench-v1", "benchmark", f"{split}-*.parquet")


def run_one(
    root: str,
    dataset: str,
    config: str,
    split: str,
    head: int,
) -> None:
    if dataset == "metamathqa":
        inspect_metamathqa(root, head)
        return

    if dataset == "natural_questions":
        pat = resolve_nq_pattern(root, config, split)
        title = f"dataset: natural_questions  config={config}  split={split}\npattern: {pat}"
        files = sorted(glob.glob(pat))
        inspect_parquet_bundle(title, files, "natural_questions", head, _preview_nq)
        return

    if dataset == "squad":
        pat = resolve_squad_pattern(root, config, split)
        title = f"dataset: squad  config={config or 'plain_text'}  split={split}\npattern: {pat}"
        files = sorted(glob.glob(pat))
        inspect_parquet_bundle(title, files, "squad", head, _preview_squad)
        return

    if dataset == "toolbench":
        pat = resolve_toolbench_pattern(root, config, split)
        exp_key = "toolbench_data" if config == "default" else "toolbench_benchmark"
        prev = _preview_toolbench_data if config == "default" else _preview_toolbench_bench
        title = f"dataset: toolbench  config={config}  split={split}\npattern: {pat}"
        files = sorted(glob.glob(pat))
        inspect_parquet_bundle(title, files, exp_key, head, prev)
        return

    raise SystemExit("unknown dataset: %s" % dataset)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="查看 pubdatasets 四数据集：行数、列数、列名、抽样（与 DATASETS_STRUCTURE_SUMMARY 对照）。"
    )
    parser.add_argument(
        "--dataset",
        choices=("all", "metamathqa", "natural_questions", "squad", "toolbench"),
        default="all",
        help="要检查的数据集；all=依次检查四个",
    )
    parser.add_argument(
        "--config",
        default="default",
        help="NQ: default|dev；SQuAD: plain_text（默认）；ToolBench: default|benchmark",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="parquet 文件名前缀：train / validation；ToolBench benchmark 时为 g1_instruction、g2_category 等",
    )
    parser.add_argument("--root", default=os.path.expanduser("~/pubdatasets"), help="pubdatasets 根目录")
    parser.add_argument("--head", type=int, default=1, help="抽样行数")
    args = parser.parse_args()

    if args.dataset == "all":
        print("root:", os.path.abspath(args.root))
        print()
        inspect_metamathqa(args.root, args.head)
        print()
        run_one(args.root, "natural_questions", "default", "train", args.head)
        print()
        run_one(args.root, "squad", "plain_text", "train", args.head)
        print()
        run_one(args.root, "toolbench", "default", "train", args.head)
        print()
        run_one(args.root, "toolbench", "benchmark", "g1_instruction", args.head)
        print()
        print("Done (all).")
        return

    print("root:", os.path.abspath(args.root))
    run_one(args.root, args.dataset, args.config, args.split, args.head)


if __name__ == "__main__":
    main()
