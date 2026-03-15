#encoding: utf-8
"""
将 pubdatasets 下 NQ / SQuAD / GSM8K / ToolBench 转为 LoRA 训练用的 src/tgt 文本。
输出：每行一条样本，src = 指令（可含 context），tgt = 回答。
用法:
  python scripts/plm/llm/pubdatasets_to_srctgt.py --pubdatasets ~/pubdatasets --out cache/llm
  或分别指定 --dataset nq|squad|gsm8k|toolbench。
  NQ 较大时可用 --workers 8 并行读 parquet，并只读 question/annotations 列以加速。
"""
import os
import re
import argparse
import glob
from concurrent.futures import ThreadPoolExecutor
from itertools import zip_longest

def _norm(t):
	if t is None or (isinstance(t, float) and (t != t or t == float("inf"))):
		return ""
	s = str(t).strip()
	s = re.sub(r"\s+", " ", s)
	return s

def _first_answer(a):
	if a is None:
		return ""
	if isinstance(a, list):
		if not a:
			return ""
		first = a[0]
		# SQuAD: answers = [{"text": "...", "answer_start": n}]
		if isinstance(first, dict) and "text" in first:
			return _norm(first["text"])
		return _norm(first)
	return _norm(a)

def _nq_extract_qa(q_raw, ann, long_answer=None):
	"""从 NQ 一行中抽出 question 与 answer 字符串。"""
	if isinstance(q_raw, dict) and "text" in q_raw:
		q = _norm(q_raw["text"])
	else:
		q = _norm(q_raw)
	if isinstance(ann, list) and ann and isinstance(ann[0], dict):
		short = ann[0].get("short_answers")
		if isinstance(short, list) and short and isinstance(short[0], dict):
			ans = _norm(short[0].get("text", ""))
		else:
			ans = _first_answer(short)
	else:
		ans = _first_answer(ann)
	if not ans and long_answer is not None:
		ans = _norm(long_answer)[:500]
	return q, ans

def load_gsm8k(main_dir, split="train", max_samples=None):
	import pandas as pd
	pat = os.path.join(main_dir, "main", f"{split}-*.parquet")
	files = sorted(glob.glob(pat))
	if not files:
		return [], []
	dfs = [pd.read_parquet(f) for f in files]
	df = pd.concat(dfs, ignore_index=True)
	if max_samples:
		df = df.head(max_samples)
	# columns: question, answer
	src = [_norm(q) for q in df["question"]]
	tgt = [_norm(a) for a in df["answer"]]
	return src, tgt

def load_squad(plain_dir, split="train", max_samples=None):
	import pandas as pd
	name = "train" if split == "train" else "validation"
	pat = os.path.join(plain_dir, "plain_text", f"{name}-*.parquet")
	files = sorted(glob.glob(pat))
	if not files:
		return [], []
	dfs = [pd.read_parquet(f) for f in files]
	df = pd.concat(dfs, ignore_index=True)
	if max_samples:
		df = df.head(max_samples)
	# 常见列：context, question, answers 或 answer
	ctx = df["context"].astype(str) if "context" in df.columns else [""] * len(df)
	q = df["question"].astype(str) if "question" in df.columns else [""] * len(df)
	src = [_norm("Context: %s\nQuestion: %s" % (c, qq)) for c, qq in zip(ctx, q)]
	if "answers" in df.columns:
		tgt = [_first_answer(a) for a in df["answers"]]
	elif "answer" in df.columns:
		tgt = [_norm(a) for a in df["answer"]]
	else:
		tgt = [""] * len(src)
	return src, tgt

def _nq_read_one_file(fpath, columns=None, max_samples=None):
	"""读单个 NQ parquet，只取 question/annotations（及 long_answer 若有），返回 (src_list, tgt_list)。"""
	import pandas as pd
	try:
		df = pd.read_parquet(fpath, columns=columns)
	except Exception:
		return [], []
	q_col = df["question"]
	ann_col = df["annotations"]
	has_long = "long_answer" in df.columns
	n = min(len(df), max_samples) if max_samples else len(df)
	src, tgt = [], []
	for i in range(n):
		long_val = df["long_answer"].iloc[i] if has_long else None
		q, a = _nq_extract_qa(q_col.iloc[i], ann_col.iloc[i], long_val)
		src.append(q)
		tgt.append(a)
	return src, tgt

def load_nq(base_dir, split="train", max_samples=None, workers=1):
	import pandas as pd
	if split == "train":
		pat = os.path.join(base_dir, "natural_questions", "default", "train-*.parquet")
	else:
		d = os.path.join(base_dir, "natural_questions", "dev")
		if not os.path.isdir(d):
			d = os.path.join(base_dir, "natural_questions")
		pat = os.path.join(d, "*.parquet")
		if not glob.glob(pat):
			pat = os.path.join(d, "validation-*.parquet")
	files = sorted(glob.glob(pat))
	if not files:
		return [], []
	# 只读必要列，减小 I/O 与内存；用 schema 检测 long_answer 避免读入整文件
	try:
		import pyarrow.parquet as pq
		schema = pq.read_schema(files[0])
		columns = ["question", "annotations"]
		if "long_answer" in schema.names:
			columns.append("long_answer")
	except Exception:
		columns = ["question", "annotations"]

	src_all, tgt_all = [], []
	if workers is None or workers <= 1:
		# 顺序：按批读入，避免一次 concat 全部（内存爆炸）
		per_file_limit = (max_samples + len(files) - 1) // len(files) if max_samples else None
		collected = 0
		for f in files:
			if max_samples and collected >= max_samples:
				break
			limit = (max_samples - collected) if max_samples else None
			s, t = _nq_read_one_file(f, columns=columns, max_samples=limit)
			src_all.extend(s)
			tgt_all.extend(t)
			collected += len(s)
	else:
		# 并行：按文件顺序提交，保证输出顺序与文件顺序一致
		def _read_one(fpath):
			return _nq_read_one_file(fpath, columns=columns)
		with ThreadPoolExecutor(max_workers=workers) as ex:
			results = list(ex.map(_read_one, files))
		for s, t in results:
			src_all.extend(s)
			tgt_all.extend(t)
		if max_samples:
			src_all = src_all[:max_samples]
			tgt_all = tgt_all[:max_samples]
	return src_all, tgt_all

def load_toolbench(data_dir, split="train", max_samples=None):
	import pandas as pd
	import json
	name = "train" if split == "train" else "validation"
	pat = os.path.join(data_dir, "data", f"{name}-*.parquet")
	files = sorted(glob.glob(pat))
	if not files:
		return [], []
	dfs = [pd.read_parquet(f) for f in files]
	df = pd.concat(dfs, ignore_index=True)
	if max_samples:
		df = df.head(max_samples)
	# ToolBench: conversations 可能是 (1) dict with 'from'/'value' 数组 (HuggingFace)，或 (2) list of {from, value}
	if "conversations" in df.columns:
		src, tgt = [], []
		for _, row in df.iterrows():
			conv = row.get("conversations")
			user_val = ""
			assistant_val = ""
			if isinstance(conv, dict) and "from" in conv and "value" in conv:
				# HuggingFace 格式: {'from': array([...]), 'value': array([...])}
				roles = conv["from"]
				values = conv["value"]
				try:
					for role, val in zip_longest(roles, values, fillvalue=""):
						role = str(role).lower() if role is not None else ""
						val = _norm(val) if val is not None else ""
						if role == "user":
							user_val = val
						elif role == "assistant":
							assistant_val = val
				except (TypeError, AttributeError):
					pass
			elif isinstance(conv, str):
				conv = json.loads(conv) if conv else []
				for msg in (conv if isinstance(conv, list) else []):
					if isinstance(msg, dict):
						role = msg.get("from", msg.get("role", ""))
						val = msg.get("value", msg.get("content", ""))
						if str(role).lower() == "user":
							user_val = _norm(val)
						elif str(role).lower() == "assistant":
							assistant_val = _norm(val)
			elif isinstance(conv, list):
				for msg in conv:
					if isinstance(msg, dict):
						role = msg.get("from", msg.get("role", ""))
						val = msg.get("value", msg.get("content", ""))
						if str(role).lower() == "user":
							user_val = _norm(val)
						elif str(role).lower() == "assistant":
							assistant_val = _norm(val)
			src.append(user_val or " ")
			tgt.append(assistant_val or " ")
		return src, tgt
	in_col = "instruction" if "instruction" in df.columns else "query" if "query" in df.columns else "question"
	out_col = "output" if "output" in df.columns else "response" if "response" in df.columns else "answer"
	src = [_norm(row.get(in_col, "")) for _, row in df.iterrows()]
	tgt = [_norm(row.get(out_col, "")) for _, row in df.iterrows()]
	return src, tgt

def main():
	ap = argparse.ArgumentParser(description="Convert pubdatasets to src/tgt text for LoRA.")
	ap.add_argument("--pubdatasets", default=os.path.expanduser("~/pubdatasets"), help="Path to pubdatasets root")
	ap.add_argument("--out", default="cache/llm", help="Output root (e.g. cache/llm)")
	ap.add_argument("--dataset", choices=("nq", "squad", "gsm8k", "toolbench", "all"), default="all")
	ap.add_argument("--max-train", type=int, default=None, help="Cap train samples per dataset")
	ap.add_argument("--max-dev", type=int, default=None, help="Cap dev samples per dataset")
	ap.add_argument("--workers", type=int, default=1, help="NQ 并行读 parquet 的线程数，建议 4~16")
	args = ap.parse_args()
	root = args.pubdatasets
	out_root = args.out

	datasets = ["nq", "squad", "gsm8k", "toolbench"] if args.dataset == "all" else [args.dataset]
	for ds in datasets:
		if ds == "gsm8k":
			src_train, tgt_train = load_gsm8k(os.path.join(root, "gsm8k"), "train", args.max_train)
			src_dev, tgt_dev = load_gsm8k(os.path.join(root, "gsm8k"), "test", args.max_dev)
		elif ds == "squad":
			src_train, tgt_train = load_squad(os.path.join(root, "squad"), "train", args.max_train)
			src_dev, tgt_dev = load_squad(os.path.join(root, "squad"), "validation", args.max_dev)
		elif ds == "nq":
			base = os.path.join(root)
			src_train, tgt_train = load_nq(base, "train", args.max_train, workers=args.workers)
			src_dev, tgt_dev = load_nq(base, "validation", args.max_dev, workers=args.workers)
		elif ds == "toolbench":
			base = os.path.join(root, "toolbench-v1")
			src_train, tgt_train = load_toolbench(base, "train", args.max_train)
			src_dev, tgt_dev = load_toolbench(base, "validation", args.max_dev)
		else:
			continue
		# 过滤空回答
		def filter_empty(sr, tg):
			return [(s, t) for s, t in zip(sr, tg) if t]
		train_pairs = filter_empty(src_train, tgt_train)
		dev_pairs = filter_empty(src_dev, tgt_dev)
		src_train, tgt_train = [p[0] for p in train_pairs], [p[1] for p in train_pairs]
		src_dev, tgt_dev = [p[0] for p in dev_pairs], [p[1] for p in dev_pairs]
		wkd = os.path.join(out_root, "pubdatasets_%s" % ds)
		os.makedirs(wkd, exist_ok=True)
		def write_txt(path, lines):
			with open(path, "w", encoding="utf-8") as f:
				for line in lines:
					f.write(line.replace("\n", " ").strip() + "\n")
		write_txt(os.path.join(wkd, "src.train.txt"), src_train)
		write_txt(os.path.join(wkd, "tgt.train.txt"), tgt_train)
		write_txt(os.path.join(wkd, "src.dev.txt"), src_dev)
		write_txt(os.path.join(wkd, "tgt.dev.txt"), tgt_dev)
		print("Wrote %s: train %d, dev %d" % (wkd, len(src_train), len(src_dev)))

if __name__ == "__main__":
	main()
