#encoding: utf-8
"""
将 pubdatasets 下 NQ / SQuAD / MetaMathQA / ToolBench 转为 LoRA 训练用的 src/tgt 文本。
字段语义与 `pubdatasets/DATASETS_STRUCTURE_SUMMARY.md` **§2.6 闭卷表** 一致。

**NQ、SQuAD 仅生成论文闭卷设定**（与 MEFT/ACL 叙述一致）：
  - NQ：`src` = 仅 `question.text`；`tgt` = `short_answers` 或 `yes_no_answer` → YES/NO
  - SQuAD：`src` = 仅 `question`；`tgt` = `answers.text[0]`
  输出目录：`pubdatasets_nq_closed`、`pubdatasets_squad_closed`

用法:
  python scripts/plm/llm/pubdatasets_to_srctgt.py --pubdatasets ~/pubdatasets --out cache/llm
  或分别指定 --dataset nq|squad|metamathqa|toolbench。
  NQ 可用 --workers 并行；只读 question/annotations 列。
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


def _squad_extract_answer(answers):
	"""SQuAD plain_text：answers 多为 struct { text: list[str], answer_start: list[int] }，与 DATASETS_STRUCTURE_SUMMARY 2.5 一致。"""
	if answers is None:
		return ""
	if isinstance(answers, float) and answers != answers:
		return ""
	if hasattr(answers, "__len__") and len(answers) == 0:
		return ""
	# HuggingFace / pyarrow 常见：dict，text 与 answer_start 为等长列表
	if isinstance(answers, dict):
		texts = answers.get("text")
		if isinstance(texts, (list, tuple)) and texts:
			return _norm(texts[0])
		if isinstance(texts, str):
			return _norm(texts)
		# 少数实现：嵌套 list of dict
		return _first_answer(texts if texts is not None else answers)
	# 兼容 list[{"text": ...}]
	if isinstance(answers, list) and answers:
		return _first_answer(answers)
	return _norm(answers)


def _seq_to_list(x):
	"""把 pyarrow/numpy 的 array 转成 list，便于迭代。"""
	if x is None:
		return []
	try:
		import numpy as np
		if isinstance(x, np.ndarray):
			return x.tolist()
	except Exception:
		pass
	if hasattr(x, "tolist"):
		try:
			return list(x.tolist())
		except Exception:
			pass
	return list(x) if isinstance(x, (list, tuple)) else [x]


def _nq_coerce_text_field(text):
	"""NQ 里 short_answers[].text 可能是 str，也可能是 numpy 单元素数组。"""
	if text is None:
		return ""
	if isinstance(text, str):
		return _norm(text)
	if isinstance(text, (bytes, bytearray)):
		return _norm(text.decode("utf-8", errors="replace"))
	if hasattr(text, "item"):
		try:
			return _norm(text.item())
		except Exception:
			pass
	if hasattr(text, "__iter__") and not isinstance(text, dict):
		try:
			if isinstance(text, (list, tuple)) and len(text) == 0:
				return ""
			for x in text:
				t = _nq_coerce_text_field(x)
				if t:
					return t
		except Exception:
			pass
		# 空 list/ndarray 等勿落到 str([])=="[]" 被当成有效短答
		if isinstance(text, (list, tuple)) and len(text) == 0:
			return ""
		try:
			import numpy as np
			if isinstance(text, np.ndarray) and text.size == 0:
				return ""
		except Exception:
			pass
	return _norm(text)


def _nq_yes_no_scalar(yn):
	if yn is None:
		return ""
	if hasattr(yn, "item"):
		try:
			yn = yn.item()
		except Exception:
			pass
	if isinstance(yn, (list, tuple)) and yn:
		yn = yn[0]
	if isinstance(yn, str):
		s = yn.strip().upper()
	else:
		s = str(yn).strip().upper()
	if s in ("1", "YES", "TRUE"):
		return "YES"
	if s in ("0", "NO", "FALSE"):
		return "NO"
	try:
		v = int(float(yn))
		if v == 1:
			return "YES"
		if v == 0:
			return "NO"
	except (TypeError, ValueError):
		pass
	return ""


def _nq_short_answer_from_annotations(ann):
	"""遍历 annotations 中的 short_answers[*].text，取首个非空。

	兼容两种落地格式：
	1) list[dict]：HuggingFace datasets / 部分读法，每项含 short_answers 列表。
	2) dict[str, ndarray]：pandas.read_parquet 对 struct-list 展成「列名 -> 数组」。
	"""
	if ann is None:
		return ""
	# 格式 A：pandas.read_parquet 将 struct-list 展成 dict，字段值为 ndarray（与本地 NQ parquet 一致）
	if isinstance(ann, dict) and "short_answers" in ann:
		for sa in _seq_to_list(ann.get("short_answers")):
			if isinstance(sa, dict):
				t = _nq_coerce_text_field(sa.get("text"))
				if t:
					return t
		return ""
	# 格式 B：list[annotator_dict]（HuggingFace / 其它读法）
	if isinstance(ann, list):
		for item in ann:
			if not isinstance(item, dict):
				continue
			short = item.get("short_answers")
			if not isinstance(short, list):
				# 也可能是 ndarray
				for sa in _seq_to_list(short):
					if not isinstance(sa, dict):
						continue
					t = _nq_coerce_text_field(sa.get("text"))
					if t:
						return t
				continue
			for sa in short:
				if isinstance(sa, dict):
					t = _nq_coerce_text_field(sa.get("text"))
				else:
					t = _norm(sa)
				if t:
					return t
	return ""


def _nq_yes_no_from_annotations(ann):
	"""无短答案时，若有 yes_no 标注则返回 YES/NO 字符串。"""
	if ann is None:
		return ""
	if isinstance(ann, dict) and "yes_no_answer" in ann:
		for yn in _seq_to_list(ann.get("yes_no_answer")):
			out = _nq_yes_no_scalar(yn)
			if out:
				return out
		return ""
	if isinstance(ann, list):
		for item in ann:
			if not isinstance(item, dict):
				continue
			yn = item.get("yes_no_answer")
			out = _nq_yes_no_scalar(yn)
			if out:
				return out
	return ""


def _nq_question_text(q_raw):
	if isinstance(q_raw, dict) and "text" in q_raw:
		return _norm(q_raw["text"])
	return _norm(q_raw)


def _nq_row_closed(q_raw, ann):
	"""MEFT/论文 closed-book NQ：src = 仅 question.text；tgt = 短答或判断题 YES/NO。

	判断题常无 `short_answers`，仅有 `yes_no_answer`（见 NQ 官方 schema）；故先取首个非空
	short answer，再回退到 `yes_no_answer` → 字符串 ``YES`` / ``NO``（与 `_nq_yes_no_scalar` 一致）。
	"""
	q = _nq_question_text(q_raw)
	if not q:
		return "", ""
	tgt = _nq_short_answer_from_annotations(ann)
	if not tgt:
		tgt = _nq_yes_no_from_annotations(ann)
	if not tgt:
		return "", ""
	return q, tgt


def load_metamathqa(base_dir, max_train=None, max_dev=None):
	"""MetaMathQA：`datasets` 的 save_to_disk 目录（含 `dataset_dict.json` 与 `train/`）。

	官方仅有 **train** split；本函数将尾部若干条划为 **dev**（与 MEFT 等用 MetaMathQA 作数学增强训练、无单独 test 的设定一致）。
	字段：`query` → src，`response` → tgt（见 `DATASETS_STRUCTURE_SUMMARY.md` §2.1）。
	"""
	try:
		from datasets import load_from_disk
	except ImportError as e:
		raise ImportError("加载 MetaMathQA 需要安装: pip install datasets") from e
	path = os.path.join(base_dir, "metamathqa")
	if not os.path.isdir(path) or not os.path.isfile(os.path.join(path, "dataset_dict.json")):
		return [], [], [], []
	try:
		d = load_from_disk(path)
	except Exception:
		return [], [], [], []
	if hasattr(d, "keys") and "train" in d:
		train = d["train"]
	else:
		train = d
	n = len(train)
	if n == 0:
		return [], [], [], []
	if "query" not in train.column_names or "response" not in train.column_names:
		return [], [], [], []
	# 尾部留出验证集（默认最多 5000 或约 1%）
	dev_n = max_dev if max_dev is not None else min(5000, max(1, n // 100))
	dev_n = min(dev_n, max(1, n - 1))
	train_cap = n - dev_n
	if max_train is not None:
		train_cap = min(train_cap, max_train)
	rows = train.select(list(range(train_cap)))
	src_tr = [_norm(q) for q in rows["query"]]
	tgt_tr = [_norm(r) for r in rows["response"]]
	dev_slice = train.select(list(range(n - dev_n, n)))
	src_dv = [_norm(q) for q in dev_slice["query"]]
	tgt_dv = [_norm(r) for r in dev_slice["response"]]
	return src_tr, tgt_tr, src_dv, tgt_dv

def load_squad(plain_dir, split="train", max_samples=None):
	"""SQuAD plain_text：仅论文 closed-book — `question`→src，`answers.text[0]`→tgt。"""
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
	# plain_text：id, title, context, question, answers{...}（见 DATASETS_STRUCTURE_SUMMARY §2.5 / §2.6）
	q = df["question"].astype(str) if "question" in df.columns else [""] * len(df)
	src = [_norm(qq) for qq in q]
	if "answers" in df.columns:
		tgt = [_squad_extract_answer(a) for a in df["answers"]]
	elif "answer" in df.columns:
		tgt = [_norm(a) for a in df["answer"]]
	else:
		tgt = [""] * len(src)
	return src, tgt

def _nq_read_one_file(fpath, columns=None, max_samples=None):
	"""读单个 NQ parquet；closed-book 仅需 question / annotations。"""
	import pandas as pd
	try:
		df = pd.read_parquet(fpath, columns=columns)
	except Exception:
		return [], []
	q_col = df["question"]
	ann_col = df["annotations"]
	n = min(len(df), max_samples) if max_samples else len(df)
	src, tgt = [], []
	for i in range(n):
		s, a = _nq_row_closed(q_col.iloc[i], ann_col.iloc[i])
		src.append(s)
		tgt.append(a)
	return src, tgt

def load_nq(base_dir, split="train", max_samples=None, workers=1):
	import pandas as pd
	if split == "train":
		pat = os.path.join(base_dir, "natural_questions", "default", "train-*.parquet")
	else:
		# 验证集优先 default/validation-*（与 HuggingFace 数据卡、DATASETS_STRUCTURE_SUMMARY 一致）
		pat = os.path.join(base_dir, "natural_questions", "default", "validation-*.parquet")
		if not glob.glob(pat):
			d = os.path.join(base_dir, "natural_questions", "dev")
			if os.path.isdir(d):
				pat = os.path.join(d, "validation-*.parquet")
				if not glob.glob(pat):
					pat = os.path.join(d, "*.parquet")
			else:
				pat = os.path.join(base_dir, "natural_questions", "validation-*.parquet")
	files = sorted(glob.glob(pat))
	if not files:
		return [], []
	columns = ["question", "annotations"]

	src_all, tgt_all = [], []
	if workers is None or workers <= 1:
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
			# parquet 读入后可能为 NaN / NA
			if conv is None:
				pass
			elif isinstance(conv, float) and conv != conv:
				conv = None
			rid = row.get("id")
			user_val = ""
			assistant_val = ""
			if conv is None:
				user_val, assistant_val = "", ""
			elif isinstance(conv, dict) and "from" in conv and "value" in conv:
				# HuggingFace 格式: {'from': array([...]), 'value': array([...])}（见 DATASETS_STRUCTURE_SUMMARY 2.5）
				roles = _seq_to_list(conv["from"])
				values = _seq_to_list(conv["value"])
				try:
					for role, val in zip_longest(roles, values, fillvalue=""):
						role = str(role).lower() if role is not None else ""
						val = _norm(val) if val is not None else ""
						if role == "user" and val:
							user_val = val
						elif role == "assistant" and val:
							assistant_val = val
				except (TypeError, AttributeError):
					pass
			elif isinstance(conv, str):
				try:
					conv = json.loads(conv) if conv.strip() else []
				except Exception:
					conv = []
				for msg in (conv if isinstance(conv, list) else []):
					if isinstance(msg, dict):
						role = msg.get("from", msg.get("role", ""))
						val = msg.get("value", msg.get("content", ""))
						v = _norm(val)
						if str(role).lower() == "user" and v:
							user_val = v
						elif str(role).lower() == "assistant" and v:
							assistant_val = v
			elif isinstance(conv, list):
				for msg in conv:
					if isinstance(msg, dict):
						role = msg.get("from", msg.get("role", ""))
						val = msg.get("value", msg.get("content", ""))
						v = _norm(val)
						if str(role).lower() == "user" and v:
							user_val = v
						elif str(role).lower() == "assistant" and v:
							assistant_val = v
			# 部分样本 user 在首轮为空：用 id 作为任务描述作指令侧（与文档说明一致）
			if not user_val and rid is not None:
				rid_s = _norm(rid)
				if rid_s:
					user_val = rid_s
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
	ap.add_argument("--dataset", choices=("nq", "squad", "metamathqa", "toolbench", "all"), default="all")
	ap.add_argument("--max-train", type=int, default=None, help="Cap train samples per dataset")
	ap.add_argument("--max-dev", type=int, default=None, help="Cap dev samples per dataset")
	ap.add_argument("--workers", type=int, default=1, help="NQ 并行读 parquet 的线程数，建议 4~16")
	args = ap.parse_args()
	root = args.pubdatasets
	out_root = args.out

	datasets = ["nq", "squad", "metamathqa", "toolbench"] if args.dataset == "all" else [args.dataset]
	for ds in datasets:
		if ds == "metamathqa":
			src_train, tgt_train, src_dev, tgt_dev = load_metamathqa(
				root, max_train=args.max_train, max_dev=args.max_dev
			)
			_nq_subs = [("metamathqa", src_train, tgt_train, src_dev, tgt_dev)]
		elif ds == "squad":
			src_train, tgt_train = load_squad(os.path.join(root, "squad"), "train", args.max_train)
			src_dev, tgt_dev = load_squad(os.path.join(root, "squad"), "validation", args.max_dev)
			_nq_subs = [("squad_closed", src_train, tgt_train, src_dev, tgt_dev)]
		elif ds == "nq":
			base = os.path.join(root)
			st_tr, tg_tr = load_nq(base, "train", args.max_train, workers=args.workers)
			st_dv, tg_dv = load_nq(base, "validation", args.max_dev, workers=args.workers)
			_nq_subs = [("nq_closed", st_tr, tg_tr, st_dv, tg_dv)]
		elif ds == "toolbench":
			base = os.path.join(root, "toolbench-v1")
			src_train, tgt_train = load_toolbench(base, "train", args.max_train)
			src_dev, tgt_dev = load_toolbench(base, "validation", args.max_dev)
			_nq_subs = [("toolbench", src_train, tgt_train, src_dev, tgt_dev)]
		else:
			continue
		# 过滤空回答
		def filter_empty(sr, tg):
			return [(s, t) for s, t in zip(sr, tg) if t]
		for sub_name, src_train, tgt_train, src_dev, tgt_dev in _nq_subs:
			train_pairs = filter_empty(src_train, tgt_train)
			dev_pairs = filter_empty(src_dev, tgt_dev)
			src_train, tgt_train = [p[0] for p in train_pairs], [p[1] for p in train_pairs]
			src_dev, tgt_dev = [p[0] for p in dev_pairs], [p[1] for p in dev_pairs]
			wkd = os.path.join(out_root, "pubdatasets_%s" % sub_name)
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
