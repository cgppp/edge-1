#encoding: utf-8
"""
生成 LoRA 训练用 HDF5：从双文件（指令 .ids + 回答 .ids）按行对齐读入，拼成「指令+回答」序列并写入 HDF5。

输出 HDF5 结构（与 utils/train/llm.py 的 PMaskDataConverter、utils/fmt/plm/llmdec/dual.py 约定一致）：
- src 组：key 为 "0","1",...；每 key 对应一个 (batch_size, seq_len) 的 token 矩阵，每条样本为「指令 token + 回答 token」经 pad 对齐。
- tgt 组：同 key；每 key 对应 (batch_size, 2)，每行为该条样本回答区间在拼接序列中的 [start+1, end+1]（1-based），供训练时只对回答算 loss。
- ndata：标量，写入的 batch 总数。

用法（命令行）:
  python tools/plm/mkiodata_llmdec.py <指令.ids> <回答.ids> <输出.h5> <minbsize>
例如:
  python tools/plm/mkiodata_llmdec.py $WKD/src.train.srt.ids $WKD/tgt.train.srt.ids $WKD/train.h5 1
"""
import sys
import os
from numpy import array as np_array, int32 as np_int32

from utils.fmt.plm.llmdec.dual import batch_padder
from utils.h5serial import h5File

from cnfg.ihyp import *
from cnfg.vocab.plm.qwen.v3 import pad_id


def _count_lines(fpath):
	"""
	统计文本文件行数，支持明文、.gz、.xz。

	参数:
		fpath (str): 文件路径。

	返回:
		int: 行数；若文件不存在返回 -1；若读取出错返回 -2；空文件返回 0。
	"""
	if not os.path.isfile(fpath):
		return -1
	n = 0
	try:
		if fpath.endswith(".xz"):
			import lzma
			with lzma.open(fpath, "rt", encoding="utf-8") as f:
				for _ in f:
					n += 1
					if n > 0 and n % 100000 == 0:
						pass  # 大文件只统计行数时可在此处 break 做近似
		elif fpath.endswith(".gz"):
			import gzip
			with gzip.open(fpath, "rt", encoding="utf-8") as f:
				for _ in f:
					n += 1
		else:
			with open(fpath, "r", encoding="utf-8") as f:
				for _ in f:
					n += 1
	except Exception:
		return -2
	return n

def handle(finput, ftarget, frs, minbsize=1, expand_for_mulgpu=True, bsize=max_sentences_gpu, maxpad=max_pad_tokens_sentence, maxpart=normal_tokens_vs_pad_tokens, maxtoken=max_tokens_gpu, pad_id=pad_id, **kwargs):
	"""
	从「指令 .ids」与「回答 .ids」双文件生成 LoRA 用 HDF5，写入 src/tgt 组及 ndata。

	参数:
		finput (str): 指令侧 .ids 文件路径，每行为空格分隔的 token id。
		ftarget (str): 回答侧 .ids 文件路径，与 finput 行一一对应。
		frs (str): 输出 HDF5 文件路径（如 train.h5、dev.h5）。
		minbsize (int): batch_padder 的 minbsize，至少凑满多少条再成一个 batch；命令行常传 1。
		expand_for_mulgpu (bool): 若 True，实际 bsize/maxtoken 使用 bsize*minbsize、maxtoken*minbsize（多卡时放大）。
		bsize (int): 期望每 batch 样本数，默认来自 cnfg.ihyp.max_sentences_gpu。
		maxpad (int): 允许的 padding 上限，默认 max_pad_tokens_sentence。
		maxpart (float): 控制 maxlen 的系数，默认 normal_tokens_vs_pad_tokens。
		maxtoken (int): 每 batch 的 token 数上限，默认 max_tokens_gpu。
		pad_id (int): padding 使用的 token id，默认 Qwen pad_id（<|endoftext|>）。
		**kwargs: 保留，未使用。

	无返回值。出错时打印到 stderr 并 sys.exit(1)。
	"""
	# ----- 检查输入文件存在性与行数 -----
	n_src = _count_lines(finput)   # 指令文件行数，-1 表示不存在，-2 表示读错，0 表示空
	n_tgt = _count_lines(ftarget)  # 回答文件行数
	if n_src == -1:
		print("Error: input file not found: %s" % finput, file=sys.stderr)
		sys.exit(1)
	if n_tgt == -1:
		print("Error: target file not found: %s" % ftarget, file=sys.stderr)
		sys.exit(1)
	if n_src == -2 or n_tgt == -2:
		print("Error: cannot read file (wrong format or permission): %s / %s" % (finput, ftarget), file=sys.stderr)
		sys.exit(1)
	if n_src == 0 or n_tgt == 0:
		print("Error: empty .ids file (0 lines). Run map first: tools/plm/map/qwen/v3.py ... -> .ids", file=sys.stderr)
		print("  src lines: %d, tgt lines: %d" % (n_src, n_tgt), file=sys.stderr)
		sys.exit(1)
	if n_src != n_tgt:
		print("Warning: src and tgt line counts differ (src=%d, tgt=%d), using minimum." % (n_src, n_tgt), file=sys.stderr)
	print("Input lines: src=%d, tgt=%d" % (n_src, n_tgt))

	# ----- 多卡时放大 bsize / maxtoken -----
	if expand_for_mulgpu:
		_bsize = bsize * minbsize
		_maxtoken = maxtoken * minbsize
	else:
		_bsize = bsize
		_maxtoken = maxtoken

	# ----- 打开 HDF5，创建 src / tgt 组，逐 batch 写入 -----
	with h5File(frs, "w", **h5_fileargs) as rsf:
		src_grp = rsf.create_group("src")   # 存每个 batch 的 token 矩阵
		tgt_grp = rsf.create_group("tgt")   # 存每个 batch 的 [start+1, end+1] 矩阵
		curd = 0                             # 当前已写入的 batch 编号，同时用作 dataset 的 key（"0","1",...）
		for i_d, ll in batch_padder(finput, ftarget, _bsize, maxpad, maxpart, _maxtoken, minbsize, pad_id=pad_id):
			# i_d: 当前 batch 的 token 矩阵 (batch_size, seq_len)，每条样本为「指令+回答」拼接并 pad 对齐
			# ll: 与 i_d 一一对应的 list of [lid, lgth]，lid=指令长度，lgth=总长度；回答区间 0-based 为 [lid, lgth)
			rid = np_array(i_d, dtype=np_int32)   # 转为 numpy 以写入 HDF5
			rtd = np_array([[lid + 1, lgth + 1] for lid, lgth in ll], dtype=np_int32)  # 转为 1-based [start+1, end+1]，与 PMaskDataConverter 约定一致
			wid = str(curd)   # dataset 的 key，如 "0", "1", ...
			src_grp.create_dataset(wid, data=rid, **h5datawargs)
			tgt_grp.create_dataset(wid, data=rtd, **h5datawargs)
			curd += 1
		rsf["ndata"] = np_array([curd], dtype=np_int32)   # 记录总 batch 数，训练脚本可据此遍历

	print("Number of batches: %d" % curd)
	if curd == 0:
		print("Error: no batch produced. Check that each line of .ids files is space-separated token IDs (integers).", file=sys.stderr)
		sys.exit(1)


if __name__ == "__main__":
	# 命令行: python mkiodata_llmdec.py <指令.ids> <回答.ids> <输出.h5> <minbsize>
	handle(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))
