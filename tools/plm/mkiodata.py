#encoding: utf-8
"""
将「指令 .ids + 回答 .ids」打成 HDF5（与 `mkiodata_llmdec.py`、`utils/train/llm.PMaskDataConverter` 约定一致）。

**为何不用 `utils.fmt.plm.roberta.dual` / `cnfg.vocab.plm.roberta.pad_id`**
- **Roberta.dual**：面向 **Encoder 侧** 或另一套「源/目标」矩阵格式，与「指令 token 与回答 token 拼成一条序列 + 记录回答区间」的 **llm decoder** 数据流不一致。
- **本仓库 LoRA+Qwen3**：应使用 **`utils.fmt.plm.llmdec.dual.batch_padder`**，与 `pubdatasets_to_srctgt` → `map` → 本脚本 → `train_lora_qwen.py` 整条链一致。
- **pad_id**：须与 **Qwen 词表** 中 padding（通常为 `<|endoftext|>`）一致，故用 **`cnfg.vocab.plm.qwen.v3.pad_id`**，不能用 Roberta 的 pad id。

---------------------------------------------------------------------------


1. **数据从哪来**  
   `utils/fmt/plm/llmdec/dual.py` 里 `batch_padder` 第二个返回值 **`ll`**：每条样本为 **`[lid, lgth]`**。  
   含义是：在「**指令 + 回答**」**拼成的一条** token 序列上，回答占 **0-based 半开区间** **`[lid, lgth)`**（`lid=len(指令)`，`lgth=len(指令)+len(回答)`）。

2. **训练端怎么吃**  
   `utils/train/llm.py` 里 **`PMaskDataConverter`** 从 HDF5 读出 `seq_o`，做 **`seq_o - 1`** 得到与 **`ind` 同口径** 的半开区间边界（`ind` 长度为 `seql-1`，对应「上一词预测下一词」的列下标 0…seql-2）。  
   **含义**：`dual` 给的 `[lid, lgth]` 是答案在 **`seq_batch` 里的下标范围**（半开 `[lid, lgth)`）；而 loss 作用在 **`oi`/`ind` 坐标**上，预测「下一 token」的位置比目标 token 在 `seq_batch` 里的下标 **少 1**，所以要 **`seq_o - 1`** 把 `[lid, lgth]` 映到 **`[lid-1, lgth-1)`**（在 `ind` 上的半开区间）。  
   **这与「长度 9 → 下标 0～8」是同一类坐标变换**：整段长 `seql`，`oi` 只有 `seql-1` 列，边界都要落到 **预测步** 的索引系里。

3. **写盘时直接存 `[lid, lgth]`（与 `llmdec.dual` 一致）**  
   **不要**再对磁盘上的 `tgt` 做 **`+1`**。`seq_o - 1` **不是**用来抵消写盘时的 `+1`，而是 **seq_batch 下标 → 预测步 `ind` 下标** 的转换。

4. **和 `utils/fmt/plm/` 下「通用 dual」（Roberta 封装）** 的区别  
   - **`utils/fmt/plm/dual.py`**：两份矩阵，没有「一条长序列上的回答起止」。  
   - **`llmdec/dual.py`**：一条拼接序列 + `[lid, lgth]` → 原样写入 HDF5 `tgt` 两列。

5. **一句话**  
   「**`tgt` 存 `dual` 的 `[lid, lgth]`；`forward` 里 `seq_o - 1` 映到 `ind` 上半开区间，只对回答段的 next-token 算 loss。**」

用法:
  python tools/plm/mkiodata.py <src.ids> <tgt.ids> <out.h5> <minbsize>
"""

import sys
from numpy import array as np_array, int32 as np_int32

from utils.fmt.plm.llmdec.dual import batch_padder
from utils.h5serial import h5File

from cnfg.ihyp import *
from cnfg.vocab.plm.qwen.v3 import pad_id


def handle(
	finput,
	ftarget,
	frs,
	minbsize=1,
	expand_for_mulgpu=True,
	bsize=max_sentences_gpu,
	maxpad=max_pad_tokens_sentence,
	maxpart=normal_tokens_vs_pad_tokens,
	maxtoken=max_tokens_gpu,
	minfreq=False,
	vsize=False,
	pad_id=pad_id,
	**kwargs,
):
	if expand_for_mulgpu:
		_bsize = bsize * minbsize
		_maxtoken = maxtoken * minbsize
	else:
		_bsize = bsize
		_maxtoken = maxtoken
	with h5File(frs, "w", **h5_fileargs) as rsf:
		src_grp = rsf.create_group("src")
		tgt_grp = rsf.create_group("tgt")
		curd = 0
		for i_d, ll in batch_padder(
			finput,
			ftarget,
			_bsize,
			maxpad,
			maxpart,
			_maxtoken,
			minbsize,
			pad_id=pad_id,
		):
			rid = np_array(i_d, dtype=np_int32)
			# 与 llmdec.dual 一致：回答在 seq_batch 上半开 [lid,lgth)；PMaskDataConverter 内 seq_o-1 映到 ind 坐标
			rtd = np_array([[lid, lgth] for lid, lgth in ll], dtype=np_int32)
			wid = str(curd)
			src_grp.create_dataset(wid, data=rid, **h5datawargs)
			tgt_grp.create_dataset(wid, data=rtd, **h5datawargs)
			curd += 1
		rsf["ndata"] = np_array([curd], dtype=np_int32)
	print("Number of batches: %d" % curd)


if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))
