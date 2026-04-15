#encoding: utf-8
"""
本模块为「指令 + 回答」拼接序列的 **部分监督**（仅对回答算 loss）提供数据变换。

**在流水线中的位置**
- **上游**：`tools/plm/mkiodata_llmdec.py` 或 **`tools/plm/mkiodata.py`**（已与 llmdec 对齐）把
  `map` 后的 `src.ids` / `tgt.ids` 打成 HDF5：
  - `src` 组：每个 batch key 对应矩阵 `(batch, seq_len)`，每行为 **指令 token + 回答 token**（右端 pad）。
  - `tgt` 组：同 key 对应 `(batch, 2)`，每行为 **`[lid, lgth]`**（与 `utils/fmt/plm/llmdec/dual.py` 一致：回答在整段中为半开 **`[lid, lgth)`**）。
- **本模块**：`PMaskDataConverter` 读入 `seq_batch`（即 HDF5 `src`）与 `seq_o`（即 HDF5 `tgt`），
  构造 **下一词预测** 的输入 `oi`、**仅回答区间** 上的 `pred_mask`、以及目标 token `ot`，供交叉熵等 loss 使用。

**`seq_o`（HDF5 `tgt`）的含义（必须与 `dual.py` / mkiodata* 一致）**
- `dual.batch_loader` 对每条样本记录 `lid = len(指令)`，`lgth = len(指令)+len(回答)`，回答在整段中的区间为 **半开区间** `[lid, lgth)`（0-based 下标）。
- **HDF5 直接存 `[lid, lgth]` 两整数**。`forward` 里 **`seq_o - 1`** 把「`seq_batch` 里答案 token 的下标范围」映到 **`oi`/`ind` 坐标**（长度为 `seql-1`）：预测步 `ind` 与目标 token 在 `seq_batch` 中的下标相差 1，故边界为 **`[lid-1, lgth-1)`**（在 `ind` 上的半开区间，见 `forward` 内 `_seq_ind`）。

**为何不能对整段 `seq_batch` 直接算 loss**
- 若对「指令 + 回答」全部位置算 next-token loss，模型会强烈拟合 **指令文本**，与「只学生成回答」的目标不符。
- `pred_mask` 把 loss 限制在 **回答对应的 next-token 位置**（在 `oi` 的坐标系下）。

**与其它训练脚本**
- `adv/train/plm/train_lora_qwen.py`：`data_converter = PMaskDataConverter(...)`，读 `train.h5`/`dev.h5`。
- 若 HDF5 由旧逻辑生成（`tgt` 误存为 **`[lid+1, lgth+1]`**），则 mask 会错位，需重新 `mkiodata*`。
"""

import torch
from torch import nn

from utils.torch.comp import torch_all_wodim

from cnfg.ihyp import cache_len_default


class PMaskDataConverter(nn.Module):
	"""
	将「整段 token 序列 + 回答区间」转为 (下一词输入, 预测掩码, 目标 token)。

	**输入**
	- `seq_batch`: `(batch, seq_len)`，每条样本为 **指令与回答拼接**（与 map 模板一致），已 pad。
	- `seq_o`: `(batch, 2)` 或 `(batch,)`（LM 特例）。常规为两列 **`[lid, lgth]`**，与 `dual` 相同：回答在 `seq_batch` 上为半开 **`[lid, lgth)`**；`forward` 内 **`seq_o - 1`** 得到 **`ind` 坐标系**上的半开边界（预测下一词用的列下标）。
	- `seq_o_sub_len`: 可选；若提供，内部在减 1 的基础上再叠加偏移（如固定 prefix 长度），用于验证时带前缀。

	**输出**
	- `oi`: `(batch, seq_len-1)`，**上一词为输入** 的左移序列（与自回归训练一致）。
	- `pred_mask`: 布尔或 None。非 None 时与 `ot` 行数一致，表示在 **展平后的有效预测位置** 上哪些参与 loss。
	- `ot`: 由 `seq_batch` 右移一位后按 `pred_mask` **筛选** 得到的 **目标 token**（仅回答段对应的 next-token）。

	**内部坐标（与注释中的 a/b/c 示例一致）**
	- 设 `seq_batch` 一行长度为 `seql`，则 `oi`/`ot` 对应长度为 `lo = seql-1`。
	- `ind` 为 `0 .. lo-1`，对应 `oi` 中「预测下一 token」的位置下标。
	"""

	def __init__(self, xseql=cache_len_default, dtype=torch.int32, device=None, **kwargs):

		super(PMaskDataConverter, self).__init__()
		self.xseql = xseql
		self.register_buffer("ind", torch.arange(xseql, dtype=dtype, device=device), persistent=False)

	# ----- 下为「单条样本、长度 12」的示意（字母仅作占位 token；与 forward 中 oi/ot/ind 对齐）-----
	# seq_batch 一行：自回归输入侧可见的整段 token（指令+回答+pad），长度 = seql = 12。
	#a b c d j k l n o p x y (seq_batch)
	#
	# tgt（下一词目标）：与 oi 同长 lo=seql-1=11；每列是「在上一位置预测出的 token」，
	# 即 seq_batch 右移一位，对应 oi[t] -> 预测 tgt[t]=seq_batch[t+1]。
	#b c d j k l n o p x y (tgt，与 oi 对齐的预测目标)
	#
	# ind：在长度 lo 上的位置下标 0..lo-1，与 oi 的列一一对应；pred_mask 在 ind 上选「哪些列参与 loss」。
	#0 1 2 3 4 5 6 7 8 9 10 (ind)
	#
	# len：若 batch 内多条样本长度不同（示意），可记每条样本有效长度；此处为 4 维 batch 的示例长度。
	#4 3 3 2 (len)
	#
	# seq_o：HDF5 的 tgt 组，每行两列 **[lid, lgth]**（与 dual：seq_batch 上半开 [lid,lgth)）；两样本示例。
	#4 7 10 12 (seq_o)  表示样本1 回答下标 [4,7)，样本2 [10,12)
	#
	# seq_ind = seq_o - 1：映到 **ind**（oi 列下标）上半开区间 [_sid, _eid)；若有 prefix 再减 seq_o_sub_len。
	#3 6 9 11 (seq_ind)

	def forward(self, seq_batch, seq_o, seq_o_sub_len=None, **kwargs):

		_bsize, _seql = seq_batch.size()
		lo = _seql - 1
		# 自回归：输入为 t 时刻及之前，预测 t+1；故与标签差一位对齐
		oi, ot = seq_batch.narrow(1, 0, lo), seq_batch.narrow(1, 1, lo)
		_is_lm = seq_o.dim() == 1
		# 整段 LM 且无 padding 特例：seq_o 全等于 seql 时不做局部 mask（由外部保证）
		if _is_lm and torch_all_wodim(seq_o.eq(_seql)).item():
			pred_mask = None
		else:
			# `ind`：在长度 lo 上 0..lo-1，与 oi 的列下标一一对应
			_ind = self.ind.narrow(0, 0, lo) if lo <= self.xseql else torch.arange(lo, dtype=self.ind.dtype, device=self.ind.device)
			_ = 1
			if seq_o_sub_len is not None:
				_ += seq_o_sub_len
			# 将 seq_batch 上的 [lid,lgth) 端点转为与 `ind`（预测步下标）同口径的半开边界
			_seq_ind = seq_o.to(seq_batch.device, non_blocking=True) - _
			if _is_lm:
				pred_mask = _ind.lt(_seq_ind.unsqueeze(-1))
			elif seq_o.size(-1) == 2:
				_sid, _eid = _seq_ind.view(_bsize, 1, 2).unbind(-1)
				# 在 `ind` 上保留 [_sid, _eid) 内的位置 —— 对应「回答」上的 next-token 预测
				pred_mask = _ind.ge(_sid)
				pred_mask &= _ind.lt(_eid)
			else:
				_sid, _eid = _seq_ind.view(_bsize, -1, 1, 2).unbind(-1)
				pred_mask = _ind.ge(_sid)
				pred_mask &= _ind.lt(_eid)
				pred_mask = pred_mask.to(torch.int32, non_blocking=True).sum(1).gt(0)
			ot = ot[pred_mask]

		return oi, pred_mask, ot
