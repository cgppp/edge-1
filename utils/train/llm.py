#encoding: utf-8
"""
LLM 训练用数据转换：只对「指定区间」内的 token 算 loss（如指令+回答中只对回答算 loss）。

本模块定义 PMaskDataConverter：
- 输入：整段 token 序列 seq_batch（如「指令 + 回答」拼接），以及 seq_o（区间信息）。
- 输出：oi（输入 token）、pred_mask（在哪些位置算 loss）、ot（目标 token，若启用 mask 则已按 mask 压成一维）。
- 用途：LoRA 等场景下只对「回答区间」做 next-token 预测，不对指令部分算 loss。
"""

import torch
from torch import nn

from utils.torch.comp import torch_all_wodim

from cnfg.ihyp import cache_len_default


class PMaskDataConverter(nn.Module):
	"""
	将 (seq_batch, seq_o) 转为 (oi, pred_mask, ot)，供 loss 只对 pred_mask 为 True 的位置计算。

	- seq_batch: (batch, seq_len)，整段 token id（如 指令 token + 回答 token）。
	- seq_o: 区间信息。两种形态：
	  - (batch,) 或 (batch, 1)：LM 模式，每个样本一个「有效长度」或上界，只对该长度之前的位置算 loss。
	  - (batch, 2)：区间模式，每行 [start+1, end+1] 表示只对 [start, end) 位置算 loss（与 HDF5 tgt 组约定一致）。
	- 返回的 oi/ot 用于 next-token 预测：oi = seq_batch[:, :-1]，ot = seq_batch[:, 1:]；若 pred_mask 非空，ot 会先被 mask 再返回（只保留要算 loss 的位置）。
	"""

	def __init__(self, xseql=cache_len_default, dtype=torch.int32, device=None, **kwargs):
		"""
		参数:
			xseql (int): 支持的最大序列长度（用于预分配位置下标 ind）。默认来自 cnfg.ihyp.cache_len_default。
			dtype: ind 缓冲区的 dtype，默认 torch.int32。
			device: ind 所在的设备，None 表示不指定（后续随输入移动）。
			**kwargs: 保留，未使用。
		属性:
			self.xseql: 即参数 xseql，供 forward 里判断是否复用 self.ind。
			self.ind: 缓冲 tensor [0, 1, ..., xseql-1]，与 seq_o 比较生成 pred_mask；persistent=False 不写入 checkpoint。
		"""
		super(PMaskDataConverter, self).__init__()
		self.xseql = xseql
		self.register_buffer("ind", torch.arange(xseql, dtype=dtype, device=device), persistent=False)

	# 形状示意（区间模式，seq_o 每行 [start+1, end+1]）：
	# seq_batch: [a b c d j k l n o p x y]  整段（指令+回答）
	# oi:        [a b c d j k l n o p x]    输入（预测下一个）
	# ot:        [b c d j k l n o p x y]    目标（被预测的下一个）
	# ind:       [0 1 2 3 4 5 6 7 8 9 10]  位置下标
	# seq_o:     [4, 7] 表示只对位置 3,4,5,6 算 loss（即 [start+1,end+1] -> start=3,end=7）
	# pred_mask: 仅 [3,4,5,6] 为 True -> ot 只保留这些位置上的 token，用于 loss

	def forward(self, seq_batch, seq_o, seq_o_sub_len=None, **kwargs):
		"""
		参数:
			seq_batch (Tensor): 形状 (batch_size, seq_len)，整段 token id（如 指令+回答 拼接）。
			seq_o (Tensor): 区间/长度信息。
				- (batch_size,) 或 (batch_size, 1)：LM 模式，每样本一个标量，表示「有效长度」或上界（1-based），只对该长度之前的 token 算 loss。
				- (batch_size, 2)：区间模式，每行 [start+1, end+1]（1-based），只对 [start, end) 位置算 loss（与 HDF5 tgt 组一致）。
				- (batch_size, 2*K)：多区间时每行多组 [s+1, e+1]，任意区间覆盖到的位置都算 loss。
			seq_o_sub_len (int, optional): 若提供，会从 seq_o 中再减去该值，得到 0-based 区间（用于兼容额外 offset）。
			**kwargs: 保留，未使用。

		返回:
			oi (Tensor): 形状 (batch_size, seq_len-1)，next-token 的输入 token，即 seq_batch[:, :-1]。
			pred_mask (Tensor | None): 形状 (batch_size, seq_len-1)，True 表示该位置参与 loss；若整段都算则为 None。
			ot (Tensor): 目标 token。若 pred_mask 为 None 则形状 (batch_size, seq_len-1)，即 seq_batch[:, 1:]；
				否则为 1 维，只包含 pred_mask 为 True 位置上的 token，供 loss 对「回答区间」做 mean。
		"""
		_bsize, _seql = seq_batch.size()   # batch 大小与序列长度
		lo = _seql - 1                      # next-token 的有效长度（输入/目标各 lo 个 token）
		oi = seq_batch.narrow(1, 0, lo)    # 输入：第 0～lo-1 个 token
		ot = seq_batch.narrow(1, 1, lo)    # 目标：第 1～lo 个 token（与 oi 错位 1，做 next-token 预测）

		_is_lm = seq_o.dim() == 1           # True 表示 LM 模式（seq_o 为 1 维）
		if _is_lm and torch_all_wodim(seq_o.eq(_seql)).item():
			# LM 且 seq_o 全等于 _seql：不截断，整段都算 loss
			pred_mask = None
		else:
			# 位置下标 [0, 1, ..., lo-1]；若 lo > self.xseql 则临时创建，否则复用 self.ind
			_ind = self.ind.narrow(0, 0, lo) if lo <= self.xseql else torch.arange(lo, dtype=self.ind.dtype, device=self.ind.device)
			# 将 seq_o 转为 0-based：HDF5 里存的是 start+1, end+1，这里减 _ 得到 start, end
			_ = 1
			if seq_o_sub_len is not None:
				_ += seq_o_sub_len
			_seq_ind = seq_o.to(seq_batch.device, non_blocking=True) - _

			if _is_lm:
				# LM：只对位置 < _seq_ind 算 loss
				pred_mask = _ind.lt(_seq_ind.unsqueeze(-1))
			elif seq_o.size(-1) == 2:
				# 区间模式：每样本一行 [start, end)，只对 [start, end) 算 loss
				_sid, _eid = _seq_ind.view(_bsize, 1, 2).unbind(-1)  # start / end，各 (batch_size, 1)
				pred_mask = _ind.ge(_sid)
				pred_mask &= _ind.lt(_eid)
			else:
				# 多区间：每样本多组 [s,e]，位置落在任意 [s,e) 内则算 loss
				_sid, _eid = _seq_ind.view(_bsize, -1, 1, 2).unbind(-1)
				pred_mask = _ind.ge(_sid)
				pred_mask &= _ind.lt(_eid)
				pred_mask = pred_mask.to(torch.int32, non_blocking=True).sum(1).gt(0)
			# 只保留 pred_mask 为 True 位置上的目标 token，后续 loss 对 ot 做 mean 即只对回答区间算
			ot = ot[pred_mask]

		return oi, pred_mask, ot
