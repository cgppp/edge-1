#encoding: utf-8
"""
LoRA 模块层：在标准 nn.Linear / nn.Embedding 上增加低秩旁路，仅训练 lora_wa、lora_wb。

前向：output = 原层(x) + scaling * (x 经 lora_wa、lora_wb 的变换)。
主权重 weight 默认冻结；可选的 bias（仅 Linear）可配置是否更新。
"""

import torch
from math import sqrt
from torch import nn
from torch.nn import functional as nnFunc

from utils.torch.comp import torch_no_grad


# ------------------------------------------------------------------------------
# LoRA 线性层：y = Wx + b + scaling * (x @ lora_wa @ lora_wb)
# ------------------------------------------------------------------------------

class Linear(nn.Linear):
	"""
	带 LoRA 的线性层。主权重 W 冻结，可训练参数为 lora_wa (in_features, r)、lora_wb (r, out_features)。
	前向：out = W @ x + b + scaling * (x @ lora_wa @ lora_wb)。
	"""

	def __init__(self, in_features, out_features, bias=True, lora_features=None, lora_alpha=None, scaling=1.0, update_bias=True, **kwargs):
		"""
		参数:
			in_features: 输入特征维度。
			out_features: 输出特征维度。
			bias: 是否使用 bias。
			lora_features: LoRA 秩 r（lora_wa 的列数 / lora_wb 的行数）。
			lora_alpha: 缩放分子，实际 scaling = lora_alpha / lora_features；若为 None 则用下面的 scaling。
			scaling: 直接指定的 LoRA 输出缩放，仅当 lora_alpha 为 None 时生效。
			update_bias: 是否在微调时更新 bias（True=更新，False=冻结）。
		"""
		super(Linear, self).__init__(in_features, out_features, bias=bias)
		# 主权重冻结，只训练 LoRA 部分
		self.weight.requires_grad_(False)
		self.update_bias = update_bias
		if self.bias is not None:
			self.bias.requires_grad_(update_bias)
		# scaling：若提供 lora_alpha 则为 lora_alpha/r，否则用传入的 scaling
		self.lora_features, self.scaling = lora_features, (scaling if lora_alpha is None else float(lora_alpha) / float(lora_features))
		# 低秩矩阵 A (in_features, r)、B (r, out_features)，前向中相当于 x @ A @ B
		self.lora_wa = nn.Parameter(torch.empty(in_features, lora_features, dtype=self.weight.dtype, device=self.weight.device))
		self.lora_wb = nn.Parameter(torch.zeros(lora_features, out_features, dtype=self.weight.dtype, device=self.weight.device))
		self.fix_init = self.reset_parameters
		self.reset_parameters()

	def forward(self, x, **kwargs):
		"""前向：out = Wx + b + scaling * (x @ lora_wa @ lora_wb)。"""
		out = nnFunc.linear(x, self.weight, self.bias)
		# x: (..., in_f) -> view(-1, in_f) @ lora_wa (in_f, r) @ lora_wb (r, out_f) -> view 回 out 的 shape，再乘 scaling 加到 out
		out.add_(x.view(-1, x.size(-1)).mm(self.lora_wa).mm(self.lora_wb).view(out.size()), alpha=self.scaling)
		return out

	def reset_parameters(self):
		"""按 Kaiming 风格初始化主权重与 bias，再初始化 LoRA 参数。"""
		with torch_no_grad():
			_ = 1.0 / sqrt(self.weight.size(-1))
			self.weight.uniform_(-_, _)
			if self.bias is not None:
				self.bias.zero_()
		self.init_lora()

	def init_lora(self):
		"""仅初始化 lora_wa（均匀）、lora_wb（零），用于训练前或 acc_lora 合并后重置。"""
		with torch_no_grad():
			_ = 1.0 / sqrt(self.weight.size(-1))
			if hasattr(self, "lora_wa"):
				self.lora_wa.uniform_(-_, _)
			if hasattr(self, "lora_wb"):
				self.lora_wb.zero_()

	def acc_lora(self):
		"""将 LoRA 合并进主权重：weight += (lora_wa @ lora_wb).t()，然后重新初始化 lora_wa/lora_wb（便于导出或继续训练）。"""
		with torch_no_grad():
			# (in_f, r) @ (r, out_f) -> (in_f, out_f)；Linear 的 weight 形状为 (out_f, in_f)，故需 .t()
			self.weight.add_(self.lora_wa.mm(self.lora_wb).t())
		self.init_lora()

	def from_std(self, m):
		"""从标准 nn.Linear 拷贝：共用 weight/bias，冻结 weight，bias 是否可训由 update_bias 决定。"""
		self.weight = m.weight
		self.weight.requires_grad_(False)
		if m.bias is None:
			if self.bias is not None:
				self.register_parameter("bias", None)
		else:
			self.bias = m.bias
			self.bias.requires_grad_(self.update_bias)
		self.to(device=m.weight.device, dtype=m.weight.dtype, non_blocking=True)

	def to_std(self):
		"""合并 LoRA 到 weight，返回一个标准 nn.Linear（权重可训），用于导出或还原。"""
		out_features, in_features = self.weight.size()
		rs = nn.Linear(in_features, out_features, bias=self.bias is not None)
		self.acc_lora()
		rs.weight = self.weight
		rs.weight.requires_grad_(True)
		if self.bias is not None:
			rs.bias = self.bias
			rs.bias.requires_grad_(True)
		rs.to(device=self.weight.device, dtype=self.weight.dtype, non_blocking=True)
		return rs

	def extra_repr(self):
		return "in_features={}, lora_features={}, out_features={}, bias={}".format(self.in_features, self.lora_features, self.out_features, self.bias is not None)


# ------------------------------------------------------------------------------
# LoRA 嵌入层：out = E[x] + scaling * (E_lora_wa[x] @ lora_wb)
# ------------------------------------------------------------------------------

class Embedding(nn.Embedding):
	"""
	带 LoRA 的嵌入层。主权重冻结；可训练 lora_wa (num_embeddings, r)、lora_wb (r, embedding_dim)。
	前向：对索引 x 查表得到 out = weight[x] + scaling * (lora_wa[x] @ lora_wb)。
	"""

	def __init__(self, num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None, lora_features=None, lora_alpha=None, scaling=1.0, **kwargs):
		"""
		参数:
			num_embeddings: 词表大小。
			embedding_dim: 嵌入维度。
			padding_idx / max_norm / norm_type / scale_grad_by_freq / sparse: 与 nn.Embedding 一致。
			_weight: 若提供则作为初始 weight（如从标准 Embedding 迁移）；否则由 reset_parameters 初始化。
			lora_features: LoRA 秩 r。
			lora_alpha: 同 Linear，实际 scaling = lora_alpha / lora_features。
			scaling: lora_alpha 为 None 时使用的缩放。
		"""
		super(Embedding, self).__init__(num_embeddings, embedding_dim, padding_idx=padding_idx, max_norm=max_norm, norm_type=norm_type, scale_grad_by_freq=scale_grad_by_freq, sparse=sparse, _weight=_weight)
		self.weight.requires_grad_(False)
		self.lora_features, self.scaling = lora_features, (scaling if lora_alpha is None else float(lora_alpha) / float(lora_features))
		self.lora_wa = nn.Parameter(torch.empty(num_embeddings, lora_features, dtype=self.weight.dtype, device=self.weight.device))
		self.lora_wb = nn.Parameter(torch.zeros(lora_features, embedding_dim, dtype=self.weight.dtype, device=self.weight.device))
		self.fix_init = self.reset_parameters
		if _weight is None:
			self.reset_parameters()

	def forward(self, x):
		"""前向：查主表得 out，再加上 scaling * (lora_wa[x] @ lora_wb)。

		注意：上游有时会产生非 contiguous 的索引张量（例如经过 view/narrow/transpose 等），
		这里使用 reshape(-1) 和 reshape(out.size()) 而不是 view(...)，避免 PyTorch 的
		“view size is not compatible ... Use .reshape(...) instead” 报错。
		"""
		out = nnFunc.embedding(x, self.weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)
		# 对 x 在 lora_wa 上查表得到 (batch, r)，再 @ lora_wb (r, emb_dim) -> (batch, emb_dim)，reshape 成 out.shape 后乘 scaling 加上去
		flat_x = x.reshape(-1)
		lora_out = nnFunc.embedding(flat_x, self.lora_wa, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse).mm(self.lora_wb).reshape(out.size())
		out.add_(lora_out, alpha=self.scaling)
		return out

	def reset_parameters(self):
		"""初始化主 weight（及 padding_idx 置零），再初始化 LoRA。"""
		with torch_no_grad():
			_ = 1.0 / sqrt(self.weight.size(-1))
			self.weight.uniform_(-_, _)
			if self.padding_idx is not None:
				self.weight[self.padding_idx].zero_()
		self.init_lora()

	def init_lora(self):
		"""仅初始化 lora_wa（均匀）、lora_wb（零）。"""
		with torch_no_grad():
			_ = 1.0 / sqrt(self.weight.size(-1))
			if hasattr(self, "lora_wa"):
				self.lora_wa.uniform_(-_, _)
			if hasattr(self, "lora_wb"):
				self.lora_wb.zero_()

	def acc_lora(self):
		"""将 LoRA 合并进主权重：weight += lora_wa @ lora_wb（形状均为 (num_embeddings, embedding_dim)），再 init_lora。"""
		with torch_no_grad():
			self.weight.add_(self.lora_wa.mm(self.lora_wb))
		self.init_lora()

	def from_std(self, m):
		"""从标准 nn.Embedding 拷贝：共用 weight，冻结。"""
		self.weight = m.weight
		self.weight.requires_grad_(False)
		self.to(device=m.weight.device, dtype=m.weight.dtype, non_blocking=True)

	def to_std(self):
		"""合并 LoRA 到 weight，返回标准 nn.Embedding。"""
		self.acc_lora()
		num_embeddings, embedding_dim = self.weight.size()
		rs = nn.Embedding(num_embeddings, embedding_dim, padding_idx=self.padding_idx, max_norm=self.max_norm, norm_type=self.norm_type, scale_grad_by_freq=self.scale_grad_by_freq, sparse=self.sparse, _weight=self.weight)
		rs.weight.requires_grad_(True)
		rs.to(device=self.weight.device, dtype=self.weight.dtype, non_blocking=True)
		return rs

	def extra_repr(self):
		s = "{num_embeddings}, {embedding_dim}, {lora_features}"
		if self.padding_idx is not None:
			s += ", padding_idx={padding_idx}"
		if self.max_norm is not None:
			s += ", max_norm={max_norm}"
		if self.norm_type != 2.0:
			s += ", norm_type={norm_type}"
		if self.scale_grad_by_freq is not False:
			s += ", scale_grad_by_freq={scale_grad_by_freq}"
		if self.sparse is not False:
			s += ", sparse=True"
		return s.format(**self.__dict__)
