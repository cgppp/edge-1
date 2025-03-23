#encoding: utf-8

import torch
from math import sqrt
from torch import nn
from torch.nn import functional as nnFunc

from utils.torch.comp import torch_no_grad

class Linear(nn.Linear):

	def __init__(self, in_features, out_features, bias=True, lora_features=None, lora_alpha=None, scaling=1.0, update_bias=True, **kwargs):

		super(Linear, self).__init__(in_features, out_features, bias=bias, **kwargs)
		self.weight.requires_grad_(False)
		self.update_bias = update_bias
		if self.bias is not None:
			self.bias.requires_grad_(update_bias)
		self.lora_features, self.scaling = lora_features, (scaling if lora_alpha is None else float(lora_alpha) / float(lora_features))
		self.lora_wa = nn.Parameter(torch.Tensor(in_features, lora_features, dtype=self.weight.dtype, device=self.weight.device))
		self.lora_wb = nn.Parameter(torch.zeros(lora_features, out_features, dtype=self.weight.dtype, device=self.weight.device))
		self.fix_init = self.reset_parameters
		self.reset_parameters()

	def forward(self, x, **kwargs):

		out = nnFunc.linear(x, self.weight, self.bias)
		out.add_(x.view(-1, x.size(-1)).mm(self.lora_wa).mm(self.lora_wb).view(out.size()), alpha=self.scaling)

		return out

	def reset_parameters(self):

		with torch_no_grad():
			_ = 1.0 / sqrt(self.weight.size(-1))
			self.weight.uniform_(-_, _)
			if self.bias is not None:
				self.bias.zero_()
		self.init_lora()

	def init_lora(self):

		with torch_no_grad():
			_ = 1.0 / sqrt(self.weight.size(-1))
			if hasattr(self, "lora_wa"):
				self.lora_wa.uniform_(-_, _)
			if hasattr(self, "lora_wb"):
				self.lora_wb.zero_()

	def acc_lora(self):

		with torch_no_grad():
			self.weight.add_(self.lora_wa.mm(self.lora_wb).t())
		self.init_lora()

	def from_std(self, m):

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

class Embedding(nn.Embedding):

	def __init__(self, num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None, lora_features=None, lora_alpha=None, scaling=1.0, **kwargs):

		super(Embedding, self).__init__(num_embeddings, embedding_dim, padding_idx=padding_idx, max_norm=max_norm, norm_type=norm_type, scale_grad_by_freq=scale_grad_by_freq, sparse=sparse, _weight=_weight, **kwargs)
		self.weight.requires_grad_(False)
		self.lora_features, self.scaling = lora_features, (scaling if lora_alpha is None else float(lora_alpha) / float(lora_features))
		self.lora_wa = nn.Parameter(torch.Tensor(num_embeddings, lora_features, dtype=self.weight.dtype, device=self.weight.device))
		self.lora_wb = nn.Parameter(torch.zeros(lora_features, embedding_dim, dtype=self.weight.dtype, device=self.weight.device))
		self.fix_init = self.reset_parameters
		if _weight is None:
			self.reset_parameters()

	def forward(self, x):

		out = nnFunc.embedding(x, self.weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)
		out.add_(nnFunc.embedding(x.view(-1), self.lora_wa, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse).mm(self.lora_wb).view(out.size()), alpha=self.scaling)

		return out

	def reset_parameters(self):

		with torch_no_grad():
			_ = 1.0 / sqrt(self.weight.size(-1))
			self.weight.uniform_(-_, _)
			if self.padding_idx is not None:
				self.weight[self.padding_idx].zero_()
		self.init_lora()

	def init_lora(self):

		with torch_no_grad():
			_ = 1.0 / sqrt(self.weight.size(-1))
			if hasattr(self, "lora_wa"):
				self.lora_wa.uniform_(-_, _)
			if hasattr(self, "lora_wb"):
				self.lora_wb.zero_()
	def acc_lora(self):

		with torch_no_grad():
			self.weight.add_(self.lora_wa.mm(self.lora_wb))
		self.init_lora()

	def from_std(self, m):

		self.weight = m.weight
		self.weight.requires_grad_(False)
		self.to(device=m.weight.device, dtype=m.weight.dtype, non_blocking=True)

	def to_std(self):

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
