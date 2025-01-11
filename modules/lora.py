#encoding: utf-8

import torch
from math import sqrt
from torch import nn
from torch.nn import functional as nnFunc

from utils.torch.comp import torch_no_grad

class Linear(nn.Linear):

	def __init__(self, in_features, out_features, bias=True, lora_features=None, update_bias=True, **kwargs):

		super(Linear, self).__init__(in_features, out_features, bias=bias, **kwargs)
		self.weight.requires_grad_(False)
		if self.bias is not None:
			self.bias.requires_grad_(update_bias)
		self.lora_features = lora_features
		self.lora_wa = nn.Parameter(torch.Tensor(in_features, lora_features))
		self.lora_wb = nn.Parameter(torch.Tensor(lora_features, out_features))
		self.reset_parameters()

	def forward(self, x, **kwargs):

		out = nnFunc.linear(x, self.weight, self.bias)
		out.add_(x.view(-1, x.size(-1)).mm(self.lora_wa).mm(self.lora_wb).view(out.size()))

		return out

	def reset_parameters(self):

		with torch_no_grad():
			_ = 1.0 / sqrt(self.weight.size(-1))
			self.weight.uniform_(-_, _)
			if hasattr(self, "lora_wa"):
				self.lora_wa.uniform_(-_, _)
			if hasattr(self, "lora_wb"):
				_ = 1.0 / sqrt(self.lora_wb.size(0))
				self.lora_wb.uniform_(-_, _)
			if self.bias is not None:
				self.bias.zero_()

	def from_linear(self, m):

		with torch_no_grad():
			self.weight.copy_(m.weight)
			if m.bias is None:
				if self.bias is not None:
					self.register_parameter("bias", None)
			else:
				if self.bias is None:
					self.bias = nn.Parameter(self.weight.new_empty(self.weight.size(0)))
				self.bias.copy_(m.bias)

	def to_linear(self):

		out_features, in_features = self.weight.size()
		rs = nn.Linear(in_features, out_features, bias=self.bias is not None)
		rs.to(device=self.weight.device, dtype=self.weight.dtype, non_blocking=True)
		with torch_no_grad():
			rs.weight.copy_(self.weight + self.lora_wa.mm(self.lora_wb).t())
			if self.bias is not None:
				rs.bias.copy_(self.bias)

		return rs

	def extra_repr(self):

		return "in_features={}, lora_features={}, out_features={}, bias={}".format(self.in_features, self.lora_features, self.out_features, self.bias is not None)
