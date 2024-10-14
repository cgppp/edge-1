#encoding: utf-8

import torch
from math import sqrt
from torch import nn
from torch.nn import functional as nnFunc
from torch.nn.init import _calculate_fan_in_and_fan_out

from utils.init.base import kaiming_uniform_
from utils.torch.comp import torch_no_grad

class IPLinear(nn.Linear):

	def reset_parameters(self):

		super(IPLinear, self).reset_parameters()
		self.fix_init()

	def fix_init(self):

		with torch_no_grad():
			torch.eye(self.weight.size(0), m=self.weight.size(1), out=self.weight, dtype=self.weight.dtype, device=self.weight.device)

# input size == output size, can compute with both weight and its transpose
class BiPLinear(IPLinear):

	def forward(self, x, reverse=False, **kwargs):

		if reverse:
			isize = list(x.size())
			edim = isize[-1]
			_x = x.view(-1, edim)
			out = _x.mm(self.weight) if self.bias is None else self.bias.addmm(_x, self.weight)
			isize[-1] = -1
			return out.view(isize)
		else:
			return nnFunc.linear(x, self.weight, self.bias)

# multi-args linear
class MALinear(nn.Linear):

	def __init__(self, in_features, out_features, bias=True, args_take=None, kwargs_take=None, add_out_func=None, **kwargs):

		super(MALinear, self).__init__(in_features, out_features, bias=bias)

		self.args_index, self.kwargs_key, self.add_out_func = args_take, kwargs_take, add_out_func

	def forward(self, *inputs, **kwargs):

		out = nnFunc.linear(inputs[self.args_index] if self.kwargs_key is None else kwargs[self.kwargs_key], self.weight, self.bias)
		if self.add_out_func is None:
			return out
		else:
			_add_out = self.add_out_func(*inputs, **kwargs)
			if isinstance(_add_out, tuple):
				return (out,) + _add_out
			else:
				return out, _add_out

class Linear(nn.Module):

	def __init__(self, in_features, out_features, bias=True, hidden_features=None, nbias=1, **kwargs):
		super(Linear, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.hidden_features = min(in_features, out_features) if hidden_features is None else hidden_features
		self.weight_h = nn.Parameter(torch.Tensor(self.hidden_features, in_features))
		self.weight = nn.Parameter(torch.Tensor(out_features, self.hidden_features))
		self.nbias = nbias
		if bias:
			self.bias = nn.Parameter(torch.Tensor(nbias, out_features)) if nbias > 1 else nn.Parameter(torch.Tensor(out_features))
		else:
			self.register_parameter("bias", None)
		self.reset_parameters()

	def reset_parameters(self):
		with torch_no_grad():
			kaiming_uniform_(self.weight_h.data, gain=sqrt(1.0/3.0))
			kaiming_uniform_(self.weight.data, gain=sqrt(1.0/3.0))
			if self.bias is not None:
				fan_in, _ = _calculate_fan_in_and_fan_out(self.weight_h.data)
				bound = 1.0 / sqrt(fan_in)
				if self.nbias > 1:
					bound /= float(self.nbias)
				self.bias.data.uniform_(-bound, bound)

	def forward(self, input, **kwargs):

		if self.bias is None:
			_bias = None
		else:
			_bias = self.bias.sum(0) if self.nbias > 1 else self.bias

		return nnFunc.linear(nnFunc.linear(input, self.weight_h, None), self.weight, _bias)

	def extra_repr(self):

		return "in_features={}, hidden_features={}, out_features={}, bias={}".format(self.in_features, self.hidden_features, self.out_features, self.bias is not None)

	def fix_init(self):
		#self.reset_parameters()
		if self.bias is not None:
			with torch_no_grad():
				self.bias.zero_()

	def to_std(self):

		_rsm = nn.Linear(self.in_features, self.out_features, bias=self.bias is not None)
		with torch_no_grad():
			_rsm.weight.copy_(self.weight.mm(self.weight_h))
			if self.bias is not None:
				_rsm.bias.copy_(self.bias.sum(0) if self.nbias > 1 else self.bias)

		return _rsm
