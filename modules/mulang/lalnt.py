#encoding: utf-8

import torch
from numbers import Integral
from torch import nn
from torch.nn import functional as nnFunc

from modules.mulang.base import MBLinear
from utils.torch.comp import torch_std_mean

class MWLinear(MBLinear):

	def __init__(self, in_features, out_features, nbias, bias=True, **kwargs):

		super(MWLinear, self).__init__(in_features, out_features, nbias, bias=False)

		self.weight = nn.Parameter(torch.empty(nbias, in_features, out_features).uniform_(- sqrt(1.0 / in_features), sqrt(1.0 / in_features)))
		if bias:
			self.bias = nn.Parameter(torch.zeros(nbias, 1, out_features))

	def forward(self, x, taskid, **kwargs):

		_isize = list(x.size())
		_w = self.weight.index_select(0, taskid)
		_input = x.view(_isize[0], -1, _isize[-1])
		if self.bias is None:
			out = _input.bmm(_w)
		else:
			out = self.bias.index_select(0, taskid).baddbmm(_input, _w)
		_isize[-1] = self.weight.size(-1)

		return out.view(_isize)

	def fix_init(self):

		_isize = self.weight.size(1)
		with torch_no_grad():
			self.weight.data.uniform_(- sqrt(1.0 / _isize), sqrt(1.0 / _isize))
		super(MWLinear, self).fix_init()

class LayerNorm(nn.LayerNorm):

	def __init__(self, normalized_shape, ntask=None, eps=1e-5, elementwise_affine=True, **kwargs):

		if isinstance(normalized_shape, Integral):
			normalized_shape = (ntask, normalized_shape,)
		else:
			normalized_shape = tuple([ntask, *normalized_shape])

		super(LayerNorm, self).__init__(normalized_shape, eps=eps, elementwise_affine=elementwise_affine, **kwargs)

		self.normalized_shape = self.normalized_shape[1:]

	def forward(self, x, taskid=None, **kwargs):

		if (self.weight is None) and (self.bias is None):
			_xn = nnFunc.layer_norm(x, self.normalized_shape, None, None, self.eps)
		else:
			_std, _mean = torch_std_mean(x, dim=-1, unbiased=False, keepdim=True)
			_xn = (x - _mean) / (_std + self.eps)
			_bsize = [1 for i in range(x.dim() - len(self.normalized_shape))] + list(self.normalized_shape)
			_bsize[0] = x.size(0)
			if self.weight is not None:
				_xn = _xn * self.weight.index_select(0, taskid).view(_bsize)
			if self.bias is not None:
				_xn = _xn.add_(self.bias.index_select(0, taskid).view(_bsize))

		return _xn
