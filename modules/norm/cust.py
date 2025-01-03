#encoding: utf-8

from math import sqrt
from numbers import Integral
from torch import nn

from utils.torch.comp import torch_no_grad, torch_std_mean

from cnfg.ihyp import *

class LayerNorm(nn.LayerNorm):

	def __init__(self, normalized_shape, eps=ieps_ln_default, elementwise_affine=True, bias=True, **kwargs):

		super(LayerNorm, self).__init__(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)

		if not bias:
			self.register_parameter("bias", None)

	def forward(self, x, **kwargs):

		_std, _mean = torch_std_mean(x, dim=-1, unbiased=False, keepdim=True)#.detach()
		_xn = (x - _mean) / (_std + self.eps)
		if self.weight is not None:
			if self.bias is None:
				_xn.mul_(self.weight)
			else:
				_xn = self.bias.addcmul(self.weight, _xn)
		elif self.bias is not None:
			_xn.add_(self.bias)

		return _xn

	def reset_parameters(self):

		with torch_no_grad():
			if self.weight is not None:
				self.weight.fill_(1.0)
			if self.bias is not None:
				self.bias.zero_()

	def fix_init(self):

		self.reset_parameters()

	def load_base(self, base_ln):

		with torch_no_grad():
			if self.weight is not None and base_ln.weight is not None:
				self.weight.copy_(base_ln.weight)
				self.weight.requires_grad_(base_ln.weight.requires_grad)
			if self.bias is not None and base_ln.bias is not None:
				self.bias.copy_(base_ln.bias)
				self.bias.requires_grad_(base_ln.bias.requires_grad)

class RMSNorm(LayerNorm):

	def __init__(self, normalized_shape, eps=ieps_ln_default, elementwise_affine=True, **kwargs):

		super(RMSNorm, self).__init__(normalized_shape, eps=eps, elementwise_affine=elementwise_affine, bias=False, **kwargs)
		self.k = 1.0 / sqrt(float(normalized_shape if isinstance(normalized_shape, Integral) else normalized_shape[-1]))

	def forward(self, x, **kwargs):

		_xn = x.div(x.norm(p=2, dim=-1, keepdim=True).mul(self.k).add_(self.eps))
		if self.weight is not None:
			_xn = _xn.mul_(self.weight)

		return _xn

def rms_norm(x, normalized_shape, weight, eps):

	_xn = x.div(x.norm(p=2, dim=-1, keepdim=True).div(sqrt(float(normalized_shape if isinstance(normalized_shape, Integral) else normalized_shape[-1]))).add_(eps))
		if weight is not None:
			_xn = _xn.mul_(weight)

		return _xn
