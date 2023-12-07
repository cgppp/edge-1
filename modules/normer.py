#encoding: utf-8

from torch import nn

from cnfg.ihyp import ieps_default

def min_normer(x, dim=-1, bias=0.0, eps=ieps_default):

	_ = x.amin(dim=dim, keepdim=True)
	if bias > 0.0:
		_.sub_(bias)
	_p = x - _
	_ = _p.sum(dim=dim, keepdim=True).add_(eps)

	return _p / _

def max_relu_normer(x, dim=-1, scale=0.5, eps=ieps_default):

	_p = x - x.amin(dim=dim, keepdim=True)
	_p = (_p - _p.amax(dim=dim, keepdim=True).mul(scale)).relu_()
	_ = _p.sum(dim=dim, keepdim=True).add_(eps)

	return _p / _

class MinNormer(nn.Module):

	def __init__(self, dim=-1, bias=0.0, eps=ieps_default, **kwargs):

		super(MinNormer, self).__init__()

		self.dim, self.bias, self.eps = dim, bias, eps

	def forward(self, x, **kwargs):

		return min_normer(x, dim=self.dim, bias=self.bias, eps=self.eps)

class MRNormer(nn.Module):

	def __init__(self, dim=-1, scale=0.5, eps=ieps_default, **kwargs):

		super(MRNormer, self).__init__()

		self.dim, self.scale, self.eps = dim, scale, eps

	def forward(self, x, **kwargs):

		return max_relu_normer(x, dim=self.dim, scale=self.scale, eps=self.eps)
