#encoding: utf-8

import torch
from math import sqrt
from torch import nn

from utils.propos import pos2p
from utils.torch.comp import torch_no_grad

class PropEmb(nn.Module):

	def __init__(self, isize, num_pos=64, scale=1.0, **kwargs):

		super(PropEmb, self).__init__()

		self.scale = scale
		self.weight = nn.Parameter(torch.Tensor(num_pos, isize))
		self.reset_parameters()

	# x: (bsize, seql, ...)
	# mask: (bsize, seql)
	def forward(self, x, mask=None, **kwargs):

		_bsize, _length = x.size()[:2]
		_num_pos, _isize = self.weight.size()
		if mask is None:
			return pos2p(_num_pos, _length, scale=self.scale, sid=_length - 1, device=self.weight.device, dtype=self.weight.dtype).squeeze(0).mm(self.weight)
		else:
			_r_len = _length - mask.long().sum(-1)
			_min_len = _r_len.min().item()
			return pos2p(_num_pos, _length, scale=self.scale, sid=_min_len - 1, device=self.weight.device, dtype=self.weight.dtype).index_select(0, _r_len - _min_len).view(_bsize * _length, _num_pos).mm(self.weight).view(_bsize, _length, _isize)

	def reset_parameters(self):

		_n, _i = self.weight.size()
		_ = sqrt(2.0 / (_n + _i))
		with torch_no_grad():
			self.weight.uniform_(-_, _)

	def fix_init(self):

		self.reset_parameters()
