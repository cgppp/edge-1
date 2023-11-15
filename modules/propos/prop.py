#encoding: utf-8

import torch
from math import sqrt
from torch import nn

from utils.torch.comp import torch_no_grad

from cnfg.ihyp import *

class Prop(nn.Module):

	def __init__(self, num_pos=64, scale=1.0, xseql=cache_len_default, **kwargs):

		super(Prop, self).__init__()

		self.register_buffer("ids", torch.arange(xseql, dtype=torch.float).unsqueeze(0), persistent=False)
		self.register_buffer("prop", torch.arange(num_pos, dtype=torch.float).div_(num_pos - 1), persistent=False)
		self.scale, self.xseql = -scale, xseql

	def forward(self, x, mask=None, **kwargs):

		seql = x.size(-1)
		_ = seql - 1.0
		if mask is not None:
			_ = _ - mask.to(self.ids.dtype).sum(-1, keepdim=True)
		_dis = ((self.ids.narrow(1, 0, seql) / _).unsqueeze(-1) - self.prop).abs_()

		return (_dis.neg_() if self.scale == -1.0 else _dis.mul_(self.scale)).softmax(-1)
