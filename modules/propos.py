#encoding: utf-8

import torch
from math import sqrt
from torch import Tensor, nn

from modules.act import Custom_Act, LGLU, get_act
from modules.base import Linear
from modules.dropout import Dropout
from modules.normer import MRNormer, MinNormer
from utils.fmt.parser import parse_none
from utils.propos import pos2p
from utils.torch.comp import torch_no_grad

from cnfg.ihyp import *

class PropEmb(nn.Module):

	def __init__(self, isize, num_pos=None, scale=1.0, **kwargs):

		super(PropEmb, self).__init__()

		_num_pos = parse_none(num_pos, isize)
		self.scale = scale
		self.weight = nn.Parameter(torch.Tensor(_num_pos, isize))
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

class InferEmb(PropEmb):

	def __init__(self, isize, hsize=None, num_pos=None, scale=1.0, act_drop=0.0, prev_act_ln=True, custom_act=use_adv_act_default, enable_bias=enable_prev_ln_bias_default, use_glu=use_glu_ffn, **kwargs):

		_num_pos = parse_none(num_pos, isize)

		super(InferEmb, self).__init__(isize, num_pos=_num_pos, scale=scale, **kwargs)

		_hsize = isize * 4 if hsize is None else hsize

		if (use_glu is not None) and (_hsize % 2 == 1):
			_hsize += 1

		_drop_ind = 2
		_ = [Linear(isize + isize, _hsize, bias=(not prev_act_ln) or enable_bias)]
		if prev_act_ln:
			_.append(nn.LayerNorm(_hsize, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters))
			_drop_ind += 1
		if use_glu is None:
			_.extend([Custom_Act() if custom_act else nn.ReLU(inplace=True), Linear(_hsize, _num_pos)])
		else:
			use_glu = use_glu.lower()
			if use_glu == "glu":
				_.append(nn.GLU())
			else:
				_act = get_act(use_glu, None)
				if _act is not None:
					_.append(_act())
					_drop_ind += 1
				_.append(LGLU())
			_.append(Linear(_hsize // 2, isize, bias=enable_bias))
		if scale == 1.0:
			_.append(nn.Softmax(-1))
		if act_drop > 0.0:
			_.insert(_drop_ind, Dropout(act_drop, inplace=inplace_after_Custom_Act))
		self.net = nn.Sequential(*_)
		self.normer_csum = nn.LayerNorm(isize, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters)
		self.normer_sum = nn.LayerNorm(isize, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters)
		self.normer = nn.Softmax(-1)

	# x: (bsize, seql, isize)
	# s: (bsize, 1, isize)
	def forward(self, x, s, states=None, **kwargs):

		if states is None:
			states_return = x.cumsum(dim=1)
		else:
			states_return = x
			if isinstance(states, Tensor):
				states_return = states_return + states

		out = self.net(torch.cat((self.normer_csum(states_return), self.normer_sum(s).expand_as(states_return),), dim=-1))
		if self.scale != 1.0:
			out = out.mul_(self.scale)
		out = self.normer(out)
		_osize = out.size()

		out = out.mm(self.weight) if len(_osize) == 2 else out.view(-1, _osize[-1]).mm(self.weight).view(*_osize[:-1], self.weight.size(-1))

		return out if states is None else (out, states_return,)
