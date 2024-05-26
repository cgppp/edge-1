#encoding: utf-8

import torch
from math import sqrt

from modules.kd.base import MGradientAdapterFunc
from transformer.SelfKD.FeatEncoder import Encoder as EncoderBase
from utils.kd.self.feat import get_iter_kd_loss, get_kd_loss

from cnfg.ihyp import *

kd_loss_l = [get_kd_loss, get_iter_kd_loss]

class Encoder(EncoderBase):

	def forward(self, inputs, mask=None, gold=None, **kwargs):

		out = self.wemb(inputs)
		if self.pemb is not None:
			out = self.pemb(inputs, expand=False).add(out, alpha=sqrt(out.size(-1)))

		if self.drop is not None:
			out = self.drop(out)

		_use_kd = self.training and (gold is not None)
		if _use_kd:
			kd_o = []
			_num_kd_loss = len(kd_loss_l)
			_k = _num_kd_loss + 1
		for prev_layer_ind, net in enumerate(self.nets):
			if _use_kd and (prev_layer_ind in self.kd_layers):
				_ = MGradientAdapterFunc(out, self.min_sim, _k)
				out = _[0]
				kd_o.append(_[1:])
			out = net(out, mask)

		if self.out_normer is not None:
			out = self.out_normer(out)

		if _use_kd:
			if (prev_layer_ind + 1) in self.kd_layers:
				kd_o.append(tuple(out for _ in range(_num_kd_loss)))
			if len(kd_o) > 1:
				_mask = None if mask is None else mask.squeeze(1)
				return out, sum([_kd_loss(torch.stack(_, dim=0), mask=_mask) for _kd_loss, _ in zip(kd_loss_l, kd_o)])
			else:
				return out, out.new_zeros(1)
		else:
			return out
