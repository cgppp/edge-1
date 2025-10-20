#encoding: utf-8

import torch
from torch import nn

from utils.torch.comp import torch_all_wodim

from cnfg.ihyp import cache_len_default

class PMaskDataConverter(nn.Module):

	def __init__(self, xseql=cache_len_default, dtype=torch.int32, device=None, **kwargs):

		super(PMaskDataConverter, self).__init__()
		self.xseql = xseql
		self.register_buffer("ind", torch.arange(xseql, dtype=dtype, device=device), persistent=False)

	#a b c d j k l n o p x y (seq_batch)
	#b c d j k l n o p x y (tgt)
	#0 1 2 3 4 5 6 7 8 9 10 (ind)
	#4 3 3 2 (len)
	#4 7 10 12 (seq_o)
	#3 6 9 11 (seq_ind)

	def forward(self, seq_batch, seq_o, seq_o_sub_len=None, **kwargs):

		_bsize, _seql = seq_batch.size()
		lo = _seql - 1
		oi, ot = seq_batch.narrow(1, 0, lo), seq_batch.narrow(1, 1, lo)
		_is_lm = seq_o.dim() == 1
		# for LM without padding
		if _is_lm and torch_all_wodim(seq_o.eq(_seql)).item():
			pred_mask = None
		else:
			_ind = self.ind.narrow(0, 0, lo) if lo <= self.xseql else torch.arange(lo, dtype=self.ind.dtype, device=self.ind.device)
			_ = 1
			if seq_o_sub_len is not None:
				_ += seq_o_sub_len
			_seq_ind = seq_o.to(seq_batch.device, non_blocking=True) - _
			if _is_lm:
				pred_mask = _ind.lt(_seq_ind.unsqueeze(-1))
			elif seq_o.size(-1) == 2:
				_sid, _eid = _seq_ind.view(_bsize, 1, 2).unbind(-1)
				pred_mask = _ind.ge(_sid)
				pred_mask &= _ind.lt(_eid)
			else:
				_sid, _eid = _seq_ind.view(_bsize, -1, 1, 2).unbind(-1)
				pred_mask = _ind.ge(_sid)
				pred_mask &= _ind.lt(_eid)
				pred_mask = pred_mask.to(torch.int32, non_blocking=True).sum(1).gt(0)
			ot = ot[pred_mask]

		return oi, pred_mask, ot
