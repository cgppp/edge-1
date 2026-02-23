#encoding: utf-8

import torch
from torch import nn

from cnfg.ihyp import cache_len_default

def head2tail(inpute, ilen):

	return torch.as_tensor([(_[_l:] + _[:_l]) for _, _l in zip(inpute.tolist(), ilen.tolist())], dtype=inpute.dtype, device=inpute.device)

class iLentoMask(nn.Module):

	def __init__(self, xseql=cache_len_default, dtype=torch.int32, device=None, **kwargs):

		super(iLentoMask, self).__init__()
		self.xseql = xseql
		self.register_buffer("ind", torch.arange(xseql, dtype=dtype, device=device), persistent=False)

	def forward(self, ilen, nquery, **kwargs):

		if ilen is None:

			return None
		else:
			_ind = self.ind.narrow(0, 0, nquery) if nquery <= self.xseql else torch.arange(lo, dtype=self.ind.dtype, device=self.ind.device)

			return _ind.ge(ilen.unsqueeze(-1))

class H2TiLentoMask(nn.Module):

	def __init__(self, xseql=cache_len_default, dtype=torch.int32, device=None, **kwargs):

		super(H2TiLentoMask, self).__init__()
		self.xseql = xseql
		self.register_buffer("ind", torch.arange(xseql, 0, step=-1, dtype=dtype, device=device), persistent=False)

	def forward(self, ilen, nquery, **kwargs):

		if ilen is None:

			return None
		else:
			_ind = self.ind.narrow(0, self.xseql - nquery, nquery) if nquery <= self.xseql else torch.arange(nquery, 0, step=-1, dtype=self.ind.dtype, device=self.ind.device)

			return _ind.gt(ilen.unsqueeze(-1))
