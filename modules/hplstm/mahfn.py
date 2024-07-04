#encoding: utf-8

import torch

from modules.hplstm.LGate import LGateFunc
from modules.hplstm.MvAvg import RS1MvAvg
from modules.hplstm.hfn import BiHPLSTM as BiHPLSTMBase, HPLSTM as HPLSTMBase, MHPLSTMCore as MHPLSTMCoreBase, ResHPLSTM as ResHPLSTMBase
from utils.base import float2odd
from utils.fmt.parser import parse_none
from utils.math import mvavg_dist2beta

class MHPLSTMCore(MHPLSTMCoreBase):

	def __init__(self, isize, num_head=8, osize=None, fhsize=None, dropout=0.0, act_drop=None, ma_beta=0.9, ma_dist=None, **kwargs):

		super(MHPLSTMCore, self).__init__(isize, num_head=num_head, osize=osize, fhsize=fhsize, dropout=dropout, act_drop=act_drop, **kwargs)

		self.ma_beta = ma_beta if ma_dist is None else mvavg_dist2beta(ma_dist)

	def forward(self, heads_input, states=None, head_mask=None, **kwargs):

		bsize, seql, nheads, adim = heads_input.size()
		if states is None:
			csum = self.normer_csum(RS1MvAvg(heads_input, self.ma_beta, bsize, seql, nheads, adim))
		else:
			_init_state = (states == "init")
			if _init_state:
				csum = self.normer_csum(heads_input.new_zeros(1, 1, nheads, adim)).expand(bsize, 1, nheads, adim)
				csum_state_return = heads_input * (1.0 - self.ma_beta)
			else:
				_csum_state = states[0]
				csum = self.normer_csum(_csum_state)
				csum_state_return = _csum_state.mul_(self.ma_beta).add_(heads_input, alpha=1.0 - self.ma_beta)
		gh_input = torch.cat((heads_input, csum,), dim=-1)
		(igate, fgate,), hidden = self.normer_ifg(self.trans_ifg(gh_input).view(bsize, seql, nheads, 2, -1)).sigmoid().unbind(-2), self.trans_hid(gh_input)
		igh = igate * hidden
		if head_mask is not None:
			fgate = fgate.masked_fill(head_mask, 1.0)
			igh.masked_fill_(head_mask, 0.0)

		cell = LGateFunc(fgate, igh, self.init_cx, True) if states is None else igh.addcmul_(fgate, self.init_cx if _init_state else states[-1])
		out = self.trans_og(torch.cat((heads_input, cell), dim=-1)).sigmoid() * cell

		if states is None:
			return out
		else:
			return out, (csum_state_return, cell,)

class HPLSTM(HPLSTMBase):

	def __init__(self, isize, num_head=8, osize=None, fhsize=None, dropout=0.0, act_drop=None, MHPLSTMCore=MHPLSTMCore, **kwargs):

		_osize = parse_none(osize, isize)
		o_hsize = float2odd(float(_osize) / num_head) * num_head
		_fhsize = float2odd(float(o_hsize * 4 if fhsize is None else fhsize) / num_head) * num_head

		super(HPLSTM, self).__init__(isize, num_head=num_head, osize=_osize, fhsize=_fhsize, dropout=dropout, act_drop=act_drop, MHPLSTMCore=MHPLSTMCore, **kwargs)

class BiHPLSTM(BiHPLSTMBase):

	def __init__(self, isize, num_head=8, osize=None, fhsize=None, dropout=0.0, act_drop=None, MHPLSTMCore=MHPLSTMCore, **kwargs):

		_osize = parse_none(osize, isize)
		o_hsize = float2odd(float(_osize) / num_head) * num_head
		_fhsize = float2odd(float(o_hsize * 4 if fhsize is None else fhsize) / num_head) * num_head

		# modules.hplstm.hfn.BiHPLSTM will double num_head, osize and fhsize
		super(BiHPLSTM, self).__init__(isize, num_head=num_head, osize=_osize, fhsize=_fhsize, dropout=dropout, act_drop=act_drop, MHPLSTMCore=MHPLSTMCore, **kwargs)

class ResHPLSTM(ResHPLSTMBase):

	def __init__(self, isize, num_head=8, fhsize=None, dropout=0.0, act_drop=None, HPLSTM=HPLSTM, **kwargs):

		super(ResHPLSTM, self).__init__(isize, num_head=num_head, fhsize=fhsize, dropout=dropout, act_drop=act_drop, HPLSTM=HPLSTM, **kwargs)

class ResBiHPLSTM(ResHPLSTMBase):

	def __init__(self, isize, num_head=8, fhsize=None, dropout=0.0, act_drop=None, HPLSTM=BiHPLSTM, **kwargs):

		super(ResBiHPLSTM, self).__init__(isize, num_head=num_head, fhsize=fhsize, dropout=dropout, act_drop=act_drop, HPLSTM=HPLSTM, **kwargs)
