#encoding: utf-8

import torch

from modules.hplstm.mem.soft.snbase import BiHPLSTM as BiHPLSTMBase, HPLSTM as HPLSTMBase, MHPLSTMCore as MHPLSTMCoreBase, ResHPLSTM as ResHPLSTMBase
from utils.fmt.parser import parse_none
from utils.hplstm.MemLGate import LGateFunc
from utils.hplstm.RS1MvAvg import RS1MvAvgFunc
from utils.math import mvavg_dist2beta

class MHPLSTMCore(MHPLSTMCoreBase):

	def __init__(self, isize, num_head=8, osize=None, act_drop=0.0, nbank=None, ma_beta=0.9, ma_dist=None, **kwargs):

		super(MHPLSTMCore, self).__init__(isize, num_head=num_head, osize=osize, act_drop=act_drop, nbank=nbank, **kwargs)

		self.ma_beta = ma_beta if ma_dist is None else mvavg_dist2beta(ma_dist)

	def forward(self, heads_input, states=None, head_mask=None, **kwargs):

		bsize, seql, nheads, adim = heads_input.size()
		if states is None:
			csum = self.normer_csum(RS1MvAvgFunc(heads_input.detach(), self.ma_beta))
		else:
			_init_state = (states == "init")
			if _init_state:
				csum = self.normer_csum(heads_input.new_zeros(1, 1, nheads, adim)).expand(bsize, 1, nheads, adim)
				csum_state_return = heads_input.detach() * (1.0 - self.ma_beta)
			else:
				_csum_state = states[0]
				csum = self.normer_csum(_csum_state)
				csum_state_return = _csum_state.mul_(self.ma_beta).add_(heads_input.detach(), alpha=1.0 - self.ma_beta)
		o_head_dim, nbank = self.init_cx.size()[1:]
		_ = self.trans_hid(torch.cat((heads_input, csum,), dim=-1))
		_nh = nheads * 3 * o_head_dim
		(igate, fgate, hidden,), (bigate, bfgate, bog,) = _.narrow(-1, 0, _nh).view(bsize, seql, nheads, 3, -1).unbind(-2), _.narrow(-1, _nh, nheads * 3 * nbank).view(bsize, seql, nheads, 3, nbank).unbind(-2)#.sigmoid() if bog uses sigmoid instead of softmax
		fgate, bigate, bfgate, bog = fgate.sigmoid(), bigate.sigmoid(), bfgate.sigmoid(), bog.softmax(-1)
		hidden = self.act(hidden)

		if self.drop is not None:
			hidden = self.drop(hidden)
		igh = igate.sigmoid() * hidden
		if head_mask is not None:
			fgate = fgate.masked_fill(head_mask, 1.0)
			igh.masked_fill_(head_mask, 0.0)
			bfgate = bfgate.masked_fill(head_mask, 1.0)

		eigh = igh.unsqueeze(-1).mul(bigate.unsqueeze(-2))
		cell = LGateFunc(fgate, eigh, self.init_cx, bfgate) if states is None else eigh.addcmul(fgate.unsqueeze(-1) * bfgate.unsqueeze(-2), self.init_cx if _init_state else states[-1])
		if self.m_drop is not None:
			bog = self.m_drop(bog)
		out = self.trans_og(torch.cat((heads_input, cell), dim=-1)).sigmoid() * cell.matmul(bog.unsqueeze(-1)).squeeze(-1)

		return out if states is None else (out, (csum_state_return, cell,),)

class HPLSTM(HPLSTMBase):

	def __init__(self, isize, num_head=8, osize=None, act_drop=0.0, nbank=None, MHPLSTMCore=MHPLSTMCore, **kwargs):

		super(HPLSTM, self).__init__(isize, num_head=num_head, osize=osize, act_drop=act_drop, nbank=nbank, MHPLSTMCore=MHPLSTMCore, **kwargs)

class BiHPLSTM(BiHPLSTMBase):

	def __init__(self, isize, num_head=8, osize=None, act_drop=0.0, nbank=None, MHPLSTMCore=MHPLSTMCore, **kwargs):

		super(BiHPLSTM, self).__init__(isize, num_head=num_head, osize=osize, act_drop=act_drop, nbank=nbank, MHPLSTMCore=MHPLSTMCore, **kwargs)

class ResHPLSTM(ResHPLSTMBase):

	def __init__(self, isize, num_head=8, dropout=0.0, act_drop=None, nbank=None, HPLSTM=HPLSTM, **kwargs):

		super(ResHPLSTM, self).__init__(isize, num_head=num_head, dropout=dropout, act_drop=parse_none(act_drop, dropout), nbank=nbank, HPLSTM=HPLSTM, **kwargs)

class ResBiHPLSTM(ResHPLSTM):

	def __init__(self, isize, num_head=8, dropout=0.0, act_drop=None, nbank=None, HPLSTM=BiHPLSTM, **kwargs):

		super(ResBiHPLSTM, self).__init__(isize, num_head=num_head, dropout=dropout, act_drop=act_drop, nbank=nbank, HPLSTM=HPLSTM, **kwargs)
