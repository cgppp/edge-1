#encoding: utf-8

import torch

from modules.hplstm.base import BiHPLSTM as BiHPLSTMBase, HPLSTM as HPLSTMBase, MHPLSTMCore as MHPLSTMCoreBase, ResHPLSTM as ResHPLSTMBase
from utils.fmt.parser import parse_none
from utils.hplstm.LGate import LGateFunc
from utils.hplstm.RS1mvavgstatnorm import RS1mvavgstatnorm
from utils.math import mvavg_dist2beta

class MHPLSTMCore(MHPLSTMCoreBase):

	def __init__(self, isize, num_head=8, osize=None, act_drop=0.0, ma_beta=0.9, ma_dist=None, **kwargs):

		super(MHPLSTMCore, self).__init__(isize, num_head=num_head, osize=osize, act_drop=act_drop, **kwargs)

		self.ma_beta = ma_beta if ma_dist is None else mvavg_dist2beta(ma_dist)

	def forward(self, heads_input, states=None, head_mask=None, **kwargs):

		bsize, seql, nheads, adim = heads_input.size()
		csum, csum_state_return = RS1mvavgstatnorm(heads_input, self.normer_csum, states=states, ma_beta=self.ma_beta)
		igate, fgate, hidden = self.normer_hid(self.trans_hid(torch.cat((heads_input, csum,), dim=-1)).view(bsize, seql, nheads, 3, -1)).unbind(-2)
		fgate = fgate.sigmoid()
		hidden = self.act(hidden)

		if self.drop is not None:
			hidden = self.drop(hidden)
		igh = igate.sigmoid() * hidden
		if head_mask is not None:
			fgate = fgate.masked_fill(head_mask, 1.0)
			igh.masked_fill_(head_mask, 0.0)

		is_seq_input = seql > 1
		cell = LGateFunc(fgate, igh, self.init_cx.unsqueeze(0).expand(bsize, nheads, -1) if states == "init" else states[-1].squeeze(1), True) if (states is None) or is_seq_input else igh.addcmul_(fgate, self.init_cx if states == "init" else states[-1])
		out = self.trans_og(torch.cat((heads_input, cell), dim=-1)).sigmoid() * cell

		return out if states is None else (out, (csum_state_return, cell.narrow(1, seql - 1, 1) if is_seq_input else cell,),)

class HPLSTM(HPLSTMBase):

	def __init__(self, isize, num_head=8, osize=None, act_drop=0.0, MHPLSTMCore=MHPLSTMCore, **kwargs):

		super(HPLSTM, self).__init__(isize, num_head=num_head, osize=osize, act_drop=act_drop, MHPLSTMCore=MHPLSTMCore, **kwargs)

class BiHPLSTM(BiHPLSTMBase):

	def __init__(self, isize, num_head=8, osize=None, act_drop=0.0, MHPLSTMCore=MHPLSTMCore, **kwargs):

		super(BiHPLSTM, self).__init__(isize, num_head=num_head, osize=osize, act_drop=act_drop, MHPLSTMCore=MHPLSTMCore, **kwargs)

class ResHPLSTM(ResHPLSTMBase):

	def __init__(self, isize, num_head=8, dropout=0.0, act_drop=None, HPLSTM=HPLSTM, **kwargs):

		super(ResHPLSTM, self).__init__(isize, num_head=num_head, dropout=dropout, act_drop=parse_none(act_drop, dropout), HPLSTM=HPLSTM, **kwargs)

class ResBiHPLSTM(ResHPLSTM):

	def __init__(self, isize, num_head=8, dropout=0.0, act_drop=None, HPLSTM=BiHPLSTM, **kwargs):

		super(ResBiHPLSTM, self).__init__(isize, num_head=num_head, dropout=dropout, act_drop=act_drop, HPLSTM=HPLSTM, **kwargs)
