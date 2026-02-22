#encoding: utf-8

import torch

from modules.group.base import GroupLinear
from modules.hplstm.snbase import BiHPLSTM as BiHPLSTMBase, HPLSTM as HPLSTMBase, MHPLSTMCore as MHPLSTMCoreBase, ResHPLSTM as ResHPLSTMBase
from utils.base import float2odd
from utils.fmt.parser import parse_none
from utils.hplstm.LGate import LGateFunc

from cnfg.ihyp import *

class MHPLSTMCore(MHPLSTMCoreBase):

	def __init__(self, isize, num_head=8, osize=None, act_drop=0.0, custom_act=use_adv_act_default, **kwargs):

		_osize = parse_none(osize, isize)
		i_head_dim = float2odd(float(isize) / num_head)
		i_hsize = num_head * i_head_dim
		o_head_dim = float2odd(float(_osize) / num_head)
		o_hsize = num_head * o_head_dim

		super(MHPLSTMCore, self).__init__(isize, num_head=num_head, osize=_osize, act_drop=act_drop, custom_act=custom_act, **kwargs)

		self.trans_hid = GroupLinear(i_hsize, 3 * o_hsize, num_head, bias=True, shuffle=False, trans_input=False, flatten_output=False)

	def forward(self, heads_input, states=None, head_mask=None, **kwargs):

		bsize, seql, nheads, adim = heads_input.size()
		igate, fgate, hidden = self.trans_hid(heads_input).view(bsize, seql, nheads, 3, -1).unbind(-2)
		fgate = fgate.sigmoid()
		hidden = self.act(hidden)

		if self.drop is not None:
			hidden = self.drop(hidden)
		igh = igate.sigmoid() * hidden
		if head_mask is not None:
			fgate = fgate.masked_fill(head_mask, 1.0)
			igh.masked_fill_(head_mask, 0.0)

		is_seq_input, use_init_states = seql > 1, (states is None) or (states == "init")
		cell = LGateFunc(fgate, igh, self.init_cx.unsqueeze(0).expand(bsize, nheads, -1) if use_init_states else states[-1].squeeze(1), True) if is_seq_input else igh.addcmul_(fgate, self.init_cx if use_init_states else states[-1])
		out = self.trans_og(torch.cat((heads_input, cell), dim=-1)).sigmoid() * cell

		return out if states is None else (out, cell.narrow(1, seql - 1, 1) if is_seq_input else cell,)

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
