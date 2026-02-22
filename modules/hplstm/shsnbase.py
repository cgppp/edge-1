#encoding: utf-8

import torch
from torch import nn

from modules.act import Custom_Act
from modules.base import Dropout, Linear
from modules.hplstm.snbase import BiHPLSTM as BiHPLSTMBase, HPLSTM as HPLSTMBase, ResHPLSTM as ResHPLSTMBase
from utils.base import float2odd
from utils.fmt.parser import parse_none
from utils.hplstm.LGate import LGateFunc
from utils.hplstm.RS1cumsumstatnorm import RS1cumsumstatnorm
from utils.torch.comp import torch_no_grad

from cnfg.ihyp import *

class MHPLSTMCore(nn.Module):

	def __init__(self, isize, num_head=8, osize=None, act_drop=0.0, custom_act=use_adv_act_default, **kwargs):

		super(MHPLSTMCore, self).__init__()

		_osize = parse_none(osize, isize)

		i_head_dim = float2odd(float(isize) / num_head)
		o_head_dim = float2odd(float(_osize) / num_head)

		self.trans_hid = Linear(i_head_dim + i_head_dim, 3 * o_head_dim, bias=True)
		self.trans_og = Linear(i_head_dim + o_head_dim, o_head_dim, bias=True)

		self.normer_csum = nn.LayerNorm((num_head, i_head_dim), eps=ieps_ln_default, elementwise_affine=enable_ln_parameters)

		self.act = Custom_Act() if custom_act else nn.ReLU()
		self.drop = Dropout(act_drop, inplace=inplace_after_Custom_Act) if act_drop > 0.0 else None
		self.init_cx = nn.Parameter(torch.zeros(num_head, o_head_dim))

	def forward(self, heads_input, states=None, head_mask=None, **kwargs):

		bsize, seql, nheads, adim = heads_input.size()
		csum, csum_state_return = RS1cumsumstatnorm(heads_input, self.normer_csum, states=states)
		igate, fgate, hidden = self.trans_hid(torch.cat((heads_input, csum,), dim=-1)).view(bsize, seql, nheads, 3, -1).unbind(-2)
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

		return out if states is None else (out, (csum_state_return, cell.narrow(1, seql - 1, 1) if is_seq_input else cell,),)

	def fix_init(self):

		with torch_no_grad():
			self.init_cx.zero_()

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
