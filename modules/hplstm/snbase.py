#encoding: utf-8

import torch
from torch import nn

from modules.act import Custom_Act
from modules.base import Dropout
from modules.group.base import GroupLinear
from modules.hplstm.base import BiHPLSTM as BiHPLSTMBase, HPLSTM as HPLSTMBase, ResHPLSTM as ResHPLSTMBase
from utils.base import float2odd
from utils.fmt.parser import parse_none
from utils.hplstm.LGate import LGateFunc
from utils.hplstm.RS1cumsum import RS1cumsumFunc
from utils.torch.comp import torch_no_grad

from cnfg.ihyp import *

class MHPLSTMCore(nn.Module):

	def __init__(self, isize, num_head=8, osize=None, act_drop=0.0, custom_act=use_adv_act_default, enable_bias=enable_prev_ln_bias_default, **kwargs):

		super(MHPLSTMCore, self).__init__()

		_osize = parse_none(osize, isize)

		i_head_dim = float2odd(float(isize) / num_head)
		i_hsize = i_head_dim * num_head
		o_head_dim = float2odd(float(_osize) / num_head)
		o_hsize = o_head_dim * num_head

		self.trans_hid = GroupLinear(i_hsize + i_hsize, o_hsize * 3, num_head, bias=enable_bias, shuffle=False, trans_input=False, flatten_output=False)
		self.trans_og = GroupLinear(i_hsize + o_hsize, o_hsize, num_head, bias=enable_bias, shuffle=False, trans_input=False, flatten_output=False)

		self.normer_csum = nn.LayerNorm((num_head, i_head_dim), eps=ieps_ln_default, elementwise_affine=enable_ln_parameters)

		self.act = Custom_Act() if custom_act else nn.ReLU()
		self.drop = Dropout(act_drop, inplace=inplace_after_Custom_Act) if act_drop > 0.0 else None
		self.init_cx = nn.Parameter(torch.zeros(num_head, o_head_dim))

	def forward(self, heads_input, states=None, head_mask=None, **kwargs):

		bsize, seql, nheads, adim = heads_input.size()
		if states is None:
			csum = self.normer_csum(RS1cumsumFunc(heads_input))
		else:
			_init_state = (states == "init")
			if _init_state:
				csum = self.normer_csum(heads_input.new_zeros(1, 1, nheads, adim)).expand(bsize, 1, nheads, adim)
				csum_state_return = heads_input
			else:
				_csum_state = states[0]
				csum = self.normer_csum(_csum_state)
				csum_state_return = _csum_state + heads_input
		igate, fgate, hidden = self.trans_hid(torch.cat((heads_input, csum,), dim=-1)).view(bsize, seql, nheads, 3, -1).unbind(-2)
		fgate = fgate.sigmoid()
		hidden = self.act(hidden)

		if self.drop is not None:
			hidden = self.drop(hidden)
		igh = igate.sigmoid() * hidden
		if head_mask is not None:
			fgate = fgate.masked_fill(head_mask, 1.0)
			igh.masked_fill_(head_mask, 0.0)

		cell = LGateFunc(fgate, igh, self.init_cx, True) if states is None else igh.addcmul_(fgate, self.init_cx if _init_state else states[-1])
		out = self.trans_og(torch.cat((heads_input, cell), dim=-1)).sigmoid() * cell

		if states is None:
			return out
		else:
			return out, (csum_state_return, cell,)

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

	def __init__(self, isize, num_head=8, dropout=0.0, act_drop=None, HPLSTM=HPLSTM, norm_residual=norm_residual_default, **kwargs):

		super(ResHPLSTM, self).__init__(isize, num_head=num_head, dropout=dropout, act_drop=parse_none(act_drop, dropout), HPLSTM=HPLSTM, **kwargs)

		self.normer = nn.LayerNorm(isize, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters)
		self.norm_residual = norm_residual

	def forward(self, iQ, *inputs, **kwargs):

		_iQ = self.normer(iQ)

		outs = self.net(_iQ, *inputs, **kwargs)

		if isinstance(outs, tuple):
			_out = outs[0]

			if self.drop is not None:
				_out = self.drop(_out)

			return _out + (_iQ if self.norm_residual else iQ), *outs[1:]

		else:
			if self.drop is not None:
				outs = self.drop(outs)

			return outs + (_iQ if self.norm_residual else iQ)

class ResBiHPLSTM(ResHPLSTM):

	def __init__(self, isize, num_head=8, dropout=0.0, act_drop=None, HPLSTM=BiHPLSTM, norm_residual=norm_residual_default, **kwargs):

		super(ResBiHPLSTM, self).__init__(isize, num_head=num_head, dropout=dropout, act_drop=act_drop, HPLSTM=HPLSTM, norm_residual=norm_residual, **kwargs)
