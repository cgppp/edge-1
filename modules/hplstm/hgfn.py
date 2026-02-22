#encoding: utf-8

import torch
from torch import nn

from modules.act import Custom_Act, get_act
from modules.base import Dropout
from modules.group.base import GroupLinear
from modules.hplstm.base import BiHPLSTM as BiHPLSTMBase, HPLSTM as HPLSTMBase, MHPLSTMCore as MHPLSTMCoreBase, ResHPLSTM as ResHPLSTMBase
from utils.base import float2odd
from utils.fmt.parser import parse_none
from utils.hplstm.LGate import LGateFunc
from utils.hplstm.RS1cumsumstatnorm import RS1cumsumstatnorm

from cnfg.ihyp import *

class MHPLSTMCore(MHPLSTMCoreBase):

	def __init__(self, isize, num_head=8, osize=None, fhsize=None, act_drop=0.0, custom_act=use_adv_act_default, enable_bias=enable_prev_ln_bias_default, enable_proj_bias=enable_proj_bias_default, use_glu=use_glu_ffn, **kwargs):

		_osize = parse_none(osize, isize)
		i_hsize = num_head * float2odd(float(isize) / num_head)
		o_hsize = num_head * float2odd(float(_osize) / num_head)
		_fhsize = (4 * o_hsize) if fhsize is None else fhsize
		_head_fhsize = float2odd(float(_fhsize) / num_head)
		_fhsize = num_head * _head_fhsize

		super(MHPLSTMCore, self).__init__(isize, num_head=num_head, osize=_osize, act_drop=act_drop, custom_act=custom_act, enable_bias=enable_bias)

		self.act = None
		_ = [GroupLinear(i_hsize + i_hsize, _fhsize, num_head, bias=enable_bias, shuffle=False, trans_input=False, flatten_output=False), nn.LayerNorm((num_head, _head_fhsize), eps=ieps_ln_default, elementwise_affine=enable_ln_parameters)]
		_drop_ind = 3
		if use_glu is None:
			_.extend([Custom_Act() if custom_act else nn.ReLU(inplace=True), GroupLinear(_fhsize, o_hsize * 3, num_head, bias=enable_proj_bias, shuffle=False, trans_input=False, flatten_output=False)])
		else:
			_.append(get_act(use_glu)())
			_.append(GroupLinear(_fhsize // 2, o_hsize * 3, num_head, bias=enable_proj_bias, shuffle=False, trans_input=False, flatten_output=False))
		if act_drop > 0.0:
			_.insert(_drop_ind, Dropout(act_drop, inplace=inplace_after_Custom_Act))
		self.trans_hid = nn.Sequential(*_)

	def forward(self, heads_input, states=None, head_mask=None, **kwargs):

		bsize, seql, nheads, adim = heads_input.size()
		csum, csum_state_return = RS1cumsumstatnorm(heads_input, self.normer_csum, states=states)
		igate, fgate, hidden = self.normer_hid(self.trans_hid(torch.cat((heads_input, csum,), dim=-1)).view(bsize, seql, nheads, 3, -1)).unbind(-2)
		fgate = fgate.sigmoid()

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

class HPLSTM(HPLSTMBase):

	def __init__(self, isize, num_head=8, osize=None, fhsize=None, act_drop=0.0, MHPLSTMCore=MHPLSTMCore, **kwargs):

		_osize = parse_none(osize, isize)
		o_hsize = num_head * float2odd(float(_osize) / num_head)
		_fhsize = num_head * float2odd(float(_osize * 4 if fhsize is None else fhsize) / num_head)

		super(HPLSTM, self).__init__(isize, num_head=num_head, osize=_osize, act_drop=act_drop, MHPLSTMCore=MHPLSTMCore, fhsize=_fhsize, **kwargs)

class BiHPLSTM(BiHPLSTMBase):

	def __init__(self, isize, num_head=8, osize=None, fhsize=None, act_drop=0.0, **kwargs):

		_osize = parse_none(osize, isize)
		_fhsize = num_head * float2odd(float(_osize * 4 if fhsize is None else fhsize) / num_head)

		super(BiHPLSTM, self).__init__(isize, num_head=num_head, osize=_osize, act_drop=act_drop, MHPLSTMCore=MHPLSTMCore, fhsize=_fhsize + _fhsize, **kwargs)

class ResHPLSTM(ResHPLSTMBase):

	def __init__(self, isize, num_head=8, fhsize=None, dropout=0.0, act_drop=None, HPLSTM=HPLSTM, **kwargs):

		super(ResHPLSTM, self).__init__(isize, num_head=num_head, dropout=dropout, act_drop=act_drop, HPLSTM=HPLSTM, fhsize=fhsize, **kwargs)

class ResBiHPLSTM(ResHPLSTMBase):

	def __init__(self, isize, num_head=8, fhsize=None, dropout=0.0, act_drop=None, HPLSTM=BiHPLSTM, **kwargs):

		super(ResBiHPLSTM, self).__init__(isize, num_head=num_head, dropout=dropout, act_drop=act_drop, HPLSTM=HPLSTM, fhsize=fhsize, **kwargs)
