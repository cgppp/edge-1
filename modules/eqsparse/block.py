#encoding: utf-8

from torch import nn

from modules.act import Custom_Act, LGLU, get_act
from modules.base import CrossAttn as CrossAttnBase, PositionwiseFF as PositionwiseFFBase, ResCrossAttn as ResCrossAttnBase, ResSelfAttn as ResSelfAttnBase, SelfAttn as SelfAttnBase
from modules.dropout import Dropout
try:
	from modules.eqsparse.cbe import EqsLinear
except Exception as e:
	print(e)
	from modules.eqsparse.einsumbe import EqsLinear
from utils.fmt.parser import parse_none

from cnfg.ihyp import *

class PositionwiseFF(PositionwiseFFBase):

	def __init__(self, isize, hsize=None, dropout=0.0, act_drop=None, norm_residual=norm_residual_default, num_conn=None, custom_act=use_adv_act_default, enable_bias=enable_prev_ln_bias_default, use_glu=use_glu_ffn, **kwargs):

		_hsize = isize * 4 if hsize is None else hsize
		_act_drop = parse_none(act_drop, dropout)

		if (use_glu is not None) and (_hsize % 2 == 1):
			_hsize += 1

		super(PositionwiseFF, self).__init__(isize, hsize=_hsize, dropout=dropout, act_drop=act_drop, norm_residual=norm_residual, custom_act=custom_act, enable_bias=enable_bias, use_glu=use_glu, **kwargs)

		_ = [EqsLinear(isize, _hsize, num_conn)]
		_drop_ind = 2
		if use_glu is None:
			_.extend([Custom_Act() if custom_act else nn.ReLU(inplace=True), EqsLinear(_hsize, isize, num_conn, bias=enable_bias)])
		else:
			use_glu = use_glu.lower()
			if use_glu == "glu":
				_.append(nn.GLU())
			else:
				_act = get_act(use_glu, None)
				if _act is not None:
					_.append(_act())
					_drop_ind += 1
				_.append(LGLU())
			_.append(EqsLinear(_hsize // 2, isize, num_conn, bias=enable_bias))
		if dropout > 0.0:
			_.append(Dropout(dropout, inplace=True))
		if _act_drop > 0.0:
			_.insert(_drop_ind, Dropout(_act_drop, inplace=inplace_after_Custom_Act))
		self.net = nn.Sequential(*_)

class SelfAttn(SelfAttnBase):

	def __init__(self, isize, hsize, osize, num_head=8, dropout=0.0, enable_bias=enable_prev_ln_bias_default, enable_proj_bias=enable_proj_bias_default, num_conn=None, **kwargs):

		super(SelfAttn, self).__init__(isize, hsize, osize, num_head=num_head, dropout=dropout, enable_bias=enable_bias, enable_proj_bias=enable_proj_bias, **kwargs)

		self.adaptor = EqsLinear(isize, self.hsize * 3, num_conn, bias=enable_proj_bias)

		self.outer = EqsLinear(self.hsize, osize, num_conn, bias=enable_bias)

class CrossAttn(CrossAttnBase):

	def __init__(self, isize, hsize, osize, num_head=8, dropout=0.0, k_isize=None, enable_bias=enable_prev_ln_bias_default, enable_proj_bias=enable_proj_bias_default, num_conn=None, **kwargs):

		super(CrossAttn, self).__init__(isize, hsize, osize, num_head=num_head, dropout=dropout, k_isize=k_isize, enable_bias=enable_bias, enable_proj_bias=enable_proj_bias, **kwargs)

		self.query_adaptor = EqsLinear(isize, self.hsize, num_conn, bias=enable_proj_bias)

		self.kv_adaptor = EqsLinear(isize if k_isize is None else k_isize, self.hsize * 2, num_conn, bias=enable_proj_bias)

		self.outer = EqsLinear(self.hsize, osize, num_conn, bias=enable_bias)

class ResSelfAttn(ResSelfAttnBase):

	def __init__(self, isize, hsize, num_head=8, dropout=0.0, norm_residual=norm_residual_default, num_conn=None, **kwargs):

		super(ResSelfAttn, self).__init__(isize, hsize, num_head=num_head, dropout=dropout, norm_residual=norm_residual, **kwargs)

		self.net = SelfAttn(isize, hsize, isize, num_head=num_head, dropout=dropout, num_conn=num_conn, **kwargs)

class ResCrossAttn(ResCrossAttnBase):

	def __init__(self, isize, hsize, num_head=8, dropout=0.0, norm_residual=norm_residual_default, num_conn=None, **kwargs):

		super(ResCrossAttn, self).__init__(isize, hsize, num_head=num_head, dropout=dropout, norm_residual=norm_residual, **kwargs)

		self.net = CrossAttn(isize, hsize, isize, num_head=num_head, dropout=dropout, num_conn=num_conn, **kwargs)
