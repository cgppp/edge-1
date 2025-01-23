#encoding: utf-8

import torch
from math import log, pi, sqrt

from modules.attn.gqa import SelfAttn as SelfAttnBase
from modules.base import PositionwiseFF as PositionwiseFFBase, ResSelfAttn as ResSelfAttnBase
from modules.norm.base import RMSNorm

from cnfg.plm.llama.v3.ihyp import *

class SelfAttn(SelfAttnBase):

	def __init__(self, isize, hsize=None, osize=None, num_head=8, dropout=0.0, enable_bias=enable_prev_ln_bias_default, enable_proj_bias=enable_proj_bias_default, num_kv_head=None, k_rel_pos=use_k_relative_position, uni_direction_reduction=False, is_left_to_right_reduction=True, zero_reduction=relpos_reduction_with_zeros, max_bucket_distance=0, use_rope=use_rope, rope_pos_offset=0, rope_dim_offset=0, rope_alpha=1.0, use_alibi=use_alibi, sparsenorm=False, xseql=cache_len_default, **kwargs):

		super(SelfAttn, self).__init__(isize, hsize=hsize, osize=osize, num_head=num_head, dropout=dropout, enable_bias=enable_bias, enable_proj_bias=enable_proj_bias, num_kv_head=num_kv_head, k_rel_pos=k_rel_pos, uni_direction_reduction=uni_direction_reduction, is_left_to_right_reduction=is_left_to_right_reduction, zero_reduction=zero_reduction, max_bucket_distance=max_bucket_distance, use_rope=use_rope, rope_pos_offset=rope_pos_offset, rope_dim_offset=rope_dim_offset, rope_alpha=rope_alpha, use_alibi=use_alibi, sparsenorm=sparsenorm, xseql=xseql, **kwargs)

	def rope_build(self, length, sid=0, dtype=None, device=None):

		poff, doff, adim = self.rope_poff, self.rope_doff, self.attn_dim

		pos = torch.arange(sid + poff, length + poff, dtype=torch.float32, device=device).unsqueeze(1)
		rdiv_term = (torch.arange(doff, adim + doff, 2, dtype=torch.float32, device=device) * -(log(sinusoid_base_frequency) / adim)).exp()

		low_freq_wavelen = rope_original_max_position_embeddings / rope_low_freq_factor
		high_freq_wavelen = rope_original_max_position_embeddings / rope_high_freq_factor
		wavelen = 2 * pi / rdiv_term
		# wavelen < high_freq_wavelen: do nothing
		# wavelen > low_freq_wavelen: divide by factor
		rdiv_term[wavelen.gt(low_freq_wavelen)] /= rope_factor
		# otherwise: interpolate between the two, using a smooth factor
		_is_medium_freq = wavelen.ge(high_freq_wavelen) & wavelen.le(low_freq_wavelen)
		smooth_factor = (rope_original_max_position_embeddings / wavelen[_is_medium_freq] - rope_low_freq_factor) / (rope_high_freq_factor - rope_low_freq_factor)
		_ = rdiv_term[_is_medium_freq]
		rdiv_term[_is_medium_freq] = _.mul(1.0 - smooth_factor).div(rope_factor).addcmul(smooth_factor, _)

		_tmp = pos * rdiv_term
		if self.rope_alpha != 1.0:
			_tmp.mul_(self.rope_alpha)
		_s, _c = _tmp.sin(), _tmp.cos()
		if dtype is not None:
			_s, _c = _s.to(dtype, non_blocking=True), _c.to(dtype, non_blocking=True)

		return _s, _c

class ResSelfAttn(ResSelfAttnBase):

	def __init__(self, isize, hsize=None, num_head=8, dropout=0.0, norm_residual=norm_residual_default, **kwargs):

		super(ResSelfAttn, self).__init__(isize, hsize=hsize, num_head=num_head, dropout=dropout, norm_residual=norm_residual, **kwargs)

		self.net = SelfAttn(isize, hsize=hsize, osize=isize, num_head=num_head, dropout=dropout, **kwargs)
		self.normer = RMSNorm(isize, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters)

class PositionwiseFF(PositionwiseFFBase):

	def __init__(self, isize, hsize=None, dropout=0.0, act_drop=None, norm_residual=norm_residual_default, custom_act=use_adv_act_default, enable_bias=enable_prev_ln_bias_default, use_glu=use_glu_ffn, disable_ffn_bias=disable_ffn_bias, **kwargs):

		super(PositionwiseFF, self).__init__(isize, hsize=hsize, dropout=dropout, act_drop=act_drop, norm_residual=norm_residual, custom_act=custom_act, enable_bias=enable_bias, use_glu=use_glu, **kwargs)

		self.normer = RMSNorm(isize, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters)
		if disable_ffn_bias:
			self.net[0].bias = None
