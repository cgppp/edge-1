#encoding: utf-8

import torch
from math import log, sqrt
from torch import nn

from modules.attn.gqa import SelfAttn as SelfAttnBase
from modules.base import PositionwiseFF as PositionwiseFFBase, ResSelfAttn as ResSelfAttnBase
from modules.norm.base import RMSNorm
from utils.fmt.parser import parse_none
from utils.relpos.rope import apply_rope

from cnfg.plm.glm.v4.ihyp import *

class SelfAttn(SelfAttnBase):

	def __init__(self, isize, hsize=None, osize=None, num_head=8, dropout=0.0, enable_bias=enable_prev_ln_bias_default, enable_proj_bias=enable_proj_bias_default, num_kv_head=None, add_attn_qkv_bias=add_attn_qkv_bias, k_rel_pos=use_k_relative_position, uni_direction_reduction=False, is_left_to_right_reduction=True, zero_reduction=relpos_reduction_with_zeros, max_bucket_distance=0, use_rope=use_rope, rope_pos_offset=0, rope_dim_offset=0, rope_alpha=1.0, sinusoid_base_frequency=sinusoid_base_frequency, use_alibi=use_alibi, sparsenorm=False, xseql=cache_len_default, **kwargs):

		super(SelfAttn, self).__init__(isize, hsize=hsize, osize=osize, num_head=num_head, dropout=dropout, enable_bias=enable_bias, enable_proj_bias=enable_proj_bias, num_kv_head=num_kv_head, k_rel_pos=k_rel_pos, uni_direction_reduction=uni_direction_reduction, is_left_to_right_reduction=is_left_to_right_reduction, zero_reduction=zero_reduction, max_bucket_distance=max_bucket_distance, use_rope=use_rope, rope_pos_offset=rope_pos_offset, rope_dim_offset=rope_dim_offset, rope_alpha=rope_alpha, sinusoid_base_frequency=sinusoid_base_frequency, use_alibi=use_alibi, sparsenorm=sparsenorm, xseql=xseql, **kwargs)

		if add_attn_qkv_bias and (self.adaptor.bias is None):
			self.adaptor.bias = nn.Parameter(torch.zeros(self.adaptor.weight.size(0)))

class ResSelfAttn(ResSelfAttnBase):

	def __init__(self, isize, hsize=None, num_head=8, dropout=0.0, norm_residual=norm_residual_default, **kwargs):

		super(ResSelfAttn, self).__init__(isize, hsize=hsize, num_head=num_head, dropout=dropout, norm_residual=norm_residual, **kwargs)

		self.net = SelfAttn(isize, hsize=hsize, osize=isize, num_head=num_head, dropout=dropout, **kwargs)
		self.normer = RMSNorm(isize, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters)

# same as modules.plm.llama.v3.PositionwiseFF, but using default kwargs from cnfg.plm.glm.v4.ihyp
class PositionwiseFF(PositionwiseFFBase):

	def __init__(self, isize, hsize=None, dropout=0.0, act_drop=None, norm_residual=norm_residual_default, custom_act=use_adv_act_default, enable_bias=enable_prev_ln_bias_default, use_glu=use_glu_ffn, disable_ffn_bias=disable_ffn_bias, **kwargs):

		super(PositionwiseFF, self).__init__(isize, hsize=hsize, dropout=dropout, act_drop=act_drop, norm_residual=norm_residual, custom_act=custom_act, enable_bias=enable_bias, use_glu=use_glu, **kwargs)

		self.normer = RMSNorm(isize, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters)
		if disable_ffn_bias:
			self.net[0].bias = None
