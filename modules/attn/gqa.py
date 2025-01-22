#encoding: utf-8

import torch
from math import sqrt

from modules.base import CrossAttn as CrossAttnBase, Linear, ResCrossAttn as ResCrossAttnBase, ResSelfAttn as ResSelfAttnBase, SelfAttn as SelfAttnBase
from utils.fmt.parser import parse_none

from cnfg.ihyp import *

class SelfAttn(SelfAttnBase):

	def __init__(self, isize, hsize=None, osize=None, num_head=8, dropout=0.0, enable_bias=enable_prev_ln_bias_default, enable_proj_bias=enable_proj_bias_default, num_kv_head=None, **kwargs):

		super(SelfAttn, self).__init__(isize, hsize=hsize, osize=osize, num_head=num_head, dropout=dropout, enable_bias=enable_bias, enable_proj_bias=enable_proj_bias, **kwargs)
		self.num_kv_head = parse_none(num_kv_head, num_head)
		self.num_qkv_head = num_head + self.num_kv_head + self.num_kv_head
		if self.num_kv_head != num_head:
			self.adaptor = Linear(isize, self.attn_dim * self.num_qkv_head, bias=enable_proj_bias)

	def forward(self, iQ, mask=None, states=None, **kwargs):

		bsize, nquery = iQ.size()[:2]
		nheads = self.num_head
		adim = self.attn_dim
		kv_nheads = self.num_kv_head

		_ = self.adaptor(iQ).view(bsize, nquery, self.num_qkv_head, adim)
		real_iQ, real_iK, real_iV = _.narrow(2, 0, nheads), _.narrow(2, nheads, kv_nheads), _.narrow(2, nheads + kv_nheads, kv_nheads)
		_h_real_iK, seql, sid = None, nquery, 0
		if self.rope_sin is None:
			real_iQ, real_iK, real_iV = real_iQ.transpose(1, 2), real_iK.permute(0, 2, 3, 1), real_iV.transpose(1, 2)
			if states is not None:
				_h_real_iK, _h_real_iV = states
				if _h_real_iK is not None:
					sid = _h_real_iK.size(-1)
					seql = nquery + sid
					real_iK, real_iV = torch.cat((_h_real_iK, real_iK,), dim=-1), torch.cat((_h_real_iV, real_iV,), dim=2)
		else:
			if states is not None:
				_h_real_iK, _h_real_iV = states
				if _h_real_iK is not None:
					sid = _h_real_iK.size(-1)
					seql = nquery + sid
			if self.ref_ropem is None:
				_rope_sin, _rope_cos = self.get_rope(seql, sid=sid)
				self.rope_sin_cache, self.rope_cos_cache = _rope_sin.unsqueeze(1), _rope_cos.unsqueeze(1)
			else:
				self.rope_sin_cache, self.rope_cos_cache = self.ref_ropem.rope_sin_cache, self.ref_ropem.rope_cos_cache
			real_iQ, real_iK = apply_rope(real_iQ, self.rope_sin_cache, self.rope_cos_cache), apply_rope(real_iK, self.rope_sin_cache, self.rope_cos_cache)
			real_iQ, real_iK, real_iV = real_iQ.transpose(1, 2), real_iK.permute(0, 2, 3, 1), real_iV.transpose(1, 2)
			if _h_real_iK is not None:
				real_iK, real_iV = torch.cat((_h_real_iK, real_iK,), dim=-1), torch.cat((_h_real_iV, real_iV,), dim=2)

		# (bsize, kv_nheads, nheads // kv_nheads, nquery, adim) * (bsize, kv_nheads, 1, adim, seql) = > (bsize, kv_nheads, nheads // kv_nheads, nquery, seql)
		scores = real_iQ.view(bsize, kv_nheads, -1, nquery, adim).matmul(real_iK.unsqueeze(2))

		if self.rel_pemb is not None:
			if states is None:
				self.rel_pos_cache = self.get_rel_pos(nquery).contiguous() if self.ref_rel_posm is None else self.ref_rel_posm.rel_pos_cache
				scores += (real_iQ.permute(2, 0, 1, 3).contiguous().view(nquery, bsize * nheads, adim).bmm(self.rel_pemb(self.rel_pos_cache).transpose(1, 2)).view(nquery, bsize, nheads, nquery).permute(1, 2, 0, 3) if self.rel_pos_map is None else self.rel_pemb(self.rel_pos_cache).permute(2, 0, 1)).unsqueeze(2)
			else:
				self.rel_pos_cache = self.get_rel_pos(seql, sid=sid).contiguous() if self.ref_rel_posm is None else self.ref_rel_posm.rel_pos_cache
				scores += (real_iQ.permute(2, 0, 1, 3).contiguous().view(nquery, bsize * nheads, adim).bmm(self.rel_pemb(self.rel_pos_cache).transpose(1, 2)).view(nquery, bsize, nheads, seql).permute(1, 2, 0, 3) if self.rel_pos_map is None else self.rel_pemb(self.rel_pos_cache).permute(2, 0, 1)).unsqueeze(2)
		if self.alibi is not None:
			self.alibi_cache = (self.get_alibi(nquery) if states is None else self.get_alibi(seql, sid=sid)).contiguous().unsqueeze(2) if self.ref_alibim is None else self.ref_alibim.alibi_cache
			scores += self.alibi_cache

		scores = scores / sqrt(adim)

		if mask is not None:
			_ = mask.size()
			scores.masked_fill_(mask.view(_[0], 1, 1, *_[1:]), -inf_default)

		scores = self.normer(scores)

		if self.drop is not None:
			scores = self.drop(scores)

		# (bsize, kv_nheads, nheads // kv_nheads, nquery, seql) * (bsize, kv_nheads, 1, seql, adim) = > (bsize, kv_nheads, nheads // kv_nheads, nquery, adim)
		out = self.outer(scores.matmul(real_iV.unsqueeze(2)).view(bsize, nheads, nquery, adim).transpose(1, 2).contiguous().view(bsize, nquery, self.hsize))

		return out if states is None else (out, (real_iK, real_iV,),)

class CrossAttn(CrossAttnBase):

	def __init__(self, isize, hsize=None, osize=None, num_head=8, dropout=0.0, k_isize=None, enable_bias=enable_prev_ln_bias_default, enable_proj_bias=enable_proj_bias_default, num_kv_head=None, **kwargs):

		super(CrossAttn, self).__init__(isize, hsize=hsize, osize=osize, num_head=num_head, dropout=dropout, k_isize=k_isize, enable_bias=enable_bias, enable_proj_bias=enable_proj_bias, **kwargs)
		self.num_kv_head = parse_none(num_kv_head, num_head)
		if self.num_kv_head != num_head:
			self.kv_adaptor = Linear(isize if k_isize is None else k_isize, 2 * self.num_kv_head * self.attn_dim, bias=enable_proj_bias)

	def forward(self, iQ, iK, mask=None, **kwargs):

		bsize, nquery = iQ.size()[:2]
		seql = iK.size(1)
		nheads = self.num_head
		adim = self.attn_dim
		kv_nheads = self.num_kv_head

		real_iQ = self.query_adaptor(iQ).view(bsize, nquery, nheads, adim).transpose(1, 2)
		if (self.real_iK is not None) and self.iK.is_set_to(iK) and self.is_decoding:
			real_iK, real_iV = self.real_iK, self.real_iV
		else:
			real_iK, real_iV = self.kv_adaptor(iK).view(bsize, seql, 2, kv_nheads, adim).unbind(2)
			real_iK, real_iV = real_iK.permute(0, 2, 3, 1), real_iV.transpose(1, 2)
			if self.is_decoding:
				self.iK, self.real_iK, self.real_iV = iK, real_iK, real_iV

		scores = real_iQ.view(bsize, -1, kv_nheads, nquery, adim).matmul(real_iK.unsqueeze(1)) / sqrt(adim)

		if mask is not None:
			_ = mask.size()
			scores.masked_fill_(mask.view(_[0], 1, 1, *_[1:]), -inf_default)

		scores = self.normer(scores)

		if self.drop is not None:
			scores = self.drop(scores)

		return self.outer(scores.matmul(real_iV.unsqueeze(1)).view(bsize, nheads, nquery, adim).transpose(1, 2).contiguous().view(bsize, nquery, self.hsize))

class ResSelfAttn(ResSelfAttnBase):

	def __init__(self, isize, hsize=None, num_head=8, dropout=0.0, norm_residual=norm_residual_default, **kwargs):

		super(ResSelfAttn, self).__init__(isize, hsize=hsize, num_head=num_head, dropout=dropout, norm_residual=norm_residual, **kwargs)

		self.net = SelfAttn(isize, hsize=hsize, osize=isize, num_head=num_head, dropout=dropout, **kwargs)

class ResCrossAttn(ResCrossAttnBase):

	def __init__(self, isize, hsize=None, num_head=8, dropout=0.0, norm_residual=norm_residual_default, **kwargs):

		super(ResCrossAttn, self).__init__(isize, hsize=hsize, num_head=num_head, dropout=dropout, norm_residual=norm_residual, **kwargs)

		self.net = CrossAttn(isize, hsize=hsize, osize=isize, num_head=num_head, dropout=dropout, **kwargs)
