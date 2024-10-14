#encoding: utf-8

from math import sqrt
from torch import nn

from modules.mulang.eff.lalnt import LayerNorm, MWLinear, PositionwiseFF, ResSelfAttn
from transformer.Encoder import Encoder as EncoderBase, EncoderLayer as EncoderLayerBase
from utils.fmt.parser import parse_none

from cnfg.ihyp import *

class EncoderLayer(EncoderLayerBase):

	def __init__(self, isize, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, ahsize=None, ntask=None, k_rel_pos=use_k_relative_position_encoder, **kwargs):

		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize
		self.task_id_shift, _nwd = (nwd, (nwd + ntask),) if merge_lang_vcb else (0, nwd,)

		super(EncoderLayer, self).__init__(isize, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, ahsize=_ahsize, **kwargs)

		self.attn = ResSelfAttn(isize, hsize=_ahsize, num_head=num_head, dropout=attn_drop, norm_residual=self.attn.norm_residual, k_rel_pos=k_rel_pos, ntask=ntask)
		self.ff = PositionwiseFF(isize, hsize=_fhsize, dropout=dropout, act_drop=act_drop, norm_residual=self.ff.norm_residual, ntask=ntask)

	def forward(self, inputs, taskid=None, mask=None, **kwargs):

		context = self.attn(inputs, taskid=taskid, mask=mask)

		context = self.ff(context, taskid=taskid)

		return context

class Encoder(EncoderBase):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, ntask=None, merge_lang_vcb=True, use_task_emb=False, ngroup=None, share_layer=False, **kwargs):

		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize
		self.task_id_shift, _nwd = (nwd, (nwd + ntask),) if merge_lang_vcb else (0, nwd,)

		super(Encoder, self).__init__(isize, _nwd, num_layer, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, xseql=xseql, ahsize=_ahsize, norm_output=norm_output, share_layer=share_layer, **kwargs)

		self.task_emb = nn.Embedding(ntask, isize, padding_idx=None) if use_task_emb else None
		self.transo = MWLinear(isize, isize, ntask, bias=enable_proj_bias_default)

		if share_layer:
			_shared_layer = EncoderLayer(isize, _fhsize, dropout, attn_drop, act_drop, num_head, _ahsize, ntask=ntask)
			self.nets = nn.ModuleList([_shared_layer for i in range(num_layer)])
		else:
			self.nets = nn.ModuleList([EncoderLayer(isize, _fhsize, dropout, attn_drop, act_drop, num_head, _ahsize, ntask=ntask) for i in range(num_layer)])

		self.out_normer = LayerNorm(isize, ntask=ntask, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters) if norm_output else None

	def forward(self, inputs, taskid=None, mask=None, **kwargs):

		if self.task_id_shift > 0:
			inputs.select(1, 0).fill_(taskid + self.task_id_shift)

		out = self.wemb(inputs)
		if self.task_emb is not None:
			out = out + self.task_emb.weight[taskid]
		if self.pemb is not None:
			out = self.pemb(inputs, expand=False).add(out, alpha=sqrt(out.size(-1)))

		if self.drop is not None:
			out = self.drop(out)

		for net in self.nets:
			out = net(out, taskid=taskid, mask=mask)

		if self.out_normer is not None:
			out = self.out_normer(out, taskid=taskid)

		return self.transo(out, taskid)
