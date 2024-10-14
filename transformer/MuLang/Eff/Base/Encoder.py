#encoding: utf-8

from math import sqrt
from torch import nn

from transformer.Encoder import Encoder as EncoderBase
from utils.fmt.parser import parse_none

from cnfg.ihyp import *

class Encoder(EncoderBase):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, ntask=None, merge_lang_vcb=True, use_task_emb=False, ngroup=None, share_layer=False, **kwargs):

		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize
		self.task_id_shift, _nwd = (nwd, (nwd + ntask),) if merge_lang_vcb else (0, nwd,)

		super(Encoder, self).__init__(isize, _nwd, num_layer, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, xseql=xseql, ahsize=_ahsize, norm_output=norm_output, share_layer=share_layer, **kwargs)

		self.task_emb = nn.Embedding(ntask, isize, padding_idx=None) if use_task_emb else None

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
			out = net(out, mask=mask)

		if self.out_normer is not None:
			out = self.out_normer(out)

		return out
