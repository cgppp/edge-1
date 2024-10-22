#encoding: utf-8

import torch
from math import sqrt

from modules.kd.base import GradientAdapterFunc
from transformer.MuLang.Eff.LALNT.Decoder import Decoder as DecoderBase
from utils.kd.self.feat import get_kd_loss

from cnfg.ihyp import *
from cnfg.vocab.base import pad_id

class Decoder(DecoderBase):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, emb_w=None, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, bindemb=True, forbidden_index=None, ntask=None, merge_lang_vcb=True, use_task_emb=False, task_emb_w=None, kd_layers=None, min_sim=0.0, share_layer=False, **kwargs):

		super(Decoder, self).__init__(isize, nwd, num_layer, fhsize=fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, emb_w=emb_w, num_head=num_head, xseql=xseql, ahsize=ahsize, norm_output=norm_output, bindemb=bindemb, forbidden_index=forbidden_index, ntask=ntask, merge_lang_vcb=merge_lang_vcb, use_task_emb=use_task_emb, task_emb_w=task_emb_w, share_layer=share_layer, **kwargs)

		self.kd_layers = set() if kd_layers is None else (kd_layers if isinstance(kd_layers, set) else set(kd_layers))
		self.min_sim = min_sim

	def forward(self, inpute, inputo, taskid=None, src_pad_mask=None, gold=None, gold_pad_mask=None, **kwargs):

		if self.task_id_shift > 0:
			inputo.select(1, 0).fill_(taskid + self.task_id_shift)

		nquery = inputo.size(-1)

		out = self.wemb(inputo)
		if self.task_emb is not None:
			out = out + self.task_emb.weight[taskid]
		if self.pemb is not None:
			out = self.pemb(inputo, expand=False).add(out, alpha=sqrt(out.size(-1)))
		if self.drop is not None:
			out = self.drop(out)

		_mask = self._get_subsequent_mask(nquery)

		_use_kd = self.training and (gold is not None)
		if _use_kd:
			kd_o = []
		for prev_layer_ind, net in enumerate(self.nets):
			if _use_kd and (prev_layer_ind in self.kd_layers):
				out, _ = GradientAdapterFunc(out, self.min_sim)
				kd_o.append(_)
			out = net(inpute, out, taskid=taskid, src_pad_mask=src_pad_mask, tgt_pad_mask=_mask)

		if self.out_normer is not None:
			out = self.out_normer(out, taskid=taskid)

		_last_layer_out = out
		out = self.lsm(self.classifier(out, taskid))

		if _use_kd:
			if (prev_layer_ind + 1) in self.kd_layers:
				kd_o.append(_last_layer_out)
			return out, get_kd_loss(torch.stack(kd_o, dim=0), mask=gold.eq(pad_id) if gold_pad_mask is None else gold_pad_mask) if len(kd_o) > 1 else out.new_zeros(1)
		else:
			return out
