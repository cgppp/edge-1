#encoding: utf-8

from transformer.MuLang.Eff.NMT import NMT as NMTBase
from transformer.MuLang.Eff.SelfKD.Decoder import Decoder
from transformer.MuLang.Eff.SelfKD.Encoder import Encoder
from utils.fmt.parser import parse_double_value_tuple
from utils.relpos.base import share_rel_pos_cache

from cnfg.ihyp import *
from cnfg.vocab.base import pad_id

class NMT(NMTBase):

	def __init__(self, isize, snwd, tnwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, global_emb=False, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, bindDecoderEmb=True, forbidden_index=None, ntask=None, merge_lang_vcb=True, use_task_emb=False, kd_layers=None, min_sim=0.0, **kwargs):

		enc_layer, dec_layer = parse_double_value_tuple(num_layer)

		super(NMT, self).__init__(isize, snwd, tnwd, (enc_layer, dec_layer,), fhsize=fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, global_emb=global_emb, num_head=num_head, xseql=xseql, ahsize=ahsize, norm_output=norm_output, bindDecoderEmb=bindDecoderEmb, forbidden_index=None, **kwargs)

		_kd_layers = [] if kd_layers is None else kd_layers
		if _kd_layers and isinstance(_kd_layers[0], (list, tuple)):
			kd_enc_layers, kd_dec_layers = tuple(set(_) for _ in _kd_layers)
		else:
			kd_enc_layers = kd_dec_layers = set(_kd_layers)

		self.enc = Encoder(isize, snwd, enc_layer, fhsize=fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, xseql=xseql, ahsize=ahsize, norm_output=norm_output, ntask=ntask, merge_lang_vcb=merge_lang_vcb, use_task_emb=use_task_emb, kd_layers=kd_enc_layers, min_sim=min_sim)

		if global_emb:
			emb_w = self.enc.wemb.weight
			task_emb_w = self.enc.task_emb.weight if use_task_emb else None
		else:
			emb_w = task_emb_w = None

		self.dec = Decoder(isize, tnwd, dec_layer, fhsize=fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, emb_w=emb_w, num_head=num_head, xseql=xseql, ahsize=ahsize, norm_output=norm_output, bindemb=bindDecoderEmb, forbidden_index=forbidden_index, ntask=ntask, merge_lang_vcb=merge_lang_vcb, use_task_emb=use_task_emb, task_emb_w=task_emb_w, kd_layers=kd_dec_layers, min_sim=min_sim)

		if rel_pos_enabled:
			share_rel_pos_cache(self)

	def forward(self, inpute, inputo, taskid=None, mask=None, gold=None, gold_pad_mask=None, **kwargs):

		_mask = inpute.eq(pad_id).unsqueeze(1) if mask is None else mask

		if self.training and (gold is not None):
			ence, _enc_kd_loss = self.enc(inpute, taskid=taskid, mask=_mask, gold=gold)
			out, _dec_kd_loss = self.dec(ence, inputo, taskid=taskid, src_pad_mask=_mask, gold=gold, gold_pad_mask=gold_pad_mask)
			return out, _enc_kd_loss + _dec_kd_loss
		else:
			return self.dec(self.enc(inpute, taskid=taskid, mask=_mask), inputo, taskid=taskid, src_pad_mask=_mask)
