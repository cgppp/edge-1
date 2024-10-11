#encoding: utf-8

from transformer.MuLang.Eff.Base.NMT import NMT as NMTBase
from transformer.MuLang.Eff.RFN.Decoder import Decoder
from transformer.MuLang.Eff.RFN.Encoder import Encoder
from utils.fmt.parser import parse_double_value_tuple
from utils.relpos.base import share_rel_pos_cache
from utils.rfn import share_LSTMCell

from cnfg.ihyp import *

class NMT(NMTBase):

	def __init__(self, isize, snwd, tnwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, global_emb=False, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, bindDecoderEmb=True, forbidden_index=None, ntask=None, merge_lang_vcb=True, use_task_emb=False, **kwargs):

		enc_layer, dec_layer = parse_double_value_tuple(num_layer)

		super(NMT, self).__init__(isize, snwd, tnwd, (enc_layer, dec_layer,), fhsize=fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, global_emb=global_emb, num_head=num_head, xseql=xseql, ahsize=ahsize, norm_output=norm_output, bindDecoderEmb=bindDecoderEmb, forbidden_index=forbidden_index, ntask=ntask, merge_lang_vcb=merge_lang_vcb, use_task_emb=use_task_emb, **kwargs)

		self.enc = Encoder(isize, snwd, enc_layer, fhsize=fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, xseql=xseql, ahsize=ahsize, norm_output=norm_output, ntask=ntask, merge_lang_vcb=merge_lang_vcb, use_task_emb=use_task_emb)

		if global_emb:
			emb_w = self.enc.wemb.weight
			task_emb_w = self.enc.task_emb.weight if use_task_emb else None
		else:
			emb_w = task_emb_w = None

		self.dec = Decoder(isize, tnwd, dec_layer, fhsize=fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, emb_w=emb_w, num_head=num_head, xseql=xseql, ahsize=ahsize, norm_output=norm_output, bindemb=bindDecoderEmb, forbidden_index=forbidden_index, ntask=ntask, merge_lang_vcb=merge_lang_vcb, use_task_emb=use_task_emb, task_emb_w=task_emb_w)

		if rel_pos_enabled:
			share_rel_pos_cache(self)
		share_LSTMCell(self, share_all=False)
