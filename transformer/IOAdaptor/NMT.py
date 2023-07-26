#encoding: utf-8

from math import ceil

from transformer.IOAdaptor.Decoder import Decoder
from transformer.IOAdaptor.Encoder import Encoder
from transformer.NMT import NMT as NMTBase
from utils.adaptor import share_ioadaptor_net
from utils.fmt.parser import parse_double_value_tuple, parse_none
from utils.relpos.base import share_rel_pos_cache

from cnfg.ihyp import *

class NMT(NMTBase):

	def __init__(self, isize, snwd, tnwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, global_emb=False, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, bindDecoderEmb=True, forbidden_index=None, ioadaptor_hsize=None, **kwargs):

		enc_layer, dec_layer = parse_double_value_tuple(num_layer)
		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		if ioadaptor_hsize is None:
			_ffn_factor = 1.5 if use_glu_ffn else 2.0
			_ioadaptor_hsize = ceil(((enc_layer - 1) * (isize * _ahsize * 4 + isize * _fhsize * _ffn_factor) + (dec_layer - 1) * (isize * _ahsize * 8 + isize * _fhsize * _ffn_factor)) / (enc_layer * 4 + dec_layer * 7) / isize / _ffn_factor)
			if _ioadaptor_hsize % 2 == 1:
				_ioadaptor_hsize += 1
			_ioadaptor_hsize = max(_ioadaptor_hsize, 2)
		else:
			_ioadaptor_hsize = ioadaptor_hsize

		super(NMT, self).__init__(isize, snwd, tnwd, (enc_layer, dec_layer,), fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, global_emb=global_emb, num_head=num_head, xseql=xseql, ahsize=_ahsize, norm_output=norm_output, bindDecoderEmb=bindDecoderEmb, forbidden_index=forbidden_index, **kwargs)

		self.enc = Encoder(isize, snwd, enc_layer, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, xseql=xseql, ahsize=_ahsize, norm_output=norm_output, ioadaptor_hsize=_ioadaptor_hsize)

		emb_w = self.enc.wemb.weight if global_emb else None

		self.dec = Decoder(isize, tnwd, dec_layer, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, emb_w=emb_w, num_head=num_head, xseql=xseql, ahsize=_ahsize, norm_output=norm_output, bindemb=bindDecoderEmb, forbidden_index=forbidden_index, ioadaptor_hsize=_ioadaptor_hsize)

		if rel_pos_enabled:
			share_rel_pos_cache(self)
		share_ioadaptor_net(self)
