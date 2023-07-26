#encoding: utf-8

from torch import nn

from modules.adaptor import IOAdaptor
from transformer.Decoder import Decoder as DecoderBase, DecoderLayer as DecoderLayerBase
from utils.fmt.parser import parse_none

from cnfg.ihyp import *

class DecoderLayer(DecoderLayerBase):

	def __init__(self, isize, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, ahsize=None, ioadaptor_hsize=None, **kwargs):

		super(DecoderLayer, self).__init__(isize, fhsize=fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, ahsize=ahsize, **kwargs)

		self.self_attn = IOAdaptor(self.self_attn, isize, hsize=ioadaptor_hsize, dropout=dropout, act_drop=act_drop, num_ia=1, **kwargs)
		self.cross_attn = IOAdaptor(self.cross_attn, isize, hsize=ioadaptor_hsize, dropout=dropout, act_drop=act_drop, num_ia=2, **kwargs)
		self.ff = IOAdaptor(self.ff, isize, hsize=ioadaptor_hsize, dropout=dropout, act_drop=act_drop, num_ia=1, **kwargs)

class Decoder(DecoderBase):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, emb_w=None, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, bindemb=True, forbidden_index=None, share_layer=False, ioadaptor_hsize=None, **kwargs):

		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(Decoder, self).__init__(isize, nwd, num_layer, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, emb_w=emb_w, num_head=num_head, xseql=xseql, ahsize=_ahsize, norm_output=norm_output, bindemb=bindemb, forbidden_index=forbidden_index, share_layer=share_layer, **kwargs)

		if share_layer:
			_shared_layer = DecoderLayer(isize, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, ahsize=_ahsize, ioadaptor_hsize=ioadaptor_hsize)
			self.nets = nn.ModuleList([_shared_layer for i in range(num_layer)])
		else:
			self.nets = nn.ModuleList([DecoderLayer(isize, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, ahsize=_ahsize, ioadaptor_hsize=ioadaptor_hsize) for i in range(num_layer)])
