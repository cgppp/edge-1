#encoding: utf-8

#from math import sqrt

from modules.propos import InferEmb
from transformer.Encoder import Encoder as EncoderBase
from utils.fmt.parser import parse_none

from cnfg.ihyp import *

class Encoder(EncoderBase):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, share_layer=False, disable_pemb=disable_std_pemb_encoder, num_pos=64, scale=1.0, **kwargs):

		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize
		_act_drop = parse_none(act_drop, dropout)

		super(Encoder, self).__init__(isize, nwd, num_layer, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=_act_drop, num_head=num_head, xseql=xseql, ahsize=_ahsize, norm_output=norm_output, share_layer=share_layer, disable_pemb=disable_pemb, **kwargs)

		if not disable_pemb:
			self.pemb = InferEmb(isize, hsize=_fhsize, num_pos=num_pos, scale=scale, act_drop=_act_drop, **kwargs)

	def forward(self, inputs, mask=None, **kwargs):

		out = self.wemb(inputs)
		if self.pemb is not None:
			if mask is not None:
				out = out.masked_fill_(mask.view(*inputs.size(), 1), 0.0)
			out = self.pemb(out, out.sum(1, keepdim=True)).add_(out)#alpha=sqrt(out.size(-1))

		if self.drop is not None:
			out = self.drop(out)

		for net in self.nets:
			out = net(out, mask)

		return out if self.out_normer is None else self.out_normer(out)
