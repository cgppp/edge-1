#encoding: utf-8

import torch
from torch import nn

from modules.base import Dropout
from modules.hplstm.snbase import ResBiHPLSTM
from transformer.Encoder import Encoder as EncoderBase
from utils.fmt.parser import parse_none
from utils.torch.comp import flip_mask

from cnfg.ihyp import *

class EncoderLayer(ResBiHPLSTM):

	def __init__(self, isize, fhsize=None, dropout=0.0, act_drop=None, num_head=8, norm_residual=norm_residual_default, **kwargs):

		_fhsize = isize * 4 if fhsize is None else fhsize

		super(EncoderLayer, self).__init__(isize, num_head=num_head, fhsize=_fhsize, dropout=dropout, act_drop=act_drop, norm_residual=norm_residual, **kwargs)

class Encoder(EncoderBase):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, share_layer=False, **kwargs):

		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(Encoder, self).__init__(isize, nwd, num_layer, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, xseql=xseql, ahsize=_ahsize, norm_output=norm_output, share_layer=share_layer, disable_pemb=True, **kwargs)

		if share_layer:
			_shared_layer = EncoderLayer(isize, _fhsize, dropout, act_drop, num_head)
			self.nets = nn.ModuleList([_shared_layer for i in range(num_layer)])
		else:
			self.nets = nn.ModuleList([EncoderLayer(isize, _fhsize, dropout, act_drop, num_head) for i in range(num_layer)])

	def forward(self, inputs, mask=None, **kwargs):

		if mask is None:
			_rmask = None
		else:
			bsize, seql = inputs.size()[:2]
			nheads = self.nets[0].net.num_head
			_rmask = torch.cat((mask.new_zeros(1, 1, 1, 1).expand(bsize, seql, nheads, 1), flip_mask(mask.view(bsize, seql, 1, 1), 1).expand(bsize, seql, nheads, 1),), dim=2)

		out = self.wemb(inputs)

		if self.drop is not None:
			out = self.drop(out)

		for net in self.nets:
			out = net(out, pad_reversed_mask=_rmask)

		return out if self.out_normer is None else self.out_normer(out)
