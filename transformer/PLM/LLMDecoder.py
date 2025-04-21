#encoding: utf-8

# API define for decoder-only LLM, based on LLaMa.v3

import torch
from math import sqrt
from torch import nn

from transformer.Decoder import Decoder as DecoderBase
from utils.fmt.parser import parse_none
from utils.plm.base import copy_plm_parameter, load_plm_wrapper
from utils.torch.comp import torch_no_grad

class Decoder(DecoderBase):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, emb_w=None, num_head=8, xseql=None, ahsize=None, norm_output=True, bindemb=True, forbidden_index=None, share_layer=False, remove_classifier_bias=None, model_name="model", **kwargs):

		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(Decoder, self).__init__(isize, nwd, num_layer, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, emb_w=emb_w, num_head=num_head, xseql=xseql, ahsize=_ahsize, norm_output=norm_output, bindemb=bindemb, forbidden_index=forbidden_index, share_layer=share_layer, **kwargs)

		self.model_name = model_name
		if remove_classifier_bias:
			self.classifier.bias = None
		self.remove_classifier_bias = remove_classifier_bias

	def forward(self, inputo, word_prediction=True, pred_mask=None, **kwargs):

		nquery = inputo.size(-1)

		out = self.wemb(inputo)

		if self.pemb is not None:
			out = self.pemb(inputo, expand=False).add(out, alpha=sqrt(out.size(-1)))
		if self.drop is not None:
			out = self.drop(out)

		_mask = self._get_subsequent_mask(nquery)

		for net in self.nets:
			out = net(out, _mask)

		if self.out_normer is not None:
			out = self.out_normer(out)

		if word_prediction:
			if pred_mask is not None:
				out = out[pred_mask]
			out = self.lsm(self.classifier(out))

		return out

	def build_states(self, inpute, states=None, return_last_hidden=False, block_size=0, **kwargs):

		_states = {} if states is None else states
		nquery = inpute.size(-1)
		_ = _states.get(0, (None, None,))[0]
		_slen = 0 if _ is None else _.size(-1)
		_rslen = _slen + nquery

		out = self.wemb(inpute)

		if self.pemb is not None:
			out = self.pemb.get_range(_rslen, sid=_slen).add(out, alpha=sqrt(out.size(-1)))
		if self.drop is not None:
			out = self.drop(out)

		if (block_size > 0) and (nquery > block_size):
			_sid = _slen
			while _sid < _rslen:
				_eid = min(_sid + block_size, _rslen)
				_mask = self._get_subsequent_mask(_eid, sid=_sid)
				_out = out.narrow(1, _sid - _slen, _eid - _sid)
				for _tmp, net in enumerate(self.nets):
					_out, _state = net(_states.get(_tmp, (None, None,)), _mask, _out)
					_states[_tmp] = _state
				_sid = _eid
			out = _out
		else:
			_mask = self._get_subsequent_mask(_rslen, sid=_slen)
			for _tmp, net in enumerate(self.nets):
				out, _state = net(_states.get(_tmp, (None, None,)), _mask, out)
				_states[_tmp] = _state

		if return_last_hidden:
			out = out.narrow(1, -1, 1)
			if self.out_normer is not None:
				out = self.out_normer(out)
			return out, _states

		return _states

	def decode(self, inpute, beam_size=1, max_len=512, length_penalty=0.0, fill_pad=False, ilen=None, states=None, **kwargs):

		return self.beam_decode(inpute, beam_size=beam_size, max_len=max_len, length_penalty=length_penalty, fill_pad=fill_pad, ilen=ilen, states=states, **kwargs) if beam_size > 1 else self.greedy_decode(inpute, max_len=max_len, fill_pad=fill_pad, ilen=ilen, states=states, **kwargs)

	def get_sos_emb(self, inpute, bsize=None):

		return self.wemb(inpute)

	@load_plm_wrapper()
	def load_plm(self, plm_parameters, model_name=None, **kwargs):

		_model_name = parse_none(model_name, self.model_name)
		with torch_no_grad():
			if "lm_head.weight" in plm_parameters:
				copy_plm_parameter(self.classifier.weight, plm_parameters, "lm_head.weight")
			copy_plm_parameter(self.wemb.weight, plm_parameters, "%s.embed_tokens.weight" % _model_name)
			copy_plm_parameter(self.out_normer.weight, plm_parameters, "%s.norm.weight" % _model_name)
			if (not self.remove_classifier_bias) and ("final_logits_bias" in plm_parameters):
				if self.classifier.bias is None:
					self.classifier.bias = nn.Parameter(torch.zeros(self.classifier.weight.size(0)))
				copy_plm_parameter(self.classifier.bias, plm_parameters, "final_logits_bias")
			for i, net in enumerate(self.nets):
				if hasattr(net, "load_plm"):
					net.load_plm(plm_parameters, model_name=_model_name, layer_idx=i, **kwargs)
