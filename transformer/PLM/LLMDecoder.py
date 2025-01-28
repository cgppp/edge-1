#encoding: utf-8

# API define for decoder-only LLM, based on LLaMa.v3

from math import sqrt

from transformer.Decoder import Decoder as DecoderBase

class Decoder(DecoderBase):

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
		out = self.wemb(inpute)

		if self.pemb is not None:
			sqrt_isize = sqrt(out.size(-1))
			out = self.pemb.get_pos(0).add(out, alpha=sqrt_isize)
		if self.drop is not None:
			out = self.drop(out)

		nquery = inpute.size(-1)
		_ = _states.get(0, (None, None,))[0]
		_slen = 0 if _ is None else _.size(-1)
		if (block_size > 0) and (nquery > block_size):
			_sid = _slen
			_tid = _sid + nquery
			while _sid < _tid:
				_eid = min(_sid + block_size, _tid)
				_mask = self._get_subsequent_mask(_eid, sid=_sid)
				_out = out.narrow(1, _sid - _slen, _eid - _sid)
				for _tmp, net in enumerate(self.nets):
					_out, _state = net(_states.get(_tmp, (None, None,)), _mask, _out)
					_states[_tmp] = _state
				_sid = _eid
			out = _out
		else:
			_mask = self._get_subsequent_mask(_slen + nquery, sid=_slen)
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
