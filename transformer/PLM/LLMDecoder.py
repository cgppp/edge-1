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

	def build_states(self, inpute, states=None):

		rs = {} if states is None else states
		out = self.wemb(inpute)

		if self.pemb is not None:
			sqrt_isize = sqrt(out.size(-1))
			out = self.pemb.get_pos(0).add(out, alpha=sqrt_isize)
		if self.drop is not None:
			out = self.drop(out)

		nquery = inpute.size(-1)
		_ = rs.get(0, (None, None,))[0]
		sid = 0 if _ is None else _.size(-1)
		_mask = self._get_subsequent_mask(sid + nquery, sid=sid)

		for _tmp, net in enumerate(self.nets):
			out, _state = net(rs.get(_tmp, (None, None,)), _mask, out)
			rs[_tmp] = _state

		return rs

	def decode(self, inpute, beam_size=1, max_len=512, length_penalty=0.0, fill_pad=False, states=None, **kwargs):

		return self.beam_decode(inpute, beam_size, max_len, length_penalty, fill_pad=fill_pad, states=states, **kwargs) if beam_size > 1 else self.greedy_decode(inpute, max_len, fill_pad=fill_pad, states=states, **kwargs)

	def get_sos_emb(self, inpute, bsize=None):

		return self.wemb(inpute)
