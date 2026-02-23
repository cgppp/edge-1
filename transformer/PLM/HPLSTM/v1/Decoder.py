#encoding: utf-8

import torch
from torch import nn

from modules.hplstm.snbase import ResHPLSTM
from transformer.PLM.LLMDecoder import Decoder as DecoderBase
from utils.base import index_tensors, select_zero_
from utils.decode.beam import expand_bsize_for_beam
from utils.decode.repan import is_penalty_enabled as is_repenalty_enabled, penalty as repenalty
from utils.fmt.parser import parse_none
from utils.hplstm.ilen import H2TiLentoMask, head2tail
from utils.norm.rms import ln2rms
from utils.sampler import SampleMax
from utils.torch.comp import all_done

from cnfg.plm.hplstm.v1.base import *
from cnfg.plm.hplstm.v1.ihyp import *
from cnfg.vocab.plm.qwen.v3 import eos_id, pad_id, vocab_size

class Decoder(DecoderBase):

	def __init__(self, isize=isize, nwd=vocab_size, num_layer=nlayer, dropout=drop, act_drop=act_drop, emb_w=None, num_head=nhead, xseql=cache_len_default, norm_output=norm_output, bindemb=bindDecoderEmb, forbidden_index=None, share_layer=False, disable_pemb=disable_std_pemb_decoder, remove_classifier_bias=remove_classifier_bias, model_name=model_name, norm_residual=norm_residual_default, i_hsize=hplstm_i_hsize, o_hsize=hplstm_o_hsize, use_rmsnorm=use_rmsnorm, **kwargs):

		super(Decoder, self).__init__(isize, nwd, num_layer, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, emb_w=emb_w, num_head=num_head, xseql=xseql, norm_output=norm_output, bindemb=bindemb, forbidden_index=forbidden_index, share_layer=True, disable_pemb=True, remove_classifier_bias=remove_classifier_bias, model_name=model_name, **kwargs)

		self.mask = None

		if share_layer:
			_shared_layer = ResHPLSTM(isize, num_head=num_head, dropout=dropout, act_drop=act_drop, norm_residual=norm_residual, i_hsize=i_hsize, o_hsize=o_hsize)
			self.nets = nn.ModuleList([_shared_layer for i in range(num_layer)])
		else:
			self.nets = nn.ModuleList([ResHPLSTM(isize, num_head=num_head, dropout=dropout, act_drop=act_drop, norm_residual=norm_residual, i_hsize=i_hsize, o_hsize=o_hsize) for i in range(num_layer)])

		self.h2tilen2mask = H2TiLentoMask()

		if use_rmsnorm:
			ln2rms(self)

	def forward(self, inputo, word_prediction=True, pred_mask=None, states=None, **kwargs):

		out = self.wemb(inputo)

		if self.drop is not None:
			out = self.drop(out)

		if states is None:
			for net in self.nets:
				out = net(out)
		else:
			for _tmp, net in enumerate(self.nets):
				out = net(out, states=states.get(_tmp, "init"))[0]

		if pred_mask is not None:
			out = out[pred_mask]

		if self.out_normer is not None:
			out = self.out_normer(out)

		if word_prediction:
			out = self.lsm(self.classifier(out))

		return out

	def build_states(self, inpute, states=None, return_last_hidden=False, block_size=0, ilen=None, head2tail_inpute=None, **kwargs):

		_states = {} if states is None else states.copy()# prevent changing states
		bsize, nquery = inpute.size()
		if ilen is None:
			head_mask = None
		else:
			inpute = head2tail(inpute) if head2tail_inpute is None else head2tail_inpute
			head_mask = self.h2tilen2mask(ilen, nquery).view(bsize, nquery, 1, 1)

		out = self.wemb(inpute)

		if self.drop is not None:
			out = self.drop(out)

		if (block_size > 0) and (nquery > block_size):
			_sid = 0
			while _sid < nquery:
				_eid = min(_sid + block_size, nquery)
				_ = _eid - _sid
				_out = out.narrow(1, _sid, _)
				for _tmp, net in enumerate(self.nets):
					_out, _state = net(_out, states=_states.get(_tmp, "init"), head_mask=None if head_mask is None else head_mask.narrow(1, _sid, _))
					_states[_tmp] = _state
				_sid = _eid
			out = _out
		else:
			for _tmp, net in enumerate(self.nets):
				out, _state = net(out, states=_states.get(_tmp, "init"), head_mask=head_mask)
				_states[_tmp] = _state

		if return_last_hidden:
			out = out.narrow(1, -1, 1)
			if self.out_normer is not None:
				out = self.out_normer(out)
			return out, _states

		return _states

	def greedy_decode(self, inpute, max_len=512, fill_pad=False, sample=False, top_k=1, top_p=0.0, temp=1.0, repetition_penalty=1.0, ilen=None, post_ilen_rs=True, states=None, head2tail_inpute=None, **kwargs):

		_states = {} if states is None else states
		out, _states = self.build_states(inpute, states=_states, return_last_hidden=True, ilen=ilen, head2tail_inpute=head2tail_inpute)

		out = self.classifier(out)
		wds = SampleMax(out, dim=-1, keepdim=False, sample=sample, top_k=top_k, top_p=top_p, temp=temp)
		done_trans = wds.eq(eos_id)
		_use_repan = is_repenalty_enabled(repetition_penalty)
		trans = wds if _use_repan else [wds]

		for i in range(1, max_len):

			out = self.wemb(wds)

			if self.drop is not None:
				out = self.drop(out)

			for _tmp, net in enumerate(self.nets):
				out, _state = net(out, states=_states.get(_tmp, "init"))
				_states[_tmp] = _state

			if self.out_normer is not None:
				out = self.out_normer(out)

			out = self.classifier(out)
			if _use_repan:
				out = repenalty(out, trans.unsqueeze(1), penalty=repetition_penalty, dim=-1, inplace=True)
			wds = SampleMax(out, dim=-1, keepdim=False, sample=sample, top_k=top_k, top_p=top_p, temp=temp)

			_ = wds.masked_fill(done_trans, pad_id) if fill_pad else wds
			if _use_repan:
				trans = torch.cat((trans, _,), 1)
			else:
				trans.append(_)

			done_trans |= wds.eq(eos_id)
			if all_done(done_trans, bsize):
				break

		if not _use_repan:
			trans = torch.cat(trans, 1)
		if post_ilen_rs:
			trans = trans.tolist() if ilen is None else [_t[_:] for _t, _ in zip(trans.tolist(), (ilen - _nquery).tolist())]

		return trans

	def beam_decode(self, inpute, beam_size=8, max_len=512, length_penalty=0.0, return_all=False, clip_beam=clip_beam_with_lp, fill_pad=False, repetition_penalty=1.0, ilen=None, post_ilen_rs=True, states=None, head2tail_inpute=None, **kwargs):

		bsize, seql = inpute.size()
		_states = {} if states is None else states
		beam_size2 = beam_size * beam_size
		bsizeb2 = bsize * beam_size2
		real_bsize = bsize * beam_size

		out, _states = self.build_states(inpute, states=_states, return_last_hidden=True, ilen=ilen, head2tail_inpute=head2tail_inpute)

		if length_penalty > 0.0:
			lpv = out.new_ones(real_bsize, 1)
			lpv_base = 6.0 ** length_penalty

		out = self.lsm(self.classifier(out))

		scores, wds = out.topk(beam_size, dim=-1)
		done_trans = wds.view(bsize, beam_size).eq(eos_id)
		_use_repan = is_repenalty_enabled(repetition_penalty)
		trans = wds = wds.view(real_bsize, 1)
		scores = scores.squeeze(1)
		sum_scores = scores
		_inds_add_beam2 = torch.arange(0, bsizeb2, beam_size2, dtype=wds.dtype, device=wds.device).unsqueeze(1).expand(bsize, beam_size)
		_inds_add_beam = torch.arange(0, real_bsize, beam_size, dtype=wds.dtype, device=wds.device).unsqueeze(1).expand(bsize, beam_size)

		states = expand_bsize_for_beam(states, beam_size=beam_size)

		for step in range(1, max_len):

			out = self.wemb(wds)

			if self.drop is not None:
				out = self.drop(out)

			for _tmp, net in enumerate(self.nets):
				out, _state = net(out, states=_states.get(_tmp, "init"))
				_states[_tmp] = _state

			if self.out_normer is not None:
				out = self.out_normer(out)

			out = self.lsm(repenalty(self.classifier(out), trans.unsqueeze(1), penalty=repetition_penalty, dim=-1, inplace=True)).view(bsize, beam_size, -1)

			_scores, _wds = out.topk(beam_size, dim=-1)
			_done_trans_unsqueeze = done_trans.unsqueeze(2)
			_scores = (_scores.masked_fill(_done_trans_unsqueeze.expand(bsize, beam_size, beam_size), 0.0) + sum_scores.unsqueeze(2).repeat(1, 1, beam_size).masked_fill_(select_zero_(_done_trans_unsqueeze.repeat(1, 1, beam_size), -1, 0), -inf_default))

			if length_penalty > 0.0:
				lpv.masked_fill_(~done_trans.view(real_bsize, 1), ((step + 6.0) ** length_penalty) / lpv_base)

			if clip_beam and (length_penalty > 0.0):
				scores, _inds = (_scores.view(real_bsize, beam_size) / lpv.expand(real_bsize, beam_size)).view(bsize, beam_size2).topk(beam_size, dim=-1)
				_tinds = (_inds + _inds_add_beam2).view(real_bsize)
				sum_scores = _scores.view(bsizeb2).index_select(0, _tinds).view(bsize, beam_size)
			else:
				scores, _inds = _scores.view(bsize, beam_size2).topk(beam_size, dim=-1)
				_tinds = (_inds + _inds_add_beam2).view(real_bsize)
				sum_scores = scores

			wds = _wds.view(bsizeb2).index_select(0, _tinds).view(real_bsize, 1)

			_inds = (_inds // beam_size + _inds_add_beam).view(real_bsize)

			trans = torch.cat((trans.index_select(0, _inds), wds.masked_fill(done_trans.view(real_bsize, 1), pad_id) if fill_pad else wds), 1)

			done_trans = (done_trans.view(real_bsize).index_select(0, _inds) | wds.eq(eos_id).squeeze(1)).view(bsize, beam_size)

			_done = False
			if length_penalty > 0.0:
				lpv = lpv.index_select(0, _inds)
			elif (not return_all) and all_done(done_trans.select(1, 0), bsize):
				_done = True

			if _done or all_done(done_trans, real_bsize):
				break

			states = index_tensors(states, indices=_inds, dim=0)

		if (not clip_beam) and (length_penalty > 0.0):
			scores = scores / lpv.view(bsize, beam_size)
			scores, _inds = scores.topk(beam_size, dim=-1)
			_inds = (_inds + _inds_add_beam).view(real_bsize)
			trans = trans.view(real_bsize, -1).index_select(0, _inds)

		trans = trans.view(bsize, beam_size, -1)

		if return_all:

			return trans.tolist() if post_ilen_rs else trans, scores
		else:
			trans = trans.select(1, 0)

			return trans.tolist() if post_ilen_rs else trans
