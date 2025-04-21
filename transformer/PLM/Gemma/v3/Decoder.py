#encoding: utf-8

import torch
from math import sqrt
from numbers import Integral
from torch import nn

from modules.base import Dropout, Linear
from modules.norm.base import RMSNorm
from modules.plm.gemma.v3 import PositionwiseFF, ResSelfAttn
from transformer.Decoder import DecoderLayer as DecoderLayerBase
from transformer.PLM.LLMDecoder import Decoder as DecoderBase
from utils.base import index_tensors, lfind_last_module, select_zero_
from utils.decode.beam import expand_bsize_for_beam
from utils.decode.repan import is_penalty_enabled as is_repenalty_enabled, penalty as repenalty
from utils.fmt.parser import parse_none
from utils.plm.base import copy_plm_parameter, load_plm_wrapper
from utils.relpos.base import share_rel_pos_cache
from utils.sampler import SampleMax
from utils.torch.comp import all_done, torch_any_wodim, torch_no_grad

from cnfg.plm.gemma.v3.ihyp import *
from cnfg.vocab.plm.gemma.v3 import eos_id, pad_id, sos_id

class DecoderLayer(DecoderLayerBase):

	def __init__(self, isize, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, ahsize=None, norm_residual=norm_residual_default, k_rel_pos=use_k_relative_position_decoder, max_bucket_distance=relative_position_max_bucket_distance_decoder, num_kv_head=None, sliding_window=sliding_window, disable_ffn_bias=disable_ffn_bias, add_self_attn_postnorm=add_self_attn_postnorm, add_pffn_postnorm=add_pffn_postnorm, add_self_attn_qknorm=add_self_attn_qknorm, rope_linear_scaling=rope_linear_scaling, sinusoid_base_frequency=sinusoid_base_frequency, model_name="model", **kwargs):

		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(DecoderLayer, self).__init__(isize, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, ahsize=_ahsize, norm_residual=norm_residual, k_rel_pos=k_rel_pos, max_bucket_distance=max_bucket_distance, **kwargs)

		self.model_name = model_name
		self.cross_attn = None
		self.self_attn = ResSelfAttn(isize, hsize=_ahsize, num_head=num_head, dropout=attn_drop, norm_residual=norm_residual, k_rel_pos=k_rel_pos, uni_direction_reduction=True, max_bucket_distance=max_bucket_distance, num_kv_head=num_kv_head, sliding_window=sliding_window, add_self_attn_postnorm=add_self_attn_postnorm, add_self_attn_qknorm=add_self_attn_qknorm, rope_linear_scaling=rope_linear_scaling, sinusoid_base_frequency=sinusoid_base_frequency)
		self.ff = PositionwiseFF(isize, hsize=_fhsize, dropout=dropout, act_drop=act_drop, norm_residual=norm_residual, disable_ffn_bias=disable_ffn_bias, add_pffn_postnorm=add_pffn_postnorm)

	def forward(self, inputo, tgt_pad_mask=None, query_unit=None, slen=None, sliding_window_khead=None, **kwargs):

		if query_unit is None:
			context = self.self_attn(inputo, mask=tgt_pad_mask, slen=slen, sliding_window_khead=sliding_window_khead)
		else:
			context, states_return = self.self_attn(query_unit, mask=tgt_pad_mask, states=inputo, slen=slen, sliding_window_khead=sliding_window_khead)

		context = self.ff(context)

		return context if query_unit is None else (context, states_return,)

	@load_plm_wrapper()
	def load_plm(self, plm_parameters, model_name=None, layer_idx=None, **kwargs):

		_model_name = parse_none(model_name, self.model_name)
		with torch_no_grad():
			copy_plm_parameter(self.self_attn.net.adaptor.weight, plm_parameters, ["%s.layers.%d.self_attn.q_proj.weight" % (_model_name, layer_idx,), "%s.layers.%d.self_attn.k_proj.weight" % (_model_name, layer_idx,), "%s.layers.%d.self_attn.v_proj.weight" % (_model_name, layer_idx,)], func=torch.cat, func_kwargs={"dim": 0})
			_bias_key = "%s.layers.%d.self_attn.q_proj.bias" % (_model_name, layer_idx,)
			if (self.self_attn.net.adaptor.bias is None) and (_bias_key in plm_parameters):
				self.self_attn.net.adaptor.bias = nn.Parameter(torch.zeros(self.attn.net.adaptor.weight.size(0)))
			if self.self_attn.net.adaptor.bias is not None:
				copy_plm_parameter(self.self_attn.net.adaptor.bias, plm_parameters, [_bias_key, "%s.layers.%d.self_attn.k_proj.bias" % (_model_name, layer_idx,), "%s.layers.%d.self_attn.v_proj.bias" % (_model_name, layer_idx,)], func=torch.cat, func_kwargs={"dim": 0})
			copy_plm_parameter(self.self_attn.net.outer.weight, plm_parameters, "%s.layers.%d.self_attn.o_proj.weight" % (_model_name, layer_idx,))
			_bias_key = "%s.layers.%d.self_attn.o_proj.bias" % (_model_name, layer_idx,)
			if (self.self_attn.net.outer.bias is None) and (_bias_key in plm_parameters):
				self.self_attn.net.outer.bias = nn.Parameter(torch.zeros(self.attn.net.outer.weight.size(0)))
			if self.self_attn.net.outer.bias is not None:
				copy_plm_parameter(self.self_attn.net.outer.bias, plm_parameters, _bias_key)
			copy_plm_parameter(self.self_attn.normer.weight, plm_parameters, "%s.layers.%d.input_layernorm.weight" % (_model_name, layer_idx,))
			_bias_key = "%s.layers.%d.post_attention_layernorm.weight" % (_model_name, layer_idx,)
			if _bias_key in plm_parameters:
				if self.self_attn.post_normer is None:
					self.self_attn.post_normer = RMSNorm(self.self_attn.normer.weight.size(), eps=ieps_ln_default, elementwise_affine=enable_ln_parameters)
				copy_plm_parameter(self.self_attn.post_normer.weight, plm_parameters, _bias_key)
			elif self.self_attn.post_normer is not None:
				self.self_attn.post_normer = None
			_bias_key = "%s.layers.%d.self_attn.q_norm.weight" % (_model_name, layer_idx,)
			if _bias_key in plm_parameters:
				if self.self_attn.net.q_normer is None:
					self.self_attn.net.q_normer = RMSNorm(self.self_attn.net.attn_dim, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters)
				copy_plm_parameter(self.self_attn.net.q_normer.weight, plm_parameters, _bias_key)
			elif self.self_attn.net.q_normer is not None:
				self.self_attn.net.q_normer = None
			_bias_key = "%s.layers.%d.self_attn.k_norm.weight" % (_model_name, layer_idx,)
			if _bias_key in plm_parameters:
				if self.self_attn.net.k_normer is None:
					self.self_attn.net.k_normer = RMSNorm(self.self_attn.net.attn_dim, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters)
				copy_plm_parameter(self.self_attn.net.k_normer.weight, plm_parameters, _bias_key)
			elif self.self_attn.net.k_normer is not None:
				self.self_attn.net.k_normer = None
			copy_plm_parameter(self.ff.net[0].weight, plm_parameters, ["%s.layers.%d.mlp.gate_proj.weight" % (_model_name, layer_idx,), "%s.layers.%d.mlp.up_proj.weight" % (_model_name, layer_idx,)], func=torch.cat, func_kwargs={"dim": 0})
			_l = lfind_last_module(self.ff.net, Linear)[-1]
			copy_plm_parameter(_l.weight, plm_parameters, "%s.layers.%d.mlp.down_proj.weight" % (_model_name, layer_idx,))
			copy_plm_parameter(self.ff.normer.weight, plm_parameters, "%s.layers.%d.post_attention_layernorm.weight" % (_model_name, layer_idx,))
			_bias_key = "%s.layers.%d.post_feedforward_layernorm.weight" % (_model_name, layer_idx,)
			_pffn_net_ldrop = isinstance(self.ff.net[-1], Dropout)
			_pffn_pnorm_ind = -2 if _pffn_net_ldrop else -1
			_pffn_has_pnorm = isinstance(self.ff.net[_pffn_pnorm_ind], RMSNorm)
			if _bias_key in plm_parameters:
				if not _pffn_has_pnorm:
					if _pffn_net_ldrop:
						self.ff.net.insert(-1, RMSNorm(self.ff.normer.weight.size(), eps=ieps_ln_default, elementwise_affine=enable_ln_parameters))
					else:
						self.ff.net.append(RMSNorm(self.ff.normer.weight.size(), eps=ieps_ln_default, elementwise_affine=enable_ln_parameters))
				copy_plm_parameter(self.ff.net[_pffn_pnorm_ind].weight, plm_parameters, _bias_key)
			elif _pffn_has_pnorm:
				del self.ff.net[_pffn_pnorm_ind]

class Decoder(DecoderBase):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, emb_w=None, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, bindemb=True, forbidden_index=None, share_layer=False, disable_pemb=disable_std_pemb_decoder, num_kv_head=None, add_attn_qkv_bias=add_attn_qkv_bias, sliding_window=sliding_window, disable_ffn_bias=disable_ffn_bias, remove_classifier_bias=remove_classifier_bias, model_name="model", **kwargs):

		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(Decoder, self).__init__(isize, nwd, num_layer, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, emb_w=emb_w, num_head=num_head, xseql=xseql, ahsize=_ahsize, norm_output=norm_output, bindemb=bindemb, forbidden_index=forbidden_index, share_layer=share_layer, disable_pemb=disable_pemb, remove_classifier_bias=remove_classifier_bias, model_name=model_name, **kwargs)

		self.sliding_window = sliding_window
		self.wemb.padding_idx = pad_id
		if share_layer:
			_shared_layer = DecoderLayer(isize, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, ahsize=_ahsize, num_kv_head=num_kv_head, sliding_window=sliding_window, disable_ffn_bias=disable_ffn_bias, add_self_attn_postnorm=add_self_attn_postnorm, add_pffn_postnorm=add_pffn_postnorm, add_self_attn_qknorm=add_self_attn_qknorm, rope_linear_scaling=rope_linear_scaling, sinusoid_base_frequency=sinusoid_base_frequency, model_name=model_name)
			self.nets = nn.ModuleList([_shared_layer for i in range(num_layer)])
		else:
			self.nets = nn.ModuleList([DecoderLayer(isize, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, ahsize=_ahsize, num_kv_head=num_kv_head, sliding_window=-1 if ((i + 1) % sliding_window_layerid_pattern == 0) else sliding_window, disable_ffn_bias=disable_ffn_bias, add_self_attn_postnorm=add_self_attn_postnorm, add_pffn_postnorm=add_pffn_postnorm, add_self_attn_qknorm=add_self_attn_qknorm, rope_linear_scaling=rope_linear_scaling if ((i + 1) % sliding_window_layerid_pattern == 0) else 1.0, sinusoid_base_frequency=sinusoid_base_frequency if ((i + 1) % sliding_window_layerid_pattern == 0) else sinusoid_base_frequency_sliding_window, model_name=model_name) for i in range(num_layer)])

		self.out_normer = RMSNorm(isize, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters) if norm_output else None

		if rel_pos_enabled:
			share_rel_pos_cache(self)

	def build_states(self, inpute, states=None, return_last_hidden=False, block_size=0, slen=None, sliding_window_khead=None, **kwargs):

		_states = {} if states is None else states
		nquery = inpute.size(-1)
		_sliding_window_khead = None if sliding_window_khead is None else (sliding_window_khead if isinstance(sliding_window_khead, Integral) else nquery)
		if slen is None:
			_ = _states.get(0, (None, None,))[0]
			_slen = 0 if _ is None else _.size(-1)
		else:
			_slen = slen
		_rslen = _slen + nquery

		out = self.wemb(inpute)

		if self.pemb is not None:
			sqrt_isize = sqrt(out.size(-1))
			out = self.pemb.get_range(_rslen, sid=_slen).add(out, alpha=sqrt_isize)
		if self.drop is not None:
			out = self.drop(out)

		if (block_size > 0) and (nquery > block_size):
			_sid = _slen
			while _sid < _rslen:
				_eid = min(_sid + block_size, _rslen)
				_lsid = (_sid - (0 if (_states.get(0, (None, None,))[0] is None) else (_states.get(0, (None, None,))[0].size(-1)))) if self.sliding_window > 0 else 0
				_mask = self._get_subsequent_mask(_eid, sid=_sid, lsid=_lsid)
				_full_mask = self._get_subsequent_mask(_eid, sid=_sid, lsid=_lsid, sliding_window=-1) if self.sliding_window >= 0 else _mask
				_out = out.narrow(1, _sid - _slen, _eid - _sid)
				for _tmp, net in enumerate(self.nets):
					_out, _state = net(_states.get(_tmp, (None, None,)), _full_mask if ((_tmp + 1) % sliding_window_layerid_pattern == 0) else _mask, _out, slen=_sid, sliding_window_khead=_sliding_window_khead)
					_states[_tmp] = _state
				_sid = _eid
			out = _out
		else:
			_mask = self._get_subsequent_mask(_rslen, sid=_slen)
			_full_mask = self._get_subsequent_mask(_rslen, sid=_slen, sliding_window=-1) if self.sliding_window >= 0 else _mask
			for _tmp, net in enumerate(self.nets):
				out, _state = net(_states.get(_tmp, (None, None,)), _full_mask if ((_tmp + 1) % sliding_window_layerid_pattern == 0) else _mask, out, slen=_slen, sliding_window_khead=_sliding_window_khead)
				_states[_tmp] = _state

		if return_last_hidden:
			out = out.narrow(1, -1, 1)
			if self.out_normer is not None:
				out = self.out_normer(out)
			return out, _states

		return _states

	def greedy_decode(self, inpute, max_len=512, fill_pad=False, sample=False, top_k=1, top_p=0.0, temp=1.0, repetition_penalty=1.0, ilen=None, post_ilen_rs=True, states=None, slen=None, sliding_window_khead=None, **kwargs):

		bsize, nquery = inpute.size()
		_states = {} if states is None else states
		_sliding_window_khead = None if sliding_window_khead is None else (sliding_window_khead if isinstance(sliding_window_khead, Integral) else nquery)
		if slen is None:
			_ = _states.get(0, (None, None,))[0]
			_slen = 0 if _ is None else _.size(-1)
		else:
			_slen = slen
		if ilen is None:
			_nquery, _inpute = nquery, inpute
		else:
			_nquery = ilen.min().item()
			_inpute = inpute.narrow(-1, 0, _nquery)
		out, _states = self.build_states(_inpute, states=_states, return_last_hidden=True, slen=_slen, sliding_window_khead=_sliding_window_khead)
		_slen += _nquery

		out = self.classifier(out)
		wds = SampleMax(out, dim=-1, keepdim=False, sample=sample, top_k=top_k, top_p=top_p, temp=temp)
		done_trans = wds.eq(eos_id)
		_use_repan = is_repenalty_enabled(repetition_penalty)
		if ilen is None:
			trans = wds if _use_repan else [wds]
		else:
			_ = ilen.gt(_nquery).unsqueeze(-1)
			wds[_] = inpute.narrow(-1, _nquery, 1)[_]
			done_trans &= ~_
			trans = wds.masked_fill(_, sos_id) if _use_repan else [wds]

		for i in range(_nquery, nquery + max_len):

			out = self.wemb(wds)
			if self.pemb is not None:
				out = self.pemb.get_pos(i).add(out, alpha=sqrt_isize)
			if self.drop is not None:
				out = self.drop(out)

			for _tmp, net in enumerate(self.nets):
				out, _state = net(_states[_tmp], None, out, slen=_slen, sliding_window_khead=_sliding_window_khead)
				_states[_tmp] = _state

			if self.out_normer is not None:
				out = self.out_normer(out)

			out = self.classifier(out)
			if _use_repan:
				out = repenalty(out, trans.unsqueeze(1), penalty=repetition_penalty, dim=-1, inplace=True)
			wds = SampleMax(out, dim=-1, keepdim=False, sample=sample, top_k=top_k, top_p=top_p, temp=temp)
			_done_trans = wds.eq(eos_id)

			_ni = i + 1
			if _ni < nquery:
				_ = ilen.gt(_ni).unsqueeze(-1)
				wds[_] = inpute.narrow(-1, _ni, 1)[_]
				_done_trans &= ~_
				if _use_repan:
					_ = wds.masked_fill(_, sos_id)
					if fill_pad:
						_.masked_fill_(done_trans, pad_id)
					trans = torch.cat((trans, _,), 1)
				else:
					trans.append(wds.masked_fill(done_trans, pad_id) if fill_pad else wds)
			else:
				_ = wds.masked_fill(done_trans, pad_id) if fill_pad else wds
				if _use_repan:
					trans = torch.cat((trans, _,), 1)
				else:
					trans.append(_)

			_slen += 1

			done_trans |= _done_trans
			if all_done(done_trans, bsize):
				break

		if not _use_repan:
			trans = torch.cat(trans, 1)
		if post_ilen_rs and (ilen is not None):
			trans = [_t[_:] for _t, _ in zip(trans.tolist(), (ilen - _nquery).tolist())]

		return trans

	def beam_decode(self, inpute, beam_size=8, max_len=512, length_penalty=0.0, return_all=False, clip_beam=clip_beam_with_lp, fill_pad=False, repetition_penalty=1.0, ilen=None, post_ilen_rs=True, states=None, slen=None, sliding_window_khead=None, **kwargs):

		bsize, nquery = inpute.size()
		_states = {} if states is None else states
		_sliding_window_khead = None if sliding_window_khead is None else (sliding_window_khead if isinstance(sliding_window_khead, Integral) else nquery)
		beam_size2 = beam_size * beam_size
		bsizeb2 = bsize * beam_size2
		real_bsize = bsize * beam_size
		if slen is None:
			_ = _states.get(0, (None, None,))[0]
			_slen = 0 if _ is None else _.size(-1)
		else:
			_slen = slen
		if ilen is None:
			_nquery, _inpute, _lpv_rs = nquery, inpute, None
		else:
			_nquery = ilen.min().item()
			_inpute, _csteps = inpute.narrow(-1, 0, _nquery), (ilen - _nquery)
			_lpv_rs = _csteps
		out, _states = self.build_states(_inpute, states=_states, return_last_hidden=True, slen=_slen, sliding_window_khead=_sliding_window_khead)
		_slen += _nquery

		if length_penalty > 0.0:
			lpv = out.new_ones(real_bsize, 1)
			lpv_base = 6.0 ** length_penalty
			if _lpv_rs is not None:
				_lpv_rs = _lpv_rs.to(out.dtype, non_blocking=True).unsqueeze(-1).repeat(1, beam_size).view(real_bsize, 1)

		out = self.lsm(self.classifier(out))

		scores, wds = out.topk(beam_size, dim=-1)
		done_trans = wds.view(bsize, beam_size).eq(eos_id)
		_use_repan = is_repenalty_enabled(repetition_penalty)
		if ilen is None:
			trans = wds = wds.view(real_bsize, 1)
		else:
			_ = ilen.gt(_nquery).view(bsize, 1, 1)
			scores.masked_fill_(_, 0.0)
			done_trans &= ~(_.squeeze(-1))
			_ = _.expand(-1, -1, beam_size)
			wds[_] = inpute.narrow(-1, _nquery, 1).unsqueeze(-1).expand(-1, -1, beam_size)[_]
			if _use_repan:
				trans, wds = wds.masked_fill(_, sos_id).view(real_bsize, 1), wds.view(real_bsize, 1)
			else:
				trans = wds = wds.view(real_bsize, 1)
		scores = scores.squeeze(1)
		sum_scores = scores
		_inds_add_beam2 = torch.arange(0, bsizeb2, beam_size2, dtype=wds.dtype, device=wds.device).unsqueeze(1).expand(bsize, beam_size)
		_inds_add_beam = torch.arange(0, real_bsize, beam_size, dtype=wds.dtype, device=wds.device).unsqueeze(1).expand(bsize, beam_size)

		_states = expand_bsize_for_beam(_states, beam_size=beam_size)

		for step in range(_nquery, nquery + max_len):

			out = self.wemb(wds)
			if self.pemb is not None:
				out = self.pemb.get_pos(step).add(out, alpha=sqrt_isize)

			if self.drop is not None:
				out = self.drop(out)

			for _tmp, net in enumerate(self.nets):
				out, _state = net(_states[_tmp], None, out, slen=_slen, sliding_window_khead=_sliding_window_khead)
				_states[_tmp] = _state

			if self.out_normer is not None:
				out = self.out_normer(out)

			out = self.lsm(repenalty(self.classifier(out), trans.unsqueeze(1), penalty=repetition_penalty, dim=-1, inplace=True)).view(bsize, beam_size, -1)
			_slen += 1

			_scores, _wds = out.topk(beam_size, dim=-1)
			_done_trans_unsqueeze = done_trans.unsqueeze(2)
			_scores = (_scores.masked_fill(_done_trans_unsqueeze.expand(bsize, beam_size, beam_size), 0.0) + sum_scores.unsqueeze(2).repeat(1, 1, beam_size).masked_fill_(select_zero_(_done_trans_unsqueeze.repeat(1, 1, beam_size), -1, 0), -inf_default))

			if length_penalty > 0.0:
				if _lpv_rs is None:
					lpv.masked_fill_(~done_trans.view(real_bsize, 1), ((step + 6.0) ** length_penalty) / lpv_base)
				else:
					_ = ~done_trans.view(real_bsize, 1)
					lpv[_] = (((step + 6.0 - _lpv_rs) ** length_penalty) / lpv_base)[_]

			if clip_beam and (length_penalty > 0.0):
				scores, _inds = (_scores.view(real_bsize, beam_size) / lpv.expand(real_bsize, beam_size)).view(bsize, beam_size2).topk(beam_size, dim=-1)
				_tinds = (_inds + _inds_add_beam2).view(real_bsize)
				sum_scores = _scores.view(bsizeb2).index_select(0, _tinds).view(bsize, beam_size)
			else:
				scores, _inds = _scores.view(bsize, beam_size2).topk(beam_size, dim=-1)
				_tinds = (_inds + _inds_add_beam2).view(real_bsize)
				sum_scores = scores

			wds = _wds.view(bsizeb2).index_select(0, _tinds).view(real_bsize, 1)
			_done_trans = wds.eq(eos_id).squeeze(1)

			_inds = (_inds // beam_size + _inds_add_beam).view(real_bsize)
			trans = trans.index_select(0, _inds)

			_nstep = step + 1
			if _nstep <= nquery:
				if _nstep < nquery:
					_ = ilen.gt(_nstep).unsqueeze(-1)
					sum_scores.masked_fill_(_, 0.0)
					_rbm = _.repeat(1, beam_size).view(real_bsize, 1)
					wds.masked_scatter_(_rbm, inpute.narrow(-1, _nstep, 1)[_].unsqueeze(-1).expand(-1, beam_size))
					_done_trans &= (~_).repeat(1, beam_size).view(real_bsize)
				else:
					_rbm = None
				_ = ilen.eq(_nstep)
				if torch_any_wodim(_):
					wds.view(bsize, beam_size)[_] = _wds[_].select(1, 0)
					sum_scores[_] = _scores[_].select(1, 0)
				if _use_repan:
					_ = wds if _rbm is None else wds.masked_fill(_rbm, sos_id)
					if fill_pad:
						_ = _.masked_fill(done_trans.view(real_bsize, 1), pad_id)
					trans = torch.cat((trans, _,), 1)
				else:
					trans = torch.cat((trans, wds.masked_fill(done_trans.view(real_bsize, 1), pad_id) if fill_pad else wds), 1)
			else:
				trans = torch.cat((trans, wds.masked_fill(done_trans.view(real_bsize, 1), pad_id) if fill_pad else wds), 1)

			done_trans = (done_trans.view(real_bsize).index_select(0, _inds) | _done_trans).view(bsize, beam_size)

			_done = False
			if length_penalty > 0.0:
				lpv = lpv.index_select(0, _inds)
			elif (not return_all) and all_done(done_trans.select(1, 0), bsize):
				_done = True

			if _done or all_done(done_trans, real_bsize):
				break

			_states = index_tensors(_states, indices=_inds, dim=0)

		if (not clip_beam) and (length_penalty > 0.0):
			scores = scores / lpv.view(bsize, beam_size)
			scores, _inds = scores.topk(beam_size, dim=-1)
			_inds = (_inds + _inds_add_beam).view(real_bsize)
			trans = trans.view(real_bsize, -1).index_select(0, _inds)

		trans = trans.view(bsize, beam_size, -1)

		if return_all:
			if post_ilen_rs and (ilen is not None):
				trans = [[_tu[_:] for _tu in _t] for _t, _ in zip(trans.tolist(), _csteps.tolist())]

			return trans, scores
		else:
			trans = trans.select(1, 0)
			if post_ilen_rs and (ilen is not None):
				trans = [_t[_:] for _t, _ in zip(trans.tolist(), _csteps.tolist())]

			return trans

	def _get_subsequent_mask(self, length, sid=0, lsid=0, sliding_window=None, **kwargs):

		_ = length - sid
		_l = length - lsid
		_mask = self.mask.narrow(1, sid, _).narrow(2, lsid, _l).contiguous() if length <= self.xseql else self.mask.new_ones(_, _l).triu(1 + sid - lsid).unsqueeze(0)
		_sliding_window = parse_none(sliding_window, self.sliding_window)
		if (_sliding_window > 0) and (_l > _sliding_window):
			_mask = _mask | _mask.new_ones(_, _l).tril(sid - _sliding_window - lsid).unsqueeze(0)

		return _mask
