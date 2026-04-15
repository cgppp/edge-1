#encoding: utf-8
"""
Qwen3 Decoder-only 结构：单层 DecoderLayer（Self-Attn + FFN）与完整 Decoder（embed + N 层 + norm + classifier）。

【在项目中的位置】
- 训练入口：adv/train/plm/train_lora_qwen.py 里通过「from transformer.PLM.QWen.v3.Decoder import Decoder as NMT」
  把本文件的 Decoder 当作「整个模型」使用（NMT 实际只有 Decoder，无 Encoder）。
- 前向主流程在基类 transformer/PLM/LLMDecoder.py 的 forward：inputo（token id）-> wemb -> 逐层 nets -> pred_mask 筛选 -> out_normer -> classifier -> logits。
- 加载基座权重：train_lora_qwen.py 先 load_plm(pre_trained_m)，会逐层调用本文件 DecoderLayer.load_plm，从 .h5/.bin 的 state_dict 按 key 拷贝到本层；
  若启用 LoRA，之后会再对整网做 std2lora（见 utils/lora/base.py），把部分 Linear/Embedding 换成 LoRA 版。

本文件定义（LORA_LAYER_SELECTION_GUIDE §11 方案 A）：
- DecoderLayer：`ResSelfAttn` 与 `HPLSTM` 并行（同取 `inputo` / `query_unit`），`context = self_attn(...)` 再 `context = context + h`（`h = hplstm(...)`），最后 `ff`（首段 `context` 已含 attention 子层残差，勿再 `+ inputo`）。
- 另见 `Decoder_2.py`（方案 B）、`Decoder_3.py`（方案 C）。
- Decoder：继承 LLMDecoder；build_states / greedy_decode / beam_decode 等同 `Decoder.py`。
"""

import torch
from math import sqrt
from numbers import Integral
from torch import nn

# 项目内模块：Dropout、RMSNorm、Qwen3 的 FFN 与自注意力
from modules.base import Dropout
from modules.norm.base import RMSNorm
from modules.hplstm.snbase import HPLSTM
from modules.hplstm.snbase import ResHPLSTM
from modules.plm.qwen.v3 import PositionwiseFF, ResSelfAttn
# 基类：DecoderLayer 只要求 isize/fhsize 等；Decoder 继承 LLMDecoder，forward 在基类里
from transformer.Decoder import DecoderLayer as DecoderLayerBase
from transformer.PLM.LLMDecoder import Decoder as DecoderBase
# 工具：beam 解码时扩展状态、按索引取 tensor、repetition penalty、解析 None 等
from utils.base import index_tensors, select_zero_
from utils.decode.beam import expand_bsize_for_beam
from utils.decode.repan import is_penalty_enabled as is_repenalty_enabled, penalty as repenalty
from utils.fmt.parser import parse_none
# 加载预训练：按 key 拷贝参数、对齐 bias、load_plm_wrapper 装饰器
from utils.plm.base import align_linear_bias, copy_plm_parameter, load_plm_wrapper
from utils.relpos.base import share_rel_pos_cache
from utils.sampler import SampleMax
from utils.torch.comp import all_done, torch_any_wodim, torch_no_grad

# 配置与词表：Qwen3 超参来自 cnfg.plm.qwen.v3.ihyp；特殊 token id 来自 cnfg.vocab.plm.qwen.v3
from cnfg.plm.qwen.v3.ihyp import *
from cnfg.vocab.plm.qwen.v3 import eos_id, pad_id, sos_id


class DecoderLayer(DecoderLayerBase):
	"""
	方案 A：与 Decoder.py 相同先用 `context = self.self_attn(...)`，再 `h = HPLSTM(inputo/query_unit)`，
	`context = context + h`，再 `ff`。
	"""

	def __init__(self, isize, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, ahsize=None, norm_residual=norm_residual_default, k_rel_pos=use_k_relative_position_decoder, max_bucket_distance=relative_position_max_bucket_distance_decoder, num_kv_head=None, add_attn_qkv_bias=add_attn_qkv_bias, add_self_attn_qknorm=add_self_attn_qknorm, sliding_window=sliding_window, disable_ffn_bias=disable_ffn_bias, model_name="model", **kwargs):
		# isize=隐藏维度，fhsize=FFN 中间维（默认 4*ahsize），ahsize=注意力头维度
		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize
		super(DecoderLayer, self).__init__(isize, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, ahsize=_ahsize, norm_residual=norm_residual, k_rel_pos=k_rel_pos, max_bucket_distance=max_bucket_distance, **kwargs)
		self.model_name = model_name   # 加载 plm 时 key 前缀，如 "model"
		self.cross_attn = None   # Decoder-only 无 encoder，故无 cross attention
		# 自注意力：内部含 input_layernorm、q/k/v 投影、q_norm/k_norm（Qwen3）、o_proj、残差
		self.self_attn = ResSelfAttn(isize, hsize=_ahsize, num_head=num_head, dropout=attn_drop, norm_residual=norm_residual, k_rel_pos=k_rel_pos, uni_direction_reduction=True, max_bucket_distance=max_bucket_distance, num_kv_head=num_kv_head, add_attn_qkv_bias=add_attn_qkv_bias, add_self_attn_qknorm=add_self_attn_qknorm, sliding_window=sliding_window)
		# FFN：SwiGLU 风格 gate/up -> act -> down，含 post_attention_layernorm
		self.ff = PositionwiseFF(isize, hsize=_fhsize, dropout=dropout, act_drop=act_drop, norm_residual=norm_residual, disable_ffn_bias=disable_ffn_bias)
		self.normer = RMSNorm(isize, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters)
		# 见 modules/hplstm/base.HPLSTM；与 ResSelfAttn 同 num_head / act_drop 即可，其余 kwargs 勿乱传以免 MHPLSTMCore 报错
		self.hplstm = HPLSTM(isize, num_head=num_head, act_drop=attn_drop)
		self.reslstm = ResHPLSTM(isize, num_head=num_head, dropout=dropout, act_drop=act_drop, HPLSTM=HPLSTM)

	def forward(self, inputo, tgt_pad_mask=None, query_unit=None, slen=None, sliding_window_khead=None, **kwargs):
		"""
		两种用法：
		- query_unit is None（训练或整段前向）：inputo 为 (batch, seq, dim)，先 self_attn 再 ff，返回 (context,)。
		- query_unit 非空（自回归一步）：inputo 为上一段的 KV states，query_unit 为本步 query hidden；返回 (context, states_return) 供下一步用。
		"""
		if query_unit is None:
			context = self.self_attn(inputo, mask=tgt_pad_mask, slen=slen, sliding_window_khead=sliding_window_khead)
			h = self.hplstm(inputo)
		else:
			context, states_return = self.self_attn(query_unit, mask=tgt_pad_mask, states=inputo, slen=slen, sliding_window_khead=sliding_window_khead)
			h = self.hplstm(query_unit)
		context = context + h
		context = self.ff(context)
		return context if query_unit is None else (context, states_return,)

	@load_plm_wrapper()
	def load_plm(self, plm_parameters, model_name=None, layer_idx=None, **kwargs):
		"""
		从 plm_parameters（由 train_lora_qwen 里 load_plm(.h5/.bin) 读入的 state_dict）按 key 拷贝到本层。
		Key 命名与 HuggingFace Qwen 一致：model.layers.{layer_idx}.self_attn.* / input_layernorm / mlp.* / post_attention_layernorm。
		本项目中 self_attn 把 q/k/v 合并成一个 adaptor，FFN 把 gate/up 合并成 ff.net[0]，故多处用 copy_plm_parameter(..., func=torch.cat)。
		"""
		_model_name = parse_none(model_name, self.model_name)
		with torch_no_grad():
			# ---------- 自注意力：HF key -> 本层子模块。q/k/v 合并到 adaptor，o_proj -> outer，input_layernorm -> self_attn.normer ----------
			copy_plm_parameter(self.self_attn.net.adaptor.weight, plm_parameters, ["%s.layers.%d.self_attn.q_proj.weight" % (_model_name, layer_idx,), "%s.layers.%d.self_attn.k_proj.weight" % (_model_name, layer_idx,), "%s.layers.%d.self_attn.v_proj.weight" % (_model_name, layer_idx,)], func=torch.cat, func_kwargs={"dim": 0})
			_bias_key = "%s.layers.%d.self_attn.q_proj.bias" % (_model_name, layer_idx,)
			align_linear_bias(self.self_attn.net.adaptor, plm_parameters, _bias_key)
			if self.self_attn.net.adaptor.bias is not None:
				copy_plm_parameter(self.self_attn.net.adaptor.bias, plm_parameters, [_bias_key, "%s.layers.%d.self_attn.k_proj.bias" % (_model_name, layer_idx,), "%s.layers.%d.self_attn.v_proj.bias" % (_model_name, layer_idx,)], func=torch.cat, func_kwargs={"dim": 0})
			copy_plm_parameter(self.self_attn.net.outer.weight, plm_parameters, "%s.layers.%d.self_attn.o_proj.weight" % (_model_name, layer_idx,))
			_bias_key = "%s.layers.%d.self_attn.o_proj.bias" % (_model_name, layer_idx,)
			align_linear_bias(self.self_attn.net.outer, plm_parameters, _bias_key)
			if self.self_attn.net.outer.bias is not None:
				copy_plm_parameter(self.self_attn.net.outer.bias, plm_parameters, _bias_key)
			copy_plm_parameter(self.self_attn.normer.weight, plm_parameters, "%s.layers.%d.input_layernorm.weight" % (_model_name, layer_idx,))
			# Qwen3 可选：q_norm / k_norm（RMSNorm），若 plm 里有则创建并拷贝，否则删掉
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
			# ---------- FFN：gate_proj+up_proj 合并到 ff.net[0]，down_proj 在 ff 最后一层，post_attention_layernorm -> ff.normer ----------
			copy_plm_parameter(self.ff.net[0].weight, plm_parameters, ["%s.layers.%d.mlp.gate_proj.weight" % (_model_name, layer_idx,), "%s.layers.%d.mlp.up_proj.weight" % (_model_name, layer_idx,)], func=torch.cat, func_kwargs={"dim": 0})
			_l = self.ff.net[-2] if isinstance(self.ff.net[-1], Dropout) else self.ff.net[-1]
			copy_plm_parameter(_l.weight, plm_parameters, "%s.layers.%d.mlp.down_proj.weight" % (_model_name, layer_idx,))
			copy_plm_parameter(self.ff.normer.weight, plm_parameters, "%s.layers.%d.post_attention_layernorm.weight" % (_model_name, layer_idx,))


class Decoder(DecoderBase):
	"""
	Qwen3 Decoder-only 完整模型：wemb（词嵌入，可能被 std2lora 换成 LoRA Embedding）+ N×DecoderLayer + out_normer + classifier（lm_head）。
	训练时 forward 在基类 LLMDecoder 中；本文件实现 build_states（跑 prefix 得到 KV）、greedy_decode、beam_decode、_get_subsequent_mask，供验证/推理用。
	"""

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, emb_w=None, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, bindemb=True, forbidden_index=None, share_layer=False, disable_pemb=disable_std_pemb_decoder, num_kv_head=None, add_attn_qkv_bias=add_attn_qkv_bias, sliding_window=sliding_window, disable_ffn_bias=disable_ffn_bias, remove_classifier_bias=remove_classifier_bias, model_name="model", **kwargs):
		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize
		super(Decoder, self).__init__(isize, nwd, num_layer, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, emb_w=emb_w, num_head=num_head, xseql=xseql, ahsize=_ahsize, norm_output=norm_output, bindemb=bindemb, forbidden_index=forbidden_index, share_layer=True, disable_pemb=disable_pemb, remove_classifier_bias=remove_classifier_bias, model_name=model_name, **kwargs)
		self.sliding_window = sliding_window   # 注意力滑动窗口长度，用于 _get_subsequent_mask
		self.wemb.padding_idx = pad_id   # 与 cnfg.vocab 一致，padding 不参与梯度
		# nets：N 个 DecoderLayer。share_layer=True 时所有层共用同一套参数（本项目 Qwen 使用共享层）
		if share_layer:
			_shared_layer = DecoderLayer(isize, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, ahsize=_ahsize, num_kv_head=num_kv_head, add_attn_qkv_bias=add_attn_qkv_bias, sliding_window=sliding_window, disable_ffn_bias=disable_ffn_bias, model_name=model_name)
			self.nets = nn.ModuleList([_shared_layer for i in range(num_layer)])
		else:
			self.nets = nn.ModuleList([DecoderLayer(isize, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, ahsize=_ahsize, num_kv_head=num_kv_head, add_attn_qkv_bias=add_attn_qkv_bias, sliding_window=sliding_window, disable_ffn_bias=disable_ffn_bias, model_name=model_name) for i in range(num_layer)])
		self.out_normer = RMSNorm(isize, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters) if norm_output else None   # 最后一层 norm，对应 HF 的 model.norm
		if rel_pos_enabled:
			share_rel_pos_cache(self)   # 若用相对位置编码，在多层间共享 RoPE 等缓存

	def build_states(self, inpute, states=None, return_last_hidden=False, block_size=0, slen=None, sliding_window_khead=None, **kwargs):
		"""
		对一段 token 做前向，更新并返回每层的 KV 状态（及可选「最后位置 hidden」），供自回归解码或验证时复用 prefix。
		【调用方】eva() 里会先对「指令+回答」中的指令部分跑 build_states(..., return_last_hidden=True) 得到 prefix_states，再只对「回答」部分算 loss；greedy_decode/beam_decode 也会先对 inpute 跑 build_states 再逐步生成。
		参数: inpute (batch, nquery), states=上一段的状态 dict, return_last_hidden=是否返回最后位置 hidden, block_size=分块长度(>0 时), slen=已有长度, sliding_window_khead=滑动窗口.
		返回: 若 return_last_hidden 则 (最后位置 hidden, _states)；否则 _states。
		"""
		_states = {} if states is None else states.copy()
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

		# 若 block_size>0 且 nquery 大，则按块过每层并更新 _states（省显存）；否则整段一次过
		if (block_size > 0) and (nquery > block_size):
			_sid = _slen
			while _sid < _rslen:
				_eid = min(_sid + block_size, _rslen)
				_mask = self._get_subsequent_mask(_eid, sid=_sid, lsid=(_sid - (0 if (_states.get(0, (None, None,))[0] is None) else (_states.get(0, (None, None,))[0].size(-1)))) if self.sliding_window > 0 else 0)
				_out = out.narrow(1, _sid - _slen, _eid - _sid)
				for _tmp, net in enumerate(self.nets):
					_out, _state = net(_states.get(_tmp, (None, None,)), tgt_pad_mask=_mask, query_unit=_out, slen=_sid, sliding_window_khead=_sliding_window_khead)
					_states[_tmp] = _state
				_sid = _eid
			out = _out
		else:
			_mask = self._get_subsequent_mask(_rslen, sid=_slen)
			for _tmp, net in enumerate(self.nets):
				out, _state = net(_states.get(_tmp, (None, None,)), tgt_pad_mask=_mask, query_unit=out, slen=_slen, sliding_window_khead=_sliding_window_khead)
				_states[_tmp] = _state   # 每层保存 (k, v) 等，供下一步自回归时 query_unit 用

		if return_last_hidden:
			out = out.narrow(1, -1, 1)
			if self.out_normer is not None:
				out = self.out_normer(out)
			return out, _states
		return _states

	def greedy_decode(self, inpute, max_len=512, fill_pad=False, sample=False, top_k=1, top_p=0.0, temp=1.0, repetition_penalty=1.0, ilen=None, post_ilen_rs=True, states=None, slen=None, sliding_window_khead=None, **kwargs):
		"""
		自回归贪心或采样解码：先对 inpute（如 prompt）跑 build_states 得到 last hidden，再一步步过 wemb -> 各层 query_unit -> norm -> classifier -> 采样，直到达到 max_len 或全部生成 eos。
		【调用方】基类 LLMDecoder.decode() 在 beam_size==1 时调用本方法；推理/预测脚本（如 adv/predict/plm/qwen/）会用到。
		参数: inpute (bsize, nquery), max_len, fill_pad=是否用 pad_id 填已结束序列, sample/top_k/top_p/temp=采样, repetition_penalty, ilen=每样本有效长度, post_ilen_rs=是否按 ilen 截断返回, states/slen/sliding_window_khead.
		返回: trans，list 或 tensor 视 post_ilen_rs 与 ilen。
		"""
		bsize, nquery = inpute.size()
		_states = {} if states is None else states
		_sliding_window_khead = None if sliding_window_khead is None else (sliding_window_khead if isinstance(sliding_window_khead, Integral) else nquery)
		if slen is None:
			_ = _states.get(0, (None, None,))[0]
			_slen = 0 if _ is None else _.size(-1)
		else:
			_slen = slen
		_inpute_slen = _slen
		if ilen is None:
			_nquery, _inpute = nquery, inpute
		else:
			_nquery = ilen.min().item()
			_inpute = inpute.narrow(-1, 0, _nquery - _inpute_slen)
		out, _states = self.build_states(_inpute, states=_states, return_last_hidden=True, slen=_slen, sliding_window_khead=_sliding_window_khead)
		if self.pemb is not None:
			sqrt_isize = sqrt(out.size(-1))
		_slen += _nquery
		# 第一步：用 prefix 最后位置的 hidden 过 classifier 得到第一个生成 token 的 logits
		out = self.classifier(out)
		wds = SampleMax(out, dim=-1, keepdim=False, sample=sample, top_k=top_k, top_p=top_p, temp=temp)
		done_trans = wds.eq(eos_id)
		_use_repan = is_repenalty_enabled(repetition_penalty)
		if ilen is None:
			trans = wds if _use_repan else [wds]
		else:
			_ = ilen.gt(_nquery).unsqueeze(-1)
			wds[_] = inpute.narrow(-1, _nquery - _inpute_slen, 1)[_]
			done_trans &= ~_
			trans = wds.masked_fill(_, sos_id) if _use_repan else [wds]

		# 自回归步：每步用上一步采样的 token id (wds) 过 wemb -> 各层 query_unit（带 _states）-> norm -> classifier -> 采样，并更新 _states
		for i in range(_nquery, nquery + max_len):
			out = self.wemb(wds)
			if self.pemb is not None:
				out = self.pemb.get_pos(i).add(out, alpha=sqrt_isize)
			if self.drop is not None:
				out = self.drop(out)
			for _tmp, net in enumerate(self.nets):
				out, _state = net(_states[_tmp], tgt_pad_mask=None, query_unit=out, slen=_slen, sliding_window_khead=_sliding_window_khead)
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
				wds[_] = inpute.narrow(-1, _ni - _inpute_slen, 1)[_]
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
		if post_ilen_rs:
			trans = trans.tolist() if ilen is None else [_t[_:] for _t, _ in zip(trans.tolist(), (ilen - _nquery).tolist())]

		return trans

	def beam_decode(self, inpute, beam_size=8, max_len=512, length_penalty=0.0, return_all=False, clip_beam=clip_beam_with_lp, fill_pad=False, repetition_penalty=1.0, ilen=None, post_ilen_rs=True, states=None, slen=None, sliding_window_khead=None, **kwargs):
		"""
		Beam search 解码：先 build_states 得到 last hidden，再每步扩展 beam_size 个候选，按 score（可选 length_penalty）保留 top beam_size，直至达 max_len 或全部 eos。
		【调用方】基类 LLMDecoder.decode() 在 beam_size>1 时调用本方法。
		参数: inpute, beam_size, max_len, length_penalty, return_all=是否返回所有 beam, clip_beam, fill_pad, repetition_penalty, ilen, post_ilen_rs, states, slen, sliding_window_khead.
		返回: trans（若 return_all 则 (trans, scores)）。
		"""
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
		_inpute_slen = _slen
		if ilen is None:
			_nquery, _inpute, _lpv_rs = nquery, inpute, None
		else:
			_nquery = ilen.min().item()
			_inpute, _csteps = inpute.narrow(-1, 0, _nquery - _inpute_slen), (ilen - _nquery)
			_lpv_rs = _csteps
		out, _states = self.build_states(_inpute, states=_states, return_last_hidden=True, slen=_slen, sliding_window_khead=_sliding_window_khead)
		if self.pemb is not None:
			sqrt_isize = sqrt(out.size(-1))
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
			wds[_] = inpute.narrow(-1, _nquery - _inpute_slen, 1).unsqueeze(-1).expand(-1, -1, beam_size)[_]
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
				out, _state = net(_states[_tmp], tgt_pad_mask=None, query_unit=out, slen=_slen, sliding_window_khead=_sliding_window_khead)
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
					wds.masked_scatter_(_rbm, inpute.narrow(-1, _nstep - _inpute_slen, 1)[_].unsqueeze(-1).expand(-1, beam_size))
					_done_trans &= (~_).repeat(1, beam_size).view(real_bsize)
				else:
					_rbm = None
				_ = ilen.eq(_nstep)
				if torch_any_wodim(_).item():
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
			if post_ilen_rs:
				trans = trans.tolist() if ilen is None else [[_tu[_:] for _tu in _t] for _t, _ in zip(trans.tolist(), _csteps.tolist())]

			return trans, scores
		else:
			trans = trans.select(1, 0)
			if post_ilen_rs:
				trans = trans.tolist() if ilen is None else [_t[_:] for _t, _ in zip(trans.tolist(), _csteps.tolist())]

			return trans

	def _get_subsequent_mask(self, length, sid=0, lsid=0, **kwargs):
		"""
		构造因果（+ 可选滑动窗口）的 attention mask：下三角为可见，其余位置 mask 掉；若 self.sliding_window>0 则只保留窗口内的 key 可见（用于长序列省显存）。
		在 build_states、greedy_decode、beam_decode 里会传给每层 DecoderLayer.forward 的 tgt_pad_mask。
		参数: length=总长, sid=当前段起始位置, lsid=key 的起始位置。
		返回: (1, length-sid, length-lsid) 的 mask，True/1 表示要 mask 掉的位置。
		"""
		_ = length - sid
		_l = length - lsid
		# 因果 mask：下三角可见。若预分配了 self.mask 且 length<=xseql 则切片；否则现场 triu 得到「上三角为 1」
		_mask = self.mask.narrow(1, sid, _).narrow(2, lsid, _l).contiguous() if length <= self.xseql else self.mask.new_ones(_, _l).triu(1 + sid - lsid).unsqueeze(0)
		_sliding_window = self.sliding_window
		# 滑动窗口：只保留最近 _sliding_window 个 key 可见，更早的也 mask 掉
		if (_sliding_window > 0) and (_l > _sliding_window):
			_mask = _mask | _mask.new_ones(_, _l).tril(sid - _sliding_window - lsid).unsqueeze(0)
		return _mask
