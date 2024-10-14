#encoding: utf-8

import torch
from math import ceil

from parallel.parallelMT import DataParallelMT
from transformer.GECToR.KBNMT import NMT
from utils.fmt.base import dict_insert_set, get_bsize, iter_dict_sort
from utils.fmt.base4torch import parse_cuda_decode
from utils.fmt.gec.kb.base import merge_src_kb
from utils.fmt.gec.kb.dyndual import batch_padder
from utils.fmt.plm.custbert.token import Tokenizer
from utils.fmt.vocab.base import reverse_dict
from utils.io import load_model_cpu
from utils.torch.comp import torch_autocast, torch_compile, torch_inference_mode

from cnfg.ihyp import *
from cnfg.vocab.plm.custbert import eos_id, sos_id, vocab_size

def load_fixing(module):
	if hasattr(module, "fix_load"):
		module.fix_load()

def sorti(lin):

	data = {}
	for ls in lin:
		data = dict_insert_set(data, ls, len(ls[0]))
	for _ in iter_dict_sort(data, free=True):
		yield from _

class Handler:

	def __init__(self, modelfs, cnfg, minbsize=1, expand_for_mulgpu=True, bsize=max_sentences_gpu, maxpad=max_pad_tokens_sentence, maxpart=normal_tokens_vs_pad_tokens, maxtoken=max_tokens_gpu, norm_u8=False, **kwargs):

		self.tokenizer = Tokenizer(cnfg.plm_vcb, norm_u8=norm_u8, post_norm_func=None, split=False)
		self.vcbt = reverse_dict(self.tokenizer.vcb)

		if expand_for_mulgpu:
			self.bsize = bsize * minbsize
			self.maxtoken = maxtoken * minbsize
		else:
			self.bsize = bsize
			self.maxtoken = maxtoken
		self.maxpad = maxpad
		self.maxpart = maxpart
		self.minbsize = minbsize

		model = NMT(cnfg.isize, vocab_size, vocab_size, cnfg.nlayer, fhsize=cnfg.ff_hsize, dropout=cnfg.drop, attn_drop=cnfg.attn_drop, act_drop=cnfg.act_drop, global_emb=cnfg.share_emb, num_head=cnfg.nhead, xseql=cache_len_default, ahsize=cnfg.attn_hsize, norm_output=cnfg.norm_output, bindDecoderEmb=cnfg.bindDecoderEmb, forbidden_index=cnfg.forbidden_indexes)
		model.build_task_model(fix_init=False)
		model = load_model_cpu(modelfs, model)
		model.apply(load_fixing)
		model.eval()
		self.use_cuda, self.cuda_device, cuda_devices, self.multi_gpu = parse_cuda_decode(cnfg.use_cuda, cnfg.gpuid, cnfg.multi_gpu_decoding)
		if self.use_cuda:
			model.to(self.cuda_device, non_blocking=True)
			if self.multi_gpu:
				model = DataParallelMT(model, device_ids=cuda_devices, output_device=self.cuda_device.index, host_replicate=True, gather_output=False)
		self.net = torch_compile(model, *torch_compile_args, **torch_compile_kwargs)
		self.use_amp = cnfg.use_amp and self.use_cuda
		self.beam_size = cnfg.beam_size
		self.length_penalty = cnfg.length_penalty
		self.op_keep_bias = cnfg.op_keep_bias
		self.edit_thres = cnfg.edit_thres

	def __call__(self, sentences_iter, **kwargs):

		_tok_ids = [merge_src_kb(tuple(self.tokenizer(_s)), tuple(self.tokenizer(_k))) for _s, _k in sentences_iter]
		_sorted_token_ids = list(sorti(_tok_ids))
		_vcbt, _cuda_device, _multi_gpu, _use_amp = self.vcbt, self.cuda_device, self.multi_gpu, self.use_amp
		rs = []
		with torch_inference_mode():
			for seq_batch, seq_kb in batch_padder(_sorted_token_ids, self.bsize, self.maxpad, self.maxpart, self.maxtoken, self.minbsize):
				_seq_batch = torch.as_tensor(seq_batch, dtype=torch.long, device=_cuda_device)
				_seq_kb = torch.as_tensor(seq_kb, dtype=torch.long, device=_cuda_device)
				with torch_autocast(enabled=_use_amp):
					output = self.net.decode(_seq_batch, kb=_seq_kb, beam_size=self.beam_size, max_len=None, length_penalty=self.length_penalty, op_keep_bias=self.op_keep_bias, edit_thres=self.edit_thres)
				if _multi_gpu:
					tmp = []
					for ou in output:
						tmp.extend(ou)
					output = tmp
				for tran in output:
					tmp = []
					_ = tran.tolist()
					if _[0] == sos_id:
						_ = _[1:]
					for tmpu in _:
						if tmpu == eos_id:
							break
						else:
							tmp.append(_vcbt[tmpu])
					rs.append("".join(tmp))
				_seq_batch = _seq_kb = None
		_mapd = {_k: _v for _k, _v in zip(_sorted_token_ids, rs)}

		return [_mapd.get(_, "") for _ in _tok_ids]
