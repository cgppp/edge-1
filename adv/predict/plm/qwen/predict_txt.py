#encoding: utf-8

# usage: python predict.py $rsf $/path/to/tokenizer $srcf $model

import sys
import torch
from transformers import AutoTokenizer

from transformer.PLM.QWen.v3.Decoder import Decoder as NMT
from utils.base import set_random_seed
from utils.fmt.base import merge_rchar, sys_open
from utils.fmt.base4torch import parse_cuda_decode
from utils.io import load_model_cpu
from utils.norm.mp.f import convert as make_mp_model
from utils.plm.inference import get_of_common_prefix
from utils.quant.s.base import quant
from utils.torch.comp import torch_autocast, torch_compile, torch_inference_mode
from utils.tqdm import tqdm

import cnfg.lora as lcnfg
import cnfg.plm.qwen.v3.base as cnfg
import cnfg.quant as qcnfg
from cnfg.plm.qwen.v3.ihyp import *
from cnfg.vocab.plm.qwen.v3 import eos_id, eot_id, instruct_task_template, vocab_size

max_len = 512
line_func = lambda x: instruct_task_template("You are a helpful assistant.", x)
ext_txt = lambda x: merge_rchar((" " if _ in "\n\r" else _ for _ in x), rchar=" ")

def ext_ids(lin):

	rs = []
	for _ in lin:
		if _ == eot_id:
			rs.clear()
		elif _ == eos_id:
			break
		else:
			rs.append(_)

	return rs

def load_fixing(module):

	if hasattr(module, "fix_load"):
		module.fix_load()

use_cuda, cuda_device, cuda_devices, multi_gpu, use_amp, use_cuda_bfmp, use_cuda_fp16 = parse_cuda_decode(cnfg.use_cuda, gpuid=cnfg.gpuid, use_amp=cnfg.use_amp, multi_gpu_decoding=cnfg.multi_gpu_decoding, use_cuda_bfmp=cnfg.use_cuda_bfmp)
set_random_seed(cnfg.seed, use_cuda)

tokenizer = AutoTokenizer.from_pretrained(sys.argv[2])

mymodel = NMT(cnfg.isize, vocab_size, cnfg.nlayer, fhsize=cnfg.ff_hsize, dropout=cnfg.drop, attn_drop=cnfg.attn_drop, act_drop=cnfg.act_drop, emb_w=None, num_head=cnfg.nhead, xseql=cache_len_default, ahsize=cnfg.attn_hsize, norm_output=cnfg.norm_output, bindemb=cnfg.bindDecoderEmb, num_kv_head=cnfg.kv_nhead, model_name=cnfg.model_name)

if len(sys.argv) < 5:
	if cnfg.pre_trained_m is not None:
		mymodel.load_plm(cnfg.pre_trained_m)
else:
	mymodel = load_model_cpu(sys.argv[4], mymodel)
	mymodel.apply(load_fixing)
mymodel.eval()

if use_cuda_bfmp:
	make_mp_model(mymodel)
elif use_cuda_fp16:
	mymodel.to(torch.float16, non_blocking=True)
if qcnfg.use_quant:
	mymodel = quant(mymodel, quant_linear=qcnfg.quant_linear, quant_embedding=qcnfg.quant_embedding, quant_normer=qcnfg.quant_normer, quant_log_shift=qcnfg.quant_log_shift, quant_dim=qcnfg.quant_dim, quant_weight=qcnfg.quant_normer_weight, quant_bias=qcnfg.quant_bias, quant_io=qcnfg.quant_io, name_cfunc=qcnfg.name_cfunc, keep_tying=True)[0]
if cuda_device:
	mymodel.to(cuda_device, non_blocking=True)

mymodel = torch_compile(mymodel, *torch_compile_args, **torch_compile_kwargs)

beam_size = cnfg.beam_size
length_penalty = cnfg.length_penalty

ens = "\n".encode("utf-8")
with sys_open(sys.argv[3], "rb") as fsrc, sys_open(sys.argv[1], "wb") as frs, torch_inference_mode():
	prefix_ids = lcnfg.prefix_ids
	if prefix_ids:
		prefix_len = len(prefix_ids)
	elif lcnfg.find_common_prefix:
		prefix_ids = get_of_common_prefix(fsrc, line_func=lambda x: tokenizer.encode(line_func(x.decode("utf-8"))), reset=True)
		prefix_len = len(prefix_ids) if prefix_ids else None
	else:
		prefix_len = prefix_states = None
	if prefix_len is not None:
		seq_batch = torch.as_tensor(prefix_ids, dtype=torch.int32).unsqueeze(0)
		if cuda_device:
			seq_batch = seq_batch.to(cuda_device, non_blocking=True)
		seq_batch = seq_batch.to(torch.int64, non_blocking=True)
		prefix_states = mymodel.build_states(seq_batch, states=None, return_last_hidden=False)
	for line in tqdm(fsrc, mininterval=tqdm_mininterval):
		_ = line.strip()
		if _:
			_ = tokenizer.encode(line_func(_.decode("utf-8")))
			if prefix_len is not None:
				_ = _[prefix_len:]
			seq_batch = torch.as_tensor(_, dtype=torch.int32).unsqueeze(0)
			if cuda_device:
				seq_batch = seq_batch.to(cuda_device, non_blocking=True)
			seq_batch = seq_batch.to(torch.int64, non_blocking=True)
			with torch_autocast(enabled=use_amp):
				output = mymodel.decode(seq_batch, beam_size=beam_size, max_len=max_len, length_penalty=length_penalty, post_ilen_rs=False, states=prefix_states)
			output = ext_ids(output.squeeze().tolist())
			frs.write(ext_txt(tokenizer.decode(output, skip_special_tokens=False, clean_up_tokenization_spaces=False).strip()).strip().encode("utf-8"))
		frs.write(ens)
