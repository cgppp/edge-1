#encoding: utf-8

# usage: python rank.py $rsf $h5f $model

norm_token = True

import sys
import torch

from loss.base import LabelSmoothingLoss
from transformer.PLM.QWen.v3.Decoder import Decoder as NMT
from utils.base import set_random_seed
from utils.fmt.base import iter_to_str, sys_open
from utils.fmt.base4torch import parse_cuda
from utils.h5serial import h5File
from utils.io import load_model_cpu
from utils.norm.mp.f import convert as make_mp_model
from utils.plm.inference import get_h5g_common_prefix, prepare_states_bsize
from utils.quant.s.base import quant
from utils.torch.comp import torch_autocast, torch_compile, torch_inference_mode
from utils.torch.ext import squeeze_sum
from utils.tqdm import tqdm
from utils.train.llm import PMaskDataConverter

import cnfg.lora as lcnfg
import cnfg.plm.qwen.v3.base as cnfg
import cnfg.quant as qcnfg
from cnfg.plm.qwen.v3.ihyp import *
from cnfg.vocab.plm.qwen.v3 import vocab_size

def load_fixing(module):

	if hasattr(module, "fix_load"):
		module.fix_load()

use_cuda, cuda_device, cuda_devices, multi_gpu, use_amp, use_cuda_bfmp, use_cuda_fp16 = parse_cuda(cnfg.use_cuda, gpuid=cnfg.gpuid, use_amp=cnfg.use_amp, use_cuda_bfmp=cnfg.use_cuda_bfmp)
set_random_seed(cnfg.seed, use_cuda)

data_converter = PMaskDataConverter(xseql=cache_len_default, device=cuda_device)

td = h5File(sys.argv[2], "r", **h5_fileargs)

ntest = td["ndata"][()].item()

prefix_ids = lcnfg.prefix_ids
if prefix_ids:
	prefix_len = len(prefix_ids)
elif lcnfg.find_common_prefix:
	prefix_ids = get_h5g_common_prefix(td["src"])
	prefix_len = len(prefix_ids) if prefix_ids else None
else:
	prefix_len = None

mymodel = NMT(cnfg.isize, vocab_size, cnfg.nlayer, fhsize=cnfg.ff_hsize, dropout=cnfg.drop, attn_drop=cnfg.attn_drop, act_drop=cnfg.act_drop, emb_w=None, num_head=cnfg.nhead, xseql=cache_len_default, ahsize=cnfg.attn_hsize, norm_output=cnfg.norm_output, bindemb=cnfg.bindDecoderEmb, num_kv_head=cnfg.kv_nhead, model_name=cnfg.model_name)

if len(sys.argv) < 4:
	if cnfg.pre_trained_m is not None:
		mymodel.load_plm(cnfg.pre_trained_m)
else:
	mymodel = load_model_cpu(sys.argv[3], mymodel)
	mymodel.apply(load_fixing)
mymodel.eval()

lossf = LabelSmoothingLoss(vocab_size, cnfg.label_smoothing, ignore_index=-1, reduction="none", forbidden_index=cnfg.forbidden_indexes)

if use_cuda_bfmp:
	make_mp_model(mymodel)
elif use_cuda_fp16:
	mymodel.to(torch.float16, non_blocking=True)
if qcnfg.use_quant:
	mymodel = quant(mymodel, quant_linear=qcnfg.quant_linear, quant_embedding=qcnfg.quant_embedding, quant_normer=qcnfg.quant_normer, quant_log_shift=qcnfg.quant_log_shift, quant_dim=qcnfg.quant_dim, quant_weight=qcnfg.quant_normer_weight, quant_bias=qcnfg.quant_bias, quant_io=qcnfg.quant_io, name_cfunc=qcnfg.name_cfunc, keep_tying=True)[0]
if cuda_device:
	mymodel.to(cuda_device, non_blocking=True)
	lossf.to(cuda_device, non_blocking=True)

mymodel = torch_compile(mymodel, *torch_compile_args, **torch_compile_kwargs)
lossf = torch_compile(lossf, *torch_compile_args, **torch_compile_kwargs)

ens = "\n".encode("utf-8")

src_grp, tgt_grp = td["src"], td["tgt"]
with sys_open(sys.argv[1], "wb") as f, torch_inference_mode():
	if prefix_ids:
		seq_batch = torch.as_tensor(prefix_ids, dtype=torch.int32).unsqueeze(0)
		if cuda_device:
			seq_batch = seq_batch.to(cuda_device, non_blocking=True)
		seq_batch = seq_batch.to(torch.int64, non_blocking=True)
		prefix_states = mymodel.build_states(seq_batch, states=None, return_last_hidden=False)
	else:
		prefix_states = None
	for i in tqdm(range(ntest), mininterval=tqdm_mininterval):
		_curid = str(i)
		seq_batch = torch.from_numpy(src_grp[_curid][()])
		seq_o = torch.from_numpy(tgt_grp[_curid][()])
		if prefix_len:
			seq_batch = seq_batch.narrow(-1, prefix_len, seq_batch.size(-1) - prefix_len)
		if cuda_device:
			seq_batch = seq_batch.to(cuda_device, non_blocking=True)
			# seq_o device movement is handled by data_converter in case necessary
			#seq_o = seq_o.to(cuda_device, non_blocking=True)
		seq_batch = seq_batch.to(torch.int64, non_blocking=True)
		oi, pred_mask, ot = data_converter(seq_batch, seq_o, seq_o_sub_len=prefix_len)
		with torch_autocast(enabled=use_amp):
			output = mymodel(oi, word_prediction=True, pred_mask=pred_mask, states=prepare_states_bsize(prefix_states, bsize=seq_batch.size(0)))
			loss = squeeze_sum(lossf(output, ot).view(ot.size(0), -1), -1)
			if pred_mask is not None:
				loss = loss.new_zeros(oi.size(), dtype=loss.dtype, device=loss.device).masked_scatter_(pred_mask, loss).sum(-1)
		if norm_token:
			lenv = float(ot.size(-1)) if pred_mask is None else pred_mask.to(torch.int32, non_blocking=True).sum(-1).to(loss, non_blocking=True)
			loss = loss / lenv
		f.write("\n".join(iter_to_str(loss.tolist())).encode("utf-8"))
		f.write(ens)
		loss = output = oi = ot = seq_batch = seq_o = pred_mask = None

td.close()
