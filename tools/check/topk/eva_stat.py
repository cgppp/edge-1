#encoding: utf-8

""" usage:
	python eva_stat.py $input_file.h5 k $model_file.h5 ...
"""

import sys
import torch
from torch import nn

from parallel.parallelMT import DataParallelMT
from transformer.EnsembleNMT import NMT as Ensemble
from transformer.NMT import NMT
from utils.base import set_random_seed
from utils.fmt.base4torch import parse_cuda_decode
from utils.h5serial import h5File
from utils.io import load_model_cpu
from utils.norm.mp.f import convert as make_mp_model
from utils.torch.comp import torch_autocast, torch_compile, torch_inference_mode
from utils.tqdm import tqdm

import cnfg.base as cnfg
from cnfg.ihyp import *
from cnfg.vocab.base import pad_id

def load_fixing(module):

	if hasattr(module, "fix_load"):
		module.fix_load()

k = int(sys.argv[2])

use_cuda, cuda_device, cuda_devices, multi_gpu, use_amp, use_cuda_bfmp, use_cuda_fp16 = parse_cuda_decode(cnfg.use_cuda, gpuid=cnfg.gpuid, use_amp=cnfg.use_amp, multi_gpu_decoding=cnfg.multi_gpu_decoding, use_cuda_bfmp=cnfg.use_cuda_bfmp)
set_random_seed(cnfg.seed, use_cuda)

td = h5File(sys.argv[1], "r", **h5_fileargs)

ntest = td["ndata"][()].item()
nword = td["nword"][()].tolist()
nwordi, nwordt = nword[0], nword[-1]

if len(sys.argv) == 4:
	mymodel = NMT(cnfg.isize, nwordi, nwordt, cnfg.nlayer, fhsize=cnfg.ff_hsize, dropout=cnfg.drop, attn_drop=cnfg.attn_drop, act_drop=cnfg.act_drop, global_emb=cnfg.share_emb, num_head=cnfg.nhead, xseql=cache_len_default, ahsize=cnfg.attn_hsize, norm_output=cnfg.norm_output, bindDecoderEmb=cnfg.bindDecoderEmb, forbidden_index=cnfg.forbidden_indexes)

	mymodel = load_model_cpu(sys.argv[3], mymodel)
	mymodel.apply(load_fixing)
	mymodel.dec.lsm = nn.Softmax(-1)
	if use_cuda_bfmp:
		make_mp_model(mymodel)
	elif use_cuda_fp16:
		mymodel.to(torch.float16, non_blocking=True)
else:
	models = []
	for modelf in sys.argv[3:]:
		tmp = NMT(cnfg.isize, nwordi, nwordt, cnfg.nlayer, fhsize=cnfg.ff_hsize, dropout=cnfg.drop, attn_drop=cnfg.attn_drop, act_drop=cnfg.act_drop, global_emb=cnfg.share_emb, num_head=cnfg.nhead, xseql=cache_len_default, ahsize=cnfg.attn_hsize, norm_output=cnfg.norm_output, bindDecoderEmb=cnfg.bindDecoderEmb, forbidden_index=cnfg.forbidden_indexes)

		tmp = load_model_cpu(modelf, tmp)
		tmp.apply(load_fixing)
		# dec.lsm is NOT used by Ensemble
		tmp.dec.lsm = nn.Softmax(-1)
		if use_cuda_bfmp:
			make_mp_model(tmp)
		elif use_cuda_fp16:
			tmp.to(torch.float16, non_blocking=True)

		models.append(tmp)
	mymodel = Ensemble(models)

mymodel.eval()

if cuda_device:
	mymodel.to(cuda_device, non_blocking=True)
	if multi_gpu:
		mymodel = DataParallelMT(mymodel, device_ids=cuda_devices, output_device=cuda_device.index, host_replicate=True, gather_output=True)

mymodel = torch_compile(mymodel, *torch_compile_args, **torch_compile_kwargs)

beam_size = cnfg.beam_size
length_penalty = cnfg.length_penalty

src_grp, tgt_grp = td["src"], td["tgt"]
nword = 0
if cuda_device:
	cum_p = torch.zeros(k, dtype=torch.double, device=cuda_device)
	m_ind = torch.zeros(k, dtype=torch.long, device=cuda_device)
else:
	cum_p = torch.zeros(k, dtype=torch.double)
	m_ind = torch.zeros(k, dtype=torch.long)
with torch_inference_mode():
	for i in tqdm(range(ntest), mininterval=tqdm_mininterval):
		bid = str(i)
		seq_batch = torch.from_numpy(src_grp[bid][()])
		seq_o = torch.from_numpy(tgt_grp[bid][()])
		lo = seq_o.size(1) - 1
		if cuda_device:
			seq_batch = seq_batch.to(cuda_device, non_blocking=True)
			seq_o = seq_o.to(cuda_device, non_blocking=True)
		seq_batch, seq_o = seq_batch.to(torch.int64, non_blocking=True), seq_o.to(torch.int64, non_blocking=True)
		with torch_autocast(enabled=use_amp):
			output = mymodel(seq_batch, seq_o.narrow(1, 0, lo))
		tgt = seq_o.narrow(1, 1, lo)
		mask = tgt.eq(pad_id).unsqueeze(-1)
		p, ind = output.masked_fill_(mask, 0.0).topk(k, dim=-1)
		data_mask = ~mask
		cum_p.add_(p.view(-1, k).sum(0).to(torch.float64, non_blocking=True))
		m_ind.add_((ind.eq(tgt.unsqueeze(-1)) & data_mask).view(-1, k).to(torch.int64, non_blocking=True).sum(0))
		nword += data_mask.to(torch.int32, non_blocking=True).sum().item()
	nword = float(nword) / 100.0
	cum_p = cum_p.div_(nword).cumsum(-1)
	m_ind = m_ind.cumsum(-1).to(torch.float64, non_blocking=True).div_(nword)

td.close()

print(cum_p.tolist())
print(m_ind.tolist())
