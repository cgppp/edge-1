#encoding: utf-8

import sys
import torch

from parallel.parallelMT import DataParallelMT
from transformer.EnsembleNMT import NMT as Ensemble
from transformer.NMT import NMT
from utils.base import set_random_seed
from utils.h5serial import h5File
from utils.io import load_model_cpu
from utils.norm.mp.f import convert as make_mp_model
from utils.torch.comp import torch_compile, torch_inference_mode
from utils.tqdm import tqdm

import cnfg.base as cnfg
from cnfg.ihyp import *

def load_fixing(module):

	if hasattr(module, "fix_load"):
		module.fix_load()

use_cuda, cuda_device, cuda_devices, multi_gpu, use_amp, use_cuda_bfmp = parse_cuda(cnfg.use_cuda, gpuid=cnfg.gpuid, use_amp=cnfg.use_amp, use_cuda_bfmp=cnfg.use_cuda_bfmp)
set_random_seed(cnfg.seed, use_cuda)

td = h5File(cnfg.dev_data, "r", **h5_fileargs)

ntest = td["ndata"][()].item()
nword = td["nword"][()].tolist()
nwordi, nwordt = nword[0], nword[-1]

if len(sys.argv) == 2:
	mymodel = NMT(cnfg.isize, nwordi, nwordt, cnfg.nlayer, fhsize=cnfg.ff_hsize, dropout=cnfg.drop, attn_drop=cnfg.attn_drop, act_drop=cnfg.act_drop, global_emb=cnfg.share_emb, num_head=cnfg.nhead, xseql=cache_len_default, ahsize=cnfg.attn_hsize, norm_output=cnfg.norm_output, bindDecoderEmb=cnfg.bindDecoderEmb, forbidden_index=cnfg.forbidden_indexes)
	if use_cuda_bfmp:
		make_mp_model(mymodel)

	mymodel = load_model_cpu(sys.argv[1], mymodel)
	mymodel.apply(load_fixing)

else:
	models = []
	for modelf in sys.argv[1:]:
		tmp = NMT(cnfg.isize, nwordi, nwordt, cnfg.nlayer, fhsize=cnfg.ff_hsize, dropout=cnfg.drop, attn_drop=cnfg.attn_drop, act_drop=cnfg.act_drop, global_emb=cnfg.share_emb, num_head=cnfg.nhead, xseql=cache_len_default, ahsize=cnfg.attn_hsize, norm_output=cnfg.norm_output, bindDecoderEmb=cnfg.bindDecoderEmb, forbidden_index=cnfg.forbidden_indexes)
		if use_cuda_bfmp:
			make_mp_model(tmp)

		tmp = load_model_cpu(modelf, tmp)
		tmp.apply(load_fixing)

		models.append(tmp)
	mymodel = Ensemble(models)

mymodel.eval()

if cuda_device:
	mymodel.to(cuda_device, non_blocking=True)
	if multi_gpu:
		mymodel = DataParallelMT(mymodel, device_ids=cuda_devices, output_device=cuda_device.index, host_replicate=True, gather_output=False)

mymodel = torch_compile(mymodel, *torch_compile_args, **torch_compile_kwargs)

beam_size = cnfg.beam_size
length_penalty = cnfg.length_penalty

src_grp = td["src"]
with torch_inference_mode():
	for i in tqdm(range(ntest), mininterval=tqdm_mininterval):
		seq_batch = torch.from_numpy(src_grp[str(i)][()])
		if cuda_device:
			seq_batch = seq_batch.to(cuda_device, non_blocking=True)
		seq_batch = seq_batch.to(torch.int64, non_blocking=True)
		output = mymodel.decode(seq_batch, beam_size, None, length_penalty)

td.close()
