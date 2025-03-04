#encoding: utf-8

import sys
import torch

from parallel.parallelMT import DataParallelMT
from transformer.Doc.Para.Base.NMT import NMT
from transformer.EnsembleNMT import NMT as Ensemble
from utils.base import set_random_seed
from utils.fmt.base import sys_open
from utils.fmt.base4torch import parse_cuda_decode
from utils.fmt.vocab.base import reverse_dict
from utils.fmt.vocab.token import ldvocab
from utils.h5serial import h5File
from utils.io import load_model_cpu
from utils.norm.mp.f import convert as make_mp_model
from utils.torch.comp import torch_autocast, torch_compile, torch_inference_mode
from utils.tqdm import tqdm

import cnfg.docpara as cnfg
from cnfg.ihyp import *
from cnfg.vocab.base import eos_id

def load_fixing(module):

	if hasattr(module, "fix_load"):
		module.fix_load()

use_cuda, cuda_device, cuda_devices, multi_gpu, use_amp, use_cuda_bfmp = parse_cuda_decode(cnfg.use_cuda, gpuid=cnfg.gpuid, use_amp=cnfg.use_amp, multi_gpu_decoding=cnfg.multi_gpu_decoding, use_cuda_bfmp=cnfg.use_cuda_bfmp)
set_random_seed(cnfg.seed, use_cuda)

td = h5File(cnfg.test_data, "r", **h5_fileargs)

tl = [(str(nsent), str(_curd),) for nsent, ndata in zip(td["nsent"][()].tolist(), td["ndata"][()].tolist()) for _curd in range(ndata)]
nwordi = td["nword"][()].tolist()[0]
vcbt, nwordt = ldvocab(sys.argv[2])
vcbt = reverse_dict(vcbt)

if len(sys.argv) == 4:
	mymodel = NMT(cnfg.isize, nwordi, nwordt, cnfg.nlayer, fhsize=cnfg.ff_hsize, dropout=cnfg.drop, attn_drop=cnfg.attn_drop, act_drop=cnfg.act_drop, global_emb=cnfg.share_emb, num_head=cnfg.nhead, xseql=cache_len_default, ahsize=cnfg.attn_hsize, norm_output=cnfg.norm_output, bindDecoderEmb=cnfg.bindDecoderEmb, forbidden_index=cnfg.forbidden_indexes, nprev_context=cnfg.num_prev_sent, num_layer_context=cnfg.num_layer_context)
	if use_cuda_bfmp:
		make_mp_model(mymodel)

	mymodel = load_model_cpu(sys.argv[3], mymodel)
	mymodel.apply(load_fixing)

else:
	models = []
	for modelf in sys.argv[3:]:
		tmp = NMT(cnfg.isize, nwordi, nwordt, cnfg.nlayer, fhsize=cnfg.ff_hsize, dropout=cnfg.drop, attn_drop=cnfg.attn_drop, act_drop=cnfg.act_drop, global_emb=cnfg.share_emb, num_head=cnfg.nhead, xseql=cache_len_default, ahsize=cnfg.attn_hsize, norm_output=cnfg.norm_output, bindDecoderEmb=cnfg.bindDecoderEmb, forbidden_index=cnfg.forbidden_indexes, nprev_context=cnfg.num_prev_sent, num_layer_context=cnfg.num_layer_context)
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

#num_prev_sent = cnfg.num_prev_sent
beam_size = cnfg.beam_size
length_penalty = cnfg.length_penalty

ens = "\n".encode("utf-8")
ens_skip = "\n".encode("utf-8")#.join(["\n" for i in range(num_prev_sent)])

src_grp = td["src"]
with sys_open(sys.argv[1], "wb") as f, torch_inference_mode():
	for nsent, i_d in tqdm(tl, mininterval=tqdm_mininterval):
		seq_batch = torch.from_numpy(src_grp[nsent][i_d][()])
		if cuda_device:
			seq_batch = seq_batch.to(cuda_device, non_blocking=True)
		seq_batch = seq_batch.to(torch.int64, non_blocking=True)
		bsize, _nsent, seql = seq_batch.size()
		_nsent_use = _nsent - 1
		with torch_autocast(enabled=use_amp):
			output = mymodel.decode(seq_batch.narrow(1, 1, _nsent_use).contiguous(), seq_batch.narrow(1, 0, _nsent_use).contiguous(), beam_size, None, length_penalty).view(bsize, _nsent_use, -1)
		if multi_gpu:
			tmp = []
			for ou in output:
				tmp.extend(ou.tolist())
			output = tmp
		else:
			output = output.tolist()
		for doc in output:
			f.write(ens_skip)
			for tran in doc:
				tmp = []
				for tmpu in tran:
					if tmpu == eos_id:
						break
					else:
						tmp.append(vcbt[tmpu])
				f.write(" ".join(tmp).encode("utf-8"))
				f.write(ens)
			f.write(ens)

td.close()
